"""Unit tests for Multimodal Vision Ingestion (diagrams, charts, and figures)."""

import io
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw
import pytest

from app.core.config import settings
from app.ingestion.vision_processor import VisionProcessor, DEFAULT_VISION_PROMPT
from app.ingestion.loaders import PDFLoader, PptxLoader, DocxLoader, DocumentSection


def create_test_image(width: int = 300, height: int = 200, color: str = "blue") -> bytes:
    """Helper to generate in-memory PNG bytes with visual content."""
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 5, width - 5, height - 5], fill="white", outline="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_vision():
    VisionProcessor.reset_state()
    original_enabled = settings.ENABLE_VISION
    original_model = settings.VISION_MODEL
    yield
    settings.ENABLE_VISION = original_enabled
    settings.VISION_MODEL = original_model
    VisionProcessor.reset_state()


def test_vision_disabled_by_default():
    settings.ENABLE_VISION = False
    VisionProcessor.reset_state()
    assert VisionProcessor.is_available() is False


def test_is_meaningful_image_filters_tiny_or_extreme_aspect():
    settings.VISION_MIN_IMAGE_SIZE = 500
    settings.VISION_MIN_DIMENSION = 100

    # 50x50 is below MIN_DIMENSION (100)
    small_bytes = create_test_image(50, 50)
    assert VisionProcessor.is_meaningful_image(small_bytes) is False

    # Extreme aspect ratio (500x10 = 50:1)
    divider_bytes = create_test_image(500, 10)
    assert VisionProcessor.is_meaningful_image(divider_bytes) is False

    # Normal chart/diagram image (600x400)
    valid_bytes = create_test_image(600, 400)
    assert len(valid_bytes) > 500
    assert VisionProcessor.is_meaningful_image(valid_bytes) is True


def test_prepare_image_filters_solid_color_masks():
    settings.VISION_MIN_IMAGE_SIZE = 50
    settings.VISION_MIN_DIMENSION = 50

    # Solid black 1-bit / grayscale mask
    solid_img = Image.new("L", (200, 200), color=0)
    buf = io.BytesIO()
    solid_img.save(buf, format="PNG")
    assert VisionProcessor.prepare_image_for_vision(buf.getvalue()) is None

    # Solid color RGB
    solid_rgb = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf_rgb = io.BytesIO()
    solid_rgb.save(buf_rgb, format="PNG")
    assert VisionProcessor.prepare_image_for_vision(buf_rgb.getvalue()) is None


def test_prepare_image_normalizes_cmyk_to_rgb():
    settings.VISION_MIN_IMAGE_SIZE = 50
    settings.VISION_MIN_DIMENSION = 50

    # Create CMYK image with content
    cmyk_img = Image.new("CMYK", (200, 200), color=(100, 50, 0, 0))
    draw = ImageDraw.Draw(cmyk_img)
    draw.rectangle([20, 20, 180, 180], fill=(0, 100, 100, 0))
    buf = io.BytesIO()
    cmyk_img.save(buf, format="JPEG")
    cmyk_bytes = buf.getvalue()

    prepared = VisionProcessor.prepare_image_for_vision(cmyk_bytes)
    assert prepared is not None
    # Verify the output is standard RGB PNG
    with Image.open(io.BytesIO(prepared)) as res_img:
        assert res_img.format == "PNG"
        assert res_img.mode == "RGB"


def test_describe_image_with_mock_client():
    settings.ENABLE_VISION = True
    settings.VISION_MIN_IMAGE_SIZE = 100  # Lower threshold for test image
    settings.VISION_MIN_DIMENSION = 50

    img_bytes = create_test_image(300, 200, color="green")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.response = (
        "Bar Chart showing quarterly solar installations from 2020 to 2024. "
        "X-axis: Years, Y-axis: Megawatts. Shows an upward trend reaching 450MW in 2024."
    )
    mock_client.generate.return_value = mock_resp

    desc = VisionProcessor.describe_image(img_bytes, custom_client=mock_client)
    assert desc is not None
    assert "Bar Chart" in desc
    assert "450MW" in desc
    mock_client.generate.assert_called_once()

    # Second call should hit sha256 cache without calling mock_client again
    desc_cached = VisionProcessor.describe_image(img_bytes, custom_client=mock_client)
    assert desc_cached == desc
    assert mock_client.generate.call_count == 1


def test_describe_image_skips_logos_and_decorative():
    settings.ENABLE_VISION = True
    settings.VISION_MIN_IMAGE_SIZE = 100
    settings.VISION_MIN_DIMENSION = 50

    img_bytes = create_test_image(200, 200, color="red")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.response = "SKIP: Decorative corporate logo."
    mock_client.generate.return_value = mock_resp

    desc = VisionProcessor.describe_image(img_bytes, custom_client=mock_client)
    assert desc is None


def test_pdf_loader_integrates_vision_descriptions(tmp_path):
    import pymupdf

    pdf_file = tmp_path / "diagram_sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((50, 50), "Figure 1 below illustrates the workflow architecture.")

    # Insert a dummy image
    img_data = create_test_image(300, 200, color="orange")
    rect = pymupdf.Rect(50, 100, 350, 300)
    page.insert_image(rect, stream=img_data)
    doc.save(str(pdf_file))
    doc.close()

    # Test with vision enabled and mocked
    with patch.object(VisionProcessor, "is_available", return_value=True), \
         patch.object(VisionProcessor, "describe_image", return_value="Flowchart diagram detailing the 4-stage ingestion pipeline."):
        loader = PDFLoader()
        sections = loader.load(pdf_file)

        assert len(sections) == 1
        sec = sections[0]
        assert "Figure 1 below illustrates the workflow" in sec.text
        assert "### Extracted Diagrams & Visual Charts:" in sec.text
        assert "Flowchart diagram detailing the 4-stage ingestion pipeline." in sec.text
        assert sec.metadata.get("has_diagrams") is True
        assert sec.metadata.get("diagram_count") == 1
