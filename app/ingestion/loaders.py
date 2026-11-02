"""Specialized file loaders for PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, and HTML."""

from abc import ABC, abstractmethod
import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import pandas as pd
from bs4 import BeautifulSoup
import openpyxl
from docx import Document as DocxDocument
from pptx import Presentation
import pymupdf

from app.core.logging import logger
from app.core.exceptions import DocumentExtractionError
from app.ingestion.text_cleaner import TextCleaner
from app.ingestion.ocr_processor import OCRProcessor


@dataclass
class DocumentSection:
    """Represents an extracted structural slice of a document."""
    text: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for document format extractors."""

    @abstractmethod
    def load(self, file_path: Path) -> List[DocumentSection]:
        """Extract text and metadata sections from the given file."""
        pass


class PDFLoader(BaseLoader):
    """Extracts text, page numbers, and tables from PDF documents using PyMuPDF."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            doc = pymupdf.open(str(file_path))
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]

                # 1. Extract standard text
                page_text = page.get_text("text")

                # 2. Extract tables if available
                table_texts = []
                try:
                    tabs = page.find_tables()
                    for table in tabs:
                        df = table.to_pandas()
                        if not df.empty:
                            table_texts.append(df.to_markdown(index=False))
                except Exception as tab_err:
                    logger.debug("Table extraction skipped for page %d: %s", page_num, tab_err)

                # Combine text and table markdown
                combined_content = page_text
                if table_texts:
                    combined_content += "\n\n### Extracted Tables:\n" + "\n\n".join(table_texts)

                cleaned = TextCleaner.clean(combined_content)

                # 3. Optional OCR if page has minimal text and OCR is available
                if len(cleaned.strip()) < 30 and OCRProcessor.is_available():
                    pix = page.get_pixmap()
                    from PIL import Image
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = OCRProcessor.extract_text_from_image(img)
                    if ocr_text:
                        cleaned = TextCleaner.clean(ocr_text)
                        logger.info("OCR recovered %d characters on PDF page %d", len(cleaned), page_num)

                if cleaned:
                    sections.append(
                        DocumentSection(
                            text=cleaned,
                            page_number=page_num,
                            metadata={"page_count": len(doc)}
                        )
                    )
            doc.close()
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract PDF ({file_path.name}): {exc}") from exc

        return sections


class DocxLoader(BaseLoader):
    """Extracts paragraphs, headings, and tables from DOCX files."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            doc = DocxDocument(str(file_path))
            current_heading = "General"
            current_buffer: List[str] = []

            for paragraph in doc.paragraphs:
                p_text = paragraph.text.strip()
                if not p_text:
                    continue

                if paragraph.style.name.startswith("Heading"):
                    if current_buffer:
                        text_block = TextCleaner.clean("\n".join(current_buffer))
                        if text_block:
                            sections.append(DocumentSection(text=text_block, section_title=current_heading))
                        current_buffer = []
                    current_heading = p_text
                else:
                    current_buffer.append(p_text)

            if current_buffer:
                text_block = TextCleaner.clean("\n".join(current_buffer))
                if text_block:
                    sections.append(DocumentSection(text=text_block, section_title=current_heading))

            # Extract tables
            for t_idx, table in enumerate(doc.tables, start=1):
                rows_data = []
                for row in table.rows:
                    row_cells = [TextCleaner.clean(cell.text, preserve_layout=False) for cell in row.cells]
                    rows_data.append(row_cells)

                if rows_data and len(rows_data) > 1:
                    df = pd.DataFrame(rows_data[1:], columns=rows_data[0])
                    table_md = df.to_markdown(index=False)
                    sections.append(
                        DocumentSection(
                            text=f"Table {t_idx}:\n{table_md}",
                            section_title=f"Table {t_idx}",
                            metadata={"is_table": True}
                        )
                    )
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract DOCX ({file_path.name}): {exc}") from exc

        return sections


class PptxLoader(BaseLoader):
    """Extracts slides, headings, and tables from PPTX presentations."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            prs = Presentation(str(file_path))
            for slide_idx, slide in enumerate(prs.slides, start=1):
                slide_title = f"Slide {slide_idx}"
                slide_texts: List[str] = []

                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for paragraph in shape.text_frame.paragraphs:
                            line = paragraph.text.strip()
                            if line:
                                slide_texts.append(line)
                        if shape == slide.shapes.title and shape.text.strip():
                            slide_title = shape.text.strip()

                    elif shape.has_table:
                        table_data = []
                        for row in shape.table.rows:
                            row_vals = [TextCleaner.clean(cell.text, preserve_layout=False) for cell in row.cells]
                            table_data.append(row_vals)
                        if table_data and len(table_data) > 1:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            slide_texts.append(df.to_markdown(index=False))

                if slide_texts:
                    clean_text = TextCleaner.clean("\n".join(slide_texts))
                    sections.append(
                        DocumentSection(
                            text=clean_text,
                            slide_number=slide_idx,
                            section_title=slide_title,
                            metadata={"slide_count": len(prs.slides)}
                        )
                    )
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract PPTX ({file_path.name}): {exc}") from exc

        return sections


class ExcelLoader(BaseLoader):
    """Extracts worksheets, columns, rows, and relationships from XLSX files."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            wb = openpyxl.load_workbook(str(file_path), data_only=True)
            for sheetname in wb.sheetnames:
                sheet = wb[sheetname]
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    continue

                # Filter out completely empty rows
                non_empty_rows = [
                    [str(c).strip() if c is not None else "" for c in r]
                    for r in rows if any(c is not None and str(c).strip() != "" for c in r)
                ]

                if not non_empty_rows:
                    continue

                headers = non_empty_rows[0]
                data_rows = non_empty_rows[1:]

                # Create structured representation preserving header-value associations
                formatted_lines = [f"### Worksheet: {sheetname}"]
                for row_idx, r in enumerate(data_rows, start=2):
                    pairs = [f"{h}: {val}" for h, val in zip(headers, r) if val]
                    if pairs:
                        formatted_lines.append(f"Row {row_idx}: " + ", ".join(pairs))

                # Also provide markdown table format if rows are reasonable
                if len(data_rows) <= 200:
                    try:
                        df = pd.DataFrame(data_rows, columns=headers)
                        formatted_lines.append("\n" + df.to_markdown(index=False))
                    except Exception:
                        pass

                text_content = TextCleaner.clean("\n".join(formatted_lines))
                if text_content:
                    sections.append(
                        DocumentSection(
                            text=text_content,
                            sheet_name=sheetname,
                            section_title=f"Sheet: {sheetname}",
                            metadata={"row_count": len(data_rows)}
                        )
                    )
            wb.close()
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract XLSX ({file_path.name}): {exc}") from exc

        return sections


class CSVLoader(BaseLoader):
    """Extracts structured tabular data from CSV files."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            # Sniff delimiter
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(4096)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    sep = dialect.delimiter
                except Exception:
                    sep = ","

            df = pd.read_csv(file_path, sep=sep, encoding="utf-8", on_bad_lines="skip")
            headers = [str(c) for c in df.columns]

            formatted_lines = [f"File: {file_path.name} ({len(df)} records)"]
            for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
                pairs = [f"{h}: {row[h]}" for h in headers if pd.notna(row[h])]
                if pairs:
                    formatted_lines.append(f"Record {row_idx}: " + ", ".join(pairs))

            # Add markdown table overview
            if len(df) <= 150:
                formatted_lines.append("\n" + df.to_markdown(index=False))

            cleaned = TextCleaner.clean("\n".join(formatted_lines))
            if cleaned:
                sections.append(
                    DocumentSection(
                        text=cleaned,
                        section_title=file_path.stem,
                        metadata={"record_count": len(df)}
                    )
                )
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract CSV ({file_path.name}): {exc}") from exc

        return sections


class TextLoader(BaseLoader):
    """Extracts plain text with encoding detection."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None

        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if content is None:
            raise DocumentExtractionError(f"Could not decode text file {file_path.name} with standard encodings.")

        cleaned = TextCleaner.clean(content)
        if cleaned:
            sections.append(DocumentSection(text=cleaned, section_title="Content"))

        return sections


class MarkdownLoader(BaseLoader):
    """Extracts Markdown documents preserving headers and hierarchical structure."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            current_section = file_path.stem
            current_buffer: List[str] = []

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    if current_buffer:
                        text_block = TextCleaner.clean("".join(current_buffer))
                        if text_block:
                            sections.append(DocumentSection(text=text_block, section_title=current_section))
                        current_buffer = []
                    current_section = stripped.lstrip("#").strip()
                current_buffer.append(line)

            if current_buffer:
                text_block = TextCleaner.clean("".join(current_buffer))
                if text_block:
                    sections.append(DocumentSection(text=text_block, section_title=current_section))
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract Markdown ({file_path.name}): {exc}") from exc

        return sections


class HTMLLoader(BaseLoader):
    """Extracts meaningful page content and headings from HTML documents."""

    def load(self, file_path: Path) -> List[DocumentSection]:
        sections: List[DocumentSection] = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove scripts, styles, navigations
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()

            page_title = soup.title.string.strip() if soup.title and soup.title.string else file_path.stem
            current_heading = page_title
            current_buffer: List[str] = []

            for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "table", "ul", "ol"]):
                if elem.name in ["h1", "h2", "h3", "h4"]:
                    if current_buffer:
                        cleaned = TextCleaner.clean("\n".join(current_buffer))
                        if cleaned:
                            sections.append(DocumentSection(text=cleaned, section_title=current_heading))
                        current_buffer = []
                    current_heading = elem.get_text().strip()
                elif elem.name == "table":
                    rows_data = []
                    for row in elem.find_all("tr"):
                        cells = [TextCleaner.clean(c.get_text(), preserve_layout=False) for c in row.find_all(["td", "th"])]
                        if cells:
                            rows_data.append(cells)
                    if rows_data and len(rows_data) > 1:
                        try:
                            df = pd.DataFrame(rows_data[1:], columns=rows_data[0])
                            current_buffer.append(df.to_markdown(index=False))
                        except Exception:
                            current_buffer.append("\n".join(["\t".join(r) for r in rows_data]))
                else:
                    text = elem.get_text().strip()
                    if text:
                        current_buffer.append(text)

            if current_buffer:
                cleaned = TextCleaner.clean("\n".join(current_buffer))
                if cleaned:
                    sections.append(DocumentSection(text=cleaned, section_title=current_heading))
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to extract HTML ({file_path.name}): {exc}") from exc

        return sections
