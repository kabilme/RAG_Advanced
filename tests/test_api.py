"""Integration tests for FastAPI REST API endpoints."""

import io
from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Local Advanced RAG System" in resp.json()["message"]


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "vector_store_type" in data
    assert "indexed_vectors" in data
    assert "embedding_dimension" in data


def test_documents_list_endpoint():
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "total_count" in data


def test_retrieval_search_endpoint():
    resp = client.post("/retrieval/search", json={"query": "test query", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "results" in data


def test_chat_query_endpoint():
    # If no documents are uploaded, it will cleanly abstain
    resp = client.post("/chat/query", json={"question": "What is the capital of Mars?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["is_abstained"] is True
    assert "The available documents do not contain sufficient information" in data["answer"]


def test_chat_sessions_endpoint():
    resp = client.get("/chat/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_settings_acceleration_configuration():
    from app.core.config import settings
    # Ensure predictable baseline for unit test regardless of prior manual UI toggles in .env
    settings.ENABLE_ACCELERATION = False

    # 1. GET /settings should return acceleration configuration
    resp = client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "enable_acceleration" in data
    assert "acceleration_device" in data
    assert "acceleration_threads" in data
    assert "gpu_available" in data
    # By default, CPU acceleration must not be automatically activated
    assert data["enable_acceleration"] is False

    # 2. POST /settings should allow manually activating acceleration with custom threads and device
    update_payload = {
        "enable_acceleration": True,
        "acceleration_device": "cpu",
        "acceleration_threads": 8,
    }
    update_resp = client.post("/settings", json=update_payload)
    assert update_resp.status_code == 200
    updated_data = update_resp.json()
    assert updated_data["enable_acceleration"] is True
    assert updated_data["acceleration_device"] == "cpu"
    assert updated_data["acceleration_threads"] == 8

    # 3. Disable acceleration again
    disable_resp = client.post("/settings", json={"enable_acceleration": False})
    assert disable_resp.status_code == 200
    assert disable_resp.json()["enable_acceleration"] is False


def test_responsive_web_ui_endpoints():
    """Verify that the responsive HTML, CSS, and JS frontend is served correctly."""
    # 1. GET /ui delivers the HTML Single-Page Application
    ui_resp = client.get("/ui")
    assert ui_resp.status_code == 200
    assert "text/html" in ui_resp.headers["content-type"]
    assert "Local RAG" in ui_resp.text
    assert "app.css" in ui_resp.text
    assert "app.js" in ui_resp.text

    # 2. Browser request to GET / with Accept: text/html delivers the web UI
    browser_resp = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert browser_resp.status_code == 200
    assert "text/html" in browser_resp.headers["content-type"]
    assert "Local RAG" in browser_resp.text

    # 3. Static assets are served cleanly
    css_resp = client.get("/static/css/app.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]

    js_resp = client.get("/static/js/app.js")
    assert js_resp.status_code == 200
    assert "javascript" in js_resp.headers["content-type"]


