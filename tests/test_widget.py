"""Unit and integration tests for the embeddable chatbot widget endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_widget_script_endpoint():
    """GET /widget/chat-widget.js should deliver valid standalone JavaScript."""
    resp = client.get("/widget/chat-widget.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    text = resp.text
    assert "RAGChatWidget" in text
    assert "attachShadow" in text
    assert "/chat/stream" in text
    assert "<h3>$1</h3>" in text
    assert "rag-msg-bubble h3" in text


def test_static_mounted_script():
    """GET /static/chat-widget.js should be accessible via static mount."""
    resp = client.get("/static/chat-widget.js")
    assert resp.status_code == 200
    assert "RAGChatWidget" in resp.text


def test_widget_embed_endpoint():
    """GET /widget/embed should deliver responsive standalone HTML chat interface."""
    resp = client.get("/widget/embed")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "rag-container" in resp.text
    assert "chatInput" in resp.text
    assert "<h3>$1</h3>" in resp.text
    assert "no-cache" in resp.headers["cache-control"]


def test_static_mounted_embed():
    """GET /static/embed.html should be accessible via static mount."""
    resp = client.get("/static/embed.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "rag-container" in resp.text


def test_widget_demo_endpoint():
    """GET /widget/demo should deliver interactive customizer showcase."""
    resp = client.get("/widget/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Chatbot Studio" in resp.text
    assert "Widget Customizer" in resp.text
    assert "Embed Code Generator" in resp.text


def test_widget_info_endpoint():
    """GET /widget/info should return JSON metadata and ready-to-paste embed tags."""
    resp = client.get("/widget/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "api_url" in data
    assert "embed_script_tag" in data
    assert "chat-widget.js" in data["embed_script_tag"]
    assert "embed_iframe_tag" in data
    assert "widget/embed" in data["embed_iframe_tag"]
    assert "demo_studio_url" in data


def test_root_includes_widget_links():
    """Root endpoint should link to widget demo and embed URLs."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "widget_studio_url" in data
    assert "widget_embed_url" in data

