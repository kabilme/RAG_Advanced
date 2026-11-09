"""Embeddable chatbot widget endpoints: script delivery, iframe embedding, and live showcase."""

from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.api.dependencies import get_db_session, get_ollama_client
from app.database.repository import DocumentRepository

router = APIRouter(prefix="/widget", tags=["Embeddable Widget"])

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@router.get("/chat-widget.js")
def get_widget_script():
    """Serves the standalone embeddable chat widget JavaScript file."""
    script_path = STATIC_DIR / "chat-widget.js"
    if not script_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat widget script file not found.",
        )
    return FileResponse(
        path=script_path,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/embed", response_class=HTMLResponse)
def get_embed_page():
    """Serves the responsive full-page chat interface suitable for iframe embeds."""
    embed_path = STATIC_DIR / "embed.html"
    if not embed_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Embed HTML page not found.",
        )
    return FileResponse(
        path=embed_path,
        media_type="text/html",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/demo", response_class=HTMLResponse)
def get_demo_studio_page():
    """Serves the interactive live customizer showcase and embed code generator."""
    demo_path = STATIC_DIR / "demo.html"
    if not demo_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Demo studio HTML page not found.",
        )
    return FileResponse(
        path=demo_path,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/info")
def get_widget_info(
    request: Request,
    ollama_client=Depends(get_ollama_client),
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Returns widget configuration details, available documents, and embed snippets."""
    doc_repo = DocumentRepository(db)
    docs = doc_repo.list_all()

    base_url = str(request.base_url).rstrip("/")
    if "127.0.0.1" in base_url or "localhost" in base_url:
        base_url = f"http://127.0.0.1:{settings.SERVER_PORT}"

    sample_suggestions = [
        "What are the main documents?",
        "Summarize the key findings",
        "Explain key definitions",
    ]
    if docs:
        first_doc = docs[0].filename
        sample_suggestions.insert(0, f"What does {first_doc} say?")

    script_snippet = (
        f'<script\n'
        f'  src="{base_url}/static/chat-widget.js"\n'
        f'  data-api-url="{base_url}"\n'
        f'  data-title="Document Assistant"\n'
        f'  data-primary-color="#0284c7"\n'
        f'  data-position="bottom-right"\n'
        f'  defer>\n'
        f'</script>'
    )

    iframe_snippet = (
        f'<iframe\n'
        f'  src="{base_url}/widget/embed?api={base_url}&title=Document+Assistant"\n'
        f'  width="100%"\n'
        f'  height="600"\n'
        f'  frameborder="0"\n'
        f'  style="border-radius: 16px; max-width: 420px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);"\n'
        f'  title="AI Document Assistant">\n'
        f'</iframe>'
    )

    return {
        "status": "online",
        "api_url": base_url,
        "llm_model": settings.LLM_MODEL,
        "total_documents": len(docs),
        "available_documents": [d.filename for d in docs[:10]],
        "suggested_prompts": sample_suggestions[:4],
        "embed_script_tag": script_snippet,
        "embed_iframe_tag": iframe_snippet,
        "demo_studio_url": f"{base_url}/widget/demo",
        "direct_embed_url": f"{base_url}/widget/embed",
    }
