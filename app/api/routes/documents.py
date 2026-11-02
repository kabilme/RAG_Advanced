"""Document management endpoints: upload, list, delete, and re-indexing."""

import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import DocumentValidationError, DocumentExtractionError
from app.api.schemas import DocumentResponse, DocumentListResponse
from app.api.dependencies import (
    get_ingestion_engine,
    get_embedding_service,
    get_vector_store,
    get_keyword_engine,
    get_db_session,
)
from app.database.repository import DocumentRepository

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=List[DocumentResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    ingestion_engine=Depends(get_ingestion_engine),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    keyword_engine=Depends(get_keyword_engine),
    db: Session = Depends(get_db_session),
):
    """Uploads, validates, parses, chunks, and indexes one or more documents."""
    doc_repo = DocumentRepository(db)
    results: List[DocumentResponse] = []
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload_file in files:
        safe_filename = Path(upload_file.filename).name
        target_path = upload_dir / safe_filename

        # Save uploaded bytes to disk
        try:
            content = await upload_file.read()
            target_path.write_bytes(content)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to write file {safe_filename} to disk: {exc}"
            )

        # Ingestion pipeline: validation, extraction, chunking
        try:
            ingestion_result = ingestion_engine.ingest_file(target_path)
        except DocumentValidationError as val_err:
            if target_path.exists():
                target_path.unlink()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except DocumentExtractionError as ext_err:
            if target_path.exists():
                target_path.unlink()
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ext_err))

        # Generate embeddings for the chunks
        chunk_texts = [c.text for c in ingestion_result.chunks]
        embeddings = embedding_service.embed_documents(chunk_texts)

        # Index into ChromaDB vector store
        vector_store.add_chunks(ingestion_result.chunks, embeddings)

        # Index into BM25 lexical keyword engine
        keyword_engine.index_chunks(ingestion_result.chunks)

        # Persist document metadata in SQLite
        doc_record = doc_repo.upsert(
            document_id=ingestion_result.document_id,
            filename=ingestion_result.filename,
            file_type=ingestion_result.file_type,
            file_hash=ingestion_result.document_hash,
            file_size_bytes=ingestion_result.file_size_bytes,
            chunk_count=len(ingestion_result.chunks),
            status="indexed",
        )

        results.append(DocumentResponse(**doc_record.to_dict()))

    return results


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db_session)):
    """Lists all indexed documents and their chunk counts."""
    doc_repo = DocumentRepository(db)
    docs = doc_repo.list_all()
    items = [DocumentResponse(**d.to_dict()) for d in docs]
    return DocumentListResponse(documents=items, total_count=len(items))


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db_session)):
    """Retrieves metadata for a specific document."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found.")
    return DocumentResponse(**doc.to_dict())


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    vector_store=Depends(get_vector_store),
    keyword_engine=Depends(get_keyword_engine),
    db: Session = Depends(get_db_session),
):
    """Deletes a document from the vector store, BM25 index, database, and filesystem."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found.")

    # 1. Remove from ChromaDB
    deleted_vectors = vector_store.delete_document(document_id)

    # 2. Remove from BM25 index
    keyword_engine.remove_document(document_id)

    # 3. Remove uploaded file from disk if present
    file_on_disk = Path(settings.UPLOAD_DIR) / doc.filename
    if file_on_disk.exists():
        try:
            file_on_disk.unlink()
        except Exception as exc:
            logger.warning("Could not delete file %s from disk: %s", file_on_disk, exc)

    # 4. Remove from SQLite
    doc_repo.delete(document_id)

    return {
        "message": f"Successfully deleted document '{doc.filename}'",
        "document_id": document_id,
        "deleted_chunks": deleted_vectors,
    }


@router.post("/{document_id}/reindex", response_model=DocumentResponse)
def reindex_document(
    document_id: str,
    ingestion_engine=Depends(get_ingestion_engine),
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    keyword_engine=Depends(get_keyword_engine),
    db: Session = Depends(get_db_session),
):
    """Re-indexes an existing document from disk to regenerate chunks and embeddings."""
    doc_repo = DocumentRepository(db)
    doc = doc_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found.")

    file_path = Path(settings.UPLOAD_DIR) / doc.filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original document file '{doc.filename}' not found in upload directory."
        )

    # Remove previous chunks
    vector_store.delete_document(document_id)
    keyword_engine.remove_document(document_id)

    # Re-extract and re-chunk
    result = ingestion_engine.ingest_file(file_path, custom_document_id=document_id)
    chunk_texts = [c.text for c in result.chunks]
    embeddings = embedding_service.embed_documents(chunk_texts)

    # Re-add to vector store and keyword index
    vector_store.add_chunks(result.chunks, embeddings)
    keyword_engine.index_chunks(result.chunks)

    # Update database record
    updated_rec = doc_repo.upsert(
        document_id=document_id,
        filename=result.filename,
        file_type=result.file_type,
        file_hash=result.document_hash,
        file_size_bytes=result.file_size_bytes,
        chunk_count=len(result.chunks),
        status="re-indexed",
    )

    return DocumentResponse(**updated_rec.to_dict())
