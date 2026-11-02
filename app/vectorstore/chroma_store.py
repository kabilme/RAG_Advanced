"""Persistent ChromaDB vector store implementation."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import VectorStoreError
from app.ingestion.chunking import TextChunk
from app.vectorstore.base import BaseVectorStore, SearchResult


class ChromaVectorStore(BaseVectorStore):
    """Encapsulates persistent ChromaDB vector storage, indexing, and retrieval."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "default",
        embedding_dimension: int = 384,
    ):
        self._persist_dir = Path(persist_directory or settings.VECTOR_DB_PATH)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model_name = embedding_model_name
        self._dimension = embedding_dimension

        # Build clean collection name partitioned by model and dimension
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", embedding_model_name)
        self._collection_name = f"rag_{safe_model}_{embedding_dimension}"

        try:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "embedding_model": embedding_model_name,
                    "dimension": str(embedding_dimension),
                    "hnsw:space": "cosine",
                }
            )
            logger.info(
                "ChromaDB collection '%s' ready at: %s (Indexed items: %d)",
                self._collection_name, self._persist_dir, self._collection.count()
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB collection: {exc}") from exc

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        """Returns total vectors indexed in collection."""
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count ChromaDB items: {exc}") from exc

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> None:
        """Stores or updates chunks and their embeddings in ChromaDB."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})"
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            meta = chunk.to_dict()
            # Store parent_text in metadata if available for answer generation context
            if chunk.parent_text:
                meta["parent_text"] = chunk.parent_text
            metadatas.append(meta)

        try:
            # Upsert ensures idempotent indexing and prevents duplicate key errors
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Indexed %d chunks into ChromaDB collection '%s'", len(chunks), self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert chunks to ChromaDB: {exc}") from exc

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Queries ChromaDB for top-K nearest neighbors and normalizes cosine distance to similarity."""
        if not query_embedding:
            return []

        threshold = similarity_threshold if similarity_threshold is not None else 0.0

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, max(1, self.count())),
                "include": ["documents", "metadatas", "distances"],
            }
            if filter_criteria:
                query_kwargs["where"] = filter_criteria

            results = self._collection.query(**query_kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB similarity query failed: {exc}") from exc

        candidates: List[SearchResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return candidates

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # For cosine distance, similarity is max(0.0, 1.0 - distance)
            similarity = max(0.0, 1.0 - float(dist))

            if similarity < threshold:
                continue

            meta_dict = dict(meta) if meta else {}
            candidates.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(meta_dict.get("document_id", "")),
                    text=doc_text,
                    similarity_score=similarity,
                    filename=str(meta_dict.get("filename", "")),
                    page_number=meta_dict.get("page_number") if meta_dict.get("page_number", -1) != -1 else None,
                    slide_number=meta_dict.get("slide_number") if meta_dict.get("slide_number", -1) != -1 else None,
                    sheet_name=meta_dict.get("sheet_name") or None,
                    section=meta_dict.get("section") or None,
                    source=str(meta_dict.get("source", "")),
                    parent_chunk_id=meta_dict.get("parent_chunk_id") or None,
                    parent_text=meta_dict.get("parent_text") or None,
                    metadata=meta_dict,
                )
            )

        return candidates

    def delete_document(self, document_id: str) -> int:
        """Removes all chunks belonging to the given document identifier."""
        try:
            initial_count = self.count()
            self._collection.delete(where={"document_id": document_id})
            deleted = initial_count - self.count()
            logger.info("Deleted %d chunks for document %s from ChromaDB", deleted, document_id)
            return deleted
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document {document_id} from ChromaDB: {exc}") from exc

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieves raw chunks for a specific document."""
        try:
            results = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            items = []
            if results and results.get("ids"):
                for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                    items.append({"chunk_id": cid, "text": doc, "metadata": meta})
            return items
        except Exception as exc:
            raise VectorStoreError(f"Failed to get chunks for document {document_id}: {exc}") from exc

    def get_all_chunks(self) -> List[TextChunk]:
        """Retrieves all indexed chunks as TextChunk objects."""
        try:
            total = self.count()
            if total == 0:
                return []
            results = self._collection.get(include=["documents", "metadatas"])
            chunks = []
            if results and results.get("ids"):
                for cid, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                    meta_dict = dict(meta) if meta else {}
                    chunks.append(
                        TextChunk(
                            chunk_id=cid,
                            document_id=str(meta_dict.get("document_id", "")),
                            text=doc,
                            chunk_index=int(meta_dict.get("chunk_index", 0)),
                            filename=str(meta_dict.get("filename", "")),
                            file_type=str(meta_dict.get("file_type", "")),
                            document_hash=str(meta_dict.get("document_hash", "")),
                            page_number=meta_dict.get("page_number") if meta_dict.get("page_number", -1) != -1 else None,
                            slide_number=meta_dict.get("slide_number") if meta_dict.get("slide_number", -1) != -1 else None,
                            sheet_name=meta_dict.get("sheet_name") or None,
                            section=meta_dict.get("section") or None,
                            source=str(meta_dict.get("source", "")),
                            parent_chunk_id=meta_dict.get("parent_chunk_id") or None,
                            parent_text=meta_dict.get("parent_text") or None,
                            metadata=meta_dict,
                        )
                    )
            return chunks
        except Exception as exc:
            logger.warning("Failed to get all chunks from ChromaDB: %s", exc)
            return []

    def clear(self) -> None:
        """Clears all vectors in this collection."""
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={
                    "embedding_model": self._embedding_model_name,
                    "dimension": str(self._dimension),
                    "hnsw:space": "cosine",
                }
            )
            logger.info("Cleared and recreated ChromaDB collection '%s'", self._collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear ChromaDB collection: {exc}") from exc

    def backup_to_json(self, backup_file: Path) -> int:
        """Exports all collection records to a JSON backup file."""
        try:
            total = self.count()
            if total == 0:
                backup_file.write_text(json.dumps([]), encoding="utf-8")
                return 0

            records = self._collection.get(include=["documents", "metadatas", "embeddings"])
            data = []
            if records and records.get("ids"):
                for i in range(len(records["ids"])):
                    emb = records["embeddings"][i] if records.get("embeddings") is not None else []
                    if hasattr(emb, "tolist"):
                        emb_list = emb.tolist()
                    elif isinstance(emb, (list, tuple)):
                        emb_list = list(emb)
                    else:
                        emb_list = []

                    data.append({
                        "id": records["ids"][i],
                        "document": records["documents"][i],
                        "metadata": records["metadatas"][i],
                        "embedding": emb_list,
                    })
            backup_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info("Backed up %d vectors to %s", len(data), backup_file)
            return len(data)
        except Exception as exc:
            raise VectorStoreError(f"Failed to backup ChromaDB collection: {exc}") from exc

    def restore_from_json(self, backup_file: Path) -> int:
        """Restores collection records from a JSON backup file."""
        if not backup_file.exists():
            raise VectorStoreError(f"Backup file not found: {backup_file}")
        try:
            data = json.loads(backup_file.read_text(encoding="utf-8"))
            if not data:
                return 0

            ids = [item["id"] for item in data]
            docs = [item["document"] for item in data]
            metas = [item["metadata"] for item in data]
            embeddings = [item["embedding"] for item in data]

            self._collection.upsert(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeddings,
            )
            logger.info("Restored %d vectors from %s into %s", len(data), backup_file, self._collection_name)
            return len(data)
        except Exception as exc:
            raise VectorStoreError(f"Failed to restore ChromaDB collection: {exc}") from exc
