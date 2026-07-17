import hashlib
import datetime
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.config import settings
from app.rag.embeddings import EmbeddingManager
from app.utils.logger import logger

class VectorStoreManager:
    """
    Manages vector storage interactions with ChromaDB.
    Supports persistent storage, duplicate document detection via SHA-256 hashing,
    idempotent ingestion pipelines, metadata filtering, and document deletion.
    """
    
    def __init__(self):
        """Initializes ChromaDB persistent client and links the LangChain Chroma wrapper."""
        try:
            logger.info(f"Initializing persistent ChromaDB client at: '{settings.CHROMA_DB_PATH}'")
            self.embeddings = EmbeddingManager.get_embeddings()
            
            # Persistent embedded client
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.collection_name = "rag_documents"
            
            # Setup collection using Cosine Similarity space metrics
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # LangChain wrapper synchronized with the same client connection
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info("ChromaDB vector store connection initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB connection: {e}")
            raise RuntimeError(f"ChromaDB connection failure: {str(e)}") from e

    def compute_sha256(self, file_bytes: bytes) -> str:
        """
        Computes the SHA-256 hash of a file's byte stream. Used for deduplication.

        Args:
            file_bytes (bytes): Binary contents of the document.

        Returns:
            str: SHA-256 hex string.
        """
        sha256_hash = hashlib.sha256()
        # Process in chunks of 64KB to handle memory efficiently if large files are uploaded
        chunk_size = 65536
        for i in range(0, len(file_bytes), chunk_size):
            sha256_hash.update(file_bytes[i:i + chunk_size])
        return sha256_hash.hexdigest()

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Checks if a file hash is already stored and returns its metadata if found.

        Args:
            file_hash (str): The SHA-256 hash value to check.

        Returns:
            Optional[Dict[str, Any]]: Document details (document_id, filename, total_chunks) if found, else None.
        """
        try:
            results = self.collection.get(
                where={"file_hash": file_hash},
                limit=1,
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            if metadatas and len(metadatas) > 0:
                meta = metadatas[0]
                return {
                    "document_id": meta.get("document_id"),
                    "filename": meta.get("filename"),
                    "file_hash": meta.get("file_hash"),
                    "total_chunks": meta.get("total_chunks")
                }
        except Exception as e:
            logger.error(f"Error querying collection by file hash '{file_hash}': {e}")
        return None

    def add_documents(self, documents: List[Document], file_hash: str, filename: str) -> str:
        """
        Ingests split document chunks into ChromaDB.
        Annotates chunks with global document metadata (hash, filename, upload time).

        Args:
            documents (List[Document]): Split text chunks.
            file_hash (str): SHA-256 file hash.
            filename (str): Base filename of the document.

        Returns:
            str: Unique document ID (recycles the file_hash).
        """
        document_id = file_hash  # Use hash directly to enforce uniqueness and deterministic routing
        upload_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        ids = []
        texts = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            chunk_id = f"{document_id}_chunk_{idx}"
            ids.append(chunk_id)
            texts.append(doc.page_content)
            
            # Merge text splitter page metadata with global ingestion metadata
            meta = doc.metadata.copy()
            meta.update({
                "document_id": document_id,
                "file_hash": file_hash,
                "filename": filename,
                "upload_time": upload_time,
                "chunk_index": idx,
                "total_chunks": len(documents)
            })
            metadatas.append(meta)

        logger.info(f"Ingesting {len(texts)} chunks from document '{filename}' (ID: {document_id})")
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Finished ingesting document '{filename}' into vector store.")
            return document_id
        except Exception as e:
            logger.error(f"Ingestion failed for document '{filename}': {e}")
            raise ValueError(f"Ingestion error writing to vector store: {str(e)}") from e

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Returns a list of unique ingested documents.
        Deduplicates chunks stored in ChromaDB in-memory.

        Returns:
            List[Dict[str, Any]]: List of document details dicts.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", []) or []
            
            unique_docs: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                doc_id = meta.get("document_id")
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": meta.get("filename"),
                        "file_hash": meta.get("file_hash"),
                        "total_chunks": meta.get("total_chunks"),
                        "upload_time": meta.get("upload_time")
                    }
            return list(unique_docs.values())
        except Exception as e:
            logger.error(f"Failed to query all documents from vector store: {e}")
            raise RuntimeError(f"Vector store query failure: {str(e)}") from e

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes all vector store entries and text chunks linked to a document ID.

        Args:
            document_id (str): Unique identifier of the document.

        Returns:
            bool: True if chunks were found and deleted, False if document ID not found.
        """
        try:
            # Check existence first
            check_results = self.collection.get(
                where={"document_id": document_id},
                limit=1,
                include=["metadatas"]
            )
            ids = check_results.get("ids", [])
            if not ids or len(ids) == 0:
                logger.warning(f"Document ID '{document_id}' not found. Deletion skipped.")
                return False
            
            logger.info(f"Removing all chunks for document ID: '{document_id}'")
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Successfully deleted document ID '{document_id}' from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document ID '{document_id}': {e}")
            raise RuntimeError(f"Vector store deletion error: {str(e)}") from e
