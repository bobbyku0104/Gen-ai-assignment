import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from app.config import settings
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import logger

class ContextRetriever:
    """
    Retrieves semantically similar document chunks from the vector store.
    Supports top-k limits, metadata filters, and logs retrieval metrics (latency, chunk count).
    """
    
    def __init__(self, db_manager: VectorStoreManager):
        """
        Initializes the retriever with a shared VectorStoreManager instance.

        Args:
            db_manager (VectorStoreManager): Shared vector database controller.
        """
        self.db_manager = db_manager

    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Performs semantic similarity search over stored chunks.

        Args:
            query (str): Search string.
            top_k (Optional[int]): Number of chunks to retrieve. Defaults to settings.TOP_K.
            metadata_filter (Optional[Dict[str, Any]]): Key-value pairs for metadata filtering.

        Returns:
            List[Document]: Top semantic matching document chunks with their original metadata.
        """
        start_time = time.perf_counter()
        k = top_k if top_k is not None else settings.TOP_K

        logger.info(f"Querying vector store for '{query}' (top_k={k}, filter={metadata_filter})")
        
        try:
            # Langchain's Chroma wrapper similarity search
            # 'filter' parameter is forwarded directly to ChromaDB's where clause
            results = self.db_manager.vector_store.similarity_search(
                query=query,
                k=k,
                filter=metadata_filter
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Retrieved {len(results)} chunks in {latency_ms:.2f} ms."
            )
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve context from vector store: {e}")
            raise RuntimeError(f"Database query failure: {str(e)}") from e
