from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from app.utils.logger import logger

class EmbeddingManager:
    """
    Manages the loading and instantiation of the SentenceTransformers embedding model.
    Implements a thread-safe singleton pattern to ensure the model is loaded into memory only once.
    """
    _embeddings_instance: Optional[HuggingFaceEmbeddings] = None

    @classmethod
    def get_embeddings(cls) -> HuggingFaceEmbeddings:
        """
        Loads and returns the configured SentenceTransformers embedding model.
        Uses normalized embeddings (L2 normalization) so that simple dot product is equivalent to cosine similarity.

        Returns:
            HuggingFaceEmbeddings: The initialized embedding model object.
        """
        if cls._embeddings_instance is None:
            logger.info(f"Initializing embedding model: '{settings.EMBEDDING_MODEL}'")
            try:
                # Dynamic device configuration: use CUDA/GPU if available, fallback to CPU
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Embedding generation device selected: '{device}'")
                
                cls._embeddings_instance = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True}  # Standardizes vectors for similarity search
                )
                logger.info("Embedding model loaded and ready.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{settings.EMBEDDING_MODEL}': {e}")
                raise RuntimeError(f"Could not load SentenceTransformers model: {str(e)}") from e
                
        return cls._embeddings_instance
