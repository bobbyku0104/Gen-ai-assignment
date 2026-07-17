from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings
from app.utils.logger import logger

class DocumentChunker:
    """
    Responsible for splitting documents into smaller, overlapping text chunks.
    Uses RecursiveCharacterTextSplitter to ensure semantic boundaries (paragraphs, sentences) are respected.
    """

    @staticmethod
    def chunk_documents(
        pages_data: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Document]:
        """
        Splits page-level data into chunks while preserving parent metadata (e.g. source file, page number).

        Args:
            pages_data (List[Dict[str, Any]]): Extracted pages from PDFLoader.
            chunk_size (Optional[int]): Override for character chunk size. Defaults to settings.CHUNK_SIZE.
            chunk_overlap (Optional[int]): Override for overlap size. Defaults to settings.CHUNK_OVERLAP.

        Returns:
            List[Document]: LangChain Document objects ready for embedding and vector store ingestion.
        """
        # Fallback to configured default settings if parameters are omitted
        size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

        if overlap >= size:
            logger.warning(
                f"Overlap ({overlap}) is larger than or equal to chunk size ({size}). "
                "Adjusting overlap to 10% of chunk size to prevent split errors."
            )
            overlap = int(size * 0.1)

        logger.info(f"Splitting {len(pages_data)} pages of text: chunk_size={size}, overlap={overlap}")

        # Instantiate LangChain Documents from raw dictionaries
        documents = [
            Document(page_content=page["text"], metadata=page["metadata"])
            for page in pages_data
        ]

        # Set up recursive character splitter.
        # It attempts to split by paragraph ("\n\n"), then sentence ("\n"), then words (" "), and characters ("") in order.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        try:
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Completed splitting. Created {len(chunked_docs)} chunks from {len(pages_data)} pages.")
            return chunked_docs
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            raise ValueError(f"Failed during text splitting operation: {str(e)}") from e
