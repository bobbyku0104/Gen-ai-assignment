import io
import os
from typing import List, Dict, Any
from pypdf import PdfReader
from app.utils.logger import logger

class PDFLoader:
    """
    Utility class for loading, reading, and extracting raw text from PDF files.
    Supports parsing from raw byte streams (FastAPI file uploads) and local filepaths.
    """

    @staticmethod
    def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses PDF binary content and extracts text from each page.

        Args:
            file_bytes (bytes): The raw file bytes.
            filename (str): The name of the file (used in metadata citations).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing pages.
                Each dictionary contains:
                - "text": Cleaned extracted page text (str)
                - "metadata": Dict containing "source" (filename) and "page" (1-indexed int)
        """
        pages_data: List[Dict[str, Any]] = []
        try:
            logger.info(f"Extracting text from uploaded PDF bytes: {filename}")
            pdf_stream = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)
            
            # Check if PDF is encrypted
            if reader.is_encrypted:
                logger.warning(f"PDF '{filename}' is encrypted. Attempting decryption with empty password.")
                try:
                    reader.decrypt("")
                except Exception as decrypt_err:
                    raise ValueError(f"Encrypted PDF '{filename}' requires a password: {decrypt_err}")

            total_pages = len(reader.pages)
            logger.info(f"PDF '{filename}' successfully read. Total pages: {total_pages}")
            
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # Standardize newline formatting slightly
                        cleaned_text = " ".join(text.split())
                        pages_data.append({
                            "text": cleaned_text,
                            "metadata": {
                                "source": filename,
                                "page": page_num
                            }
                        })
                    else:
                        logger.debug(f"Skipping empty or image-only page {page_num} in '{filename}'.")
                except Exception as page_err:
                    logger.error(f"Error parsing page {page_num} of '{filename}': {page_err}")
                    # Continue parsing other pages instead of completely failing
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to process PDF bytes for '{filename}': {e}")
            raise ValueError(f"Failed to parse PDF document '{filename}': {str(e)}") from e

        logger.info(f"Extracted {len(pages_data)} pages of text from '{filename}'")
        return pages_data

    @staticmethod
    def load_pdf_from_path(file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a local PDF file and extracts text page-by-page.

        Args:
            file_path (str): The absolute or relative filepath to the PDF.

        Returns:
            List[Dict[str, Any]]: List of page contents and metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")
        
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            return PDFLoader.load_pdf_from_bytes(file_bytes, filename)
        except Exception as e:
            logger.error(f"Failed to read PDF file from path '{file_path}': {e}")
            raise ValueError(f"Failed to load PDF from path '{file_path}': {str(e)}") from e
