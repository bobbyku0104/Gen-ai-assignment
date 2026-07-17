import time
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from app.models.schemas import (
    HealthResponse,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DeleteResponse,
    Citation
)
from app.rag.loader import PDFLoader
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import ContextRetriever
from app.rag.generator import AnswerGenerator
from app.utils.logger import logger

# Initialize Router
router = APIRouter()

# Initialize core RAG singletons to share across routes
try:
    db_manager = VectorStoreManager()
    retriever = ContextRetriever(db_manager)
    generator = AnswerGenerator()
except Exception as e:
    logger.critical(f"Failed to initialize core RAG components: {e}")
    # We raise the exception so FastAPI startup fails early instead of serving broken routes
    raise e

@router.get(
    "/",
    summary="Root Greeting",
    response_model=Dict[str, str]
)
async def root():
    """Welcome endpoint for the Cost-Efficient RAG API."""
    return {"message": "Welcome to the Cost-Efficient RAG Application API. Go to /docs for Swagger."}

@router.get(
    "/health",
    summary="Health Probe",
    response_model=HealthResponse
)
async def health_check():
    """Checks the database connection and system operational status."""
    db_connected = False
    try:
        # Basic check to see if database client can heartbeat or describe collections
        db_manager.client.heartbeat()
        db_connected = True
    except Exception as e:
        logger.error(f"Health check failed on ChromaDB: {e}")
        
    return HealthResponse(
        status="ok" if db_connected else "degraded",
        environment="production" if not db_connected else "running",
        database_connected=db_connected
    )

@router.post(
    "/upload",
    summary="Upload PDF Document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to parse and ingest"),
    chunk_size: Optional[int] = Form(None, description="Optional override for chunk character size"),
    chunk_overlap: Optional[int] = Form(None, description="Optional override for chunk overlap character size")
):
    """
    Parses and ingests a PDF document.
    Implements SHA-256 deduplication and idempotent routing.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    try:
        # Read file bytes in memory
        file_bytes = await file.read()
        
        # Step 1: Duplicate PDF detection via SHA-256
        file_hash = db_manager.compute_sha256(file_bytes)
        existing_doc = db_manager.get_document_by_hash(file_hash)
        
        if existing_doc:
            logger.info(f"Duplicate document upload blocked. File hash: {file_hash}")
            return UploadResponse(
                message="Document already ingested (Idempotency bypass).",
                document_id=existing_doc["document_id"],
                filename=existing_doc["filename"],
                total_chunks=existing_doc["total_chunks"],
                file_hash=existing_doc["file_hash"]
            )

        # Step 2: Parse PDF using PDFLoader
        pages = PDFLoader.load_pdf_from_bytes(file_bytes, file.filename)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF contains no extractable text content."
            )

        # Step 3: Split text using DocumentChunker
        chunked_docs = DocumentChunker.chunk_documents(
            pages_data=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Step 4: Write to Vector Database (ChromaDB)
        document_id = db_manager.add_documents(
            documents=chunked_docs,
            file_hash=file_hash,
            filename=file.filename
        )

        return UploadResponse(
            message="Document uploaded and processed successfully.",
            document_id=document_id,
            filename=file.filename,
            total_chunks=len(chunked_docs),
            file_hash=file_hash
        )

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"In-flight upload failure for '{file.filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document upload: {str(e)}"
        )

@router.post(
    "/query",
    summary="Query RAG Pipeline",
    response_model=QueryResponse
)
async def query_rag(request: QueryRequest):
    """
    Performs similarity search context retrieval followed by Gemini generation.
    Returns the answer and source page citations with timing latency.
    """
    start_time = time.perf_counter()
    try:
        # Step 1: Retrieve matching chunks
        chunks = retriever.retrieve_context(
            query=request.question,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )

        # Step 2: Generate response with Gemini
        answer = generator.generate_answer(
            question=request.question,
            chunks=chunks
        )

        # Step 3: Format references/citations
        citations = [
            Citation(
                filename=doc.metadata.get("source", "Unknown"),
                page=int(doc.metadata.get("page", 1)),
                text=doc.page_content
            )
            for doc in chunks
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Query API executed successfully in {latency_ms:.2f} ms.")

        return QueryResponse(
            answer=answer,
            citations=citations,
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Query execution failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during query execution: {str(e)}"
        )

@router.get(
    "/documents",
    summary="List Ingested Documents",
    response_model=DocumentListResponse
)
async def list_documents():
    """Lists all distinct ingested PDF files from vector storage."""
    try:
        documents = db_manager.get_all_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document lists: {str(e)}"
        )

@router.delete(
    "/documents/{document_id}",
    summary="Delete Ingested Document",
    response_model=DeleteResponse
)
async def delete_document(document_id: str):
    """Deletes all chunks associated with a specific document hash ID."""
    try:
        success = db_manager.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{document_id}' was not found in the database."
            )
        
        return DeleteResponse(
            message=f"Successfully deleted document ID '{document_id}' and all associated vector chunks.",
            success=True
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to delete document ID '{document_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete document deletion: {str(e)}"
        )
