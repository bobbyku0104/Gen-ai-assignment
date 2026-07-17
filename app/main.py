from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.config import settings
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    Guarantees clean workspace logs and startup reporting.
    """
    logger.info("Initializing Cost-Efficient RAG API Server...")
    logger.info(
        f"Settings Loaded: CHUNK_SIZE={settings.CHUNK_SIZE}, "
        f"CHUNK_OVERLAP={settings.CHUNK_OVERLAP}, TOP_K={settings.TOP_K}"
    )
    yield
    logger.info("Shutting down Cost-Efficient RAG API Server...")

# Initialize FastAPI application
app = FastAPI(
    title="Cost-Efficient RAG Application",
    description=(
        "Production-ready Retrieval-Augmented Generation (RAG) backend service. "
        "Utilizes local persistent ChromaDB, SentenceTransformers (all-MiniLM-L6-v2) local embeddings, "
        "and Google Gemini API for grounded generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware configuration (critical for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to specific domains in production setups
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Intercepts schema validation failures (HTTP 422) and logs them before returning.
    """
    logger.warning(
        f"Validation failure on {request.method} {request.url.path}: {exc.errors()}"
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "message": "Input validation failed. Please check the schema requirements."
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Uncaught fallback exception interceptor.
    Prevents raw stack trace leak to client API responses, logging it securely to files instead.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}", 
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "message": str(exc) if settings.LOG_LEVEL == "DEBUG" else "Please contact administrative logs."
        }
    )
