import os
from typing import Set
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables and/or a .env file.
    Uses Pydantic settings management for type safety and validation.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    GEMINI_API_KEY: str = Field(..., description="API key for Google Gemini")
    CHROMA_DB_PATH: str = Field("chroma_db", description="Path to local Chroma DB storage directory")
    LOG_LEVEL: str = Field("INFO", description="Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    CHUNK_SIZE: int = Field(500, description="Default character size of chunk for text splitting")
    CHUNK_OVERLAP: int = Field(50, description="Default character overlap size for text splitting")
    TOP_K: int = Field(3, description="Default number of chunks to retrieve for RAG context")
    EMBEDDING_MODEL: str = Field("all-MiniLM-L6-v2", description="Local SentenceTransformers model name")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure that the LOG_LEVEL is a valid level supported by Python's logging module."""
        valid_levels: Set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return upper_v

    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K")
    @classmethod
    def validate_positive_integers(cls, v: int) -> int:
        """Ensure chunk size, chunk overlap, and top_k are positive integers."""
        if v <= 0:
            raise ValueError("Value must be a positive integer greater than zero")
        return v

# Instantiate settings singleton to be imported across the application
try:
    settings = Settings()
except Exception as e:
    # During startup/import or in non-production test setups, we might not have all environment variables.
    # We raise an informative runtime error.
    raise RuntimeError(
        f"Failed to load application configuration. Ensure that your .env file exists and contains "
        f"all required variables (specifically GEMINI_API_KEY). Details: {e}"
    ) from e
