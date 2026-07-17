import logging
import os
import sys
from app.config import settings

def setup_logger() -> logging.Logger:
    """
    Configures and initializes the application-wide logger.
    Directs output to both stdout (console) and a persistent log file.
    """
    logger = logging.getLogger("rag_app")
    logger.setLevel(settings.LOG_LEVEL)

    # Avoid adding handlers multiple times if logger is imported in different modules
    if logger.handlers:
        return logger

    # Ensure the directory for logs exists
    log_dir = "logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        # Fallback to console-only logging if logs directory is write-protected
        sys.stderr.write(f"Warning: Could not create log directory {log_dir}: {e}\n")
        log_dir = None

    # Logging format structure: [timestamp] LEVEL [name:filename:line] message
    log_format = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (writing to sys.stdout for standard application logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.LOG_LEVEL)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler (only if directory is available)
    if log_dir:
        log_filepath = os.path.join(log_dir, "app.log")
        try:
            file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
            file_handler.setLevel(settings.LOG_LEVEL)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except Exception as e:
            sys.stderr.write(f"Warning: Could not configure file handler at {log_filepath}: {e}\n")

    return logger

# Export configured logger instance
logger = setup_logger()
