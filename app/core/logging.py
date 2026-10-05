import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable to hold request_id across async coroutines
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include custom extra attributes if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_obj.update(record.extra_fields)

        return json.dumps(log_obj)


class TextFormatter(logging.Formatter):
    """Formats log records as readable colored/standard text for local development."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = request_id_ctx.get()
        req_str = f"[{req_id}] " if req_id != "-" else ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{timestamp} | {record.levelname:<7} | {req_str}{record.name}:{record.lineno} - {record.getMessage()}"
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg


def setup_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Configure root logger with the specified format and severity level."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Clear existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(name)
