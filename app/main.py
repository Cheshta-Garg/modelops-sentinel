from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.router import api_v1_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware

settings = get_settings()
setup_logging(level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifecycle management for startup and graceful shutdown."""
    logger.info(
        f"Starting {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT.upper()}]"
    )
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME} gracefully.")


def create_application() -> FastAPI:
    """Application factory for FastAPI."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production ML Model Lifecycle, Serving & Monitoring Platform",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Register Middlewares
    app.add_middleware(RequestContextMiddleware)

    # 2. Register Global Error Handlers
    register_error_handlers(app)

    # 3. Register API Routers
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    # 4. Root redirect to interactive documentation
    @app.get("/", include_in_schema=False)
    async def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_application()
