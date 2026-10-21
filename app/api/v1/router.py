from fastapi import APIRouter

from app.api.v1.endpoints import auth, health, models

api_v1_router = APIRouter()

# Register endpoint routers
api_v1_router.include_router(auth.router)
api_v1_router.include_router(health.router)
api_v1_router.include_router(models.router)

