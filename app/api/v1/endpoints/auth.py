from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.rate_limit import RateLimiter
from app.db.session import get_db_session
from app.models.user import User
from app.schemas.auth import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from app.services.auth import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication & RBAC"])

# Rate limiters for authentication endpoints to prevent brute-force attacks
auth_rate_limiter = RateLimiter(max_requests=10, window_seconds=60, prefix="auth_rate_limit")


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
    description="Registers an account with bcrypt password hashing and assigned RBAC role (admin, engineer, viewer).",
    dependencies=[Depends(auth_rate_limiter)],
)
async def register_user(
    payload: UserRegisterRequest,
    session: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    user = await auth_service.register(session=session, data=payload)
    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User login and JWT token issuance",
    description="Authenticates credentials and returns a cryptographic JWT bearer access token.",
    dependencies=[Depends(auth_rate_limiter)],
)
async def login_user(
    payload: UserLoginRequest,
    session: AsyncSession = Depends(get_db_session),
) -> TokenResponse:
    token, expires_in, user = await auth_service.login(session=session, data=payload)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current authenticated user profile",
    description="Resolves and returns identity claims and roles for the bearer token presented.",
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)
