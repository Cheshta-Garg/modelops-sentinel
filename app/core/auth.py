import uuid
from collections.abc import Callable

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppException
from app.core.security import decode_access_token
from app.db.session import get_db_session
from app.models.enums import UserRole
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """Validate JWT bearer token from request header and resolve user entity."""
    if not credentials or not credentials.credentials:
        raise AppException(
            message="Authentication credentials were not provided. Include 'Authorization: Bearer <token>' header.",
            code="UNAUTHENTICATED",
            status_code=401,
        )

    token = credentials.credentials
    payload = decode_access_token(token)

    user_id_raw = payload.get("sub")
    if not user_id_raw:
        raise AppException(
            message="Token claim payload missing user identity subject.",
            code="INVALID_TOKEN_CLAIMS",
            status_code=401,
        )

    try:
        user_id = uuid.UUID(str(user_id_raw))
    except ValueError as exc:
        raise AppException(
            message="Malformed user identifier in token.",
            code="INVALID_TOKEN_SUBJECT",
            status_code=401,
        ) from exc

    stmt = select(User).where(User.id == user_id)
    user = (await session.execute(stmt)).scalar_one_or_none()

    if not user:
        raise AppException(
            message="User account associated with this token no longer exists.",
            code="USER_NOT_FOUND",
            status_code=401,
        )

    if not user.is_active:
        raise AppException(
            message="User account is deactivated.",
            code="USER_DEACTIVATED",
            status_code=403,
        )

    return user


def require_roles(*allowed_roles: UserRole) -> Callable:
    """RBAC dependency factory.

    Rules:
    - ADMIN always has full access to all endpoints.
    - Other users must match one of the allowed roles.
    """

    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        # Admin has full access to everything
        if current_user.role == UserRole.ADMIN:
            return current_user

        if current_user.role not in allowed_roles:
            raise AppException(
                message=(
                    f"User role '{current_user.role.value}' does not have permission for this action. "
                    f"Required: {[r.value for r in allowed_roles]}"
                ),
                code="FORBIDDEN_INSUFFICIENT_PERMISSIONS",
                status_code=403,
                details={
                    "user_role": current_user.role.value,
                    "required_roles": [r.value for r in allowed_roles],
                },
            )
        return current_user

    return role_checker
