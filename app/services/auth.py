from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import AppException
from app.core.logging import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import UserLoginRequest, UserRegisterRequest
from app.services.audit import AuditService

logger = get_logger(__name__)
settings = get_settings()


class AuthService:
    """Manages user registration, credential verification, and JWT issuance."""

    async def register(
        self,
        session: AsyncSession,
        data: UserRegisterRequest,
    ) -> User:
        """Register a new user account with hashed password and audit logging."""
        # 1. Enforce unique email
        stmt = select(User).where(User.email == data.email.lower().strip())
        existing = (await session.execute(stmt)).scalar_one_or_none()
        if existing:
            raise AppException(
                message=f"User account with email '{data.email}' already exists.",
                code="USER_ALREADY_EXISTS",
                status_code=409,
            )

        # 2. Hash password with bcrypt
        hashed = hash_password(data.password)

        # 3. Create user record
        user = User(
            email=data.email.lower().strip(),
            hashed_password=hashed,
            full_name=data.full_name.strip(),
            role=data.role,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        # 4. Audit Log
        await AuditService.log_action(
            session=session,
            action="USER_REGISTERED",
            entity_type="USER",
            entity_id=user.id,
            user_id=user.id,
            payload_after={"email": user.email, "role": user.role.value},
        )

        logger.info(f"Registered new user: {user.email} (role: {user.role.value}, id: {user.id})")
        return user

    async def login(
        self,
        session: AsyncSession,
        data: UserLoginRequest,
    ) -> tuple[str, int, User]:
        """Verify user credentials and issue signed JWT access token."""
        stmt = select(User).where(User.email == data.email.lower().strip())
        user = (await session.execute(stmt)).scalar_one_or_none()

        if not user or not verify_password(data.password, user.hashed_password):
            logger.warning(f"Failed login attempt for email: {data.email}")
            raise AppException(
                message="Invalid email or password.",
                code="INVALID_CREDENTIALS",
                status_code=401,
            )

        if not user.is_active:
            raise AppException(
                message="User account is deactivated. Contact an administrator.",
                code="USER_DEACTIVATED",
                status_code=403,
            )

        # Generate JWT access token
        expires_seconds = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        token_claims = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value,
        }
        token = create_access_token(data=token_claims)

        # Audit Log
        await AuditService.log_action(
            session=session,
            action="USER_LOGGED_IN",
            entity_type="USER",
            entity_id=user.id,
            user_id=user.id,
        )

        logger.info(f"User logged in successfully: {user.email} ({user.id})")
        return token, expires_seconds, user


auth_service = AuthService()
