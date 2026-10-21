import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import UserRole


class UserRegisterRequest(BaseModel):
    """Payload to register a new user account."""

    email: EmailStr = Field(description="Unique email address", examples=["engineer@modelops.com"])
    password: str = Field(min_length=8, description="Plaintext password (min 8 chars)", examples=["SecurePassword123!"])
    full_name: str = Field(min_length=1, description="Full name of user", examples=["Jane Doe"])
    role: UserRole = Field(default=UserRole.VIEWER, description="Assigned RBAC role (admin, engineer, viewer)")


class UserLoginRequest(BaseModel):
    """Payload to authenticate and obtain a JWT bearer token."""

    email: EmailStr = Field(description="Account email", examples=["engineer@modelops.com"])
    password: str = Field(description="Account password", examples=["SecurePassword123!"])


class UserResponse(BaseModel):
    """User profile data returned to client."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Authentication token payload returned upon successful login."""

    access_token: str = Field(description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token authentication scheme")
    expires_in: int = Field(description="Token validity lifetime in seconds")
    user: UserResponse = Field(description="Authenticated user identity details")
