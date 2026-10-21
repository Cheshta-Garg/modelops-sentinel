import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.monitoring import AuditLog

logger = get_logger(__name__)


class AuditService:
    """Service to record immutable compliance and security audit logs."""

    @staticmethod
    async def log_action(
        session: AsyncSession,
        action: str,
        entity_type: str,
        entity_id: str | uuid.UUID | None = None,
        user_id: uuid.UUID | None = None,
        payload_before: dict[str, Any] | None = None,
        payload_after: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Create and commit an immutable audit log entry."""
        audit_entry = AuditLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=str(entity_id) if entity_id else None,
            payload_before=payload_before,
            payload_after=payload_after,
        )
        session.add(audit_entry)
        await session.commit()
        await session.refresh(audit_entry)
        logger.info(f"Audit log recorded: {action} on {entity_type}:{entity_id} by user:{user_id}")
        return audit_entry
