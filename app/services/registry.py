import uuid
from pathlib import Path
from typing import Any

import joblib
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.errors import AppException, NotFoundError
from app.core.logging import get_logger
from app.models.enums import ArtifactFormat, ModelStatus
from app.models.model import Model, ModelVersion
from app.schemas.model import ModelCreateRequest
from app.storage.artifact_store import LocalArtifactStore, compute_sha256

logger = get_logger(__name__)

# Maximum permissible file size for lightweight tabular artifacts (50 MB)
MAX_ARTIFACT_SIZE_BYTES = 50 * 1024 * 1024

ALLOWED_EXTENSIONS: dict[ArtifactFormat, set[str]] = {
    ArtifactFormat.JOBLIB: {".joblib", ".pkl", ".pickle"},
    ArtifactFormat.ONNX: {".onnx"},
}


class ModelRegistryService:
    """Manages model catalog lifecycle and artifact validation pipeline."""

    def __init__(self, storage: LocalArtifactStore | None = None) -> None:
        self.storage = storage or LocalArtifactStore()

    async def create_model(
        self,
        session: AsyncSession,
        data: ModelCreateRequest,
        created_by_id: uuid.UUID | None = None,
    ) -> Model:
        """Register a new logical model identity."""
        # 1. Enforce unique model name
        stmt = select(Model).where(Model.name == data.name)
        existing = (await session.execute(stmt)).scalar_one_or_none()
        if existing:
            raise AppException(
                message=f"Model with name '{data.name}' is already registered.",
                code="MODEL_ALREADY_EXISTS",
                status_code=409,
            )

        model = Model(
            name=data.name,
            description=data.description,
            task_type=data.task_type,
            created_by_id=created_by_id,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        logger.info(f"Registered new logical model: {model.name} (id: {model.id})")
        return model

    async def get_model(
        self,
        session: AsyncSession,
        model_id: uuid.UUID,
    ) -> Model:
        """Fetch model by UUID including associated versions."""
        stmt = (
            select(Model)
            .where(Model.id == model_id)
            .options(selectinload(Model.versions))
        )
        model = (await session.execute(stmt)).scalar_one_or_none()
        if not model:
            raise NotFoundError(message=f"Model with id '{model_id}' was not found.")
        return model

    async def list_models(
        self,
        session: AsyncSession,
        skip: int = 0,
        limit: int = 50,
    ) -> list[tuple[Model, int]]:
        """List registered models with their version counts."""
        # Query models along with count of associated versions
        stmt = (
            select(
                Model,
                func.count(ModelVersion.id).label("versions_count"),
            )
            .outerjoin(ModelVersion, Model.id == ModelVersion.model_id)
            .group_by(Model.id)
            .order_by(Model.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def create_model_version(
        self,
        session: AsyncSession,
        model_id: uuid.UUID,
        file: UploadFile,
        version_tag: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        artifact_format: ArtifactFormat = ArtifactFormat.JOBLIB,
        created_by_id: uuid.UUID | None = None,
    ) -> ModelVersion:
        """Execute the complete artifact ingestion, validation, and staging pipeline."""
        # 1. Verify parent model exists
        model = await self.get_model(session, model_id)

        # 2. Version tag uniqueness check
        ver_stmt = select(ModelVersion).where(
            ModelVersion.model_id == model_id,
            ModelVersion.version_tag == version_tag,
        )
        if (await session.execute(ver_stmt)).scalar_one_or_none():
            raise AppException(
                message=f"Version '{version_tag}' already exists for model '{model.name}'.",
                code="VERSION_TAG_EXISTS",
                status_code=409,
            )

        # 3. Extension validation
        filename = file.filename or "model.joblib"
        ext = Path(filename).suffix.lower()
        valid_exts = ALLOWED_EXTENSIONS.get(artifact_format, set())
        if ext not in valid_exts:
            raise AppException(
                message=f"Invalid file extension '{ext}' for format {artifact_format}. Allowed: {sorted(valid_exts)}",
                code="INVALID_ARTIFACT_EXTENSION",
                status_code=400,
            )

        # 4. Size validation
        content = await file.read()
        if len(content) == 0:
            raise AppException(
                message="Uploaded artifact file is empty.",
                code="EMPTY_ARTIFACT_FILE",
                status_code=400,
            )
        if len(content) > MAX_ARTIFACT_SIZE_BYTES:
            raise AppException(
                message=f"Artifact size ({len(content)} bytes) exceeds maximum limit of {MAX_ARTIFACT_SIZE_BYTES} bytes.",
                code="ARTIFACT_TOO_LARGE",
                status_code=413,
            )

        # 5. Cryptographic SHA-256 calculation
        sha256_hash = compute_sha256(content)
        logger.info(f"Artifact binary verified: size={len(content)} bytes, sha256={sha256_hash}")

        # 6. Schema structure validation
        if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
            raise AppException(
                message="input_schema and output_schema must be valid JSON dictionaries.",
                code="INVALID_SCHEMA_STRUCTURE",
                status_code=400,
            )

        # 7. Store artifact on disk
        relative_path = await self.storage.save(
            model_name=model.name,
            version_tag=version_tag,
            filename=filename,
            content=content,
        )

        # 8. Create version entity in REGISTERED state
        version = ModelVersion(
            model_id=model_id,
            version_tag=version_tag,
            status=ModelStatus.REGISTERED,
            artifact_path=relative_path,
            artifact_format=artifact_format,
            artifact_hash_sha256=sha256_hash,
            input_schema=input_schema,
            output_schema=output_schema,
            created_by_id=created_by_id,
        )
        session.add(version)
        await session.flush()

        # 9. Transition to VALIDATING
        version.status = ModelStatus.VALIDATING
        logger.info(f"Executing load test on {model.name}:{version_tag}...")

        # 10. Load test & callable verification
        abs_path = self.storage.get_absolute_path(relative_path)
        try:
            if artifact_format == ArtifactFormat.JOBLIB:
                loaded_obj = joblib.load(abs_path)
                if not hasattr(loaded_obj, "predict") or not callable(loaded_obj.predict):
                    raise ValueError("Deserialized model artifact does not implement a callable .predict() method.")
            elif artifact_format == ArtifactFormat.ONNX:
                # Basic ONNX verification (magic byte check)
                if not content.startswith(b"\x08"):
                    raise ValueError("Invalid ONNX protobuf payload.")
        except Exception as exc:
            # Transition to FAILED on any deserialization or validation error
            version.status = ModelStatus.FAILED
            await session.commit()
            logger.error(f"Model validation failed for {version_tag}: {exc}")
            raise AppException(
                message=f"Model validation failed: {str(exc)}",
                code="MODEL_VALIDATION_FAILED",
                status_code=422,
                details={"error": str(exc), "status": ModelStatus.FAILED},
            ) from exc

        # 11. Transition to STAGING (Ready for deployment)
        version.status = ModelStatus.STAGING
        await session.commit()
        await session.refresh(version)

        logger.info(f"Model version {model.name}:{version_tag} successfully validated and promoted to STAGING")
        return version
