import json
import uuid

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppException
from app.db.session import get_db_session
from app.models.enums import ArtifactFormat
from app.schemas.model import (
    ModelCreateRequest,
    ModelDetailResponse,
    ModelResponse,
    ModelVersionResponse,
    ModelVersionSummaryResponse,
)
from app.services.registry import ModelRegistryService

router = APIRouter(prefix="/models", tags=["Model Registry"])
registry_service = ModelRegistryService()


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new logical ML model",
    description="Registers a new model identity in the registry with name, description, and task type.",
)
async def create_model(
    payload: ModelCreateRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ModelResponse:
    model = await registry_service.create_model(session, payload)
    return ModelResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        task_type=model.task_type,
        created_at=model.created_at,
        updated_at=model.updated_at,
        versions_count=0,
    )


@router.get(
    "",
    response_model=list[ModelResponse],
    status_code=status.HTTP_200_OK,
    summary="List all registered models",
    description="Returns a paginated list of all models registered in the catalog.",
)
async def list_models(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session),
) -> list[ModelResponse]:
    model_records = await registry_service.list_models(session, skip=skip, limit=limit)
    return [
        ModelResponse(
            id=model.id,
            name=model.name,
            description=model.description,
            task_type=model.task_type,
            created_at=model.created_at,
            updated_at=model.updated_at,
            versions_count=count,
        )
        for model, count in model_records
    ]


@router.get(
    "/{model_id}",
    response_model=ModelDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get model details and versions",
    description="Fetches a specific model by UUID including all of its version iterations.",
)
async def get_model(
    model_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
) -> ModelDetailResponse:
    model = await registry_service.get_model(session, model_id)
    version_summaries = [
        ModelVersionSummaryResponse(
            id=v.id,
            version_tag=v.version_tag,
            status=v.status,
            artifact_format=v.artifact_format,
            created_at=v.created_at,
        )
        for v in (model.versions or [])
    ]
    return ModelDetailResponse(
        id=model.id,
        name=model.name,
        description=model.description,
        task_type=model.task_type,
        created_at=model.created_at,
        updated_at=model.updated_at,
        versions_count=len(version_summaries),
        versions=version_summaries,
    )


@router.post(
    "/{model_id}/versions",
    response_model=ModelVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and validate a new model artifact version",
    description=(
        "Executes the full artifact pipeline: extension check, size check, SHA-256 calculation, "
        "local storage, in-memory load testing, schema validation, and state transition to STAGING."
    ),
)
async def create_model_version(
    model_id: uuid.UUID,
    file: UploadFile = File(..., description="Serialized model binary (.joblib or .onnx)"),
    version_tag: str = Form(..., description="Semantic version tag, e.g. v1.0.0"),
    input_schema: str = Form(
        ...,
        description='JSON string of input features, e.g. {"features": ["age", "income"]}',
    ),
    output_schema: str = Form(
        ...,
        description='JSON string of target output, e.g. {"prediction": "int", "probability": "float"}',
    ),
    artifact_format: ArtifactFormat = Form(
        default=ArtifactFormat.JOBLIB,
        description="Format of the serialized model",
    ),
    session: AsyncSession = Depends(get_db_session),
) -> ModelVersionResponse:
    # Parse JSON schema strings
    try:
        parsed_input_schema = json.loads(input_schema)
        parsed_output_schema = json.loads(output_schema)
    except json.JSONDecodeError as exc:
        raise AppException(
            message=f"Invalid JSON in schema specifications: {exc}",
            code="MALFORMED_SCHEMA_JSON",
            status_code=400,
        ) from exc

    version = await registry_service.create_model_version(
        session=session,
        model_id=model_id,
        file=file,
        version_tag=version_tag.strip(),
        input_schema=parsed_input_schema,
        output_schema=parsed_output_schema,
        artifact_format=artifact_format,
    )

    return ModelVersionResponse(
        id=version.id,
        model_id=version.model_id,
        version_tag=version.version_tag,
        status=version.status,
        artifact_path=version.artifact_path,
        artifact_format=version.artifact_format,
        artifact_hash_sha256=version.artifact_hash_sha256,
        input_schema=version.input_schema,
        output_schema=version.output_schema,
        created_at=version.created_at,
        updated_at=version.updated_at,
    )
