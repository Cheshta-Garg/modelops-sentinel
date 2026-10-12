import hashlib
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ArtifactStore(Protocol):
    """Abstract interface defining the artifact storage contract."""

    async def save(
        self,
        model_name: str,
        version_tag: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Store binary artifact and return its canonical path."""
        ...

    async def load(self, artifact_path: str) -> bytes:
        """Read artifact binary from storage."""
        ...

    async def exists(self, artifact_path: str) -> bool:
        """Check if an artifact exists at the given path."""
        ...


class LocalArtifactStore:
    """Filesystem-backed implementation of ArtifactStore."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or settings.ARTIFACT_DIR).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize(self, value: str) -> str:
        """Sanitize names to prevent path traversal attacks."""
        return Path(value).name.strip().replace(" ", "_")

    def _get_target_dir(self, model_name: str, version_tag: str) -> Path:
        safe_model = self._sanitize(model_name)
        safe_version = self._sanitize(version_tag)
        target_dir = self.base_dir / safe_model / safe_version
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    async def save(
        self,
        model_name: str,
        version_tag: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Save artifact to disk and return relative path from base_dir."""
        safe_filename = self._sanitize(filename)
        target_dir = self._get_target_dir(model_name, version_tag)
        file_path = target_dir / safe_filename

        file_path.write_bytes(content)
        logger.info(f"Artifact stored locally at {file_path} ({len(content)} bytes)")

        # Return relative path for portability across environments
        return str(file_path.relative_to(self.base_dir)).replace("\\", "/")

    async def load(self, artifact_path: str) -> bytes:
        """Read artifact bytes from local storage."""
        full_path = (self.base_dir / artifact_path).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"Artifact not found at {full_path}")
        return full_path.read_bytes()

    async def exists(self, artifact_path: str) -> bool:
        """Verify artifact presence on disk."""
        full_path = (self.base_dir / artifact_path).resolve()
        return full_path.is_file()

    def get_absolute_path(self, artifact_path: str) -> Path:
        """Resolve full absolute filesystem path for loading into memory."""
        return (self.base_dir / artifact_path).resolve()


def compute_sha256(content: bytes) -> str:
    """Calculate SHA-256 cryptographic digest for binary payload."""
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()
