from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4


class ContentAddressedByteStoreError(RuntimeError):
    pass


class ContentAddressedByteStoreConflictError(ContentAddressedByteStoreError):
    pass


class ContentAddressedByteStoreCorruptionError(ContentAddressedByteStoreError):
    pass


class ContentAddressedByteStore:
    """Atomic SHA-256 addressed storage for already-canonical bytes."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def blob_path(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest

    def write(
        self,
        *,
        object_id: str,
        digest: str,
        value: bytes,
    ) -> bool:
        observed = hashlib.sha256(value).hexdigest()
        if observed != digest:
            raise ContentAddressedByteStoreConflictError(f"content hash mismatch for {object_id}")
        path = self.blob_path(digest)
        if path.exists():
            existing = path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != digest or existing != value:
                raise ContentAddressedByteStoreConflictError(f"CAS blob conflict for sha256 {digest}")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_bytes(value)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def read(self, *, object_id: str, digest: str) -> bytes:
        path = self.blob_path(digest)
        if not path.exists():
            raise ContentAddressedByteStoreCorruptionError(f"missing CAS blob for {object_id}")
        value = path.read_bytes()
        observed = hashlib.sha256(value).hexdigest()
        if observed != digest:
            raise ContentAddressedByteStoreCorruptionError(f"CAS blob hash mismatch for {object_id}")
        return value
