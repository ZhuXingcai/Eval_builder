from pathlib import Path

from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)
from eval_factory.trace.normalization import NormalizedContentBlob
from eval_factory.trace.storage.models import TraceCasConflictError, TraceStoreCorruptionError


class ContentAddressedStore:
    def __init__(self, root: Path) -> None:
        self._store = ContentAddressedByteStore(root)
        self.root = self._store.root

    def blob_path(self, digest: str) -> Path:
        return self._store.blob_path(digest)

    def write(self, blob: NormalizedContentBlob) -> bool:
        try:
            return self._store.write(
                object_id=blob.content_ref.object_id,
                digest=blob.content_ref.object_sha256,
                value=blob.canonical_bytes,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise TraceCasConflictError(str(exc)) from exc

    def read(self, object_id: str, digest: str) -> bytes:
        try:
            return self._store.read(
                object_id=object_id,
                digest=digest,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise TraceStoreCorruptionError(str(exc)) from exc
