from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import ValidationError

from eval_factory.contracts.core import ContractModel, ObjectRef


class FactoryPrivateObjectError(RuntimeError):
    pass


class FactoryPrivateObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        *,
        object_type: str,
        payload: bytes,
    ) -> ObjectRef:
        digest = hashlib.sha256(payload).hexdigest()
        reference = ObjectRef(
            object_type=object_type,
            object_id=f"{object_type}://sha256/{digest}",
            object_version="v2",
            object_sha256=digest,
        )
        path = self._path(reference)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise FactoryPrivateObjectError("private CAS content is corrupt")
            return reference
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except FileExistsError as exc:
            if not path.exists() or path.read_bytes() != payload:
                raise FactoryPrivateObjectError("private CAS write raced with different bytes") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        return reference

    def put_text(self, *, object_type: str, text: str) -> ObjectRef:
        return self.put_bytes(object_type=object_type, payload=text.encode("utf-8"))

    def put_model(self, *, object_type: str, value: ContractModel) -> ObjectRef:
        return self.put_bytes(object_type=object_type, payload=value.canonical_json())

    def get_bytes(self, reference: ObjectRef) -> bytes:
        path = self._path(reference)
        if not path.is_file() or path.is_symlink():
            raise FactoryPrivateObjectError("private CAS object is missing")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference.object_sha256:
            raise FactoryPrivateObjectError("private CAS object hash is corrupt")
        return payload

    def get_text(self, reference: ObjectRef) -> str:
        try:
            return self.get_bytes(reference).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FactoryPrivateObjectError("private CAS text is not UTF-8") from exc

    def get_model[ModelT: ContractModel](
        self,
        reference: ObjectRef,
        model_type: type[ModelT],
    ) -> ModelT:
        payload = self.get_bytes(reference)
        try:
            value = model_type.model_validate_json(payload)
        except ValidationError as exc:
            raise FactoryPrivateObjectError("private CAS model is invalid") from exc
        if value.canonical_json() != payload:
            raise FactoryPrivateObjectError("private CAS model is not canonical")
        return value

    def _path(self, reference: ObjectRef) -> Path:
        return self.root / "sha256" / reference.object_sha256[:2] / reference.object_sha256


__all__ = ["FactoryPrivateObjectError", "FactoryPrivateObjectStore"]
