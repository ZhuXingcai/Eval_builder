from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.approval.interaction_models import (
    UserCheckpointSourceContextV2,
    user_checkpoint_source_context_v2_ref,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointPresentationV2,
    user_checkpoint_presentation_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class UserCheckpointMaterialError(RuntimeError):
    pass


class UserCheckpointMaterialTypeError(UserCheckpointMaterialError):
    pass


class UserCheckpointMaterialConflictError(UserCheckpointMaterialError):
    pass


class UserCheckpointMaterialIntegrityError(UserCheckpointMaterialError):
    pass


class UserCheckpointMaterialLimitError(UserCheckpointMaterialError):
    pass


@dataclass(frozen=True, slots=True)
class UserCheckpointMaterialWrite:
    object_ref: ObjectRef
    written: bool
    canonical_size_bytes: int


class _UserCheckpointMaterialEnvelopeV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-material-envelope/private-v1"] = (
        "eval-factory/user-checkpoint-material-envelope/private-v1"
    )
    object_ref: ObjectRef
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_blob_ref.object_type != "user-checkpoint-material-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id
            != (f"user-checkpoint-material-content://sha256/{self.content_blob_ref.object_sha256}")
        ):
            raise ValueError("checkpoint material content ref is invalid")
        return self


class UserCheckpointMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_source_context_bytes: int,
        max_presentation_bytes: int,
    ) -> None:
        self.root = root.expanduser().resolve()
        if max_source_context_bytes < 2 or max_presentation_bytes < 2:
            raise UserCheckpointMaterialLimitError("checkpoint material limits must be at least two bytes")
        self.max_source_context_bytes = max_source_context_bytes
        self.max_presentation_bytes = max_presentation_bytes
        self._source_contexts = ContentAddressedByteStore(self.root / "content")
        self._presentations = ContentAddressedByteStore(self.root / "content")
        self._source_index = self.root / "source-context"
        self._presentation_index = self.root / "presentation"

    def put_source_context(
        self,
        value: UserCheckpointSourceContextV2,
    ) -> UserCheckpointMaterialWrite:
        ref = user_checkpoint_source_context_v2_ref(value)
        result = self._put(
            store=self._source_contexts,
            index_root=self._source_index,
            ref=ref,
            payload=value.canonical_json(),
            limit=self.max_source_context_bytes,
        )
        self.get_source_context(ref)
        return result

    def put_presentation(
        self,
        value: UserCheckpointPresentationV2,
    ) -> UserCheckpointMaterialWrite:
        ref = user_checkpoint_presentation_v2_ref(value)
        result = self._put(
            store=self._presentations,
            index_root=self._presentation_index,
            ref=ref,
            payload=value.canonical_json(),
            limit=self.max_presentation_bytes,
        )
        self.get_presentation(ref)
        return result

    def get_source_context(
        self,
        ref: ObjectRef,
    ) -> UserCheckpointSourceContextV2:
        self._require_ref(
            ref,
            object_type="user-checkpoint-source-context",
            object_version="private-v1",
        )
        payload = self._read(
            store=self._source_contexts,
            index_root=self._source_index,
            ref=ref,
            limit=self.max_source_context_bytes,
        )
        try:
            value = UserCheckpointSourceContextV2.model_validate_json(payload)
            observed = user_checkpoint_source_context_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise UserCheckpointMaterialIntegrityError("checkpoint source context is invalid") from exc
        if observed != ref or value.canonical_json() != payload:
            raise UserCheckpointMaterialIntegrityError("checkpoint source context differs from its reference")
        return value

    def get_presentation(
        self,
        ref: ObjectRef,
    ) -> UserCheckpointPresentationV2:
        self._require_ref(
            ref,
            object_type="user-checkpoint-presentation",
            object_version="v2",
        )
        payload = self._read(
            store=self._presentations,
            index_root=self._presentation_index,
            ref=ref,
            limit=self.max_presentation_bytes,
        )
        try:
            value = UserCheckpointPresentationV2.model_validate_json(payload)
            observed = user_checkpoint_presentation_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise UserCheckpointMaterialIntegrityError("checkpoint presentation is invalid") from exc
        if observed != ref or value.canonical_json() != payload:
            raise UserCheckpointMaterialIntegrityError("checkpoint presentation differs from its reference")
        return value

    def verify_source_context(
        self,
        ref: ObjectRef,
    ) -> None:
        self.get_source_context(ref)

    def verify_presentation(
        self,
        ref: ObjectRef,
    ) -> None:
        self.get_presentation(ref)

    @staticmethod
    def _put(
        *,
        store: ContentAddressedByteStore,
        index_root: Path,
        ref: ObjectRef,
        payload: bytes,
        limit: int,
    ) -> UserCheckpointMaterialWrite:
        if len(payload) > limit:
            raise UserCheckpointMaterialLimitError("checkpoint material exceeds its byte limit")
        existing = UserCheckpointMaterialStore._existing_write(
            store=store,
            index_root=index_root,
            ref=ref,
            limit=limit,
        )
        if existing is not None:
            return existing
        content_digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="user-checkpoint-material-content",
            object_id=(f"user-checkpoint-material-content://sha256/{content_digest}"),
            object_version="private-v1",
            object_sha256=content_digest,
        )
        envelope = _UserCheckpointMaterialEnvelopeV2(
            object_ref=ref,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        try:
            store.write(
                object_id=content_ref.object_id,
                digest=content_digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise UserCheckpointMaterialConflictError("checkpoint material write conflicted") from exc
        written = UserCheckpointMaterialStore._write_envelope(
            index_root=index_root,
            envelope=envelope,
        )
        if not written:
            existing = UserCheckpointMaterialStore._existing_write(
                store=store,
                index_root=index_root,
                ref=ref,
                limit=limit,
            )
            if existing is None:
                raise UserCheckpointMaterialIntegrityError("checkpoint material envelope disappeared")
            return existing
        return UserCheckpointMaterialWrite(
            object_ref=ref,
            written=written,
            canonical_size_bytes=len(payload),
        )

    @staticmethod
    def _existing_write(
        *,
        store: ContentAddressedByteStore,
        index_root: Path,
        ref: ObjectRef,
        limit: int,
    ) -> UserCheckpointMaterialWrite | None:
        path = UserCheckpointMaterialStore._envelope_path(
            index_root=index_root,
            ref=ref,
        )
        if not path.exists():
            return None
        envelope = UserCheckpointMaterialStore._read_envelope(
            index_root=index_root,
            ref=ref,
        )
        try:
            payload = store.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise UserCheckpointMaterialIntegrityError("checkpoint material is missing or corrupt") from exc
        if len(payload) > limit or len(payload) != envelope.canonical_size_bytes:
            raise UserCheckpointMaterialLimitError("checkpoint material size is invalid")
        return UserCheckpointMaterialWrite(
            object_ref=ref,
            written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    @staticmethod
    def _read(
        *,
        store: ContentAddressedByteStore,
        index_root: Path,
        ref: ObjectRef,
        limit: int,
    ) -> bytes:
        envelope = UserCheckpointMaterialStore._read_envelope(
            index_root=index_root,
            ref=ref,
        )
        try:
            payload = store.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise UserCheckpointMaterialIntegrityError("checkpoint material is missing or corrupt") from exc
        if len(payload) > limit or len(payload) != envelope.canonical_size_bytes:
            raise UserCheckpointMaterialLimitError("checkpoint material size is invalid")
        return payload

    @staticmethod
    def _write_envelope(
        *,
        index_root: Path,
        envelope: _UserCheckpointMaterialEnvelopeV2,
    ) -> bool:
        path = (
            index_root
            / "sha256"
            / envelope.object_ref.object_sha256[:2]
            / f"{envelope.object_ref.object_sha256}.json"
        )
        rendered = envelope.canonical_json() + b"\n"
        if path.exists():
            UserCheckpointMaterialStore._read_envelope(
                index_root=index_root,
                ref=envelope.object_ref,
            )
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    @staticmethod
    def _read_envelope(
        *,
        index_root: Path,
        ref: ObjectRef,
    ) -> _UserCheckpointMaterialEnvelopeV2:
        path = UserCheckpointMaterialStore._envelope_path(
            index_root=index_root,
            ref=ref,
        )
        if not path.exists():
            raise UserCheckpointMaterialIntegrityError("checkpoint material envelope is missing")
        try:
            envelope = _UserCheckpointMaterialEnvelopeV2.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise UserCheckpointMaterialIntegrityError("checkpoint material envelope is invalid") from exc
        if envelope.object_ref != ref:
            raise UserCheckpointMaterialIntegrityError(
                "checkpoint material envelope differs from its reference"
            )
        return envelope

    @staticmethod
    def _envelope_path(
        *,
        index_root: Path,
        ref: ObjectRef,
    ) -> Path:
        return index_root / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    @staticmethod
    def _require_ref(
        ref: ObjectRef,
        *,
        object_type: str,
        object_version: str,
    ) -> None:
        if ref.object_type != object_type or ref.object_version != object_version:
            raise UserCheckpointMaterialTypeError("checkpoint material reference has the wrong type")


__all__ = [
    "UserCheckpointMaterialConflictError",
    "UserCheckpointMaterialError",
    "UserCheckpointMaterialIntegrityError",
    "UserCheckpointMaterialLimitError",
    "UserCheckpointMaterialStore",
    "UserCheckpointMaterialTypeError",
    "UserCheckpointMaterialWrite",
]
