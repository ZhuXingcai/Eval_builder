from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator, model_validator

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.statistics.models import (
    IndependentLabelCandidatePoolV1,
    IndependentLabelPartitionMaterialV1,
    IndependentLabelTestSetMaterialV1,
    independent_label_candidate_pool_v1_ref,
    independent_label_partition_material_v1_ref,
    independent_label_test_set_material_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class IndependentLabelMaterialError(RuntimeError):
    pass


class IndependentLabelMaterialTypeError(IndependentLabelMaterialError):
    pass


class IndependentLabelMaterialConflictError(IndependentLabelMaterialError):
    pass


class IndependentLabelMaterialIntegrityError(IndependentLabelMaterialError):
    pass


class IndependentLabelMaterialLimitError(IndependentLabelMaterialError):
    pass


class IndependentLabelMaterialInjectedCrash(IndependentLabelMaterialError):
    pass


class IndependentLabelMaterialCodec(StrEnum):
    CANDIDATE_POOL = "candidate-pool"
    PARTITION = "partition"
    TEST_SET = "test-set"


class IndependentLabelMaterialStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class IndependentLabelMaterialFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: IndependentLabelMaterialStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticIndependentLabelMaterialFaultInjector:
    crash_points: frozenset[IndependentLabelMaterialStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: IndependentLabelMaterialStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise IndependentLabelMaterialInjectedCrash(
                f"injected independent label material crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class IndependentLabelMaterialWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _IndependentLabelMaterialEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-material-envelope/private-v1"] = (
        "eval-factory/independent-label-material-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: IndependentLabelMaterialCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @field_validator("codec", mode="before")
    @classmethod
    def parse_codec(cls, value: object) -> IndependentLabelMaterialCodec:
        if isinstance(value, IndependentLabelMaterialCodec):
            return value
        if isinstance(value, str):
            return IndependentLabelMaterialCodec(value)
        raise TypeError("codec must be an IndependentLabelMaterialCodec")

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_blob_ref.object_type != "independent-label-material-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id
            != (f"independent-label-material-content://sha256/{self.content_blob_ref.object_sha256}")
        ):
            raise ValueError("independent label material content ref is invalid")
        return self


_Model = (
    IndependentLabelCandidatePoolV1 | IndependentLabelPartitionMaterialV1 | IndependentLabelTestSetMaterialV1
)


class IndependentLabelMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_candidate_pool_bytes: int,
        max_partition_bytes: int,
        max_test_set_bytes: int,
        max_members: int,
        fault_injector: IndependentLabelMaterialFaultInjector | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        if (
            max_candidate_pool_bytes < 2
            or max_partition_bytes < 2
            or max_test_set_bytes < 2
            or max_members < 1
        ):
            raise IndependentLabelMaterialLimitError("independent label material limits are invalid")
        self.max_candidate_pool_bytes = max_candidate_pool_bytes
        self.max_partition_bytes = max_partition_bytes
        self.max_test_set_bytes = max_test_set_bytes
        self.max_members = max_members
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(self.root / "cas")
        self._envelopes = self.root / "envelopes"

    def put_candidate_pool(
        self,
        value: IndependentLabelCandidatePoolV1,
    ) -> IndependentLabelMaterialWrite:
        return self._put(
            value,
            codec=IndependentLabelMaterialCodec.CANDIDATE_POOL,
            ref=independent_label_candidate_pool_v1_ref(value),
            limit=self.max_candidate_pool_bytes,
            member_count=len(value.candidates),
        )

    def put_partition(
        self,
        value: IndependentLabelPartitionMaterialV1,
    ) -> IndependentLabelMaterialWrite:
        return self._put(
            value,
            codec=IndependentLabelMaterialCodec.PARTITION,
            ref=independent_label_partition_material_v1_ref(value),
            limit=self.max_partition_bytes,
            member_count=len(value.members),
        )

    def put_test_set(
        self,
        value: IndependentLabelTestSetMaterialV1,
    ) -> IndependentLabelMaterialWrite:
        return self._put(
            value,
            codec=IndependentLabelMaterialCodec.TEST_SET,
            ref=independent_label_test_set_material_v1_ref(value),
            limit=self.max_test_set_bytes,
            member_count=len(value.members),
        )

    def get_candidate_pool(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelCandidatePoolV1:
        value = self._get(
            ref,
            codec=IndependentLabelMaterialCodec.CANDIDATE_POOL,
            limit=self.max_candidate_pool_bytes,
        )
        if not isinstance(value, IndependentLabelCandidatePoolV1):
            raise IndependentLabelMaterialTypeError("candidate pool codec returned wrong model")
        return value

    def get_partition(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelPartitionMaterialV1:
        value = self._get(
            ref,
            codec=IndependentLabelMaterialCodec.PARTITION,
            limit=self.max_partition_bytes,
        )
        if not isinstance(value, IndependentLabelPartitionMaterialV1):
            raise IndependentLabelMaterialTypeError("partition codec returned wrong model")
        return value

    def get_test_set(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetMaterialV1:
        value = self._get(
            ref,
            codec=IndependentLabelMaterialCodec.TEST_SET,
            limit=self.max_test_set_bytes,
        )
        if not isinstance(value, IndependentLabelTestSetMaterialV1):
            raise IndependentLabelMaterialTypeError("test-set codec returned wrong model")
        return value

    def verify_candidate_pool(self, ref: ObjectRef) -> None:
        self.get_candidate_pool(ref)

    def verify_partition(self, ref: ObjectRef) -> None:
        self.get_partition(ref)

    def verify_test_set(self, ref: ObjectRef) -> None:
        self.get_test_set(ref)

    def _envelope_path(self, ref: ObjectRef) -> Path:
        codec = _codec_for_ref(ref)
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _put(
        self,
        value: _Model,
        *,
        codec: IndependentLabelMaterialCodec,
        ref: ObjectRef,
        limit: int,
        member_count: int,
    ) -> IndependentLabelMaterialWrite:
        if member_count > self.max_members:
            raise IndependentLabelMaterialLimitError("independent label material exceeds member limit")
        payload = value.canonical_json()
        if len(payload) > limit:
            raise IndependentLabelMaterialLimitError("independent label material exceeds byte limit")
        existing = self._existing_write(ref, codec=codec, limit=limit)
        if existing is not None:
            return existing
        content_digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="independent-label-material-content",
            object_id=f"independent-label-material-content://sha256/{content_digest}",
            object_version="private-v1",
            object_sha256=content_digest,
        )
        envelope = _IndependentLabelMaterialEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=content_digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise IndependentLabelMaterialConflictError(
                "independent label material CAS write conflicted"
            ) from exc
        self._maybe_raise(
            IndependentLabelMaterialStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        envelope_written = self._write_envelope(envelope)
        self._maybe_raise(
            IndependentLabelMaterialStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not envelope_written:
            existing = self._existing_write(ref, codec=codec, limit=limit)
            if existing is None:
                raise IndependentLabelMaterialIntegrityError(
                    "independent label material envelope disappeared"
                )
            return existing
        self._get(ref, codec=codec, limit=limit)
        return IndependentLabelMaterialWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def _existing_write(
        self,
        ref: ObjectRef,
        *,
        codec: IndependentLabelMaterialCodec,
        limit: int,
    ) -> IndependentLabelMaterialWrite | None:
        path = self._envelope_path(ref)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec=codec)
        payload = self._read_payload(envelope, limit=limit)
        self._decode(codec, payload, ref)
        return IndependentLabelMaterialWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _get(
        self,
        ref: ObjectRef,
        *,
        codec: IndependentLabelMaterialCodec,
        limit: int,
    ) -> _Model:
        _require_ref_codec(ref, codec)
        envelope = self._read_envelope(ref, codec=codec)
        payload = self._read_payload(envelope, limit=limit)
        value = self._decode(codec, payload, ref)
        member_count = (
            len(value.candidates)
            if isinstance(value, IndependentLabelCandidatePoolV1)
            else len(value.members)
        )
        if member_count > self.max_members:
            raise IndependentLabelMaterialLimitError("stored independent label material exceeds member limit")
        return value

    def _read_envelope(
        self,
        ref: ObjectRef,
        *,
        codec: IndependentLabelMaterialCodec,
    ) -> _IndependentLabelMaterialEnvelopeV1:
        path = self._envelope_path(ref)
        if not path.exists():
            raise IndependentLabelMaterialIntegrityError("independent label material envelope is missing")
        try:
            envelope = _IndependentLabelMaterialEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise IndependentLabelMaterialIntegrityError(
                "independent label material envelope is invalid"
            ) from exc
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise IndependentLabelMaterialIntegrityError(
                "independent label material envelope differs from its reference"
            )
        return envelope

    def _read_payload(
        self,
        envelope: _IndependentLabelMaterialEnvelopeV1,
        *,
        limit: int,
    ) -> bytes:
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise IndependentLabelMaterialIntegrityError(
                "independent label material is missing or corrupt"
            ) from exc
        if len(payload) > limit or len(payload) != envelope.canonical_size_bytes:
            raise IndependentLabelMaterialLimitError("stored independent label material size is invalid")
        return payload

    @staticmethod
    def _decode(
        codec: IndependentLabelMaterialCodec,
        payload: bytes,
        ref: ObjectRef,
    ) -> _Model:
        model_type: type[_Model]
        if codec is IndependentLabelMaterialCodec.CANDIDATE_POOL:
            model_type = IndependentLabelCandidatePoolV1
        elif codec is IndependentLabelMaterialCodec.PARTITION:
            model_type = IndependentLabelPartitionMaterialV1
        else:
            model_type = IndependentLabelTestSetMaterialV1
        try:
            value = model_type.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelMaterialIntegrityError(
                "independent label material payload is invalid"
            ) from exc
        if value.canonical_json() != payload or _ref_for(value) != ref:
            raise IndependentLabelMaterialIntegrityError(
                "independent label material payload differs from its reference"
            )
        return value

    def _write_envelope(
        self,
        envelope: _IndependentLabelMaterialEnvelopeV1,
    ) -> bool:
        path = self._envelope_path(envelope.object_ref)
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        rendered = envelope.canonical_json() + b"\n"
        try:
            with temporary.open("xb") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
            return True
        finally:
            temporary.unlink(missing_ok=True)

    def _maybe_raise(
        self,
        point: IndependentLabelMaterialStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _codec_for_ref(ref: ObjectRef) -> IndependentLabelMaterialCodec:
    mapping = {
        "independent-label-candidate-pool": IndependentLabelMaterialCodec.CANDIDATE_POOL,
        "independent-label-partition-material": IndependentLabelMaterialCodec.PARTITION,
        "independent-label-test-set-material": IndependentLabelMaterialCodec.TEST_SET,
    }
    codec = mapping.get(ref.object_type)
    if codec is None or ref.object_version != "private-v1":
        raise IndependentLabelMaterialTypeError("independent label material reference has the wrong type")
    return codec


def _require_ref_codec(
    ref: ObjectRef,
    codec: IndependentLabelMaterialCodec,
) -> None:
    if _codec_for_ref(ref) is not codec:
        raise IndependentLabelMaterialTypeError("independent label material reference has the wrong codec")


def _ref_for(value: _Model) -> ObjectRef:
    if isinstance(value, IndependentLabelCandidatePoolV1):
        return independent_label_candidate_pool_v1_ref(value)
    if isinstance(value, IndependentLabelPartitionMaterialV1):
        return independent_label_partition_material_v1_ref(value)
    return independent_label_test_set_material_v1_ref(value)


__all__ = [
    "IndependentLabelMaterialCodec",
    "IndependentLabelMaterialConflictError",
    "IndependentLabelMaterialError",
    "IndependentLabelMaterialFaultInjector",
    "IndependentLabelMaterialInjectedCrash",
    "IndependentLabelMaterialIntegrityError",
    "IndependentLabelMaterialLimitError",
    "IndependentLabelMaterialStore",
    "IndependentLabelMaterialStoreFaultPoint",
    "IndependentLabelMaterialTypeError",
    "IndependentLabelMaterialWrite",
    "StaticIndependentLabelMaterialFaultInjector",
]
