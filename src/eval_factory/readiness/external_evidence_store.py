from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.readiness.external_evidence_models import (
    ExternalCorpusInventoryV1,
    ExternalEvidenceMaterialClosureV1,
    ExternalPartitionInventoryV1,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceSetV1,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ExternalEvidenceStoreError(RuntimeError):
    pass


class ExternalEvidenceStoreTypeError(ExternalEvidenceStoreError):
    pass


class ExternalEvidenceStoreConflictError(ExternalEvidenceStoreError):
    pass


class ExternalEvidenceStoreIntegrityError(ExternalEvidenceStoreError):
    pass


class ExternalEvidenceStoreLimitError(ExternalEvidenceStoreError):
    pass


class ExternalEvidenceStoreInjectedCrash(ExternalEvidenceStoreError):
    pass


class ExternalEvidenceStoreCodec(StrEnum):
    INVENTORY = "inventory"
    MATERIAL_CLOSURE = "material-closure"
    OBSERVATION_SET = "observation-set"
    PARTITION = "partition"
    REFERENCE_SET = "reference-set"


class ExternalEvidenceStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class ExternalEvidenceStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ExternalEvidenceStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticExternalEvidenceStoreFaultInjector:
    crash_points: frozenset[ExternalEvidenceStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ExternalEvidenceStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ExternalEvidenceStoreInjectedCrash(
                f"injected external evidence store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ExternalEvidenceStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _ExternalEvidenceEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-evidence-envelope/private-v1"] = (
        "eval-factory/external-evidence-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: ExternalEvidenceStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> _ExternalEvidenceEnvelopeV1:
        expected_id = (
            f"external-evidence-content://{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "external-evidence-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("external evidence content ref is invalid")
        return self


_Material = (
    BlindLabelObservationSetV1
    | ExternalCorpusInventoryV1
    | ExternalEvidenceMaterialClosureV1
    | ExternalPartitionInventoryV1
    | ExternalReferenceSetV1
)


class ExternalEvidenceMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: ExternalEvidenceStoreFaultInjector | None = None,
    ) -> None:
        if max_private_bytes < 2 or max_members < 1:
            raise ExternalEvidenceStoreLimitError("external evidence store limits are invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ExternalEvidenceStoreTypeError(
                "external evidence store root must be a non-symlink directory"
            )
        resolved = candidate.resolve()
        self.max_private_bytes = max_private_bytes
        self.max_members = max_members
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"

    def put_inventory(
        self,
        value: ExternalCorpusInventoryV1,
    ) -> ExternalEvidenceStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalEvidenceStoreCodec.INVENTORY,
            member_count=len(value.members),
        )

    def put_partition(
        self,
        value: ExternalPartitionInventoryV1,
    ) -> ExternalEvidenceStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalEvidenceStoreCodec.PARTITION,
            member_count=len(value.members),
        )

    def put_material_closure(
        self,
        value: ExternalEvidenceMaterialClosureV1,
    ) -> ExternalEvidenceStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalEvidenceStoreCodec.MATERIAL_CLOSURE,
            member_count=value.material_object_count,
        )

    def put_reference_set(
        self,
        value: ExternalReferenceSetV1,
    ) -> ExternalEvidenceStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalEvidenceStoreCodec.REFERENCE_SET,
            member_count=len(value.records),
        )

    def put_observation_set(
        self,
        value: BlindLabelObservationSetV1,
    ) -> ExternalEvidenceStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalEvidenceStoreCodec.OBSERVATION_SET,
            member_count=len(value.records),
        )

    def get_inventory(
        self,
        ref: ObjectRef,
    ) -> ExternalCorpusInventoryV1:
        value = self._get(ref, ExternalEvidenceStoreCodec.INVENTORY)
        if not isinstance(value, ExternalCorpusInventoryV1):
            raise ExternalEvidenceStoreTypeError("external inventory codec returned the wrong model")
        return value

    def get_partition(
        self,
        ref: ObjectRef,
    ) -> ExternalPartitionInventoryV1:
        value = self._get(ref, ExternalEvidenceStoreCodec.PARTITION)
        if not isinstance(value, ExternalPartitionInventoryV1):
            raise ExternalEvidenceStoreTypeError("external partition codec returned the wrong model")
        return value

    def get_material_closure(
        self,
        ref: ObjectRef,
    ) -> ExternalEvidenceMaterialClosureV1:
        value = self._get(
            ref,
            ExternalEvidenceStoreCodec.MATERIAL_CLOSURE,
        )
        if not isinstance(value, ExternalEvidenceMaterialClosureV1):
            raise ExternalEvidenceStoreTypeError("external material-closure codec returned the wrong model")
        return value

    def get_reference_set(
        self,
        ref: ObjectRef,
    ) -> ExternalReferenceSetV1:
        value = self._get(ref, ExternalEvidenceStoreCodec.REFERENCE_SET)
        if not isinstance(value, ExternalReferenceSetV1):
            raise ExternalEvidenceStoreTypeError("external reference-set codec returned the wrong model")
        return value

    def get_observation_set(
        self,
        ref: ObjectRef,
    ) -> BlindLabelObservationSetV1:
        value = self._get(ref, ExternalEvidenceStoreCodec.OBSERVATION_SET)
        if not isinstance(value, BlindLabelObservationSetV1):
            raise ExternalEvidenceStoreTypeError("external observation-set codec returned the wrong model")
        return value

    def verify_inventory(self, ref: ObjectRef) -> None:
        self.get_inventory(ref)

    def verify_partition(self, ref: ObjectRef) -> None:
        self.get_partition(ref)

    def verify_material_closure(self, ref: ObjectRef) -> None:
        self.get_material_closure(ref)

    def verify_reference_set(self, ref: ObjectRef) -> None:
        self.get_reference_set(ref)

    def verify_observation_set(self, ref: ObjectRef) -> None:
        self.get_observation_set(ref)

    def _put(
        self,
        value: _Material,
        *,
        ref: ObjectRef,
        codec: ExternalEvidenceStoreCodec,
        member_count: int,
    ) -> ExternalEvidenceStoreWrite:
        if member_count > self.max_members:
            raise ExternalEvidenceStoreLimitError("external evidence material exceeds the member limit")
        payload = value.canonical_json()
        if len(payload) > self.max_private_bytes:
            raise ExternalEvidenceStoreLimitError("external evidence material exceeds the byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        content_sha256 = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="external-evidence-content",
            object_id=(f"external-evidence-content://{codec.value}/sha256/{content_sha256}"),
            object_version="private-v1",
            object_sha256=content_sha256,
        )
        envelope = _ExternalEvidenceEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=content_sha256,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ExternalEvidenceStoreConflictError("external evidence CAS write conflicted") from exc
        self._maybe_raise(
            ExternalEvidenceStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._maybe_raise(
            ExternalEvidenceStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise ExternalEvidenceStoreIntegrityError("external evidence envelope disappeared")
            return existing
        self._get(ref, codec)
        return ExternalEvidenceStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def _get(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceStoreCodec,
    ) -> _Material:
        _require_ref_codec(ref, codec)
        envelope = self._read_envelope(ref, codec)
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ExternalEvidenceStoreIntegrityError(
                "external evidence content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_private_bytes or len(payload) != envelope.canonical_size_bytes:
            raise ExternalEvidenceStoreIntegrityError("stored external evidence byte size is invalid")
        try:
            if codec is ExternalEvidenceStoreCodec.INVENTORY:
                value: _Material = ExternalCorpusInventoryV1.model_validate_json(payload)
            elif codec is ExternalEvidenceStoreCodec.PARTITION:
                value = ExternalPartitionInventoryV1.model_validate_json(payload)
            elif codec is ExternalEvidenceStoreCodec.MATERIAL_CLOSURE:
                value = ExternalEvidenceMaterialClosureV1.model_validate_json(payload)
            elif codec is ExternalEvidenceStoreCodec.REFERENCE_SET:
                value = ExternalReferenceSetV1.model_validate_json(payload)
            else:
                value = BlindLabelObservationSetV1.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ExternalEvidenceStoreIntegrityError("stored external evidence material is invalid") from exc
        if value.canonical_json() != payload or _material_ref(value) != ref:
            raise ExternalEvidenceStoreIntegrityError(
                "stored external evidence material differs from its ref"
            )
        if isinstance(
            value,
            ExternalReferenceSetV1 | BlindLabelObservationSetV1,
        ):
            count = len(value.records)
        elif isinstance(value, ExternalEvidenceMaterialClosureV1):
            count = value.material_object_count
        else:
            count = len(value.members)
        if count > self.max_members:
            raise ExternalEvidenceStoreLimitError(
                "stored external evidence material exceeds the member limit"
            )
        return value

    def _existing(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceStoreCodec,
    ) -> ExternalEvidenceStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self._get(ref, codec)
        return ExternalEvidenceStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _read_envelope(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceStoreCodec,
    ) -> _ExternalEvidenceEnvelopeV1:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            raise ExternalEvidenceStoreIntegrityError("external evidence envelope is missing")
        try:
            envelope = _ExternalEvidenceEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise ExternalEvidenceStoreIntegrityError("external evidence envelope is invalid") from exc
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise ExternalEvidenceStoreIntegrityError("external evidence envelope authority differs")
        return envelope

    def _write_envelope(
        self,
        envelope: _ExternalEvidenceEnvelopeV1,
    ) -> bool:
        path = self._envelope_path(envelope.object_ref, envelope.codec)
        if path.exists():
            self._read_envelope(envelope.object_ref, envelope.codec)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(envelope.canonical_json() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                self._read_envelope(envelope.object_ref, envelope.codec)
                return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: ExternalEvidenceStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _maybe_raise(
        self,
        point: ExternalEvidenceStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _require_ref_codec(
    ref: ObjectRef,
    codec: ExternalEvidenceStoreCodec,
) -> None:
    expected_type = {
        ExternalEvidenceStoreCodec.INVENTORY: "external-corpus-inventory",
        ExternalEvidenceStoreCodec.MATERIAL_CLOSURE: ("external-evidence-material-closure"),
        ExternalEvidenceStoreCodec.OBSERVATION_SET: "blind-label-observation-set",
        ExternalEvidenceStoreCodec.PARTITION: "external-partition-inventory",
        ExternalEvidenceStoreCodec.REFERENCE_SET: "external-reference-set",
    }[codec]
    if ref.object_type != expected_type or ref.object_version != "private-v1":
        raise ExternalEvidenceStoreTypeError("external evidence material ref has the wrong type")


def _material_ref(
    value: _Material,
) -> ObjectRef:
    return value.to_ref()


__all__ = [
    "ExternalEvidenceMaterialStore",
    "ExternalEvidenceStoreCodec",
    "ExternalEvidenceStoreConflictError",
    "ExternalEvidenceStoreError",
    "ExternalEvidenceStoreFaultInjector",
    "ExternalEvidenceStoreFaultPoint",
    "ExternalEvidenceStoreInjectedCrash",
    "ExternalEvidenceStoreIntegrityError",
    "ExternalEvidenceStoreLimitError",
    "ExternalEvidenceStoreTypeError",
    "ExternalEvidenceStoreWrite",
    "StaticExternalEvidenceStoreFaultInjector",
]
