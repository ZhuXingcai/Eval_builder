from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityReportV2,
    real_trace_stability_report_v2_ref,
    validate_real_trace_stability_report_v2_identity,
)
from eval_factory.readiness.real_trace_stability_models import (
    RealTraceStabilityCaseResultV1,
    RealTraceStabilityInventoryV1,
    RealTraceStabilityResultSetV1,
    real_trace_stability_case_result_v1_ref,
    real_trace_stability_inventory_v1_ref,
    real_trace_stability_result_set_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class RealTraceStabilityStoreError(RuntimeError):
    pass


class RealTraceStabilityStoreTypeError(RealTraceStabilityStoreError):
    pass


class RealTraceStabilityStoreConflictError(RealTraceStabilityStoreError):
    pass


class RealTraceStabilityStoreIntegrityError(RealTraceStabilityStoreError):
    pass


class RealTraceStabilityStoreLimitError(RealTraceStabilityStoreError):
    pass


class RealTraceStabilityStoreInjectedCrash(RealTraceStabilityStoreError):
    pass


class RealTraceStabilityStoreCodec(StrEnum):
    INVENTORY = "inventory"
    CASE_RESULT = "case-result"
    RESULT_SET = "result-set"
    REPORT = "report"


class RealTraceStabilityStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class RealTraceStabilityStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: RealTraceStabilityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticRealTraceStabilityStoreFaultInjector:
    crash_points: frozenset[RealTraceStabilityStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: RealTraceStabilityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise RealTraceStabilityStoreInjectedCrash(
                f"injected real-trace stability store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class RealTraceStabilityStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _RealTraceStabilityEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-envelope/private-v1"] = (
        "eval-factory/real-trace-stability-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: RealTraceStabilityStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_blob_ref.object_type != "real-trace-stability-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id
            != (
                "real-trace-stability-content://"
                f"{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}"
            )
        ):
            raise ValueError("real-trace stability content ref is invalid")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: RealTraceStabilityStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise RealTraceStabilityStoreLimitError("real-trace stability byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise RealTraceStabilityStoreTypeError(
                "real-trace stability store root must be a non-symlink directory"
            )
        resolved = candidate.resolve()
        self.max_bytes = max_bytes
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"

    def put(
        self,
        *,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
        payload: bytes,
    ) -> RealTraceStabilityStoreWrite:
        if len(payload) > self.max_bytes:
            raise RealTraceStabilityStoreLimitError("real-trace stability value exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="real-trace-stability-content",
            object_id=(f"real-trace-stability-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _RealTraceStabilityEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise RealTraceStabilityStoreConflictError("real-trace stability CAS write conflicted") from exc
        self._maybe_raise(
            RealTraceStabilityStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._maybe_raise(
            RealTraceStabilityStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise RealTraceStabilityStoreIntegrityError("real-trace stability envelope disappeared")
            return existing
        self.get(ref=ref, codec=codec)
        return RealTraceStabilityStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
    ) -> bytes:
        envelope = self._read_envelope(ref, codec)
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise RealTraceStabilityStoreIntegrityError(
                "real-trace stability content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise RealTraceStabilityStoreLimitError("stored real-trace stability byte size is invalid")
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
    ) -> RealTraceStabilityStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return RealTraceStabilityStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(
        self,
        envelope: _RealTraceStabilityEnvelopeV1,
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

    def _read_envelope(
        self,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
    ) -> _RealTraceStabilityEnvelopeV1:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            raise RealTraceStabilityStoreIntegrityError("real-trace stability envelope is missing")
        try:
            envelope = _RealTraceStabilityEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise RealTraceStabilityStoreIntegrityError("real-trace stability envelope is invalid") from exc
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise RealTraceStabilityStoreIntegrityError("real-trace stability envelope authority differs")
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _maybe_raise(
        self,
        point: RealTraceStabilityStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_Material = RealTraceStabilityInventoryV1 | RealTraceStabilityCaseResultV1 | RealTraceStabilityResultSetV1


class RealTraceStabilityMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: RealTraceStabilityStoreFaultInjector | None = None,
    ) -> None:
        if max_members < 1:
            raise RealTraceStabilityStoreLimitError("real-trace stability member limit is invalid")
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )

    def put_inventory(
        self,
        value: RealTraceStabilityInventoryV1,
    ) -> RealTraceStabilityStoreWrite:
        return self._put(
            value,
            ref=real_trace_stability_inventory_v1_ref(value),
            codec=RealTraceStabilityStoreCodec.INVENTORY,
            member_count=len(value.members),
        )

    def put_case_result(
        self,
        value: RealTraceStabilityCaseResultV1,
    ) -> RealTraceStabilityStoreWrite:
        return self._put(
            value,
            ref=real_trace_stability_case_result_v1_ref(value),
            codec=RealTraceStabilityStoreCodec.CASE_RESULT,
            member_count=1,
        )

    def put_result_set(
        self,
        value: RealTraceStabilityResultSetV1,
    ) -> RealTraceStabilityStoreWrite:
        return self._put(
            value,
            ref=real_trace_stability_result_set_v1_ref(value),
            codec=RealTraceStabilityStoreCodec.RESULT_SET,
            member_count=len(value.case_result_refs),
        )

    def get_inventory(
        self,
        ref: ObjectRef,
    ) -> RealTraceStabilityInventoryV1:
        value = self._get(ref, RealTraceStabilityStoreCodec.INVENTORY)
        if not isinstance(value, RealTraceStabilityInventoryV1):
            raise RealTraceStabilityStoreTypeError("inventory codec returned the wrong model")
        return value

    def get_case_result(
        self,
        ref: ObjectRef,
    ) -> RealTraceStabilityCaseResultV1:
        value = self._get(ref, RealTraceStabilityStoreCodec.CASE_RESULT)
        if not isinstance(value, RealTraceStabilityCaseResultV1):
            raise RealTraceStabilityStoreTypeError("case-result codec returned the wrong model")
        return value

    def get_result_set(
        self,
        ref: ObjectRef,
    ) -> RealTraceStabilityResultSetV1:
        value = self._get(ref, RealTraceStabilityStoreCodec.RESULT_SET)
        if not isinstance(value, RealTraceStabilityResultSetV1):
            raise RealTraceStabilityStoreTypeError("result-set codec returned the wrong model")
        return value

    def _put(
        self,
        value: _Material,
        *,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
        member_count: int,
    ) -> RealTraceStabilityStoreWrite:
        if member_count > self.max_members:
            raise RealTraceStabilityStoreLimitError("real-trace stability material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get(
        self,
        ref: ObjectRef,
        codec: RealTraceStabilityStoreCodec,
    ) -> _Material:
        _require_material_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            if codec is RealTraceStabilityStoreCodec.INVENTORY:
                value: _Material = RealTraceStabilityInventoryV1.model_validate_json(payload)
            elif codec is RealTraceStabilityStoreCodec.CASE_RESULT:
                value = RealTraceStabilityCaseResultV1.model_validate_json(payload)
            else:
                value = RealTraceStabilityResultSetV1.model_validate_json(payload)
            observed = _material_ref(value)
        except (ValidationError, ValueError) as exc:
            raise RealTraceStabilityStoreIntegrityError(
                "stored real-trace stability material is invalid"
            ) from exc
        if observed != ref or value.canonical_json() != payload:
            raise RealTraceStabilityStoreIntegrityError(
                "stored real-trace stability material differs from its ref"
            )
        if isinstance(value, RealTraceStabilityInventoryV1):
            count = len(value.members)
        elif isinstance(value, RealTraceStabilityResultSetV1):
            count = len(value.case_result_refs)
        else:
            count = 1
        if count > self.max_members:
            raise RealTraceStabilityStoreLimitError(
                "stored real-trace stability material exceeds member limit"
            )
        return value


class RealTraceStabilityReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        max_cases: int,
        fault_injector: RealTraceStabilityStoreFaultInjector | None = None,
    ) -> None:
        if max_cases < 1:
            raise RealTraceStabilityStoreLimitError("real-trace stability case limit is invalid")
        self.max_cases = max_cases
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )

    def put(
        self,
        value: RealTraceStabilityReportV2,
    ) -> RealTraceStabilityStoreWrite:
        try:
            validate_real_trace_stability_report_v2_identity(value)
        except ValueError as exc:
            raise RealTraceStabilityStoreConflictError(
                "real-trace stability report identity is stale"
            ) from exc
        if len(value.case_summaries) > self.max_cases:
            raise RealTraceStabilityStoreLimitError("real-trace stability report exceeds case limit")
        return self._store.put(
            ref=real_trace_stability_report_v2_ref(value),
            codec=RealTraceStabilityStoreCodec.REPORT,
            payload=value.canonical_json(),
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> RealTraceStabilityReportV2:
        if ref.object_type != "real-trace-stability-report" or ref.object_version != "v2":
            raise RealTraceStabilityStoreTypeError("real-trace stability report ref has wrong type")
        payload = self._store.get(
            ref=ref,
            codec=RealTraceStabilityStoreCodec.REPORT,
        )
        try:
            value = RealTraceStabilityReportV2.model_validate_json(payload)
            validate_real_trace_stability_report_v2_identity(value)
            observed = real_trace_stability_report_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise RealTraceStabilityStoreIntegrityError(
                "stored real-trace stability report is invalid"
            ) from exc
        if len(value.case_summaries) > self.max_cases or observed != ref or value.canonical_json() != payload:
            raise RealTraceStabilityStoreIntegrityError(
                "stored real-trace stability report differs from its ref"
            )
        return value

    def verify(self, ref: ObjectRef) -> None:
        self.get(ref)


def _require_material_ref(
    ref: ObjectRef,
    codec: RealTraceStabilityStoreCodec,
) -> None:
    expected = {
        RealTraceStabilityStoreCodec.INVENTORY: ("real-trace-stability-inventory"),
        RealTraceStabilityStoreCodec.CASE_RESULT: ("real-trace-stability-case-result"),
        RealTraceStabilityStoreCodec.RESULT_SET: ("real-trace-stability-result-set"),
    }[codec]
    if ref.object_type != expected or ref.object_version != "private-v1":
        raise RealTraceStabilityStoreTypeError("real-trace stability material ref has wrong type")


def _material_ref(value: _Material) -> ObjectRef:
    if isinstance(value, RealTraceStabilityInventoryV1):
        return real_trace_stability_inventory_v1_ref(value)
    if isinstance(value, RealTraceStabilityCaseResultV1):
        return real_trace_stability_case_result_v1_ref(value)
    return real_trace_stability_result_set_v1_ref(value)


__all__ = [
    "RealTraceStabilityMaterialStore",
    "RealTraceStabilityReportStore",
    "RealTraceStabilityStoreCodec",
    "RealTraceStabilityStoreConflictError",
    "RealTraceStabilityStoreError",
    "RealTraceStabilityStoreFaultPoint",
    "RealTraceStabilityStoreInjectedCrash",
    "RealTraceStabilityStoreIntegrityError",
    "RealTraceStabilityStoreLimitError",
    "RealTraceStabilityStoreTypeError",
    "RealTraceStabilityStoreWrite",
    "StaticRealTraceStabilityStoreFaultInjector",
]
