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
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadReportV2,
    scheduler_load_report_v2_ref,
    validate_scheduler_load_report_v2_identity,
)
from eval_factory.readiness.scheduler_load_models import (
    SchedulerLoadCaseResultV1,
    SchedulerLoadFaultObservationV1,
    SchedulerLoadResultSetV1,
    SchedulerLoadRunLedgerV1,
    SchedulerLoadWorkloadV1,
    scheduler_load_case_result_v1_ref,
    scheduler_load_fault_observation_v1_ref,
    scheduler_load_result_set_v1_ref,
    scheduler_load_run_ledger_v1_ref,
    scheduler_load_workload_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class SchedulerLoadStoreError(RuntimeError):
    pass


class SchedulerLoadStoreTypeError(SchedulerLoadStoreError):
    pass


class SchedulerLoadStoreConflictError(SchedulerLoadStoreError):
    pass


class SchedulerLoadStoreIntegrityError(SchedulerLoadStoreError):
    pass


class SchedulerLoadStoreLimitError(SchedulerLoadStoreError):
    pass


class SchedulerLoadStoreInjectedCrash(SchedulerLoadStoreError):
    pass


class SchedulerLoadStoreCodec(StrEnum):
    WORKLOAD = "workload"
    FAULT_OBSERVATION = "fault-observation"
    CASE_RESULT = "case-result"
    RESULT_SET = "result-set"
    RUN_LEDGER = "run-ledger"
    REPORT = "report"


class SchedulerLoadStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class SchedulerLoadStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: SchedulerLoadStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticSchedulerLoadStoreFaultInjector:
    crash_points: frozenset[SchedulerLoadStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: SchedulerLoadStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise SchedulerLoadStoreInjectedCrash(f"injected scheduler load store crash at {point.value}")


@dataclass(frozen=True, slots=True)
class SchedulerLoadStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _SchedulerLoadEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/scheduler-load-envelope/private-v1"] = (
        "eval-factory/scheduler-load-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: SchedulerLoadStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        if (
            self.content_blob_ref.object_type != "scheduler-load-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id
            != (f"scheduler-load-content://{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}")
        ):
            raise ValueError("scheduler load content ref is invalid")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: SchedulerLoadStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise SchedulerLoadStoreLimitError("scheduler load byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise SchedulerLoadStoreTypeError("scheduler load store root must be a non-symlink directory")
        resolved = candidate.resolve()
        self.max_bytes = max_bytes
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(resolved / "cas")
        self._envelopes = resolved / "envelopes"

    def put(
        self,
        *,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
        payload: bytes,
    ) -> SchedulerLoadStoreWrite:
        if len(payload) > self.max_bytes:
            raise SchedulerLoadStoreLimitError("scheduler load value exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="scheduler-load-content",
            object_id=(f"scheduler-load-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _SchedulerLoadEnvelopeV1(
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
            raise SchedulerLoadStoreConflictError("scheduler load CAS write conflicted") from exc
        self._maybe_raise(SchedulerLoadStoreFaultPoint.AFTER_CAS_WRITE, ref)
        written = self._write_envelope(envelope)
        self._maybe_raise(
            SchedulerLoadStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise SchedulerLoadStoreIntegrityError("scheduler load envelope disappeared")
            return existing
        self.get(ref=ref, codec=codec)
        return SchedulerLoadStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
    ) -> bytes:
        envelope = self._read_envelope(ref, codec)
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise SchedulerLoadStoreIntegrityError("scheduler load content is missing or corrupt") from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise SchedulerLoadStoreLimitError("stored scheduler load byte size is invalid")
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
    ) -> SchedulerLoadStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return SchedulerLoadStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(
        self,
        envelope: _SchedulerLoadEnvelopeV1,
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
                self._read_envelope(
                    envelope.object_ref,
                    envelope.codec,
                )
                return False
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def _read_envelope(
        self,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
    ) -> _SchedulerLoadEnvelopeV1:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            raise SchedulerLoadStoreIntegrityError("scheduler load envelope is missing")
        try:
            envelope = _SchedulerLoadEnvelopeV1.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as exc:
            raise SchedulerLoadStoreIntegrityError("scheduler load envelope is invalid") from exc
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise SchedulerLoadStoreIntegrityError("scheduler load envelope authority differs")
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _maybe_raise(
        self,
        point: SchedulerLoadStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_Material = (
    SchedulerLoadWorkloadV1
    | SchedulerLoadFaultObservationV1
    | SchedulerLoadCaseResultV1
    | SchedulerLoadResultSetV1
    | SchedulerLoadRunLedgerV1
)


class SchedulerLoadMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: SchedulerLoadStoreFaultInjector | None = None,
    ) -> None:
        if max_members != 1000:
            raise SchedulerLoadStoreLimitError("scheduler load member limit must be 1000")
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )

    def put_workload(
        self,
        value: SchedulerLoadWorkloadV1,
    ) -> SchedulerLoadStoreWrite:
        return self._put(
            value,
            ref=scheduler_load_workload_v1_ref(value),
            codec=SchedulerLoadStoreCodec.WORKLOAD,
            member_count=len(value.members),
        )

    def put_fault_observation(
        self,
        value: SchedulerLoadFaultObservationV1,
    ) -> SchedulerLoadStoreWrite:
        return self._put(
            value,
            ref=scheduler_load_fault_observation_v1_ref(value),
            codec=SchedulerLoadStoreCodec.FAULT_OBSERVATION,
            member_count=1,
        )

    def put_case_result(
        self,
        value: SchedulerLoadCaseResultV1,
    ) -> SchedulerLoadStoreWrite:
        return self._put(
            value,
            ref=scheduler_load_case_result_v1_ref(value),
            codec=SchedulerLoadStoreCodec.CASE_RESULT,
            member_count=1,
        )

    def put_result_set(
        self,
        value: SchedulerLoadResultSetV1,
    ) -> SchedulerLoadStoreWrite:
        return self._put(
            value,
            ref=scheduler_load_result_set_v1_ref(value),
            codec=SchedulerLoadStoreCodec.RESULT_SET,
            member_count=len(value.case_result_refs),
        )

    def put_run_ledger(
        self,
        value: SchedulerLoadRunLedgerV1,
    ) -> SchedulerLoadStoreWrite:
        return self._put(
            value,
            ref=scheduler_load_run_ledger_v1_ref(value),
            codec=SchedulerLoadStoreCodec.RUN_LEDGER,
            member_count=len(value.prepared_authority_refs) // 6,
        )

    def get_workload(self, ref: ObjectRef) -> SchedulerLoadWorkloadV1:
        value = self._get(ref, SchedulerLoadStoreCodec.WORKLOAD)
        if not isinstance(value, SchedulerLoadWorkloadV1):
            raise SchedulerLoadStoreTypeError("workload codec returned the wrong model")
        return value

    def get_fault_observation(
        self,
        ref: ObjectRef,
    ) -> SchedulerLoadFaultObservationV1:
        value = self._get(ref, SchedulerLoadStoreCodec.FAULT_OBSERVATION)
        if not isinstance(value, SchedulerLoadFaultObservationV1):
            raise SchedulerLoadStoreTypeError("fault codec returned the wrong model")
        return value

    def get_case_result(
        self,
        ref: ObjectRef,
    ) -> SchedulerLoadCaseResultV1:
        value = self._get(ref, SchedulerLoadStoreCodec.CASE_RESULT)
        if not isinstance(value, SchedulerLoadCaseResultV1):
            raise SchedulerLoadStoreTypeError("case-result codec returned the wrong model")
        return value

    def get_result_set(
        self,
        ref: ObjectRef,
    ) -> SchedulerLoadResultSetV1:
        value = self._get(ref, SchedulerLoadStoreCodec.RESULT_SET)
        if not isinstance(value, SchedulerLoadResultSetV1):
            raise SchedulerLoadStoreTypeError("result-set codec returned the wrong model")
        return value

    def get_run_ledger(
        self,
        ref: ObjectRef,
    ) -> SchedulerLoadRunLedgerV1:
        value = self._get(ref, SchedulerLoadStoreCodec.RUN_LEDGER)
        if not isinstance(value, SchedulerLoadRunLedgerV1):
            raise SchedulerLoadStoreTypeError("run-ledger codec returned the wrong model")
        return value

    def _put(
        self,
        value: _Material,
        *,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
        member_count: int,
    ) -> SchedulerLoadStoreWrite:
        if member_count > self.max_members:
            raise SchedulerLoadStoreLimitError("scheduler load material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get(
        self,
        ref: ObjectRef,
        codec: SchedulerLoadStoreCodec,
    ) -> _Material:
        _require_material_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = _decode_material(codec, payload)
            observed = _material_ref(value)
        except (ValidationError, ValueError) as exc:
            raise SchedulerLoadStoreIntegrityError("stored scheduler load material is invalid") from exc
        if observed != ref or value.canonical_json() != payload:
            raise SchedulerLoadStoreIntegrityError("stored scheduler load material differs from its ref")
        count = _member_count(value)
        if count > self.max_members:
            raise SchedulerLoadStoreLimitError("stored scheduler load material exceeds member limit")
        return value


class SchedulerLoadReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        max_cases: int,
        fault_injector: SchedulerLoadStoreFaultInjector | None = None,
    ) -> None:
        if max_cases != 1000:
            raise SchedulerLoadStoreLimitError("scheduler load case limit must be 1000")
        self.max_cases = max_cases
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )

    def put(
        self,
        value: SchedulerLoadReportV2,
    ) -> SchedulerLoadStoreWrite:
        try:
            validate_scheduler_load_report_v2_identity(value)
        except ValueError as exc:
            raise SchedulerLoadStoreConflictError("scheduler load report identity is stale") from exc
        if len(value.case_summaries) != self.max_cases:
            raise SchedulerLoadStoreLimitError("scheduler load report has the wrong case count")
        return self._store.put(
            ref=scheduler_load_report_v2_ref(value),
            codec=SchedulerLoadStoreCodec.REPORT,
            payload=value.canonical_json(),
        )

    def get(self, ref: ObjectRef) -> SchedulerLoadReportV2:
        if ref.object_type != "scheduler-load-report" or ref.object_version != "v2":
            raise SchedulerLoadStoreTypeError("scheduler load report ref has wrong type")
        payload = self._store.get(
            ref=ref,
            codec=SchedulerLoadStoreCodec.REPORT,
        )
        try:
            value = SchedulerLoadReportV2.model_validate_json(payload)
            validate_scheduler_load_report_v2_identity(value)
            observed = scheduler_load_report_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise SchedulerLoadStoreIntegrityError("stored scheduler load report is invalid") from exc
        if (
            len(value.case_summaries) != self.max_cases
            or observed != ref
            or value.canonical_json() != payload
        ):
            raise SchedulerLoadStoreIntegrityError("stored scheduler load report differs from its ref")
        return value

    def verify(self, ref: ObjectRef) -> None:
        self.get(ref)


def _decode_material(
    codec: SchedulerLoadStoreCodec,
    payload: bytes,
) -> _Material:
    if codec is SchedulerLoadStoreCodec.WORKLOAD:
        return SchedulerLoadWorkloadV1.model_validate_json(payload)
    if codec is SchedulerLoadStoreCodec.FAULT_OBSERVATION:
        return SchedulerLoadFaultObservationV1.model_validate_json(payload)
    if codec is SchedulerLoadStoreCodec.CASE_RESULT:
        return SchedulerLoadCaseResultV1.model_validate_json(payload)
    if codec is SchedulerLoadStoreCodec.RESULT_SET:
        return SchedulerLoadResultSetV1.model_validate_json(payload)
    if codec is SchedulerLoadStoreCodec.RUN_LEDGER:
        return SchedulerLoadRunLedgerV1.model_validate_json(payload)
    raise SchedulerLoadStoreTypeError("report is not private material")


def _require_material_ref(
    ref: ObjectRef,
    codec: SchedulerLoadStoreCodec,
) -> None:
    expected = {
        SchedulerLoadStoreCodec.WORKLOAD: "scheduler-load-workload",
        SchedulerLoadStoreCodec.FAULT_OBSERVATION: ("scheduler-load-fault-observation"),
        SchedulerLoadStoreCodec.CASE_RESULT: "scheduler-load-case-result",
        SchedulerLoadStoreCodec.RESULT_SET: "scheduler-load-result-set",
        SchedulerLoadStoreCodec.RUN_LEDGER: "scheduler-load-run-ledger",
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "private-v1":
        raise SchedulerLoadStoreTypeError("scheduler load material ref has wrong type")


def _material_ref(value: _Material) -> ObjectRef:
    if isinstance(value, SchedulerLoadWorkloadV1):
        return scheduler_load_workload_v1_ref(value)
    if isinstance(value, SchedulerLoadFaultObservationV1):
        return scheduler_load_fault_observation_v1_ref(value)
    if isinstance(value, SchedulerLoadCaseResultV1):
        return scheduler_load_case_result_v1_ref(value)
    if isinstance(value, SchedulerLoadResultSetV1):
        return scheduler_load_result_set_v1_ref(value)
    return scheduler_load_run_ledger_v1_ref(value)


def _member_count(value: _Material) -> int:
    if isinstance(value, SchedulerLoadWorkloadV1):
        return len(value.members)
    if isinstance(value, SchedulerLoadResultSetV1):
        return len(value.case_result_refs)
    if isinstance(value, SchedulerLoadRunLedgerV1):
        return len(value.prepared_authority_refs) // 6
    return 1


__all__ = [
    "SchedulerLoadMaterialStore",
    "SchedulerLoadReportStore",
    "SchedulerLoadStoreCodec",
    "SchedulerLoadStoreConflictError",
    "SchedulerLoadStoreError",
    "SchedulerLoadStoreFaultPoint",
    "SchedulerLoadStoreInjectedCrash",
    "SchedulerLoadStoreIntegrityError",
    "SchedulerLoadStoreLimitError",
    "SchedulerLoadStoreTypeError",
    "SchedulerLoadStoreWrite",
    "StaticSchedulerLoadStoreFaultInjector",
]
