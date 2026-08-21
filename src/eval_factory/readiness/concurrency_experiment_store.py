from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentReportV2,
    concurrency_experiment_report_v2_ref,
    validate_concurrency_experiment_report_v2_identity,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.readiness.concurrency_experiment_models import (
    ConcurrencyExperimentCaseSampleV1,
    ConcurrencyExperimentFaultObservationV1,
    ConcurrencyExperimentLevelResultV1,
    ConcurrencyExperimentRecoveryResultV1,
    ConcurrencyExperimentResultSetV1,
    ConcurrencyExperimentRunLedgerV1,
    ConcurrencyExperimentTrialResultV1,
    ConcurrencyExperimentWorkloadV1,
    concurrency_experiment_case_sample_v1_ref,
    concurrency_experiment_fault_observation_v1_ref,
    concurrency_experiment_level_result_v1_ref,
    concurrency_experiment_recovery_result_v1_ref,
    concurrency_experiment_result_set_v1_ref,
    concurrency_experiment_run_ledger_v1_ref,
    concurrency_experiment_trial_result_v1_ref,
    concurrency_experiment_workload_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ConcurrencyExperimentStoreError(RuntimeError):
    pass


class ConcurrencyExperimentStoreTypeError(ConcurrencyExperimentStoreError):
    pass


class ConcurrencyExperimentStoreConflictError(ConcurrencyExperimentStoreError):
    pass


class ConcurrencyExperimentStoreIntegrityError(ConcurrencyExperimentStoreError):
    pass


class ConcurrencyExperimentStoreLimitError(ConcurrencyExperimentStoreError):
    pass


class ConcurrencyExperimentStoreInjectedCrash(ConcurrencyExperimentStoreError):
    pass


class ConcurrencyExperimentStoreCodec(StrEnum):
    HOST = "host"
    WORKLOAD = "workload"
    CASE_SAMPLE = "case-sample"
    TRIAL_RESULT = "trial-result"
    LEVEL_RESULT = "level-result"
    FAULT_OBSERVATION = "fault-observation"
    RECOVERY_RESULT = "recovery-result"
    RESULT_SET = "result-set"
    RUN_LEDGER = "run-ledger"
    REPORT = "report"


class ConcurrencyExperimentStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class ConcurrencyExperimentStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ConcurrencyExperimentStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticConcurrencyExperimentStoreFaultInjector:
    crash_points: frozenset[ConcurrencyExperimentStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ConcurrencyExperimentStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ConcurrencyExperimentStoreInjectedCrash(
                f"injected concurrency store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ConcurrencyExperimentStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


@dataclass(frozen=True, slots=True)
class ConcurrencyExperimentResultSetClosure:
    result_set: ConcurrencyExperimentResultSetV1
    level_results: tuple[ConcurrencyExperimentLevelResultV1, ...]
    recovery_results: tuple[ConcurrencyExperimentRecoveryResultV1, ...]


class _ConcurrencyEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/concurrency-experiment-envelope/private-v1"] = (
        "eval-factory/concurrency-experiment-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: ConcurrencyExperimentStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        expected_id = (
            "concurrency-experiment-content://"
            f"{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "concurrency-experiment-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("concurrency experiment content ref is invalid")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: ConcurrencyExperimentStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise ConcurrencyExperimentStoreLimitError("concurrency experiment byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ConcurrencyExperimentStoreTypeError(
                "concurrency experiment root must be a non-symlink directory"
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
        codec: ConcurrencyExperimentStoreCodec,
        payload: bytes,
    ) -> ConcurrencyExperimentStoreWrite:
        if len(payload) > self.max_bytes:
            raise ConcurrencyExperimentStoreLimitError("concurrency experiment value exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="concurrency-experiment-content",
            object_id=(f"concurrency-experiment-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _ConcurrencyEnvelopeV1(
            object_ref=ref,
            codec=codec,
            content_blob_ref=content_ref,
            canonical_size_bytes=len(payload),
        )
        _require_safe_store_member(
            self._cas.root,
            self._cas.blob_path(digest),
        )
        try:
            content_written = self._cas.write(
                object_id=content_ref.object_id,
                digest=digest,
                value=payload,
            )
        except ContentAddressedByteStoreConflictError as exc:
            raise ConcurrencyExperimentStoreConflictError(
                "concurrency experiment CAS write conflicted"
            ) from exc
        self._maybe_raise(
            ConcurrencyExperimentStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._maybe_raise(
            ConcurrencyExperimentStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise ConcurrencyExperimentStoreIntegrityError("concurrency experiment envelope disappeared")
            return existing
        self.get(ref=ref, codec=codec)
        return ConcurrencyExperimentStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: ConcurrencyExperimentStoreCodec,
    ) -> bytes:
        envelope = self._read_envelope(ref, codec)
        _require_safe_store_member(
            self._cas.root,
            self._cas.blob_path(envelope.content_blob_ref.object_sha256),
        )
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ConcurrencyExperimentStoreIntegrityError(
                "concurrency experiment content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise ConcurrencyExperimentStoreLimitError("stored concurrency experiment byte size is invalid")
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: ConcurrencyExperimentStoreCodec,
    ) -> ConcurrencyExperimentStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return ConcurrencyExperimentStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(self, envelope: _ConcurrencyEnvelopeV1) -> bool:
        path = self._envelope_path(envelope.object_ref, envelope.codec)
        _require_safe_store_member(self._envelopes, path)
        if path.exists():
            self._read_envelope(envelope.object_ref, envelope.codec)
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_store_member(self._envelopes, path)
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
        codec: ConcurrencyExperimentStoreCodec,
    ) -> _ConcurrencyEnvelopeV1:
        path = self._envelope_path(ref, codec)
        _require_safe_store_member(self._envelopes, path)
        if not path.exists():
            raise ConcurrencyExperimentStoreIntegrityError("concurrency experiment envelope is missing")
        try:
            encoded = path.read_bytes()
            envelope = _ConcurrencyEnvelopeV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ConcurrencyExperimentStoreIntegrityError(
                "concurrency experiment envelope is invalid"
            ) from exc
        if encoded != envelope.canonical_json() + b"\n":
            raise ConcurrencyExperimentStoreIntegrityError("concurrency experiment envelope is not canonical")
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise ConcurrencyExperimentStoreIntegrityError(
                "concurrency experiment envelope authority differs"
            )
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: ConcurrencyExperimentStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _maybe_raise(
        self,
        point: ConcurrencyExperimentStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_Material = (
    ConcurrencyExperimentHostSummaryV2
    | ConcurrencyExperimentWorkloadV1
    | ConcurrencyExperimentCaseSampleV1
    | ConcurrencyExperimentTrialResultV1
    | ConcurrencyExperimentLevelResultV1
    | ConcurrencyExperimentFaultObservationV1
    | ConcurrencyExperimentRecoveryResultV1
    | ConcurrencyExperimentResultSetV1
    | ConcurrencyExperimentRunLedgerV1
)


class ConcurrencyExperimentMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: ConcurrencyExperimentStoreFaultInjector | None = None,
    ) -> None:
        if max_members < 384 or max_members > 1_000_000:
            raise ConcurrencyExperimentStoreLimitError("concurrency experiment member limit is invalid")
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )

    def put_host(
        self,
        value: ConcurrencyExperimentHostSummaryV2,
    ) -> ConcurrencyExperimentStoreWrite:
        digest = value.host_summary_sha256
        ref = ObjectRef(
            object_type="concurrency-experiment-host",
            object_id=f"concurrency-experiment-host://sha256/{digest}",
            object_version="private-v1",
            object_sha256=digest,
        )
        return self._put(value, ref, ConcurrencyExperimentStoreCodec.HOST, 1)

    def put_workload(
        self,
        value: ConcurrencyExperimentWorkloadV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_workload_v1_ref(value),
            ConcurrencyExperimentStoreCodec.WORKLOAD,
            len(value.trials),
        )

    def put_case_sample(
        self,
        value: ConcurrencyExperimentCaseSampleV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_case_sample_v1_ref(value),
            ConcurrencyExperimentStoreCodec.CASE_SAMPLE,
            1,
        )

    def put_trial_result(
        self,
        value: ConcurrencyExperimentTrialResultV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_trial_result_v1_ref(value),
            ConcurrencyExperimentStoreCodec.TRIAL_RESULT,
            len(value.case_sample_refs),
        )

    def put_level_result(
        self,
        value: ConcurrencyExperimentLevelResultV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_level_result_v1_ref(value),
            ConcurrencyExperimentStoreCodec.LEVEL_RESULT,
            len(value.trial_result_refs),
        )

    def put_fault_observation(
        self,
        value: ConcurrencyExperimentFaultObservationV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_fault_observation_v1_ref(value),
            ConcurrencyExperimentStoreCodec.FAULT_OBSERVATION,
            1,
        )

    def put_recovery_result(
        self,
        value: ConcurrencyExperimentRecoveryResultV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_recovery_result_v1_ref(value),
            ConcurrencyExperimentStoreCodec.RECOVERY_RESULT,
            1,
        )

    def put_result_set(
        self,
        value: ConcurrencyExperimentResultSetV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_result_set_v1_ref(value),
            ConcurrencyExperimentStoreCodec.RESULT_SET,
            len(value.level_result_refs) + len(value.recovery_result_refs),
        )

    def put_run_ledger(
        self,
        value: ConcurrencyExperimentRunLedgerV1,
    ) -> ConcurrencyExperimentStoreWrite:
        return self._put(
            value,
            concurrency_experiment_run_ledger_v1_ref(value),
            ConcurrencyExperimentStoreCodec.RUN_LEDGER,
            len(value.child_manifest_refs),
        )

    def get_host(self, ref: ObjectRef) -> ConcurrencyExperimentHostSummaryV2:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.HOST,
            ConcurrencyExperimentHostSummaryV2,
        )

    def get_workload(self, ref: ObjectRef) -> ConcurrencyExperimentWorkloadV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.WORKLOAD,
            ConcurrencyExperimentWorkloadV1,
        )

    def get_case_sample(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentCaseSampleV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.CASE_SAMPLE,
            ConcurrencyExperimentCaseSampleV1,
        )

    def get_trial_result(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentTrialResultV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.TRIAL_RESULT,
            ConcurrencyExperimentTrialResultV1,
        )

    def get_level_result(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentLevelResultV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.LEVEL_RESULT,
            ConcurrencyExperimentLevelResultV1,
        )

    def get_fault_observation(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentFaultObservationV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.FAULT_OBSERVATION,
            ConcurrencyExperimentFaultObservationV1,
        )

    def get_recovery_result(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentRecoveryResultV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.RECOVERY_RESULT,
            ConcurrencyExperimentRecoveryResultV1,
        )

    def get_result_set(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentResultSetV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.RESULT_SET,
            ConcurrencyExperimentResultSetV1,
        )

    def get_result_set_closure(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentResultSetClosure:
        result_set = self.get_result_set(ref)
        levels = tuple(self.get_level_result(level_ref) for level_ref in result_set.level_result_refs)
        for level in levels:
            for trial_ref in level.trial_result_refs:
                trial = self.get_trial_result(trial_ref)
                for case_ref in trial.case_sample_refs:
                    self.get_case_sample(case_ref)
        recoveries = tuple(
            self.get_recovery_result(recovery_ref) for recovery_ref in result_set.recovery_result_refs
        )
        for recovery in recoveries:
            self.get_fault_observation(recovery.fault_observation_ref)
        return ConcurrencyExperimentResultSetClosure(
            result_set=result_set,
            level_results=levels,
            recovery_results=recoveries,
        )

    def get_run_ledger(
        self,
        ref: ObjectRef,
    ) -> ConcurrencyExperimentRunLedgerV1:
        return self._typed_get(
            ref,
            ConcurrencyExperimentStoreCodec.RUN_LEDGER,
            ConcurrencyExperimentRunLedgerV1,
        )

    def _put(
        self,
        value: _Material,
        ref: ObjectRef,
        codec: ConcurrencyExperimentStoreCodec,
        member_count: int,
    ) -> ConcurrencyExperimentStoreWrite:
        if member_count > self.max_members:
            raise ConcurrencyExperimentStoreLimitError("concurrency experiment material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _typed_get[ModelT: ContractModelV2](
        self,
        ref: ObjectRef,
        codec: ConcurrencyExperimentStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        _require_material_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            observed = _material_ref(value)
        except (ValidationError, ValueError) as exc:
            raise ConcurrencyExperimentStoreIntegrityError(
                "stored concurrency experiment material is invalid"
            ) from exc
        if observed != ref or value.canonical_json() != payload:
            raise ConcurrencyExperimentStoreIntegrityError(
                "stored concurrency experiment material differs from its ref"
            )
        if _member_count(value) > self.max_members:
            raise ConcurrencyExperimentStoreLimitError(
                "stored concurrency experiment material exceeds member limit"
            )
        return value


class ConcurrencyExperimentReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        fault_injector: ConcurrencyExperimentStoreFaultInjector | None = None,
    ) -> None:
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )

    def put(
        self,
        value: ConcurrencyExperimentReportV2,
    ) -> ConcurrencyExperimentStoreWrite:
        try:
            validate_concurrency_experiment_report_v2_identity(value)
        except ValueError as exc:
            raise ConcurrencyExperimentStoreConflictError(
                "concurrency experiment report identity is stale"
            ) from exc
        return self._store.put(
            ref=concurrency_experiment_report_v2_ref(value),
            codec=ConcurrencyExperimentStoreCodec.REPORT,
            payload=value.canonical_json(),
        )

    def get(self, ref: ObjectRef) -> ConcurrencyExperimentReportV2:
        if ref.object_type != "concurrency-experiment-report" or ref.object_version != "v2":
            raise ConcurrencyExperimentStoreTypeError("concurrency experiment report ref has wrong type")
        payload = self._store.get(
            ref=ref,
            codec=ConcurrencyExperimentStoreCodec.REPORT,
        )
        try:
            value = ConcurrencyExperimentReportV2.model_validate_json(payload)
            validate_concurrency_experiment_report_v2_identity(value)
            observed = concurrency_experiment_report_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise ConcurrencyExperimentStoreIntegrityError(
                "stored concurrency experiment report is invalid"
            ) from exc
        if observed != ref or value.canonical_json() != payload:
            raise ConcurrencyExperimentStoreIntegrityError(
                "stored concurrency experiment report differs from its ref"
            )
        return value

    def verify(self, ref: ObjectRef) -> None:
        self.get(ref)


_MODEL_BY_CODEC: dict[
    ConcurrencyExperimentStoreCodec,
    type[ContractModelV2],
] = {
    ConcurrencyExperimentStoreCodec.HOST: ConcurrencyExperimentHostSummaryV2,
    ConcurrencyExperimentStoreCodec.WORKLOAD: ConcurrencyExperimentWorkloadV1,
    ConcurrencyExperimentStoreCodec.CASE_SAMPLE: ConcurrencyExperimentCaseSampleV1,
    ConcurrencyExperimentStoreCodec.TRIAL_RESULT: ConcurrencyExperimentTrialResultV1,
    ConcurrencyExperimentStoreCodec.LEVEL_RESULT: ConcurrencyExperimentLevelResultV1,
    ConcurrencyExperimentStoreCodec.FAULT_OBSERVATION: (ConcurrencyExperimentFaultObservationV1),
    ConcurrencyExperimentStoreCodec.RECOVERY_RESULT: (ConcurrencyExperimentRecoveryResultV1),
    ConcurrencyExperimentStoreCodec.RESULT_SET: ConcurrencyExperimentResultSetV1,
    ConcurrencyExperimentStoreCodec.RUN_LEDGER: ConcurrencyExperimentRunLedgerV1,
}


def _require_material_ref(
    ref: ObjectRef,
    codec: ConcurrencyExperimentStoreCodec,
) -> None:
    expected = {
        ConcurrencyExperimentStoreCodec.HOST: "concurrency-experiment-host",
        ConcurrencyExperimentStoreCodec.WORKLOAD: "concurrency-experiment-workload",
        ConcurrencyExperimentStoreCodec.CASE_SAMPLE: ("concurrency-experiment-case-sample"),
        ConcurrencyExperimentStoreCodec.TRIAL_RESULT: ("concurrency-experiment-trial-result"),
        ConcurrencyExperimentStoreCodec.LEVEL_RESULT: ("concurrency-experiment-level-result"),
        ConcurrencyExperimentStoreCodec.FAULT_OBSERVATION: ("concurrency-experiment-fault-observation"),
        ConcurrencyExperimentStoreCodec.RECOVERY_RESULT: ("concurrency-experiment-recovery-result"),
        ConcurrencyExperimentStoreCodec.RESULT_SET: ("concurrency-experiment-result-set"),
        ConcurrencyExperimentStoreCodec.RUN_LEDGER: ("concurrency-experiment-run-ledger"),
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "private-v1":
        raise ConcurrencyExperimentStoreTypeError("concurrency experiment material ref has wrong type")


def _material_ref(value: ContractModelV2) -> ObjectRef:
    if isinstance(value, ConcurrencyExperimentHostSummaryV2):
        digest = value.host_summary_sha256
        return ObjectRef(
            object_type="concurrency-experiment-host",
            object_id=f"concurrency-experiment-host://sha256/{digest}",
            object_version="private-v1",
            object_sha256=digest,
        )
    if isinstance(value, ConcurrencyExperimentWorkloadV1):
        return concurrency_experiment_workload_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentCaseSampleV1):
        return concurrency_experiment_case_sample_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentTrialResultV1):
        return concurrency_experiment_trial_result_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentLevelResultV1):
        return concurrency_experiment_level_result_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentFaultObservationV1):
        return concurrency_experiment_fault_observation_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentRecoveryResultV1):
        return concurrency_experiment_recovery_result_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentResultSetV1):
        return concurrency_experiment_result_set_v1_ref(value)
    if isinstance(value, ConcurrencyExperimentRunLedgerV1):
        return concurrency_experiment_run_ledger_v1_ref(value)
    raise ConcurrencyExperimentStoreTypeError("concurrency experiment material type is unsupported")


def _member_count(value: ContractModelV2) -> int:
    if isinstance(value, ConcurrencyExperimentWorkloadV1):
        return len(value.trials)
    if isinstance(value, ConcurrencyExperimentTrialResultV1):
        return len(value.case_sample_refs)
    if isinstance(value, ConcurrencyExperimentLevelResultV1):
        return len(value.trial_result_refs)
    if isinstance(value, ConcurrencyExperimentResultSetV1):
        return len(value.level_result_refs) + len(value.recovery_result_refs)
    if isinstance(value, ConcurrencyExperimentRunLedgerV1):
        return len(value.child_manifest_refs)
    return 1


def _require_safe_store_member(root: Path, path: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ConcurrencyExperimentStoreTypeError(
            "concurrency experiment store root has an invalid file type"
        )
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ConcurrencyExperimentStoreTypeError(
            "concurrency experiment store member escapes its root"
        ) from exc
    if not relative.parts:
        raise ConcurrencyExperimentStoreTypeError("concurrency experiment store member is not a file")
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise ConcurrencyExperimentStoreTypeError("concurrency experiment store member uses a symlink")
        if current.exists():
            is_leaf = index == len(relative.parts) - 1
            if (is_leaf and not current.is_file()) or (not is_leaf and not current.is_dir()):
                raise ConcurrencyExperimentStoreTypeError(
                    "concurrency experiment store member has an invalid file type"
                )


__all__ = [
    "ConcurrencyExperimentMaterialStore",
    "ConcurrencyExperimentReportStore",
    "ConcurrencyExperimentResultSetClosure",
    "ConcurrencyExperimentStoreCodec",
    "ConcurrencyExperimentStoreConflictError",
    "ConcurrencyExperimentStoreError",
    "ConcurrencyExperimentStoreFaultPoint",
    "ConcurrencyExperimentStoreInjectedCrash",
    "ConcurrencyExperimentStoreIntegrityError",
    "ConcurrencyExperimentStoreLimitError",
    "ConcurrencyExperimentStoreTypeError",
    "ConcurrencyExperimentStoreWrite",
    "StaticConcurrencyExperimentStoreFaultInjector",
]
