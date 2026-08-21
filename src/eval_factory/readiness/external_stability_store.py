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
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.readiness.external_stability import (
    ExternalRealTraceStabilityRunResult,
)
from eval_factory.readiness.external_stability_models import (
    ExternalRealTraceStabilityCaseResultV1,
    ExternalRealTraceStabilityResultSetV1,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class ExternalRealTraceStabilityStoreError(RuntimeError):
    pass


class ExternalRealTraceStabilityStoreTypeError(
    ExternalRealTraceStabilityStoreError,
):
    pass


class ExternalRealTraceStabilityStoreConflictError(
    ExternalRealTraceStabilityStoreError,
):
    pass


class ExternalRealTraceStabilityStoreIntegrityError(
    ExternalRealTraceStabilityStoreError,
):
    pass


class ExternalRealTraceStabilityStoreLimitError(
    ExternalRealTraceStabilityStoreError,
):
    pass


class ExternalRealTraceStabilityStoreInjectedCrash(
    ExternalRealTraceStabilityStoreError,
):
    pass


class ExternalRealTraceStabilityStoreCodec(StrEnum):
    CASE_RESULT = "case-result"
    RESULT_SET = "result-set"
    REPORT = "report"


class ExternalRealTraceStabilityStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"


class ExternalRealTraceStabilityStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: ExternalRealTraceStabilityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticExternalRealTraceStabilityStoreFaultInjector:
    crash_points: frozenset[ExternalRealTraceStabilityStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: ExternalRealTraceStabilityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise ExternalRealTraceStabilityStoreInjectedCrash(
                f"injected external real-trace stability store crash at {point.value}"
            )


@dataclass(frozen=True, slots=True)
class ExternalRealTraceStabilityStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _ExternalRealTraceStabilityEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-real-trace-stability-envelope/private-v1"] = (
        "eval-factory/external-real-trace-stability-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: ExternalRealTraceStabilityStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        expected_id = (
            "external-real-trace-stability-content://"
            f"{self.codec.value}/sha256/"
            f"{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "external-real-trace-stability-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("external real-trace stability content ref is invalid")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: (ExternalRealTraceStabilityStoreFaultInjector | None),
    ) -> None:
        if max_bytes < 2:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability byte limit is invalid"
            )
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise ExternalRealTraceStabilityStoreTypeError(
                "external real-trace stability store root must be a non-symlink directory"
            )
        self._root = candidate.resolve()
        self.max_bytes = max_bytes
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(self._root / "cas")
        self._envelopes = self._root / "envelopes"

    def put(
        self,
        *,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
        payload: bytes,
    ) -> ExternalRealTraceStabilityStoreWrite:
        if len(payload) > self.max_bytes:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability value exceeds byte limit"
            )
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="external-real-trace-stability-content",
            object_id=(f"external-real-trace-stability-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _ExternalRealTraceStabilityEnvelopeV1(
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
            raise ExternalRealTraceStabilityStoreConflictError(
                "external real-trace stability CAS write conflicted"
            ) from exc
        self._fault(
            ExternalRealTraceStabilityStoreFaultPoint.AFTER_CAS_WRITE,
            ref,
        )
        written = self._write_envelope(envelope)
        self._fault(
            ExternalRealTraceStabilityStoreFaultPoint.AFTER_ENVELOPE_WRITE,
            ref,
        )
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise ExternalRealTraceStabilityStoreIntegrityError(
                    "external real-trace stability envelope disappeared"
                )
            return existing
        self.get(ref=ref, codec=codec)
        return ExternalRealTraceStabilityStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
    ) -> bytes:
        envelope = self._read_envelope(ref, codec)
        try:
            payload = self._cas.read(
                object_id=envelope.content_blob_ref.object_id,
                digest=envelope.content_blob_ref.object_sha256,
            )
        except ContentAddressedByteStoreCorruptionError as exc:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external real-trace stability content is missing or corrupt"
            ) from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "stored external real-trace stability byte size is invalid"
            )
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
    ) -> ExternalRealTraceStabilityStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return ExternalRealTraceStabilityStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(
        self,
        envelope: _ExternalRealTraceStabilityEnvelopeV1,
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
        codec: ExternalRealTraceStabilityStoreCodec,
    ) -> _ExternalRealTraceStabilityEnvelopeV1:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external real-trace stability envelope is missing"
            )
        try:
            encoded = path.read_bytes()
            envelope = _ExternalRealTraceStabilityEnvelopeV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external real-trace stability envelope is invalid"
            ) from exc
        if encoded != envelope.canonical_json() + b"\n":
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external real-trace stability envelope is not canonical"
            )
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external real-trace stability envelope authority differs"
            )
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _fault(
        self,
        point: ExternalRealTraceStabilityStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_Material = ExternalRealTraceStabilityCaseResultV1 | ExternalRealTraceStabilityResultSetV1


class ExternalRealTraceStabilityMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: (ExternalRealTraceStabilityStoreFaultInjector | None) = None,
    ) -> None:
        if max_members < 1 or max_members > 100_000:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability member limit is invalid"
            )
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )

    def put_case_result(
        self,
        value: ExternalRealTraceStabilityCaseResultV1,
    ) -> ExternalRealTraceStabilityStoreWrite:
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalRealTraceStabilityStoreCodec.CASE_RESULT,
            member_count=1,
        )

    def put_result_set(
        self,
        value: ExternalRealTraceStabilityResultSetV1,
    ) -> ExternalRealTraceStabilityStoreWrite:
        if len(value.case_result_refs) > self.max_members:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability material exceeds member limit"
            )
        self._verify_result_set_cases(value)
        return self._put(
            value,
            ref=value.to_ref(),
            codec=ExternalRealTraceStabilityStoreCodec.RESULT_SET,
            member_count=len(value.case_result_refs),
        )

    def get_case_result(
        self,
        ref: ObjectRef,
    ) -> ExternalRealTraceStabilityCaseResultV1:
        value = self._get(
            ref,
            ExternalRealTraceStabilityStoreCodec.CASE_RESULT,
        )
        if not isinstance(
            value,
            ExternalRealTraceStabilityCaseResultV1,
        ):
            raise ExternalRealTraceStabilityStoreTypeError(
                "external case-result codec returned the wrong model"
            )
        return value

    def get_result_set(
        self,
        ref: ObjectRef,
    ) -> ExternalRealTraceStabilityResultSetV1:
        value = self._get(
            ref,
            ExternalRealTraceStabilityStoreCodec.RESULT_SET,
        )
        if not isinstance(value, ExternalRealTraceStabilityResultSetV1):
            raise ExternalRealTraceStabilityStoreTypeError(
                "external result-set codec returned the wrong model"
            )
        self._verify_result_set_cases(value)
        return value

    def get_result_set_closure(
        self,
        ref: ObjectRef,
    ) -> tuple[
        ExternalRealTraceStabilityResultSetV1,
        tuple[ExternalRealTraceStabilityCaseResultV1, ...],
    ]:
        result_set = self.get_result_set(ref)
        cases = tuple(self.get_case_result(case_ref) for case_ref in result_set.case_result_refs)
        return result_set, cases

    def _put(
        self,
        value: _Material,
        *,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
        member_count: int,
    ) -> ExternalRealTraceStabilityStoreWrite:
        if member_count > self.max_members:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability material exceeds member limit"
            )
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get(
        self,
        ref: ObjectRef,
        codec: ExternalRealTraceStabilityStoreCodec,
    ) -> _Material:
        _require_material_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            if codec is ExternalRealTraceStabilityStoreCodec.CASE_RESULT:
                value: _Material = ExternalRealTraceStabilityCaseResultV1.model_validate_json(payload)
            else:
                value = ExternalRealTraceStabilityResultSetV1.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "stored external real-trace stability material is invalid"
            ) from exc
        if value.to_ref() != ref or value.canonical_json() != payload:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "stored external real-trace stability material differs from its ref"
            )
        count = (
            len(value.case_result_refs)
            if isinstance(
                value,
                ExternalRealTraceStabilityResultSetV1,
            )
            else 1
        )
        if count > self.max_members:
            raise ExternalRealTraceStabilityStoreLimitError(
                "stored external real-trace stability material exceeds member limit"
            )
        return value

    def _verify_result_set_cases(
        self,
        value: ExternalRealTraceStabilityResultSetV1,
    ) -> None:
        cases = tuple(self.get_case_result(case_ref) for case_ref in value.case_result_refs)
        summaries = tuple(case.to_public_summary() for case in cases)
        if summaries != value.case_summaries:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external result set differs from stored case closure"
            )


class ExternalRealTraceStabilityReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        max_cases: int,
        fault_injector: (ExternalRealTraceStabilityStoreFaultInjector | None) = None,
    ) -> None:
        if max_cases < 1 or max_cases > 100_000:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability case limit is invalid"
            )
        self.max_cases = max_cases
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )

    def put(
        self,
        value: ExternalRealTraceStabilityReportV2,
    ) -> ExternalRealTraceStabilityStoreWrite:
        if len(value.case_summaries) > self.max_cases:
            raise ExternalRealTraceStabilityStoreLimitError(
                "external real-trace stability report exceeds case limit"
            )
        return self._store.put(
            ref=value.to_ref(),
            codec=ExternalRealTraceStabilityStoreCodec.REPORT,
            payload=value.canonical_json(),
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> ExternalRealTraceStabilityReportV2:
        if ref.object_type != "external-real-trace-stability-report" or ref.object_version != "v2":
            raise ExternalRealTraceStabilityStoreTypeError(
                "external real-trace stability report ref has wrong type"
            )
        payload = self._store.get(
            ref=ref,
            codec=ExternalRealTraceStabilityStoreCodec.REPORT,
        )
        try:
            value = ExternalRealTraceStabilityReportV2.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "stored external real-trace stability report is invalid"
            ) from exc
        if (
            len(value.case_summaries) > self.max_cases
            or value.to_ref() != ref
            or value.canonical_json() != payload
        ):
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "stored external real-trace stability report differs from its ref"
            )
        return value


class ExternalRealTraceStabilityPersistenceService:
    def __init__(
        self,
        *,
        material_store: ExternalRealTraceStabilityMaterialStore,
        report_store: ExternalRealTraceStabilityReportStore,
    ) -> None:
        _require_disjoint_roots(
            material_store._store._root,
            report_store._store._root,
        )
        self.material_store = material_store
        self.report_store = report_store

    def persist(
        self,
        value: ExternalRealTraceStabilityRunResult,
    ) -> ExternalRealTraceStabilityRunResult:
        for case in value.case_results:
            self.material_store.put_case_result(case)
        self.material_store.put_result_set(value.result_set)
        self.report_store.put(value.report)
        return self.get(value.report.to_ref())

    def get(
        self,
        report_ref: ObjectRef,
    ) -> ExternalRealTraceStabilityRunResult:
        report = self.report_store.get(report_ref)
        result_set, cases = self.material_store.get_result_set_closure(report.private_result_set_ref)
        if (
            report.policy_ref != result_set.policy_ref
            or report.private_inventory_ref != result_set.private_inventory_ref
            or report.case_summaries != result_set.case_summaries
            or len(cases) != report.unique_real_trace_count
        ):
            raise ExternalRealTraceStabilityStoreIntegrityError(
                "external stability report differs from private closure"
            )
        return ExternalRealTraceStabilityRunResult(
            case_results=cases,
            result_set=result_set,
            report=report,
        )


def _require_material_ref(
    ref: ObjectRef,
    codec: ExternalRealTraceStabilityStoreCodec,
) -> None:
    expected = {
        ExternalRealTraceStabilityStoreCodec.CASE_RESULT: ("real-trace-stability-case-result"),
        ExternalRealTraceStabilityStoreCodec.RESULT_SET: ("external-real-trace-stability-result-set"),
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "private-v1":
        raise ExternalRealTraceStabilityStoreTypeError(
            "external real-trace stability material ref has wrong type"
        )


def _require_disjoint_roots(left: Path, right: Path) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ExternalRealTraceStabilityStoreTypeError("external real-trace stability store roots overlap")


__all__ = [
    "ExternalRealTraceStabilityMaterialStore",
    "ExternalRealTraceStabilityPersistenceService",
    "ExternalRealTraceStabilityReportStore",
    "ExternalRealTraceStabilityStoreCodec",
    "ExternalRealTraceStabilityStoreConflictError",
    "ExternalRealTraceStabilityStoreError",
    "ExternalRealTraceStabilityStoreFaultInjector",
    "ExternalRealTraceStabilityStoreFaultPoint",
    "ExternalRealTraceStabilityStoreInjectedCrash",
    "ExternalRealTraceStabilityStoreIntegrityError",
    "ExternalRealTraceStabilityStoreLimitError",
    "ExternalRealTraceStabilityStoreTypeError",
    "ExternalRealTraceStabilityStoreWrite",
    "StaticExternalRealTraceStabilityStoreFaultInjector",
]
