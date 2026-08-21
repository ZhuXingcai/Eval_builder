from __future__ import annotations

import hashlib
import os
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import Field, ValidationError, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvaluationReportV2,
    label_quality_evaluation_report_v2_ref,
    validate_label_quality_evaluation_report_v2_identity,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionV2
from eval_factory.statistics.label_quality import LabelQualityEvaluator
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationCompilation,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityAcceptedRunV1,
    LabelQualityObservationSetV1,
    LabelQualityPairResultV1,
    LabelQualityResultSetV1,
    label_quality_accepted_run_v1_ref,
    label_quality_observation_set_v1_ref,
    label_quality_pair_result_v1_ref,
    label_quality_result_set_v1_ref,
)
from eval_factory.storage import (
    ContentAddressedByteStore,
    ContentAddressedByteStoreConflictError,
    ContentAddressedByteStoreCorruptionError,
)


class LabelQualityStoreError(RuntimeError):
    pass


class LabelQualityStoreTypeError(LabelQualityStoreError):
    pass


class LabelQualityStoreConflictError(LabelQualityStoreError):
    pass


class LabelQualityStoreIntegrityError(LabelQualityStoreError):
    pass


class LabelQualityStoreLimitError(LabelQualityStoreError):
    pass


class LabelQualityStoreInjectedCrash(LabelQualityStoreError):
    pass


class LabelQualityStoreCodec(StrEnum):
    OBSERVATION_SET = "observation-set"
    PAIR_RESULT = "pair-result"
    RESULT_SET = "result-set"
    ACCEPTED_RUN = "accepted-run"
    REPORT = "report"


class LabelQualityStoreFaultPoint(StrEnum):
    AFTER_CAS_WRITE = "after_cas_write"
    AFTER_ENVELOPE_WRITE = "after_envelope_write"
    BEFORE_ACCEPTANCE_INDEX = "before_acceptance_index"


class LabelQualityStoreFaultInjector(Protocol):
    def maybe_raise(
        self,
        point: LabelQualityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticLabelQualityStoreFaultInjector:
    crash_points: frozenset[LabelQualityStoreFaultPoint] = frozenset()

    def maybe_raise(
        self,
        point: LabelQualityStoreFaultPoint,
        *,
        object_ref: ObjectRef,
    ) -> None:
        del object_ref
        if point in self.crash_points:
            raise LabelQualityStoreInjectedCrash(f"injected label quality store crash at {point.value}")


@dataclass(frozen=True, slots=True)
class LabelQualityStoreWrite:
    object_ref: ObjectRef
    written: bool
    content_blob_written: bool
    canonical_size_bytes: int


class _LabelQualityEnvelopeV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-envelope/private-v1"] = (
        "eval-factory/label-quality-envelope/private-v1"
    )
    object_ref: ObjectRef
    codec: LabelQualityStoreCodec
    content_blob_ref: ObjectRef
    canonical_size_bytes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_envelope(self) -> Self:
        expected_id = (
            f"label-quality-content://{self.codec.value}/sha256/{self.content_blob_ref.object_sha256}"
        )
        if (
            self.content_blob_ref.object_type != "label-quality-content"
            or self.content_blob_ref.object_version != "private-v1"
            or self.content_blob_ref.object_id != expected_id
        ):
            raise ValueError("label quality content ref is invalid")
        return self


class _LabelQualityAcceptanceIndexV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-acceptance-index/private-v1"] = (
        "eval-factory/label-quality-acceptance-index/private-v1"
    )
    acceptance_key: Identifier
    accepted_run_ref: ObjectRef

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        if (
            self.accepted_run_ref.object_type != "label-quality-accepted-run"
            or self.accepted_run_ref.object_version != "private-v1"
        ):
            raise ValueError("acceptance index must reference an accepted run")
        return self


class _EnvelopeByteStore:
    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        fault_injector: LabelQualityStoreFaultInjector | None,
    ) -> None:
        if max_bytes < 2:
            raise LabelQualityStoreLimitError("label quality byte limit is invalid")
        candidate = root.expanduser()
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise LabelQualityStoreTypeError("label quality root must be a non-symlink directory")
        self.root = candidate.resolve()
        self.max_bytes = max_bytes
        self.fault_injector = fault_injector
        self._cas = ContentAddressedByteStore(self.root / "cas")
        self._envelopes = self.root / "envelopes"

    def put(
        self,
        *,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
        payload: bytes,
    ) -> LabelQualityStoreWrite:
        if len(payload) > self.max_bytes:
            raise LabelQualityStoreLimitError("label quality value exceeds byte limit")
        existing = self._existing(ref, codec)
        if existing is not None:
            return existing
        digest = hashlib.sha256(payload).hexdigest()
        content_ref = ObjectRef(
            object_type="label-quality-content",
            object_id=(f"label-quality-content://{codec.value}/sha256/{digest}"),
            object_version="private-v1",
            object_sha256=digest,
        )
        envelope = _LabelQualityEnvelopeV1(
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
            raise LabelQualityStoreConflictError("label quality CAS write conflicted") from exc
        self._fault(LabelQualityStoreFaultPoint.AFTER_CAS_WRITE, ref)
        written = self._write_envelope(envelope)
        self._fault(LabelQualityStoreFaultPoint.AFTER_ENVELOPE_WRITE, ref)
        if not written:
            existing = self._existing(ref, codec)
            if existing is None:
                raise LabelQualityStoreIntegrityError("label quality envelope disappeared")
            return existing
        self.get(ref=ref, codec=codec)
        return LabelQualityStoreWrite(
            object_ref=ref,
            written=True,
            content_blob_written=content_written,
            canonical_size_bytes=len(payload),
        )

    def get(
        self,
        *,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
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
            raise LabelQualityStoreIntegrityError("label quality content is missing or corrupt") from exc
        if len(payload) > self.max_bytes or len(payload) != envelope.canonical_size_bytes:
            raise LabelQualityStoreLimitError("stored label quality byte size is invalid")
        return payload

    def _existing(
        self,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
    ) -> LabelQualityStoreWrite | None:
        path = self._envelope_path(ref, codec)
        if not path.exists():
            return None
        envelope = self._read_envelope(ref, codec)
        self.get(ref=ref, codec=codec)
        return LabelQualityStoreWrite(
            object_ref=ref,
            written=False,
            content_blob_written=False,
            canonical_size_bytes=envelope.canonical_size_bytes,
        )

    def _write_envelope(self, envelope: _LabelQualityEnvelopeV1) -> bool:
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
        codec: LabelQualityStoreCodec,
    ) -> _LabelQualityEnvelopeV1:
        path = self._envelope_path(ref, codec)
        _require_safe_store_member(self._envelopes, path)
        if not path.exists():
            raise LabelQualityStoreIntegrityError("label quality envelope is missing")
        try:
            encoded = path.read_bytes()
            envelope = _LabelQualityEnvelopeV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise LabelQualityStoreIntegrityError("label quality envelope is invalid") from exc
        if encoded != envelope.canonical_json() + b"\n":
            raise LabelQualityStoreIntegrityError("label quality envelope is not canonical")
        if envelope.object_ref != ref or envelope.codec is not codec:
            raise LabelQualityStoreIntegrityError("label quality envelope authority differs")
        return envelope

    def _envelope_path(
        self,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
    ) -> Path:
        return self._envelopes / codec.value / "sha256" / ref.object_sha256[:2] / f"{ref.object_sha256}.json"

    def _fault(
        self,
        point: LabelQualityStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


_PrivateMaterial = (
    LabelQualityObservationSetV1
    | LabelQualityPairResultV1
    | LabelQualityResultSetV1
    | LabelQualityAcceptedRunV1
)


class LabelQualityMaterialStore:
    def __init__(
        self,
        root: Path,
        *,
        max_private_bytes: int,
        max_members: int,
        fault_injector: LabelQualityStoreFaultInjector | None = None,
    ) -> None:
        if max_members < 300 or max_members > 10_000_000:
            raise LabelQualityStoreLimitError("label quality member limit is invalid")
        self.root = root.expanduser()
        self.max_members = max_members
        self._store = _EnvelopeByteStore(
            self.root,
            max_bytes=max_private_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root
        self._authorities = self.root / "authorities"

    def put_observation_set(
        self,
        value: LabelQualityObservationSetV1,
    ) -> LabelQualityStoreWrite:
        return self._put(
            value,
            label_quality_observation_set_v1_ref(value),
            LabelQualityStoreCodec.OBSERVATION_SET,
            len(value.decisions),
        )

    def get_observation_set(
        self,
        ref: ObjectRef,
    ) -> LabelQualityObservationSetV1:
        return self._get(
            ref,
            LabelQualityStoreCodec.OBSERVATION_SET,
            LabelQualityObservationSetV1,
        )

    def put_pair_result(
        self,
        value: LabelQualityPairResultV1,
    ) -> LabelQualityStoreWrite:
        return self._put(
            value,
            label_quality_pair_result_v1_ref(value),
            LabelQualityStoreCodec.PAIR_RESULT,
            1,
        )

    def get_pair_result(
        self,
        ref: ObjectRef,
    ) -> LabelQualityPairResultV1:
        return self._get(
            ref,
            LabelQualityStoreCodec.PAIR_RESULT,
            LabelQualityPairResultV1,
        )

    def put_result_set(
        self,
        value: LabelQualityResultSetV1,
    ) -> LabelQualityStoreWrite:
        return self._put(
            value,
            label_quality_result_set_v1_ref(value),
            LabelQualityStoreCodec.RESULT_SET,
            len(value.pair_results),
        )

    def get_result_set(
        self,
        ref: ObjectRef,
    ) -> LabelQualityResultSetV1:
        return self._get(
            ref,
            LabelQualityStoreCodec.RESULT_SET,
            LabelQualityResultSetV1,
        )

    def put_accepted_run(
        self,
        value: LabelQualityAcceptedRunV1,
    ) -> LabelQualityStoreWrite:
        return self._put(
            value,
            label_quality_accepted_run_v1_ref(value),
            LabelQualityStoreCodec.ACCEPTED_RUN,
            1,
        )

    def get_accepted_run(
        self,
        ref: ObjectRef,
    ) -> LabelQualityAcceptedRunV1:
        return self._get(
            ref,
            LabelQualityStoreCodec.ACCEPTED_RUN,
            LabelQualityAcceptedRunV1,
        )

    def find_acceptance(
        self,
        acceptance_key: str,
    ) -> LabelQualityAcceptedRunV1 | None:
        path = self._acceptance_path(acceptance_key)
        _require_safe_store_member(self._authorities, path)
        if not path.exists():
            return None
        return self._read_acceptance(path, acceptance_key)

    def get_acceptance(
        self,
        acceptance_key: str,
    ) -> LabelQualityAcceptedRunV1:
        result = self.find_acceptance(acceptance_key)
        if result is None:
            raise LabelQualityStoreIntegrityError("label quality acceptance was not found")
        return result

    def publish_acceptance(
        self,
        value: LabelQualityAcceptedRunV1,
    ) -> LabelQualityAcceptedRunV1:
        path = self._acceptance_path(value.acceptance_key)
        _require_safe_store_member(self._authorities, path)
        existing = self.find_acceptance(value.acceptance_key)
        if existing is not None:
            return existing
        path.parent.mkdir(parents=True, exist_ok=True)
        _require_safe_store_member(self._authorities, path)
        index = _LabelQualityAcceptanceIndexV1(
            acceptance_key=value.acceptance_key,
            accepted_run_ref=value.to_ref(),
        )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(index.canonical_json() + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            with suppress(FileExistsError):
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return self._read_acceptance(path, value.acceptance_key)

    def _read_acceptance(
        self,
        path: Path,
        acceptance_key: str,
    ) -> LabelQualityAcceptedRunV1:
        try:
            encoded = path.read_bytes()
            index = _LabelQualityAcceptanceIndexV1.model_validate_json(encoded)
        except (OSError, ValidationError, ValueError) as exc:
            raise LabelQualityStoreIntegrityError("label quality acceptance index is invalid") from exc
        if encoded != index.canonical_json() + b"\n" or index.acceptance_key != acceptance_key:
            raise LabelQualityStoreIntegrityError("label quality acceptance index differs")
        accepted = self.get_accepted_run(index.accepted_run_ref)
        if accepted.acceptance_key != acceptance_key:
            raise LabelQualityStoreIntegrityError("accepted run differs from acceptance index")
        return accepted

    def _acceptance_path(self, acceptance_key: str) -> Path:
        digest = hashlib.sha256(acceptance_key.encode()).hexdigest()
        return self._authorities / "sha256" / digest[:2] / f"{digest}.json"

    def _put(
        self,
        value: _PrivateMaterial,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
        member_count: int,
    ) -> LabelQualityStoreWrite:
        if member_count > self.max_members:
            raise LabelQualityStoreLimitError("label quality material exceeds member limit")
        return self._store.put(
            ref=ref,
            codec=codec,
            payload=value.canonical_json(),
        )

    def _get[ModelT: ContractModelV2](
        self,
        ref: ObjectRef,
        codec: LabelQualityStoreCodec,
        model_type: type[ModelT],
    ) -> ModelT:
        _require_material_ref(ref, codec)
        payload = self._store.get(ref=ref, codec=codec)
        try:
            value = model_type.model_validate_json(payload)
            observed = _material_ref(value)
        except (ValidationError, ValueError) as exc:
            raise LabelQualityStoreIntegrityError("stored label quality material is invalid") from exc
        if observed != ref or value.canonical_json() != payload:
            raise LabelQualityStoreIntegrityError("stored label quality material differs from its ref")
        if _member_count(value) > self.max_members:
            raise LabelQualityStoreLimitError("stored label quality material exceeds member limit")
        return value


class LabelQualityReportStore:
    def __init__(
        self,
        root: Path,
        *,
        max_report_bytes: int,
        fault_injector: LabelQualityStoreFaultInjector | None = None,
    ) -> None:
        self._store = _EnvelopeByteStore(
            root,
            max_bytes=max_report_bytes,
            fault_injector=fault_injector,
        )
        self.root = self._store.root

    def put(
        self,
        value: LabelQualityEvaluationReportV2,
    ) -> LabelQualityStoreWrite:
        try:
            validate_label_quality_evaluation_report_v2_identity(value)
        except ValueError as exc:
            raise LabelQualityStoreConflictError("label quality report identity is stale") from exc
        return self._store.put(
            ref=value.to_ref(),
            codec=LabelQualityStoreCodec.REPORT,
            payload=value.canonical_json(),
        )

    def get(
        self,
        ref: ObjectRef,
    ) -> LabelQualityEvaluationReportV2:
        if ref.object_type != "label-quality-evaluation-report" or ref.object_version != "v2":
            raise LabelQualityStoreTypeError("label quality report ref has wrong type")
        payload = self._store.get(
            ref=ref,
            codec=LabelQualityStoreCodec.REPORT,
        )
        try:
            value = LabelQualityEvaluationReportV2.model_validate_json(payload)
            validate_label_quality_evaluation_report_v2_identity(value)
            observed = label_quality_evaluation_report_v2_ref(value)
        except (ValidationError, ValueError) as exc:
            raise LabelQualityStoreIntegrityError("stored label quality report is invalid") from exc
        if observed != ref or value.canonical_json() != payload:
            raise LabelQualityStoreIntegrityError("stored label quality report differs from its ref")
        return value


class LabelQualityAcceptedRunStore:
    def __init__(
        self,
        *,
        material_store: LabelQualityMaterialStore,
        report_store: LabelQualityReportStore,
        fault_injector: LabelQualityStoreFaultInjector | None = None,
    ) -> None:
        _require_disjoint_roots(material_store.root, report_store.root)
        self.material_store = material_store
        self.report_store = report_store
        self.fault_injector = fault_injector

    def persist(
        self,
        *,
        acceptance_key: str,
        request_sha256: str,
        compilation: LabelQualityEvaluationCompilation,
        report: LabelQualityEvaluationReportV2,
        audit: ContractAudit,
    ) -> LabelQualityAcceptedRunV1:
        existing = self.material_store.find_acceptance(acceptance_key)
        if existing is not None:
            if existing.request_sha256 != request_sha256:
                raise LabelQualityStoreConflictError("label quality acceptance request changed")
            if existing.report_ref != report.to_ref():
                raise LabelQualityStoreConflictError("label quality accepted report changed")
            self._verify_closure(existing)
            return existing
        expected_report = LabelQualityEvaluator().evaluate(
            compilation=compilation,
            policy=compilation.policy,
            audit=report.audit,
        )
        if expected_report != report:
            raise LabelQualityStoreIntegrityError("label quality report differs from private evaluation")
        observation_ref: ObjectRef | None = None
        result_ref: ObjectRef | None = None
        if compilation.observation_set is not None:
            result_set = compilation.result_set
            if result_set is None:
                raise LabelQualityStoreIntegrityError("frozen compilation result set is missing")
            self.material_store.put_observation_set(compilation.observation_set)
            for pair in result_set.pair_results:
                self.material_store.put_pair_result(pair)
            self.material_store.put_result_set(result_set)
            observation_ref = compilation.observation_set.to_ref()
            result_ref = result_set.to_ref()
        elif compilation.result_set is not None:
            raise LabelQualityStoreIntegrityError("pending compilation cannot carry a result set")
        self.report_store.put(report)
        self._verify_material_closure(
            observation_ref=observation_ref,
            result_ref=result_ref,
            report=report,
        )
        accepted = LabelQualityAcceptedRunV1.create(
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            report_ref=report.to_ref(),
            observation_set_ref=observation_ref,
            result_set_ref=result_ref,
            audit=audit,
        )
        self.material_store.put_accepted_run(accepted)
        self._fault(
            LabelQualityStoreFaultPoint.BEFORE_ACCEPTANCE_INDEX,
            accepted.to_ref(),
        )
        authoritative = self.material_store.publish_acceptance(accepted)
        if authoritative.request_sha256 != request_sha256 or authoritative.to_ref() != accepted.to_ref():
            raise LabelQualityStoreConflictError("label quality first authority conflicted")
        self._verify_closure(authoritative)
        return authoritative

    def get_accepted_report(
        self,
        acceptance_key: str,
    ) -> LabelQualityEvaluationReportV2:
        accepted = self.material_store.get_acceptance(acceptance_key)
        return self._verify_closure(accepted)

    def _verify_closure(
        self,
        accepted: LabelQualityAcceptedRunV1,
    ) -> LabelQualityEvaluationReportV2:
        report = self.report_store.get(accepted.report_ref)
        self._verify_material_closure(
            observation_ref=accepted.observation_set_ref,
            result_ref=accepted.result_set_ref,
            report=report,
        )
        return report

    def _verify_material_closure(
        self,
        *,
        observation_ref: ObjectRef | None,
        result_ref: ObjectRef | None,
        report: LabelQualityEvaluationReportV2,
    ) -> None:
        if observation_ref is None:
            if result_ref is not None:
                raise LabelQualityStoreIntegrityError("pending acceptance carries a private result-set ref")
            if report.observation_closure_sha256 is not None or report.result_closure_sha256 is not None:
                raise LabelQualityStoreIntegrityError("pending report carries private closure commitments")
            return
        if result_ref is None:
            raise LabelQualityStoreIntegrityError("accepted private result-set ref is missing")
        observation = self.material_store.get_observation_set(observation_ref)
        result_set = self.material_store.get_result_set(result_ref)
        if (
            result_set.dataset_manifest_ref != observation.dataset_manifest_ref
            or result_set.observation_set_ref != observation.to_ref()
            or report.observation_closure_sha256 != observation.observation_set_sha256
            or report.result_closure_sha256 != result_set.result_set_sha256
        ):
            raise LabelQualityStoreIntegrityError("accepted report differs from private closure")
        observations = {_decision_key(decision): decision for decision in observation.decisions}
        for embedded in result_set.pair_results:
            stored = self.material_store.get_pair_result(embedded.to_ref())
            if stored != embedded:
                raise LabelQualityStoreIntegrityError("result set pair differs from stored material")
            observed = observations.pop(_pair_key(embedded), None)
            if embedded.decision != observed:
                raise LabelQualityStoreIntegrityError("result set pair differs from observation authority")
        if observations:
            raise LabelQualityStoreIntegrityError("observation authority contains an unpaired decision")

    def _fault(
        self,
        point: LabelQualityStoreFaultPoint,
        ref: ObjectRef,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point, object_ref=ref)


def _material_ref(value: ContractModelV2) -> ObjectRef:
    if isinstance(value, LabelQualityObservationSetV1):
        return label_quality_observation_set_v1_ref(value)
    if isinstance(value, LabelQualityPairResultV1):
        return label_quality_pair_result_v1_ref(value)
    if isinstance(value, LabelQualityResultSetV1):
        return label_quality_result_set_v1_ref(value)
    if isinstance(value, LabelQualityAcceptedRunV1):
        return label_quality_accepted_run_v1_ref(value)
    raise LabelQualityStoreTypeError("label quality material type is unsupported")


def _member_count(value: ContractModelV2) -> int:
    if isinstance(value, LabelQualityObservationSetV1):
        return len(value.decisions)
    if isinstance(value, LabelQualityResultSetV1):
        return len(value.pair_results)
    return 1


def _decision_key(value: LabelDecisionV2) -> tuple[ObjectRef, ObjectRef]:
    return (value.label_spec_ref, value.trace_envelope_ref)


def _pair_key(value: LabelQualityPairResultV1) -> tuple[ObjectRef, ObjectRef]:
    return (
        value.reference.label_spec_ref,
        value.reference.trace_envelope_ref,
    )


def _require_material_ref(
    ref: ObjectRef,
    codec: LabelQualityStoreCodec,
) -> None:
    expected = {
        LabelQualityStoreCodec.OBSERVATION_SET: ("label-quality-observation-set"),
        LabelQualityStoreCodec.PAIR_RESULT: "label-quality-pair-result",
        LabelQualityStoreCodec.RESULT_SET: "label-quality-result-set",
        LabelQualityStoreCodec.ACCEPTED_RUN: "label-quality-accepted-run",
    }.get(codec)
    if expected is None or ref.object_type != expected or ref.object_version != "private-v1":
        raise LabelQualityStoreTypeError("label quality material ref has wrong type")


def _require_safe_store_member(root: Path, path: Path) -> None:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise LabelQualityStoreTypeError("label quality store root has an invalid file type")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LabelQualityStoreTypeError("label quality store member escapes its root") from exc
    if not relative.parts:
        raise LabelQualityStoreTypeError("label quality store member is not a file")
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        if current.is_symlink():
            raise LabelQualityStoreTypeError("label quality store member uses a symlink")
        if current.exists():
            is_leaf = index == len(relative.parts) - 1
            if (is_leaf and not current.is_file()) or (not is_leaf and not current.is_dir()):
                raise LabelQualityStoreTypeError("label quality store member has an invalid file type")


def _require_disjoint_roots(first: Path, second: Path) -> None:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    if (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    ):
        raise LabelQualityStoreTypeError("label quality store roots overlap")


__all__ = [
    "LabelQualityAcceptedRunStore",
    "LabelQualityMaterialStore",
    "LabelQualityReportStore",
    "LabelQualityStoreCodec",
    "LabelQualityStoreConflictError",
    "LabelQualityStoreError",
    "LabelQualityStoreFaultPoint",
    "LabelQualityStoreInjectedCrash",
    "LabelQualityStoreIntegrityError",
    "LabelQualityStoreLimitError",
    "LabelQualityStoreTypeError",
    "LabelQualityStoreWrite",
    "StaticLabelQualityStoreFaultInjector",
]
