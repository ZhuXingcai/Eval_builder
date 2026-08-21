from __future__ import annotations

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Literal, Self, TypedDict

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.trace import ParseQuality

REAL_TRACE_STABILITY_POLICY_VERSION: Literal["real-trace-stability/r8-04-v1"] = (
    "real-trace-stability/r8-04-v1"
)
REAL_TRACE_STABILITY_CLAIM_SCOPE: Literal["REAL_TRACE_STABILITY_ONLY"] = "REAL_TRACE_STABILITY_ONLY"
REAL_TRACE_STABILITY_SOURCE_MANIFEST_SHA256: Literal[
    "0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04"
] = "0d9df6c3935e00e5e15d486c9afe8cc2f581b63cf8f99ec42c1d3216cc74ee04"


class RealTraceStabilityFaultPointV2(StrEnum):
    BEFORE_SOURCE_REGISTRATION = "before_source_registration"
    AFTER_SOURCE_REGISTRATION = "after_source_registration"
    AFTER_TRACE_STORE_PERSIST = "after_trace_store_persist"
    BEFORE_STAGE_COMPLETION = "before_stage_completion"


class RealTraceStabilityCaseOutcomeV2(StrEnum):
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    INCOMPLETE = "INCOMPLETE"


class RealTraceStabilityReasonCodeV2(StrEnum):
    NONE = "NONE"
    TRACE_RUN_FAILED = "TRACE_RUN_FAILED"
    EXPECTED_FAULT_NOT_OBSERVED = "EXPECTED_FAULT_NOT_OBSERVED"
    RESUME_FAILED = "RESUME_FAILED"
    UNEXPECTED_RETRY = "UNEXPECTED_RETRY"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    DUPLICATE_COMPLETION = "DUPLICATE_COMPLETION"
    MATERIAL_INTEGRITY_ERROR = "MATERIAL_INTEGRITY_ERROR"
    CHILD_STATE_INCOMPLETE = "CHILD_STATE_INCOMPLETE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class RealTraceStabilitySampleStatusV2(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class RealTraceStabilityOutcomeV2(StrEnum):
    PASSED = "PASSED"
    STABILITY_THRESHOLD_NOT_MET = "STABILITY_THRESHOLD_NOT_MET"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"
    INCOMPLETE = "INCOMPLETE"


class _DerivedReportFields(TypedDict):
    case_summary_refs: tuple[ObjectRef, ...]
    stable_case_refs: tuple[ObjectRef, ...]
    unstable_case_refs: tuple[ObjectRef, ...]
    infrastructure_error_case_refs: tuple[ObjectRef, ...]
    incomplete_case_refs: tuple[ObjectRef, ...]
    parse_quality_counts: tuple[RealTraceStabilityCategoryCountV2, ...]
    parse_quality_unavailable_count: int
    fault_point_counts: tuple[RealTraceStabilityCategoryCountV2, ...]
    total_attempts: int
    total_unexpected_retries: int
    total_resumes: int
    total_replays: int
    total_stage_results: int
    total_completion_witnesses: int
    observed_threshold_met: bool
    sample_status: RealTraceStabilitySampleStatusV2
    outcome: RealTraceStabilityOutcomeV2


class RealTraceStabilityPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-policy/v2"] = (
        "eval-factory/real-trace-stability-policy/v2"
    )
    policy_id: Identifier
    source_manifest_ref: ObjectRef
    required_unique_real_traces: Literal[100] = 100
    required_stable_rate_bps: Literal[10_000] = 10_000
    require_typed_result: Literal[True] = True
    require_expected_fault: Literal[True] = True
    require_resume_success: Literal[True] = True
    require_replay_stability: Literal[True] = True
    max_unexpected_retries: Literal[0] = 0
    required_completion_witnesses: Literal[1] = 1
    fault_points: tuple[RealTraceStabilityFaultPointV2, ...] = tuple(RealTraceStabilityFaultPointV2)
    runner_version: Literal["serial-trace-runner/r1-09-v1"] = "serial-trace-runner/r1-09-v1"
    adapter_version: Literal["raw_traj_v1/1.0.0"] = "raw_traj_v1/1.0.0"
    trace_ir_version: Literal["v1"] = "v1"
    repair_policy_version: Literal["raw-traj-local-repair/v1"] = "raw-traj-local-repair/v1"
    recovery_policy_version: Literal["raw-traj-streaming-recovery/v1"] = "raw-traj-streaming-recovery/v1"
    normalization_policy_version: Literal["raw-traj-event-normalization/v1"] = (
        "raw-traj-event-normalization/v1"
    )
    tool_family_policy_version: Literal["raw-traj-tool-family/v1"] = "raw-traj-tool-family/v1"
    file_observation_policy_version: Literal["raw-traj-file-observation/v1"] = "raw-traj-file-observation/v1"
    segmentation_policy_version: Literal["deterministic-interaction-segments/v1"] = (
        "deterministic-interaction-segments/v1"
    )
    storage_policy_version: Literal["stored-manifest/v1"] = "stored-manifest/v1"
    max_sources: int = Field(ge=100, le=100_000)
    max_source_bytes: int = Field(ge=1, le=1_000_000_000)
    max_total_source_bytes: int = Field(ge=1, le=1_000_000_000_000)
    max_private_bytes: int = Field(ge=2, le=1_000_000_000)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("fault_points", mode="before")
    @classmethod
    def parse_fault_points(
        cls,
        value: object,
    ) -> tuple[RealTraceStabilityFaultPointV2, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("fault_points must be a tuple")
        return tuple(_parse_enum(item, RealTraceStabilityFaultPointV2, "fault_points") for item in value)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.source_manifest_ref,
            "real-trace-source-manifest",
            "csv/v1",
            "source_manifest_ref",
        )
        if self.source_manifest_ref.object_sha256 != REAL_TRACE_STABILITY_SOURCE_MANIFEST_SHA256:
            raise ValueError("source manifest differs from approved R8-04 authority")
        if self.fault_points != tuple(RealTraceStabilityFaultPointV2):
            raise ValueError("policy must bind all four R1 fault points in canonical order")
        if self.max_source_bytes > self.max_total_source_bytes:
            raise ValueError("per-source byte limit cannot exceed total byte limit")
        _require_audit(self.audit, (self.source_manifest_ref,), "stability policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "real-trace-stability-policy",
            real_trace_stability_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_manifest_ref: ObjectRef,
        max_sources: int,
        max_source_bytes: int,
        max_total_source_bytes: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> RealTraceStabilityPolicyV2:
        value = cls(
            policy_id="real-trace-stability-policy://pending",
            source_manifest_ref=source_manifest_ref,
            max_sources=max_sources,
            max_source_bytes=max_source_bytes,
            max_total_source_bytes=max_total_source_bytes,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, (source_manifest_ref,)),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "real-trace-stability-policy",
            real_trace_stability_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_policy_v2_ref(self)


class RealTraceStabilityCategoryCountV2(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-category-count/v2"] = (
        "eval-factory/real-trace-stability-category-count/v2"
    )
    category: str = Field(min_length=1, max_length=128)
    count: int = Field(ge=1, le=100_000)


class RealTraceStabilityCorpusSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-corpus-summary/v2"] = (
        "eval-factory/real-trace-stability-corpus-summary/v2"
    )
    corpus_summary_id: Identifier
    source_manifest_ref: ObjectRef
    policy_ref: ObjectRef
    private_inventory_ref: ObjectRef
    eligible_unique_trace_count: int = Field(ge=1, le=100_000)
    unique_instance_count: int = Field(ge=1, le=100_000)
    unique_raw_hash_count: int = Field(ge=1, le=100_000)
    required_unique_trace_count: Literal[100] = 100
    shortfall_count: int = Field(ge=0, le=100)
    sample_status: RealTraceStabilitySampleStatusV2
    total_raw_bytes: int = Field(ge=1, le=1_000_000_000_000)
    minimum_raw_bytes: int = Field(ge=1, le=1_000_000_000)
    maximum_raw_bytes: int = Field(ge=1, le=1_000_000_000)
    category_counts: tuple[RealTraceStabilityCategoryCountV2, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    corpus_summary_sha256: Sha256
    audit: ContractAudit

    @field_validator("sample_status", mode="before")
    @classmethod
    def parse_sample_status(
        cls,
        value: object,
    ) -> RealTraceStabilitySampleStatusV2:
        return _parse_enum(
            value,
            RealTraceStabilitySampleStatusV2,
            "sample_status",
        )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(
            self.source_manifest_ref,
            "real-trace-source-manifest",
            "csv/v1",
            "source_manifest_ref",
        )
        _require_ref(
            self.policy_ref,
            "real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.private_inventory_ref,
            "real-trace-stability-inventory",
            "private-v1",
            "private_inventory_ref",
        )
        if not (self.eligible_unique_trace_count == self.unique_instance_count == self.unique_raw_hash_count):
            raise ValueError("corpus unique counts disagree")
        expected_shortfall = max(
            self.required_unique_trace_count - self.eligible_unique_trace_count,
            0,
        )
        expected_status = (
            RealTraceStabilitySampleStatusV2.SUFFICIENT
            if expected_shortfall == 0
            else RealTraceStabilitySampleStatusV2.INSUFFICIENT
        )
        if self.shortfall_count != expected_shortfall or self.sample_status is not expected_status:
            raise ValueError("corpus sample sufficiency fields disagree")
        if self.minimum_raw_bytes > self.maximum_raw_bytes:
            raise ValueError("corpus byte bounds are reversed")
        categories = tuple(item.category for item in self.category_counts)
        if categories != tuple(sorted(set(categories))):
            raise ValueError("category counts must be sorted and unique")
        if sum(item.count for item in self.category_counts) != self.eligible_unique_trace_count:
            raise ValueError("category counts do not cover the corpus")
        refs = (
            self.source_manifest_ref,
            self.policy_ref,
            self.private_inventory_ref,
        )
        _require_audit(self.audit, refs, "corpus summary")
        _validate_identity(
            self.corpus_summary_id,
            self.corpus_summary_sha256,
            "real-trace-stability-corpus-summary",
            real_trace_stability_corpus_summary_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_manifest_ref: ObjectRef,
        policy_ref: ObjectRef,
        private_inventory_ref: ObjectRef,
        eligible_unique_trace_count: int,
        unique_instance_count: int,
        unique_raw_hash_count: int,
        total_raw_bytes: int,
        minimum_raw_bytes: int,
        maximum_raw_bytes: int,
        category_counts: tuple[RealTraceStabilityCategoryCountV2, ...],
        audit: ContractAudit,
    ) -> RealTraceStabilityCorpusSummaryV2:
        shortfall = max(100 - eligible_unique_trace_count, 0)
        value = cls(
            corpus_summary_id="real-trace-stability-corpus-summary://pending",
            source_manifest_ref=source_manifest_ref,
            policy_ref=policy_ref,
            private_inventory_ref=private_inventory_ref,
            eligible_unique_trace_count=eligible_unique_trace_count,
            unique_instance_count=unique_instance_count,
            unique_raw_hash_count=unique_raw_hash_count,
            shortfall_count=shortfall,
            sample_status=(
                RealTraceStabilitySampleStatusV2.SUFFICIENT
                if shortfall == 0
                else RealTraceStabilitySampleStatusV2.INSUFFICIENT
            ),
            total_raw_bytes=total_raw_bytes,
            minimum_raw_bytes=minimum_raw_bytes,
            maximum_raw_bytes=maximum_raw_bytes,
            category_counts=tuple(sorted(category_counts, key=lambda item: item.category)),
            corpus_summary_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (source_manifest_ref, policy_ref, private_inventory_ref),
            ),
        )
        return _finalize(
            value,
            "corpus_summary_id",
            "corpus_summary_sha256",
            "real-trace-stability-corpus-summary",
            real_trace_stability_corpus_summary_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_corpus_summary_v2_ref(self)


class RealTraceStabilityCaseSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-case-summary/v2"] = (
        "eval-factory/real-trace-stability-case-summary/v2"
    )
    case_summary_id: Identifier
    private_case_result_ref: ObjectRef
    outcome: RealTraceStabilityCaseOutcomeV2
    reason_code: RealTraceStabilityReasonCodeV2
    parse_quality: ParseQuality | None = None
    assigned_fault_point: RealTraceStabilityFaultPointV2
    fault_observed: bool
    resume_succeeded: bool
    replay_stable: bool
    attempt_count: int = Field(ge=0, le=1_000)
    unexpected_retry_count: int = Field(ge=0, le=1_000)
    resume_count: int = Field(ge=0, le=1_000)
    replay_count: int = Field(ge=0, le=1_000)
    stage_result_count: int = Field(ge=0, le=1_000)
    completion_witness_count: int = Field(ge=0, le=1_000)
    threshold_matched: bool
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    case_summary_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> RealTraceStabilityCaseOutcomeV2:
        return _parse_enum(value, RealTraceStabilityCaseOutcomeV2, "outcome")

    @field_validator("reason_code", mode="before")
    @classmethod
    def parse_reason(
        cls,
        value: object,
    ) -> RealTraceStabilityReasonCodeV2:
        return _parse_enum(value, RealTraceStabilityReasonCodeV2, "reason_code")

    @field_validator("parse_quality", mode="before")
    @classmethod
    def parse_parse_quality(cls, value: object) -> ParseQuality | None:
        if value is None:
            return None
        return _parse_enum(value, ParseQuality, "parse_quality")

    @field_validator("assigned_fault_point", mode="before")
    @classmethod
    def parse_fault(
        cls,
        value: object,
    ) -> RealTraceStabilityFaultPointV2:
        return _parse_enum(
            value,
            RealTraceStabilityFaultPointV2,
            "assigned_fault_point",
        )

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(
            self.private_case_result_ref,
            "real-trace-stability-case-result",
            "private-v1",
            "private_case_result_ref",
        )
        if self.unexpected_retry_count > self.attempt_count:
            raise ValueError("unexpected retries cannot exceed attempts")
        validate_real_trace_stability_case_classification(
            self.outcome,
            self.reason_code,
            replay_stable=self.replay_stable,
        )
        stable = _stable_case_fields(self)
        if self.outcome is RealTraceStabilityCaseOutcomeV2.STABLE:
            if not stable or not self.threshold_matched:
                raise ValueError("STABLE case requires every strict invariant")
        elif self.threshold_matched:
            raise ValueError("non-stable case cannot match the stability threshold")
        _require_audit(
            self.audit,
            (self.private_case_result_ref,),
            "case summary",
        )
        _validate_identity(
            self.case_summary_id,
            self.case_summary_sha256,
            "real-trace-stability-case-summary",
            real_trace_stability_case_summary_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        private_case_result_ref: ObjectRef,
        outcome: RealTraceStabilityCaseOutcomeV2,
        reason_code: RealTraceStabilityReasonCodeV2,
        parse_quality: ParseQuality | None,
        assigned_fault_point: RealTraceStabilityFaultPointV2,
        fault_observed: bool,
        resume_succeeded: bool,
        replay_stable: bool,
        attempt_count: int,
        unexpected_retry_count: int,
        resume_count: int,
        replay_count: int,
        stage_result_count: int,
        completion_witness_count: int,
        audit: ContractAudit,
    ) -> RealTraceStabilityCaseSummaryV2:
        strict_match = (
            outcome is RealTraceStabilityCaseOutcomeV2.STABLE
            and reason_code is RealTraceStabilityReasonCodeV2.NONE
            and parse_quality is not None
            and fault_observed
            and resume_succeeded
            and replay_stable
            and attempt_count == 1
            and unexpected_retry_count == 0
            and resume_count == 1
            and replay_count == 1
            and stage_result_count == 1
            and completion_witness_count == 1
        )
        value = cls(
            case_summary_id="real-trace-stability-case-summary://pending",
            private_case_result_ref=private_case_result_ref,
            outcome=outcome,
            reason_code=reason_code,
            parse_quality=parse_quality,
            assigned_fault_point=assigned_fault_point,
            fault_observed=fault_observed,
            resume_succeeded=resume_succeeded,
            replay_stable=replay_stable,
            attempt_count=attempt_count,
            unexpected_retry_count=unexpected_retry_count,
            resume_count=resume_count,
            replay_count=replay_count,
            stage_result_count=stage_result_count,
            completion_witness_count=completion_witness_count,
            threshold_matched=strict_match,
            case_summary_sha256="0" * 64,
            audit=_safe_audit(audit, (private_case_result_ref,)),
        )
        return _finalize(
            value,
            "case_summary_id",
            "case_summary_sha256",
            "real-trace-stability-case-summary",
            real_trace_stability_case_summary_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_case_summary_v2_ref(self)


class RealTraceStabilityReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/real-trace-stability-report/v2"] = (
        "eval-factory/real-trace-stability-report/v2"
    )
    report_id: Identifier
    policy_ref: ObjectRef
    corpus_summary: RealTraceStabilityCorpusSummaryV2
    corpus_summary_ref: ObjectRef
    private_result_set_ref: ObjectRef
    case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    case_summary_refs: tuple[ObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    stable_case_refs: tuple[ObjectRef, ...]
    unstable_case_refs: tuple[ObjectRef, ...]
    infrastructure_error_case_refs: tuple[ObjectRef, ...]
    incomplete_case_refs: tuple[ObjectRef, ...]
    parse_quality_counts: tuple[RealTraceStabilityCategoryCountV2, ...]
    parse_quality_unavailable_count: int = Field(ge=0, le=100_000)
    fault_point_counts: tuple[RealTraceStabilityCategoryCountV2, ...]
    total_attempts: int = Field(ge=0)
    total_unexpected_retries: int = Field(ge=0)
    total_resumes: int = Field(ge=0)
    total_replays: int = Field(ge=0)
    total_stage_results: int = Field(ge=0)
    total_completion_witnesses: int = Field(ge=0)
    observed_threshold_met: bool
    sample_status: RealTraceStabilitySampleStatusV2
    outcome: RealTraceStabilityOutcomeV2
    claim_scope: Literal["REAL_TRACE_STABILITY_ONLY"] = REAL_TRACE_STABILITY_CLAIM_SCOPE
    policy_version: Literal["real-trace-stability/r8-04-v1"] = REAL_TRACE_STABILITY_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @field_validator("sample_status", mode="before")
    @classmethod
    def parse_sample_status(
        cls,
        value: object,
    ) -> RealTraceStabilitySampleStatusV2:
        return _parse_enum(
            value,
            RealTraceStabilitySampleStatusV2,
            "sample_status",
        )

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_report_outcome(
        cls,
        value: object,
    ) -> RealTraceStabilityOutcomeV2:
        return _parse_enum(value, RealTraceStabilityOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.policy_ref,
            "real-trace-stability-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.corpus_summary_ref,
            "real-trace-stability-corpus-summary",
            "v2",
            "corpus_summary_ref",
        )
        _require_ref(
            self.private_result_set_ref,
            "real-trace-stability-result-set",
            "private-v1",
            "private_result_set_ref",
        )
        validate_real_trace_stability_corpus_summary_v2_identity(self.corpus_summary)
        if (
            self.corpus_summary_ref != real_trace_stability_corpus_summary_v2_ref(self.corpus_summary)
            or self.corpus_summary.policy_ref != self.policy_ref
        ):
            raise ValueError("report corpus authority is stale")
        expected = _derive_report_fields(
            self.corpus_summary,
            self.case_summaries,
        )
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"report {field_name} differs from case evidence")
        refs = (
            self.policy_ref,
            self.corpus_summary_ref,
            self.private_result_set_ref,
            *self.case_summary_refs,
        )
        _require_audit(self.audit, refs, "stability report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "real-trace-stability-report",
            real_trace_stability_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        corpus_summary: RealTraceStabilityCorpusSummaryV2,
        private_result_set_ref: ObjectRef,
        case_summaries: tuple[RealTraceStabilityCaseSummaryV2, ...],
        audit: ContractAudit,
    ) -> RealTraceStabilityReportV2:
        ordered = tuple(sorted(case_summaries, key=lambda item: item.case_summary_id))
        fields = _derive_report_fields(corpus_summary, ordered)
        corpus_ref = real_trace_stability_corpus_summary_v2_ref(corpus_summary)
        refs = (
            policy_ref,
            corpus_ref,
            private_result_set_ref,
            *fields["case_summary_refs"],
        )
        value = cls(
            report_id="real-trace-stability-report://pending",
            policy_ref=policy_ref,
            corpus_summary=corpus_summary,
            corpus_summary_ref=corpus_ref,
            private_result_set_ref=private_result_set_ref,
            case_summaries=ordered,
            **fields,
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "real-trace-stability-report",
            real_trace_stability_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return real_trace_stability_report_v2_ref(self)


def real_trace_stability_policy_v2_carried_sha256(
    value: RealTraceStabilityPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def real_trace_stability_corpus_summary_v2_carried_sha256(
    value: RealTraceStabilityCorpusSummaryV2,
) -> str:
    return _carried(
        value,
        {"corpus_summary_id", "corpus_summary_sha256", "audit"},
    )


def real_trace_stability_case_summary_v2_carried_sha256(
    value: RealTraceStabilityCaseSummaryV2,
) -> str:
    return _carried(
        value,
        {"case_summary_id", "case_summary_sha256", "audit"},
    )


def real_trace_stability_report_v2_carried_sha256(
    value: RealTraceStabilityReportV2,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={
            "report_id",
            "report_sha256",
            "audit",
            "corpus_summary",
            "case_summaries",
        },
        exclude_none=False,
    )
    return _payload_sha256(payload)


def real_trace_stability_policy_v2_ref(
    value: RealTraceStabilityPolicyV2,
) -> ObjectRef:
    validate_real_trace_stability_policy_v2_identity(value)
    return _ref(
        "real-trace-stability-policy",
        value.policy_id,
        value.policy_sha256,
    )


def real_trace_stability_corpus_summary_v2_ref(
    value: RealTraceStabilityCorpusSummaryV2,
) -> ObjectRef:
    validate_real_trace_stability_corpus_summary_v2_identity(value)
    return _ref(
        "real-trace-stability-corpus-summary",
        value.corpus_summary_id,
        value.corpus_summary_sha256,
    )


def real_trace_stability_case_summary_v2_ref(
    value: RealTraceStabilityCaseSummaryV2,
) -> ObjectRef:
    validate_real_trace_stability_case_summary_v2_identity(value)
    return _ref(
        "real-trace-stability-case-summary",
        value.case_summary_id,
        value.case_summary_sha256,
    )


def real_trace_stability_report_v2_ref(
    value: RealTraceStabilityReportV2,
) -> ObjectRef:
    validate_real_trace_stability_report_v2_identity(value)
    return _ref(
        "real-trace-stability-report",
        value.report_id,
        value.report_sha256,
    )


def validate_real_trace_stability_policy_v2_identity(
    value: RealTraceStabilityPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "real-trace-stability-policy",
        real_trace_stability_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_real_trace_stability_corpus_summary_v2_identity(
    value: RealTraceStabilityCorpusSummaryV2,
) -> None:
    _validate_identity(
        value.corpus_summary_id,
        value.corpus_summary_sha256,
        "real-trace-stability-corpus-summary",
        real_trace_stability_corpus_summary_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_real_trace_stability_case_summary_v2_identity(
    value: RealTraceStabilityCaseSummaryV2,
) -> None:
    _validate_identity(
        value.case_summary_id,
        value.case_summary_sha256,
        "real-trace-stability-case-summary",
        real_trace_stability_case_summary_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_real_trace_stability_report_v2_identity(
    value: RealTraceStabilityReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "real-trace-stability-report",
        real_trace_stability_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_real_trace_stability_case_classification(
    outcome: RealTraceStabilityCaseOutcomeV2,
    reason_code: RealTraceStabilityReasonCodeV2,
    *,
    replay_stable: bool,
) -> None:
    if outcome is RealTraceStabilityCaseOutcomeV2.STABLE:
        allowed = frozenset({RealTraceStabilityReasonCodeV2.NONE})
    elif outcome is RealTraceStabilityCaseOutcomeV2.UNSTABLE:
        allowed = frozenset(
            {
                RealTraceStabilityReasonCodeV2.TRACE_RUN_FAILED,
                RealTraceStabilityReasonCodeV2.EXPECTED_FAULT_NOT_OBSERVED,
                RealTraceStabilityReasonCodeV2.RESUME_FAILED,
                RealTraceStabilityReasonCodeV2.UNEXPECTED_RETRY,
                RealTraceStabilityReasonCodeV2.REPLAY_MISMATCH,
                RealTraceStabilityReasonCodeV2.DUPLICATE_COMPLETION,
            }
        )
    elif outcome is RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR:
        allowed = frozenset(
            {
                RealTraceStabilityReasonCodeV2.MATERIAL_INTEGRITY_ERROR,
                RealTraceStabilityReasonCodeV2.INTERNAL_ERROR,
            }
        )
    else:
        allowed = frozenset({RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE})
    if reason_code not in allowed:
        raise ValueError("case outcome and reason code disagree")
    if (
        outcome
        in {
            RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR,
            RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
        }
        and replay_stable
    ):
        raise ValueError("untrustworthy case cannot claim replay stability")


def _stable_case_fields(value: RealTraceStabilityCaseSummaryV2) -> bool:
    return (
        value.parse_quality is not None
        and value.fault_observed
        and value.resume_succeeded
        and value.replay_stable
        and value.attempt_count == 1
        and value.unexpected_retry_count == 0
        and value.resume_count == 1
        and value.replay_count == 1
        and value.stage_result_count == 1
        and value.completion_witness_count == 1
    )


def _derive_report_fields(
    corpus: RealTraceStabilityCorpusSummaryV2,
    cases: tuple[RealTraceStabilityCaseSummaryV2, ...],
) -> _DerivedReportFields:
    if len(cases) != corpus.eligible_unique_trace_count:
        raise ValueError("report case count differs from corpus")
    refs = tuple(real_trace_stability_case_summary_v2_ref(item) for item in cases)
    if refs != tuple(sorted(refs, key=_ref_key)) or len(refs) != len(set(refs)):
        raise ValueError("report case summaries must be sorted and unique")
    private_refs = tuple(item.private_case_result_ref for item in cases)
    if len(private_refs) != len(set(private_refs)):
        raise ValueError("report case summaries reuse private case results")

    partitions = {
        outcome: tuple(ref for item, ref in zip(cases, refs, strict=True) if item.outcome is outcome)
        for outcome in RealTraceStabilityCaseOutcomeV2
    }
    parse_counts = Counter(item.parse_quality.value for item in cases if item.parse_quality is not None)
    fault_counts = Counter(item.assigned_fault_point.value for item in cases)
    expected_fault_counts = Counter(
        tuple(RealTraceStabilityFaultPointV2)[index % len(RealTraceStabilityFaultPointV2)].value
        for index in range(len(cases))
    )
    if fault_counts != expected_fault_counts:
        raise ValueError("report fault distribution differs from canonical inventory rotation")
    parse_rows = tuple(
        RealTraceStabilityCategoryCountV2(category=category, count=count)
        for category, count in sorted(parse_counts.items())
    )
    fault_rows = tuple(
        RealTraceStabilityCategoryCountV2(category=category, count=count)
        for category, count in sorted(fault_counts.items())
    )
    observed_threshold_met = bool(cases) and all(item.threshold_matched for item in cases)
    incomplete = bool(
        partitions[RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR]
        or partitions[RealTraceStabilityCaseOutcomeV2.INCOMPLETE]
    )
    if incomplete:
        outcome = RealTraceStabilityOutcomeV2.INCOMPLETE
    elif corpus.sample_status is RealTraceStabilitySampleStatusV2.INSUFFICIENT:
        outcome = RealTraceStabilityOutcomeV2.STATISTICAL_GATE_PENDING
    elif not observed_threshold_met:
        outcome = RealTraceStabilityOutcomeV2.STABILITY_THRESHOLD_NOT_MET
    else:
        outcome = RealTraceStabilityOutcomeV2.PASSED

    return {
        "case_summary_refs": refs,
        "stable_case_refs": partitions[RealTraceStabilityCaseOutcomeV2.STABLE],
        "unstable_case_refs": partitions[RealTraceStabilityCaseOutcomeV2.UNSTABLE],
        "infrastructure_error_case_refs": partitions[RealTraceStabilityCaseOutcomeV2.INFRASTRUCTURE_ERROR],
        "incomplete_case_refs": partitions[RealTraceStabilityCaseOutcomeV2.INCOMPLETE],
        "parse_quality_counts": parse_rows,
        "parse_quality_unavailable_count": len(cases) - sum(parse_counts.values()),
        "fault_point_counts": fault_rows,
        "total_attempts": sum(item.attempt_count for item in cases),
        "total_unexpected_retries": sum(item.unexpected_retry_count for item in cases),
        "total_resumes": sum(item.resume_count for item in cases),
        "total_replays": sum(item.replay_count for item in cases),
        "total_stage_results": sum(item.stage_result_count for item in cases),
        "total_completion_witnesses": sum(item.completion_witness_count for item in cases),
        "observed_threshold_met": observed_threshold_met,
        "sample_status": corpus.sample_status,
        "outcome": outcome,
    }


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": tuple(sorted(set(refs), key=_ref_key))})


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    expected = tuple(sorted(set(refs), key=_ref_key))
    if audit.input_refs != expected:
        raise ValueError(f"{label} audit refs are incomplete")
    versions = tuple(
        binding for binding in audit.governing_versions if binding.component == "real-trace-stability"
    )
    if len(versions) != 1 or versions[0].version != REAL_TRACE_STABILITY_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing the stability policy")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} has the wrong type or version")


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{object_type} identity is pending")
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix} identity is stale")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "REAL_TRACE_STABILITY_CLAIM_SCOPE",
    "REAL_TRACE_STABILITY_POLICY_VERSION",
    "REAL_TRACE_STABILITY_SOURCE_MANIFEST_SHA256",
    "RealTraceStabilityCaseOutcomeV2",
    "RealTraceStabilityCaseSummaryV2",
    "RealTraceStabilityCategoryCountV2",
    "RealTraceStabilityCorpusSummaryV2",
    "RealTraceStabilityFaultPointV2",
    "RealTraceStabilityOutcomeV2",
    "RealTraceStabilityPolicyV2",
    "RealTraceStabilityReasonCodeV2",
    "RealTraceStabilityReportV2",
    "RealTraceStabilitySampleStatusV2",
    "real_trace_stability_case_summary_v2_carried_sha256",
    "real_trace_stability_case_summary_v2_ref",
    "real_trace_stability_corpus_summary_v2_carried_sha256",
    "real_trace_stability_corpus_summary_v2_ref",
    "real_trace_stability_policy_v2_carried_sha256",
    "real_trace_stability_policy_v2_ref",
    "real_trace_stability_report_v2_carried_sha256",
    "real_trace_stability_report_v2_ref",
    "validate_real_trace_stability_case_classification",
    "validate_real_trace_stability_case_summary_v2_identity",
    "validate_real_trace_stability_corpus_summary_v2_identity",
    "validate_real_trace_stability_policy_v2_identity",
    "validate_real_trace_stability_report_v2_identity",
]
