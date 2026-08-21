from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.canary_execution_v2 import (
    R6_CANARY_EXECUTION_PROFILE,
    R6_CANARY_POLICY_VERSION,
    R6_CANARY_STAGES,
    R6CanaryDatasetResultV2,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import ItemStatus, JobStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2

CANARY_REGRESSION_POLICY_VERSION: Literal["canary-regression/r8-03-v1"] = "canary-regression/r8-03-v1"
CANARY_REGRESSION_CLAIM_SCOPE: Literal["CANARY_REGRESSION_ONLY"] = "CANARY_REGRESSION_ONLY"
CANARY_REGRESSION_COHORT_SHA256: Literal[
    "1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"
] = "1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"


class CanaryRegressionExpectationV2(StrEnum):
    MUST_SUCCEED = "MUST_SUCCEED"
    ANY_TYPED_TERMINAL = "ANY_TYPED_TERMINAL"


class CanaryRegressionObservedOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"
    INCOMPLETE = "INCOMPLETE"


class CanaryRegressionReasonCodeV2(StrEnum):
    NONE = "NONE"
    LABEL_NO_MATCH = "LABEL_NO_MATCH"
    STAGE_BLOCKED_POLICY = "STAGE_BLOCKED_POLICY"
    STAGE_BLOCKED_CAPABILITY = "STAGE_BLOCKED_CAPABILITY"
    STAGE_RETRY_EXHAUSTED = "STAGE_RETRY_EXHAUSTED"
    RESOURCE_WAITING = "RESOURCE_WAITING"
    RESOURCE_BUDGET_EXHAUSTED = "RESOURCE_BUDGET_EXHAUSTED"
    RESOURCE_UNSATISFIABLE = "RESOURCE_UNSATISFIABLE"
    PROFILE_POLICY_ERROR = "PROFILE_POLICY_ERROR"
    LIFECYCLE_POLICY_ERROR = "LIFECYCLE_POLICY_ERROR"
    MATERIAL_INTEGRITY_ERROR = "MATERIAL_INTEGRITY_ERROR"
    CHILD_STATE_INCOMPLETE = "CHILD_STATE_INCOMPLETE"
    OUTCOME_MISMATCH = "OUTCOME_MISMATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class CanaryRegressionOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


class CanaryRegressionPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-policy/v2"] = (
        "eval-factory/canary-regression-policy/v2"
    )
    policy_id: Identifier
    cohort_manifest_ref: ObjectRef
    min_cases: Literal[20] = 20
    max_cases: Literal[30] = 30
    required_case_count: Literal[24] = 24
    strict_signal_quality: Literal["strict_events"] = "strict_events"
    heuristic_signal_quality: Literal["heuristic_requires_annotation"] = "heuristic_requires_annotation"
    success_verified_trait: Literal["tool.file_read"] = "tool.file_read"
    child_execution_profile: Literal["r6-canary/r1-r5-v1"] = R6_CANARY_EXECUTION_PROFILE
    child_policy_version: Literal["r6-canary-execution/r6-parent-v1"] = R6_CANARY_POLICY_VERSION
    max_case_refs: int = Field(ge=24, le=100_000)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["canary-regression/r8-03-v1"] = CANARY_REGRESSION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(
            self.cohort_manifest_ref,
            "development-canary-manifest",
            "v4",
            "cohort_manifest_ref",
        )
        if self.cohort_manifest_ref.object_sha256 != CANARY_REGRESSION_COHORT_SHA256:
            raise ValueError("canary regression policy must bind the approved v4 cohort")
        _validate_audit(
            self.audit,
            (self.cohort_manifest_ref,),
            "canary regression policy",
        )
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "canary-regression-policy",
            canary_regression_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        cohort_manifest_ref: ObjectRef,
        max_case_refs: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> CanaryRegressionPolicyV2:
        value = cls(
            policy_id="canary-regression-policy://pending",
            cohort_manifest_ref=cohort_manifest_ref,
            max_case_refs=max_case_refs,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, (cohort_manifest_ref,)),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "canary-regression-policy",
            canary_regression_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return canary_regression_policy_v2_ref(self)


class CanaryRegressionCaseResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-case-result/v2"] = (
        "eval-factory/canary-regression-case-result/v2"
    )
    case_result_id: Identifier
    instance_id: Identifier
    source_trace_ref: ObjectRef
    signal_quality: Literal["strict_events", "heuristic_requires_annotation"]
    expectation: CanaryRegressionExpectationV2
    observed_outcome: CanaryRegressionObservedOutcomeV2
    expectation_matched: bool
    reason_code: CanaryRegressionReasonCodeV2
    highest_reached_stage: StageNameV2 | None = None
    child_manifest_ref: ObjectRef
    child_dataset_result_ref: ObjectRef | None = None
    job_id: Identifier
    job_status: JobStatus | None = None
    item_id: Identifier | None = None
    item_status: ItemStatus | None = None
    stage_result_refs: tuple[ObjectRef, ...] = ()
    quality_result_refs: tuple[ObjectRef, ...] = ()
    package_manifest_refs: tuple[ObjectRef, ...] = ()
    audit_report_ref: ObjectRef | None = None
    work_unit_count: int = Field(ge=0, le=10_000)
    work_lease_count: int = Field(ge=0, le=10_000)
    stage_run_count: int = Field(ge=0, le=10_000)
    attempt_count: int = Field(ge=0, le=10_000)
    retry_count: int = Field(ge=0, le=10_000)
    resume_count: int = Field(ge=0, le=10_000)
    provider_invocation_count: int = Field(ge=0, le=10_000)
    policy_version: Literal["canary-regression/r8-03-v1"] = CANARY_REGRESSION_POLICY_VERSION
    case_result_sha256: Sha256
    audit: ContractAudit

    @field_validator("expectation", mode="before")
    @classmethod
    def parse_expectation(cls, value: object) -> CanaryRegressionExpectationV2:
        return _parse_enum(value, CanaryRegressionExpectationV2, "expectation")

    @field_validator("observed_outcome", mode="before")
    @classmethod
    def parse_observed(
        cls,
        value: object,
    ) -> CanaryRegressionObservedOutcomeV2:
        return _parse_enum(
            value,
            CanaryRegressionObservedOutcomeV2,
            "observed_outcome",
        )

    @field_validator("reason_code", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> CanaryRegressionReasonCodeV2:
        return _parse_enum(value, CanaryRegressionReasonCodeV2, "reason_code")

    @field_validator("highest_reached_stage", mode="before")
    @classmethod
    def parse_stage(cls, value: object) -> StageNameV2 | None:
        if value is None:
            return None
        return _parse_enum(value, StageNameV2, "highest_reached_stage")

    @field_validator("job_status", mode="before")
    @classmethod
    def parse_job_status(cls, value: object) -> JobStatus | None:
        if value is None:
            return None
        return _parse_enum(value, JobStatus, "job_status")

    @field_validator("item_status", mode="before")
    @classmethod
    def parse_item_status(cls, value: object) -> ItemStatus | None:
        if value is None:
            return None
        return _parse_enum(value, ItemStatus, "item_status")

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        _require_ref(self.source_trace_ref, "trace-source", None, "source_trace_ref")
        _require_ref(
            self.child_manifest_ref,
            "r6-canary-execution-manifest",
            "v2",
            "child_manifest_ref",
        )
        if self.child_dataset_result_ref is not None:
            _require_ref(
                self.child_dataset_result_ref,
                "r6-canary-dataset-result",
                "v2",
                "child_dataset_result_ref",
            )
        _require_refs(
            self.stage_result_refs,
            "stage-result",
            "stage_result_refs",
        )
        _require_refs(
            self.quality_result_refs,
            "item-quality-compilation-result",
            "quality_result_refs",
            object_version="v2",
        )
        _require_refs(
            self.package_manifest_refs,
            "final-package-manifest",
            "package_manifest_refs",
            object_version="v2",
        )
        if self.audit_report_ref is not None:
            _require_ref(
                self.audit_report_ref,
                "batch-audit-report",
                "v2",
                "audit_report_ref",
            )
        if self.highest_reached_stage is not None and self.highest_reached_stage not in R6_CANARY_STAGES:
            raise ValueError("highest reached stage must belong to the R6 canary chain")
        expected_match = _expectation_matches(
            self.expectation,
            self.observed_outcome,
        )
        if self.expectation_matched is not expected_match:
            raise ValueError("case expectation match is not derived")
        if self.retry_count > self.attempt_count or self.resume_count > self.attempt_count:
            raise ValueError("retry/resume counts cannot exceed attempts")
        self._validate_outcome_shape()
        refs = (
            self.source_trace_ref,
            self.child_manifest_ref,
            *((self.child_dataset_result_ref,) if self.child_dataset_result_ref is not None else ()),
            *self.stage_result_refs,
            *self.quality_result_refs,
            *self.package_manifest_refs,
            *((self.audit_report_ref,) if self.audit_report_ref is not None else ()),
        )
        _validate_audit(self.audit, refs, "canary regression case result")
        _validate_identity(
            self.case_result_id,
            self.case_result_sha256,
            "canary-regression-case-result",
            canary_regression_case_result_v2_carried_sha256(self),
        )
        return self

    def _validate_outcome_shape(self) -> None:
        outcome = self.observed_outcome
        if outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED:
            if (
                self.reason_code is not CanaryRegressionReasonCodeV2.NONE
                or self.highest_reached_stage is not StageNameV2.ITEM_QUALITY
                or self.child_dataset_result_ref is None
                or self.job_status is not JobStatus.SUCCEEDED
                or self.item_id is None
                or self.item_status is not ItemStatus.APPROVED
                or not self.stage_result_refs
                or len(self.quality_result_refs) != 1
                or len(self.package_manifest_refs) != 1
                or self.audit_report_ref is None
            ):
                raise ValueError("successful case requires complete R1-R5 authority")
            return
        if self.reason_code is CanaryRegressionReasonCodeV2.NONE:
            raise ValueError("non-success case requires a closed reason")
        if self.quality_result_refs or self.package_manifest_refs:
            raise ValueError("non-success case cannot carry quality or package success")
        if outcome is CanaryRegressionObservedOutcomeV2.BLOCKED:
            if (
                self.item_id is None
                or self.item_status is not ItemStatus.REJECTED
                or self.highest_reached_stage is None
                or not self.stage_result_refs
                or self.child_dataset_result_ref is not None
            ):
                raise ValueError("blocked case requires exact rejected Item evidence")
            return
        if outcome is CanaryRegressionObservedOutcomeV2.FAILED:
            if (
                self.item_id is None
                or self.item_status is not ItemStatus.FAILED
                or self.highest_reached_stage is None
                or not self.stage_result_refs
            ):
                raise ValueError("failed case requires exact terminal Item evidence")
            return
        if outcome is CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR:
            if self.child_dataset_result_ref is not None:
                raise ValueError("infrastructure failure cannot carry child dataset result")
            return
        if (
            self.reason_code is not CanaryRegressionReasonCodeV2.CHILD_STATE_INCOMPLETE
            or self.child_dataset_result_ref is not None
            or self.item_status in {ItemStatus.APPROVED, ItemStatus.REJECTED, ItemStatus.FAILED}
        ):
            raise ValueError("incomplete case cannot carry terminal product authority")

    @classmethod
    def create(
        cls,
        *,
        instance_id: str,
        source_trace_ref: ObjectRef,
        signal_quality: Literal["strict_events", "heuristic_requires_annotation"],
        expectation: CanaryRegressionExpectationV2,
        observed_outcome: CanaryRegressionObservedOutcomeV2,
        reason_code: CanaryRegressionReasonCodeV2,
        highest_reached_stage: StageNameV2 | None,
        child_manifest_ref: ObjectRef,
        child_dataset_result_ref: ObjectRef | None,
        job_id: str,
        job_status: JobStatus | None,
        item_id: str | None,
        item_status: ItemStatus | None,
        stage_result_refs: tuple[ObjectRef, ...],
        quality_result_refs: tuple[ObjectRef, ...],
        package_manifest_refs: tuple[ObjectRef, ...],
        audit_report_ref: ObjectRef | None,
        work_unit_count: int,
        work_lease_count: int,
        stage_run_count: int,
        attempt_count: int,
        retry_count: int,
        resume_count: int,
        provider_invocation_count: int,
        audit: ContractAudit,
    ) -> CanaryRegressionCaseResultV2:
        stage_refs = _sorted_refs(stage_result_refs)
        quality_refs = _sorted_refs(quality_result_refs)
        package_refs = _sorted_refs(package_manifest_refs)
        refs = (
            source_trace_ref,
            child_manifest_ref,
            *((child_dataset_result_ref,) if child_dataset_result_ref is not None else ()),
            *stage_refs,
            *quality_refs,
            *package_refs,
            *((audit_report_ref,) if audit_report_ref is not None else ()),
        )
        value = cls(
            case_result_id="canary-regression-case-result://pending",
            instance_id=instance_id,
            source_trace_ref=source_trace_ref,
            signal_quality=signal_quality,
            expectation=expectation,
            observed_outcome=observed_outcome,
            expectation_matched=_expectation_matches(expectation, observed_outcome),
            reason_code=reason_code,
            highest_reached_stage=highest_reached_stage,
            child_manifest_ref=child_manifest_ref,
            child_dataset_result_ref=child_dataset_result_ref,
            job_id=job_id,
            job_status=job_status,
            item_id=item_id,
            item_status=item_status,
            stage_result_refs=stage_refs,
            quality_result_refs=quality_refs,
            package_manifest_refs=package_refs,
            audit_report_ref=audit_report_ref,
            work_unit_count=work_unit_count,
            work_lease_count=work_lease_count,
            stage_run_count=stage_run_count,
            attempt_count=attempt_count,
            retry_count=retry_count,
            resume_count=resume_count,
            provider_invocation_count=provider_invocation_count,
            case_result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "case_result_id",
            "case_result_sha256",
            "canary-regression-case-result",
            canary_regression_case_result_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return canary_regression_case_result_v2_ref(self)


class CanaryRegressionStageSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-stage-summary/v2"] = (
        "eval-factory/canary-regression-stage-summary/v2"
    )
    stage: StageNameV2
    reached_count: int = Field(ge=0, le=30)
    terminal_count: int = Field(ge=0, le=30)

    @field_validator("stage", mode="before")
    @classmethod
    def parse_stage(cls, value: object) -> StageNameV2:
        return _parse_enum(value, StageNameV2, "stage")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.stage not in R6_CANARY_STAGES:
            raise ValueError("stage summary must belong to the R6 canary chain")
        if self.terminal_count > self.reached_count:
            raise ValueError("terminal count cannot exceed reached count")
        return self


class CanaryRegressionReportV2(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-report/v2"] = (
        "eval-factory/canary-regression-report/v2"
    )
    report_id: Identifier
    cohort_manifest_ref: ObjectRef
    policy_ref: ObjectRef
    template_ref: ObjectRef
    case_results: tuple[CanaryRegressionCaseResultV2, ...] = Field(min_length=24, max_length=24)
    case_result_refs: tuple[ObjectRef, ...] = Field(min_length=24, max_length=24)
    expected_must_succeed_ids: tuple[Identifier, ...]
    expected_typed_terminal_ids: tuple[Identifier, ...]
    succeeded_ids: tuple[Identifier, ...]
    blocked_ids: tuple[Identifier, ...]
    failed_ids: tuple[Identifier, ...]
    infrastructure_error_ids: tuple[Identifier, ...]
    incomplete_ids: tuple[Identifier, ...]
    matched_ids: tuple[Identifier, ...]
    mismatched_ids: tuple[Identifier, ...]
    stage_summaries: tuple[CanaryRegressionStageSummaryV2, ...]
    total_work_units: int = Field(ge=0)
    total_work_leases: int = Field(ge=0)
    total_stage_runs: int = Field(ge=0)
    total_attempts: int = Field(ge=0)
    total_retries: int = Field(ge=0)
    total_resumes: int = Field(ge=0)
    total_provider_invocations: int = Field(ge=0)
    outcome: CanaryRegressionOutcomeV2
    claim_scope: Literal["CANARY_REGRESSION_ONLY"] = CANARY_REGRESSION_CLAIM_SCOPE
    policy_version: Literal["canary-regression/r8-03-v1"] = CANARY_REGRESSION_POLICY_VERSION
    report_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> CanaryRegressionOutcomeV2:
        return _parse_enum(value, CanaryRegressionOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_ref(
            self.cohort_manifest_ref,
            "development-canary-manifest",
            "v4",
            "cohort_manifest_ref",
        )
        if self.cohort_manifest_ref.object_sha256 != CANARY_REGRESSION_COHORT_SHA256:
            raise ValueError("report must bind approved v4 cohort")
        _require_ref(
            self.policy_ref,
            "canary-regression-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.template_ref,
            "canary-regression-template",
            "private-v1",
            "template_ref",
        )
        for case in self.case_results:
            validate_canary_regression_case_result_v2_identity(case)
        facts = _report_facts(self.case_results)
        expected = (
            self.case_result_refs,
            self.expected_must_succeed_ids,
            self.expected_typed_terminal_ids,
            self.succeeded_ids,
            self.blocked_ids,
            self.failed_ids,
            self.infrastructure_error_ids,
            self.incomplete_ids,
            self.matched_ids,
            self.mismatched_ids,
            self.stage_summaries,
            self.total_work_units,
            self.total_work_leases,
            self.total_stage_runs,
            self.total_attempts,
            self.total_retries,
            self.total_resumes,
            self.total_provider_invocations,
            self.outcome,
        )
        if expected != facts:
            raise ValueError("canary regression report aggregates are not exact")
        refs = (
            self.cohort_manifest_ref,
            self.policy_ref,
            self.template_ref,
            *self.case_result_refs,
        )
        _validate_audit(self.audit, refs, "canary regression report")
        _validate_identity(
            self.report_id,
            self.report_sha256,
            "canary-regression-report",
            canary_regression_report_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        cohort_manifest_ref: ObjectRef,
        policy_ref: ObjectRef,
        template_ref: ObjectRef,
        case_results: tuple[CanaryRegressionCaseResultV2, ...],
        audit: ContractAudit,
    ) -> CanaryRegressionReportV2:
        cases = tuple(sorted(case_results, key=lambda value: value.instance_id))
        facts = _report_facts(cases)
        refs = (
            cohort_manifest_ref,
            policy_ref,
            template_ref,
            *facts[0],
        )
        value = cls(
            report_id="canary-regression-report://pending",
            cohort_manifest_ref=cohort_manifest_ref,
            policy_ref=policy_ref,
            template_ref=template_ref,
            case_results=cases,
            case_result_refs=facts[0],
            expected_must_succeed_ids=facts[1],
            expected_typed_terminal_ids=facts[2],
            succeeded_ids=facts[3],
            blocked_ids=facts[4],
            failed_ids=facts[5],
            infrastructure_error_ids=facts[6],
            incomplete_ids=facts[7],
            matched_ids=facts[8],
            mismatched_ids=facts[9],
            stage_summaries=facts[10],
            total_work_units=facts[11],
            total_work_leases=facts[12],
            total_stage_runs=facts[13],
            total_attempts=facts[14],
            total_retries=facts[15],
            total_resumes=facts[16],
            total_provider_invocations=facts[17],
            outcome=facts[18],
            report_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "report_id",
            "report_sha256",
            "canary-regression-report",
            canary_regression_report_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return canary_regression_report_v2_ref(self)


def canary_regression_policy_v2_carried_sha256(
    value: CanaryRegressionPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def canary_regression_case_result_v2_carried_sha256(
    value: CanaryRegressionCaseResultV2,
) -> str:
    return _carried(value, {"case_result_id", "case_result_sha256", "audit"})


def canary_regression_report_v2_carried_sha256(
    value: CanaryRegressionReportV2,
) -> str:
    return _carried(value, {"report_id", "report_sha256", "audit"})


def canary_regression_policy_v2_ref(
    value: CanaryRegressionPolicyV2,
) -> ObjectRef:
    validate_canary_regression_policy_v2_identity(value)
    return _ref("canary-regression-policy", value.policy_id, value.policy_sha256)


def canary_regression_case_result_v2_ref(
    value: CanaryRegressionCaseResultV2,
) -> ObjectRef:
    validate_canary_regression_case_result_v2_identity(value)
    return _ref(
        "canary-regression-case-result",
        value.case_result_id,
        value.case_result_sha256,
    )


def canary_regression_report_v2_ref(
    value: CanaryRegressionReportV2,
) -> ObjectRef:
    validate_canary_regression_report_v2_identity(value)
    return _ref("canary-regression-report", value.report_id, value.report_sha256)


def r6_canary_dataset_result_v2_ref(
    value: R6CanaryDatasetResultV2,
) -> ObjectRef:
    digest = value.canonical_sha256()
    return ObjectRef(
        object_type="r6-canary-dataset-result",
        object_id=f"r6-canary-dataset-result://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def validate_canary_regression_policy_v2_identity(
    value: CanaryRegressionPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "canary-regression-policy",
        canary_regression_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_canary_regression_case_result_v2_identity(
    value: CanaryRegressionCaseResultV2,
) -> None:
    _validate_identity(
        value.case_result_id,
        value.case_result_sha256,
        "canary-regression-case-result",
        canary_regression_case_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_canary_regression_report_v2_identity(
    value: CanaryRegressionReportV2,
) -> None:
    _validate_identity(
        value.report_id,
        value.report_sha256,
        "canary-regression-report",
        canary_regression_report_v2_carried_sha256(value),
        allow_pending=False,
    )


def _expectation_matches(
    expectation: CanaryRegressionExpectationV2,
    observed: CanaryRegressionObservedOutcomeV2,
) -> bool:
    if expectation is CanaryRegressionExpectationV2.MUST_SUCCEED:
        return observed is CanaryRegressionObservedOutcomeV2.SUCCEEDED
    return observed in {
        CanaryRegressionObservedOutcomeV2.SUCCEEDED,
        CanaryRegressionObservedOutcomeV2.BLOCKED,
        CanaryRegressionObservedOutcomeV2.FAILED,
    }


def _report_facts(
    cases: tuple[CanaryRegressionCaseResultV2, ...],
) -> tuple[
    tuple[ObjectRef, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[CanaryRegressionStageSummaryV2, ...],
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    CanaryRegressionOutcomeV2,
]:
    ids = tuple(case.instance_id for case in cases)
    if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)) or len(ids) != 24:
        raise ValueError("report cases must be 24 sorted unique instances")
    expected_must = _ids(
        cases,
        lambda item: item.expectation is CanaryRegressionExpectationV2.MUST_SUCCEED,
    )
    expected_any = _ids(
        cases,
        lambda item: item.expectation is CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL,
    )
    if len(expected_must) != 15 or len(expected_any) != 9:
        raise ValueError("report expectation partition must be 15 MUST and 9 ANY")
    succeeded = _ids(
        cases,
        lambda item: item.observed_outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED,
    )
    blocked = _ids(
        cases,
        lambda item: item.observed_outcome is CanaryRegressionObservedOutcomeV2.BLOCKED,
    )
    failed = _ids(
        cases,
        lambda item: item.observed_outcome is CanaryRegressionObservedOutcomeV2.FAILED,
    )
    infrastructure = _ids(
        cases,
        lambda item: item.observed_outcome is CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR,
    )
    incomplete = _ids(
        cases,
        lambda item: item.observed_outcome is CanaryRegressionObservedOutcomeV2.INCOMPLETE,
    )
    matched = _ids(cases, lambda item: item.expectation_matched)
    mismatched = _ids(cases, lambda item: not item.expectation_matched)
    stage_summaries = tuple(
        CanaryRegressionStageSummaryV2(
            stage=stage,
            reached_count=sum(_reached(case, stage) for case in cases),
            terminal_count=sum(case.highest_reached_stage is stage for case in cases),
        )
        for stage in R6_CANARY_STAGES
    )
    if incomplete or infrastructure:
        outcome = CanaryRegressionOutcomeV2.INCOMPLETE
    elif failed or mismatched:
        outcome = CanaryRegressionOutcomeV2.FAILED
    else:
        outcome = CanaryRegressionOutcomeV2.PASSED
    return (
        tuple(case.to_ref() for case in cases),
        expected_must,
        expected_any,
        succeeded,
        blocked,
        failed,
        infrastructure,
        incomplete,
        matched,
        mismatched,
        stage_summaries,
        sum(case.work_unit_count for case in cases),
        sum(case.work_lease_count for case in cases),
        sum(case.stage_run_count for case in cases),
        sum(case.attempt_count for case in cases),
        sum(case.retry_count for case in cases),
        sum(case.resume_count for case in cases),
        sum(case.provider_invocation_count for case in cases),
        outcome,
    )


def _reached(case: CanaryRegressionCaseResultV2, stage: StageNameV2) -> bool:
    if case.highest_reached_stage is None:
        return False
    return R6_CANARY_STAGES.index(case.highest_reached_stage) >= R6_CANARY_STAGES.index(stage)


def _ids(
    cases: tuple[CanaryRegressionCaseResultV2, ...],
    predicate: Callable[[CanaryRegressionCaseResultV2], bool],
) -> tuple[str, ...]:
    return tuple(case.instance_id for case in cases if predicate(case))


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
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str | None,
    field_name: str,
) -> None:
    if value.object_type != object_type or (
        object_version is not None and value.object_version != object_version
    ):
        suffix = "" if object_version is None else f" {object_version}"
        raise ValueError(f"{field_name} must reference {object_type}{suffix}")


def _require_refs(
    values: tuple[ObjectRef, ...],
    object_type: str,
    field_name: str,
    *,
    object_version: str | None = None,
) -> None:
    _require_sorted_unique_refs(values, field_name)
    for value in values:
        _require_ref(value, object_type, object_version, field_name)


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
