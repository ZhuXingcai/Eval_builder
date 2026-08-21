from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.canary_regression_v2 import (
    CANARY_REGRESSION_POLICY_VERSION,
    CanaryRegressionCaseResultV2,
    CanaryRegressionExpectationV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionOutcomeV2,
    CanaryRegressionPolicyV2,
    CanaryRegressionReasonCodeV2,
    CanaryRegressionReportV2,
    canary_regression_report_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import ItemStatus, JobStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-03/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r8-03-contract-test",
        governing_versions=(
            VersionBinding(
                component="canary-regression",
                version=CANARY_REGRESSION_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
    )


def _cohort_ref() -> ObjectRef:
    return ObjectRef(
        object_type="development-canary-manifest",
        object_id="development-canary-manifest://v4",
        object_version="v4",
        object_sha256="1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6",
    )


def _policy() -> CanaryRegressionPolicyV2:
    return CanaryRegressionPolicyV2.create(
        cohort_manifest_ref=_cohort_ref(),
        max_case_refs=1_000,
        max_report_bytes=2_000_000,
        audit=_audit(),
    )


def _success(index: int) -> CanaryRegressionCaseResultV2:
    instance_id = f"LH_{index:03d}"
    return CanaryRegressionCaseResultV2.create(
        instance_id=instance_id,
        source_trace_ref=_ref("trace-source", instance_id, version="1.0.0"),
        signal_quality="strict_events",
        expectation=CanaryRegressionExpectationV2.MUST_SUCCEED,
        observed_outcome=CanaryRegressionObservedOutcomeV2.SUCCEEDED,
        reason_code=CanaryRegressionReasonCodeV2.NONE,
        highest_reached_stage=StageNameV2.ITEM_QUALITY,
        child_manifest_ref=_ref("r6-canary-execution-manifest", instance_id),
        child_dataset_result_ref=_ref("r6-canary-dataset-result", instance_id),
        job_id=f"job://r8-03/{instance_id}",
        job_status=JobStatus.SUCCEEDED,
        item_id=f"item://r8-03/{instance_id}",
        item_status=ItemStatus.APPROVED,
        stage_result_refs=tuple(
            _ref("stage-result", f"{instance_id}/{stage.value}", version="v1")
            for stage in (
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
            )
        ),
        quality_result_refs=(_ref("item-quality-compilation-result", instance_id),),
        package_manifest_refs=(_ref("final-package-manifest", instance_id),),
        audit_report_ref=_ref("batch-audit-report", instance_id),
        work_unit_count=10,
        work_lease_count=10,
        stage_run_count=9,
        attempt_count=10,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=1,
        audit=_audit(),
    )


def _blocked(index: int) -> CanaryRegressionCaseResultV2:
    instance_id = f"LH_{index:03d}"
    return CanaryRegressionCaseResultV2.create(
        instance_id=instance_id,
        source_trace_ref=_ref("trace-source", instance_id, version="1.0.0"),
        signal_quality="heuristic_requires_annotation",
        expectation=CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL,
        observed_outcome=CanaryRegressionObservedOutcomeV2.BLOCKED,
        reason_code=CanaryRegressionReasonCodeV2.LABEL_NO_MATCH,
        highest_reached_stage=StageNameV2.LABEL,
        child_manifest_ref=_ref("r6-canary-execution-manifest", instance_id),
        child_dataset_result_ref=None,
        job_id=f"job://r8-03/{instance_id}",
        job_status=JobStatus.RUNNING,
        item_id=f"item://r8-03/{instance_id}",
        item_status=ItemStatus.REJECTED,
        stage_result_refs=(
            _ref("stage-result", f"{instance_id}/trace", version="v1"),
            _ref("stage-result", f"{instance_id}/safety", version="v1"),
            _ref("stage-result", f"{instance_id}/label", version="v1"),
        ),
        quality_result_refs=(),
        package_manifest_refs=(),
        audit_report_ref=None,
        work_unit_count=9,
        work_lease_count=3,
        stage_run_count=3,
        attempt_count=3,
        retry_count=0,
        resume_count=0,
        provider_invocation_count=0,
        audit=_audit(),
    )


def _recreate(
    case: CanaryRegressionCaseResultV2,
    **updates: object,
) -> CanaryRegressionCaseResultV2:
    fields = (
        "instance_id",
        "source_trace_ref",
        "signal_quality",
        "expectation",
        "observed_outcome",
        "reason_code",
        "highest_reached_stage",
        "child_manifest_ref",
        "child_dataset_result_ref",
        "job_id",
        "job_status",
        "item_id",
        "item_status",
        "stage_result_refs",
        "quality_result_refs",
        "package_manifest_refs",
        "audit_report_ref",
        "work_unit_count",
        "work_lease_count",
        "stage_run_count",
        "attempt_count",
        "retry_count",
        "resume_count",
        "provider_invocation_count",
    )
    values = {field: getattr(case, field) for field in fields}
    values.update(updates)
    return CanaryRegressionCaseResultV2.create(
        **values,
        audit=_audit(),
    )


def _report() -> CanaryRegressionReportV2:
    cases = tuple(
        sorted(
            (
                *(_success(index) for index in range(1, 16)),
                *(_blocked(index) for index in range(16, 25)),
            ),
            key=lambda value: value.instance_id,
        )
    )
    policy = _policy()
    return CanaryRegressionReportV2.create(
        cohort_manifest_ref=_cohort_ref(),
        policy_ref=policy.to_ref(),
        template_ref=_ref("canary-regression-template", "file-read", version="private-v1"),
        case_results=cases,
        audit=_audit(),
    )


def test_report_contracts_are_strict_frozen_and_content_addressed() -> None:
    report = _report()

    assert report.outcome is CanaryRegressionOutcomeV2.PASSED
    assert len(report.case_results) == 24
    assert len(report.succeeded_ids) == 15
    assert len(report.blocked_ids) == 9
    assert report.claim_scope == "CANARY_REGRESSION_ONLY"
    assert canary_regression_report_v2_ref(report).object_sha256 == report.report_sha256
    with pytest.raises(ValidationError):
        report.report_sha256 = "f" * 64
    with pytest.raises(ValidationError):
        CanaryRegressionReportV2.model_validate(
            {
                **report.model_dump(mode="python"),
                "raw_trace": "forbidden",
            }
        )


def test_case_state_matrix_rejects_partial_success_and_reason_drift() -> None:
    success = _success(1)
    with pytest.raises(ValidationError, match="successful case"):
        CanaryRegressionCaseResultV2.model_validate(
            success.model_copy(update={"package_manifest_refs": ()}).model_dump(mode="python")
        )
    blocked = _blocked(16)
    with pytest.raises(ValidationError, match="non-success case"):
        CanaryRegressionCaseResultV2.model_validate(
            blocked.model_copy(update={"reason_code": CanaryRegressionReasonCodeV2.NONE}).model_dump(
                mode="python"
            )
        )


def test_report_derives_failed_and_incomplete_outcomes() -> None:
    base = _report()
    failed_case = _recreate(
        _blocked(16),
        observed_outcome=CanaryRegressionObservedOutcomeV2.FAILED,
        reason_code=CanaryRegressionReasonCodeV2.STAGE_BLOCKED_POLICY,
        item_status=ItemStatus.FAILED,
    )
    failed = CanaryRegressionReportV2.create(
        cohort_manifest_ref=base.cohort_manifest_ref,
        policy_ref=base.policy_ref,
        template_ref=base.template_ref,
        case_results=tuple(
            failed_case if item.instance_id == failed_case.instance_id else item for item in base.case_results
        ),
        audit=_audit(),
    )
    assert failed.outcome is CanaryRegressionOutcomeV2.FAILED

    incomplete_case = _recreate(
        _blocked(17),
        observed_outcome=CanaryRegressionObservedOutcomeV2.INCOMPLETE,
        reason_code=CanaryRegressionReasonCodeV2.CHILD_STATE_INCOMPLETE,
        highest_reached_stage=None,
        job_status=None,
        item_id=None,
        item_status=None,
        stage_result_refs=(),
    )
    incomplete = CanaryRegressionReportV2.create(
        cohort_manifest_ref=base.cohort_manifest_ref,
        policy_ref=base.policy_ref,
        template_ref=base.template_ref,
        case_results=tuple(
            incomplete_case if item.instance_id == incomplete_case.instance_id else item
            for item in base.case_results
        ),
        audit=_audit(),
    )
    assert incomplete.outcome is CanaryRegressionOutcomeV2.INCOMPLETE


def test_report_identity_ignores_audit_actor_and_time() -> None:
    first = _report()
    later = CanaryRegressionReportV2.create(
        cohort_manifest_ref=first.cohort_manifest_ref,
        policy_ref=first.policy_ref,
        template_ref=first.template_ref,
        case_results=first.case_results,
        audit=_audit(created_at=datetime(2026, 8, 3, tzinfo=UTC)),
    )

    assert first.to_ref() == later.to_ref()


def test_public_report_contains_no_private_surface() -> None:
    serialized = _report().model_dump_json().casefold()
    for forbidden in (
        "payload_json",
        "physical_path",
        "raw_trace",
        "task_prompt",
        "private_reference",
        "grader_rule",
        "hidden_condition",
        "credential",
        "provider_payload",
        "exception_text",
        "final_output",
    ):
        assert forbidden not in serialized
