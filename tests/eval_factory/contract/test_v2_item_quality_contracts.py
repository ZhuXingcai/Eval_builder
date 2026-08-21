from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.quality_v2 import (
    ItemQualityFailureCodeV2,
    ItemQualityOutcomeV2,
    QualityReportV2,
    quality_report_v2_ref,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef, actor: str = "item-quality-contract-test") -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        created_by=actor,
        governing_versions=(VersionBinding(component="item-quality", version="r5-10"),),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _blocked_report(*, audit: ContractAudit | None = None) -> QualityReportV2:
    task_ref = _ref("r4-task-contract-set", "current")
    policy_ref = _ref("semantic-review-policy", "current")
    workflow_ref = _ref("semantic-review-workflow-result", "current")
    revision_ref = _ref("attachment-candidate-revision", "current")
    validation_ref = _ref("revision-deterministic-validation", "current")
    source_ref = _ref("deterministic-item-validation-result", "current")
    round_ref = _ref("semantic-review-round-result", "coverage")
    stage_result_ref = _ref("stage-result", "coverage", version="record/v1")
    return QualityReportV2.create(
        r4_task_contract_set_ref=task_ref,
        review_policy_ref=policy_ref,
        semantic_workflow_result_ref=workflow_ref,
        candidate_revision_ref=revision_ref,
        deterministic_validation_ref=validation_ref,
        source_deterministic_validation_ref=source_ref,
        candidate_inventory_ref=None,
        artifact_version_refs=(),
        output_refs=(),
        artifact_validation_result_refs=(),
        semantic_round_result_refs=(round_ref,),
        stage_result_refs=(stage_result_ref,),
        repair_plan_refs=(),
        repair_result_refs=(),
        current_finding_refs=(),
        stale_finding_refs=(),
        resolution_refs=(),
        accepted_artifact_refs=(),
        final_package_manifest_ref=None,
        provenance_manifest_ref=None,
        environment_spec_ref=None,
        package_sha256=None,
        input_state_only=None,
        outcome=ItemQualityOutcomeV2.BLOCKED,
        failure_code=None,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        approvable=False,
        audit=audit
        or _audit(
            task_ref,
            policy_ref,
            workflow_ref,
            revision_ref,
            validation_ref,
            source_ref,
            round_ref,
            stage_result_ref,
        ),
    )


def test_quality_report_rejects_approvable_without_final_package_truth() -> None:
    report = _blocked_report()
    values = report.model_dump(mode="python")
    values["approvable"] = True

    with pytest.raises(ValidationError, match="approvable"):
        QualityReportV2.model_validate(values)


def test_quality_report_rejects_passed_without_approvable_package() -> None:
    report = _blocked_report()
    values = report.model_dump(mode="python")
    values["outcome"] = ItemQualityOutcomeV2.PASSED

    with pytest.raises(
        ValidationError,
        match="PASSED quality report must be approvable",
    ):
        QualityReportV2.model_validate(values)


def test_quality_report_rejects_wrong_inventory_ref_type() -> None:
    report = _blocked_report()
    values = report.model_dump(mode="python")
    values["current_finding_refs"] = (_ref("attachment-output", "not-a-finding"),)

    with pytest.raises(
        ValidationError,
        match="current_finding_refs must reference one of",
    ):
        QualityReportV2.model_validate(values)


def test_quality_report_identity_is_audit_independent_and_strict() -> None:
    first = _blocked_report()
    second = _blocked_report(
        audit=_audit(
            *first.audit.input_refs,
            actor="another-item-quality-compiler",
        ).model_copy(
            update={
                "created_at": datetime(2026, 7, 30, tzinfo=UTC),
            }
        )
    )

    assert quality_report_v2_ref(first) == quality_report_v2_ref(second)
    assert first.canonical_sha256() != second.canonical_sha256()

    values = first.model_dump(mode="python")
    values["reviewer_reasoning"] = "forbidden"
    with pytest.raises(ValidationError, match="extra"):
        QualityReportV2.model_validate(values)


def test_item_quality_closed_enums_cover_package_integrity_failure() -> None:
    assert tuple(ItemQualityOutcomeV2) == (
        ItemQualityOutcomeV2.PASSED,
        ItemQualityOutcomeV2.REQUIRES_REPAIR,
        ItemQualityOutcomeV2.REJECTED,
        ItemQualityOutcomeV2.BLOCKED,
        ItemQualityOutcomeV2.UPSTREAM_INCOMPLETE,
    )
    assert ItemQualityFailureCodeV2.PROVENANCE_UNSAFE.value == "PROVENANCE_UNSAFE"
