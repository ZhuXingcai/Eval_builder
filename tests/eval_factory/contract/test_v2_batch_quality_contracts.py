from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.batch_quality_v2 import (
    BATCH_QUALITY_ITEM_LINEAGE_CHECKS,
    BatchFindingCategoryV2,
    BatchFindingScopeV2,
    BatchQualityPolicyV2,
    BatchQualityReportV2,
    BatchReviewerFindingV2,
    ItemLineageAuditOutcomeV2,
    ItemLineageAuditResultV2,
    ItemQualityReportRevisionV2,
    LineageAuditCheckCodeV2,
    LineageAuditCheckOutcomeV2,
    LineageAuditCheckV2,
    batch_quality_policy_v2_ref,
    batch_quality_report_v2_ref,
    batch_reviewer_finding_v2_ref,
    item_quality_report_revision_v2_ref,
    validate_batch_quality_report_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.task_v2 import PromptLeakageCategoryV2

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _digest(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-03/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(
    *,
    created_at: datetime = NOW,
    created_by: str = "batch-quality-contract-test",
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=created_by,
        governing_versions=(
            VersionBinding(
                component="batch-quality",
                version="r7-03-v1",
            ),
        ),
        input_refs=(),
    )


def _policy(
    *,
    audit: ContractAudit | None = None,
) -> BatchQualityPolicyV2:
    return BatchQualityPolicyV2.create(
        max_items=100,
        max_label_decisions=10_000,
        max_package_members=100_000,
        max_lineage_checks=10_000,
        max_direct_evidence_refs=100_000,
        max_findings=100_000,
        max_finding_subject_refs=1_000_000,
        max_report_revisions=100,
        max_blocked_items=100,
        audit=audit or _audit(),
    )


def _check(
    *,
    item_id: str,
    code: LineageAuditCheckCodeV2,
    outcome: LineageAuditCheckOutcomeV2 = LineageAuditCheckOutcomeV2.PASSED,
) -> LineageAuditCheckV2:
    counts = {
        LineageAuditCheckOutcomeV2.PASSED: (1, 0, 0),
        LineageAuditCheckOutcomeV2.FAILED: (1, 1, 0),
        LineageAuditCheckOutcomeV2.SKIPPED: (0, 0, 0),
        LineageAuditCheckOutcomeV2.INDETERMINATE: (1, 0, 1),
    }[outcome]
    return LineageAuditCheckV2.create(
        code=code,
        outcome=outcome,
        item_ids=(item_id,),
        subject_refs=(_ref("lineage-subject", f"{item_id}/{code.value}"),),
        evidence_refs=(
            ()
            if outcome is LineageAuditCheckOutcomeV2.SKIPPED
            else (_ref("lineage-evidence", f"{item_id}/{code.value}"),)
        ),
        evaluated_count=counts[0],
        failed_count=counts[1],
        indeterminate_count=counts[2],
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )


def _lineage(
    item_id: str,
    *,
    failed_code: LineageAuditCheckCodeV2 | None = None,
) -> ItemLineageAuditResultV2:
    checks = tuple(
        _check(
            item_id=item_id,
            code=code,
            outcome=(
                LineageAuditCheckOutcomeV2.FAILED
                if code is failed_code
                else LineageAuditCheckOutcomeV2.PASSED
            ),
        )
        for code in BATCH_QUALITY_ITEM_LINEAGE_CHECKS
    )
    return ItemLineageAuditResultV2.create(
        item_id=item_id,
        source_trace_ref=_ref("trace-source", item_id, version="raw_traj_v1"),
        trace_envelope_ref=_ref(
            "trace-envelope",
            item_id,
            version="stored-manifest/v1",
        ),
        label_decision_refs=(
            _ref(
                "label-decision",
                item_id,
                version="label-decision-merge/r3-04-v1",
            ),
        ),
        selection_context_ref=_ref("selection-context", item_id),
        task_draft_ref=_ref("task-draft", item_id),
        r4_task_contract_set_ref=_ref("r4-task-contract-set", item_id),
        item_quality_result_ref=_ref("item-quality-compilation-result", item_id),
        base_quality_report_ref=_ref("quality-report", item_id),
        final_package_manifest_ref=_ref("final-package-manifest", item_id),
        provenance_manifest_ref=_ref("provenance-manifest", item_id),
        environment_spec_ref=_ref("environment-spec", item_id),
        checks=checks,
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )


def _finding(
    category: BatchFindingCategoryV2,
    *,
    evidence_type: str,
    affected_item_ids: tuple[str, ...],
    owner_item_id: str | None = None,
    restricted_category: PromptLeakageCategoryV2 | None = None,
    direct_evidence_ref: ObjectRef | None = None,
) -> BatchReviewerFindingV2:
    return BatchReviewerFindingV2.create(
        category=category,
        owner_item_id=owner_item_id,
        affected_item_ids=affected_item_ids,
        subject_refs=tuple(_ref("finding-subject", item_id) for item_id in affected_item_ids),
        direct_evidence_ref=direct_evidence_ref or _ref(evidence_type, category.value.casefold()),
        restricted_category=restricted_category,
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )


def _revision(
    *,
    item_id: str,
    findings: tuple[BatchReviewerFindingV2, ...],
) -> ItemQualityReportRevisionV2:
    return ItemQualityReportRevisionV2.create(
        item_id=item_id,
        base_item_quality_result_ref=_ref(
            "item-quality-compilation-result",
            item_id,
        ),
        base_quality_report_ref=_ref("quality-report", item_id),
        final_package_manifest_ref=_ref("final-package-manifest", item_id),
        provenance_manifest_ref=_ref("provenance-manifest", item_id),
        environment_spec_ref=_ref("environment-spec", item_id),
        accepted_artifact_refs=(_ref("candidate-artifact-version", f"{item_id}/artifact"),),
        package_sha256=_digest(f"package:{item_id}"),
        findings=findings,
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )


def _report(
    *,
    findings: tuple[BatchReviewerFindingV2, ...] = (),
    revisions: tuple[ItemQualityReportRevisionV2, ...] = (),
    lineages: tuple[ItemLineageAuditResultV2, ...] | None = None,
    invalidated_result_refs: tuple[ObjectRef, ...] = (),
) -> BatchQualityReportV2:
    item_ids = ("item://a", "item://b")
    active_lineages = lineages or tuple(_lineage(item_id) for item_id in item_ids)
    base_quality_refs = tuple(_ref("quality-report", item_id) for item_id in item_ids)
    revision_by_item = {value.item_id: value for value in revisions}
    current_quality_refs = tuple(
        (
            item_quality_report_revision_v2_ref(revision_by_item[item_id])
            if item_id in revision_by_item
            else base_ref
        )
        for item_id, base_ref in zip(item_ids, base_quality_refs, strict=True)
    )
    duplicate_pair_refs = tuple(
        finding.direct_evidence_ref
        for finding in findings
        if finding.category
        in {
            BatchFindingCategoryV2.EXACT_TASK_DUPLICATE,
            BatchFindingCategoryV2.NEAR_TASK_DUPLICATE,
            BatchFindingCategoryV2.EXACT_ATTACHMENT_DUPLICATE,
            BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE,
        }
    )
    visible_match_refs = tuple(
        finding.direct_evidence_ref
        for finding in findings
        if finding.category
        in {
            BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION,
            BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE,
        }
    )
    answer_reuse_refs = tuple(
        finding.direct_evidence_ref
        for finding in findings
        if finding.category is BatchFindingCategoryV2.ANSWER_REUSE
    )
    batch_lineage_check = LineageAuditCheckV2.create(
        code=LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS,
        outcome=LineageAuditCheckOutcomeV2.PASSED,
        item_ids=item_ids,
        subject_refs=(_ref("resolved-job-work-graph", "batch"),),
        evidence_refs=tuple(_ref("item-quality-compilation-result", item_id) for item_id in item_ids),
        evaluated_count=1,
        failed_count=0,
        indeterminate_count=0,
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )
    return BatchQualityReportV2.create(
        batch_id="dataset-job://r7-03",
        resolved_job_work_graph_ref=_ref("resolved-job-work-graph", "batch"),
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        duplicate_policy_ref=_ref("duplicate-detection-policy", "batch"),
        duplicate_result_ref=_ref("duplicate-detection-result", "batch"),
        cross_item_policy_ref=_ref("cross-item-safety-policy", "batch"),
        cross_item_result_ref=_ref("cross-item-safety-result", "batch"),
        item_ids=item_ids,
        item_quality_result_refs=tuple(
            _ref("item-quality-compilation-result", item_id) for item_id in item_ids
        ),
        base_item_quality_report_refs=base_quality_refs,
        current_item_quality_report_refs=current_quality_refs,
        lineage_audits=active_lineages,
        batch_lineage_checks=(batch_lineage_check,),
        findings=findings,
        item_quality_report_revisions=revisions,
        duplicate_pair_refs=duplicate_pair_refs,
        duplicate_cluster_refs=(),
        visible_match_refs=visible_match_refs,
        answer_reuse_pair_refs=answer_reuse_refs,
        safety_cluster_refs=(),
        invalidated_result_refs=invalidated_result_refs,
        audit=_audit(),
    )


def test_policy_is_strict_current_and_audit_independent() -> None:
    first = _policy()
    second = _policy(
        audit=_audit(
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
            created_by="different-actor",
        )
    )

    assert first.policy_sha256 == second.policy_sha256
    assert first.batch_quality_policy_id == second.batch_quality_policy_id
    assert first.exact_task_duplicate_severity is Severity.P1
    assert first.other_duplicate_severity is Severity.P2
    assert first.answer_bearing_safety_severity is Severity.P0
    assert first.other_safety_severity is Severity.P1
    assert first.required_item_lineage_checks == BATCH_QUALITY_ITEM_LINEAGE_CHECKS
    assert first.required_batch_lineage_checks == (LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS,)
    with pytest.raises(ValidationError):
        BatchQualityPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "unexpected": True,
            }
        )


@pytest.mark.parametrize(
    ("outcome", "evaluated", "failed", "indeterminate"),
    (
        (LineageAuditCheckOutcomeV2.PASSED, 1, 0, 0),
        (LineageAuditCheckOutcomeV2.FAILED, 1, 1, 0),
        (LineageAuditCheckOutcomeV2.SKIPPED, 0, 0, 0),
        (LineageAuditCheckOutcomeV2.INDETERMINATE, 1, 0, 1),
    ),
)
def test_lineage_check_status_matrix_is_closed(
    outcome: LineageAuditCheckOutcomeV2,
    evaluated: int,
    failed: int,
    indeterminate: int,
) -> None:
    check = LineageAuditCheckV2.create(
        code=LineageAuditCheckCodeV2.SOURCE_TRACE_LABELS,
        outcome=outcome,
        item_ids=("item://a",),
        subject_refs=(_ref("trace-source", "a", version="raw_traj_v1"),),
        evidence_refs=(
            () if outcome is LineageAuditCheckOutcomeV2.SKIPPED else (_ref("label-decision", "a"),)
        ),
        evaluated_count=evaluated,
        failed_count=failed,
        indeterminate_count=indeterminate,
        policy_ref=batch_quality_policy_v2_ref(_policy()),
        audit=_audit(),
    )
    assert check.outcome is outcome

    with pytest.raises(ValidationError):
        LineageAuditCheckV2.model_validate(
            {
                **check.model_dump(mode="python"),
                "failed_count": failed + 1,
            }
        )


def test_item_lineage_requires_exact_check_inventory_and_counts() -> None:
    passed = _lineage("item://a")
    failed = _lineage(
        "item://a",
        failed_code=LineageAuditCheckCodeV2.PACKAGE_MEMBER_DERIVATION,
    )

    assert passed.outcome is ItemLineageAuditOutcomeV2.PASSED
    assert passed.passed_check_count == len(BATCH_QUALITY_ITEM_LINEAGE_CHECKS)
    assert failed.outcome is ItemLineageAuditOutcomeV2.FAILED
    assert failed.failed_check_count == 1
    with pytest.raises(ValidationError):
        ItemLineageAuditResultV2.model_validate(
            {
                **passed.model_dump(mode="python"),
                "checks": passed.checks[:-1],
            }
        )


def test_finding_policy_derives_scope_severity_and_blocked_items() -> None:
    exact_task = _finding(
        BatchFindingCategoryV2.EXACT_TASK_DUPLICATE,
        evidence_type="duplicate-pair-evidence",
        affected_item_ids=("item://a", "item://b"),
    )
    near_attachment = _finding(
        BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE,
        evidence_type="duplicate-pair-evidence",
        affected_item_ids=("item://a", "item://b"),
    )
    private_leak = _finding(
        BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE,
        evidence_type="cross-item-visible-match-evidence",
        affected_item_ids=("item://b",),
        owner_item_id="item://b",
        restricted_category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
    )
    hidden_leak = _finding(
        BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE,
        evidence_type="cross-item-visible-match-evidence",
        affected_item_ids=("item://b",),
        owner_item_id="item://b",
        restricted_category=PromptLeakageCategoryV2.HIDDEN_PASS_CONDITION,
    )

    assert exact_task.scope is BatchFindingScopeV2.BATCH
    assert exact_task.severity is Severity.P1
    assert exact_task.non_waivable is False
    assert exact_task.release_blocking is True
    assert exact_task.blocked_item_ids == exact_task.affected_item_ids
    assert near_attachment.severity is Severity.P2
    assert near_attachment.blocked_item_ids == ()
    assert near_attachment.release_blocking is False
    assert private_leak.scope is BatchFindingScopeV2.ITEM
    assert private_leak.owner_item_id == "item://b"
    assert private_leak.severity is Severity.P0
    assert private_leak.non_waivable is True
    assert hidden_leak.severity is Severity.P1

    with pytest.raises(ValidationError):
        BatchReviewerFindingV2.model_validate(
            {
                **private_leak.model_dump(mode="python"),
                "severity": Severity.P2,
            }
        )


def test_item_report_revision_preserves_base_truth_and_derives_counts() -> None:
    finding = _finding(
        BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION,
        evidence_type="cross-item-visible-match-evidence",
        affected_item_ids=("item://b",),
        owner_item_id="item://b",
        restricted_category=PromptLeakageCategoryV2.FINAL_ANSWER,
    )
    revision = _revision(
        item_id="item://b",
        findings=(finding,),
    )

    assert revision.open_p0_count == 1
    assert revision.open_p1_count == 0
    assert revision.unresolved_non_waivable_count == 1
    assert revision.approvable is False
    assert revision.release_revalidation_required is True
    assert revision.finding_refs == (batch_reviewer_finding_v2_ref(finding),)
    with pytest.raises(ValidationError):
        ItemQualityReportRevisionV2.model_validate(
            {
                **revision.model_dump(mode="python"),
                "approvable": True,
            }
        )


def test_batch_report_owns_exact_inventories_and_allows_p2_only() -> None:
    clean = _report()
    p2 = _finding(
        BatchFindingCategoryV2.NEAR_TASK_DUPLICATE,
        evidence_type="duplicate-pair-evidence",
        affected_item_ids=("item://a", "item://b"),
    )
    p2_report = _report(findings=(p2,))
    contamination = _finding(
        BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION,
        evidence_type="cross-item-visible-match-evidence",
        affected_item_ids=("item://b",),
        owner_item_id="item://b",
        restricted_category=PromptLeakageCategoryV2.FINAL_ANSWER,
    )
    revision = _revision(
        item_id="item://b",
        findings=(contamination,),
    )
    blocked = _report(
        findings=(contamination,),
        revisions=(revision,),
    )

    assert clean.approvable is True
    assert clean.finding_count == 0
    assert p2_report.approvable is True
    assert p2_report.open_p2_count == 1
    assert p2_report.affected_item_ids == ("item://a", "item://b")
    assert p2_report.blocked_item_ids == ()
    assert blocked.approvable is False
    assert blocked.item_scoped_finding_refs == (batch_reviewer_finding_v2_ref(contamination),)
    assert blocked.batch_scoped_finding_refs == ()
    assert blocked.blocked_item_ids == ("item://b",)
    assert blocked.report_revision_count == 1
    validate_batch_quality_report_v2_identity(blocked)
    assert batch_quality_report_v2_ref(blocked).object_sha256 == (blocked.report_sha256)

    with pytest.raises(ValidationError):
        BatchQualityReportV2.model_validate(
            {
                **blocked.model_dump(mode="python"),
                "blocked_item_count": 0,
            }
        )


def test_batch_report_rejects_duplicate_direct_evidence_consumption() -> None:
    shared_evidence = _ref("duplicate-pair-evidence", "shared")
    exact = _finding(
        BatchFindingCategoryV2.EXACT_TASK_DUPLICATE,
        evidence_type="duplicate-pair-evidence",
        affected_item_ids=("item://a", "item://b"),
        direct_evidence_ref=shared_evidence,
    )
    near = _finding(
        BatchFindingCategoryV2.NEAR_TASK_DUPLICATE,
        evidence_type="duplicate-pair-evidence",
        affected_item_ids=("item://a", "item://b"),
        direct_evidence_ref=shared_evidence,
    )

    with pytest.raises(ValidationError, match="direct evidence refs must be unique"):
        _report(findings=(exact, near))


def test_invalidated_source_result_keeps_batch_non_approvable() -> None:
    report = _report(
        invalidated_result_refs=(_ref("duplicate-detection-result", "invalidated"),),
    )

    assert report.invalidated_result_refs
    assert report.approvable is False


def test_new_schemas_are_closed_and_content_free() -> None:
    def property_names(value: object) -> set[str]:
        if isinstance(value, dict):
            names = set(value.get("properties", {}))
            for child in value.values():
                names.update(property_names(child))
            return names
        if isinstance(value, list):
            names: set[str] = set()
            for child in value:
                names.update(property_names(child))
            return names
        return set()

    models = (
        BatchQualityPolicyV2,
        LineageAuditCheckV2,
        ItemLineageAuditResultV2,
        BatchReviewerFindingV2,
        ItemQualityReportRevisionV2,
        BatchQualityReportV2,
    )
    forbidden = {
        "description",
        "repair_action",
        "raw_trace",
        "visible_prompt",
        "source_text",
        "evidence_content",
        "private_reference",
        "grader_rule",
        "physical_path",
        "provider_payload",
        "credential",
        "reviewer_id",
        "user_decision",
    }
    for model in models:
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert not forbidden.intersection(property_names(schema))
