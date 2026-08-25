from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    PLAN_CHECKPOINTS,
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentScopeDecision,
    EnvironmentStrategyChoice,
    FinalReviewScope,
    InvalidationScope,
    QueryPackagingChoice,
    TypedAdjustment,
    UserApprovalPolicy,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2, StageNameV2
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel
from eval_factory.contracts.release_v2 import (
    CheckpointDecisionBinding,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)

ROOT = Path(__file__).resolve().parents[3]
HASH = "a" * 64


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://example/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        created_by="contract-test",
        governing_versions=(VersionBinding(component="spec", version="approved-v2", sha256=HASH),),
    )


def _policy(
    mode: ApprovalMode,
    checkpoints: frozenset[ApprovalCheckpoint],
    scope: FinalReviewScope = FinalReviewScope.NONE,
    *,
    explicit: bool = True,
) -> UserApprovalPolicy:
    return UserApprovalPolicy(
        policy_id=f"approval-policy://{mode.value.casefold()}/v2",
        policy_version="v2",
        mode=mode,
        enabled_checkpoints=checkpoints,
        final_review_scope=scope,
        explicit_unattended_choice=explicit,
        audit=_audit(),
    )


def _job_base() -> dict[str, Any]:
    return {
        "job_id": "job://example/v2",
        "traces": (
            TraceSourceRef(
                source_trace_id="source-trace://example",
                source_uri="raw-traj://example",
                raw_sha256=HASH,
                adapter_name="raw-traj-v1",
                adapter_version="v1",
                processing_class="RESTRICTED_TRACE_RAW",
            ),
        ),
        "privacy_profile": "trusted-monitored-local",
        "model_profiles": (),
        "budget": ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=1,
            max_renderers=1,
            max_network_requests=0,
            max_storage_bytes=1024,
        ),
        "concurrency": ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        "selection_spec_ref": None,
        "approval_policy_ref": _ref("user-approval-policy"),
        "export_target": ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        "idempotency_key": "job-example-v2",
        "audit": _audit(),
    }


def _components() -> EvaluationItemComponents:
    return EvaluationItemComponents(
        query_spec_ref=_ref("query-spec"),
        environment_spec_ref=_ref("environment-spec"),
        rubric_set_ref=_ref("rubric-set"),
        evaluator_spec_ref=_ref("evaluator-spec"),
        reference_policy_ref=_ref("reference-policy"),
        tool_policy_ref=_ref("tool-policy"),
        provenance_manifest_ref=_ref("provenance-manifest"),
        quality_report_ref=_ref("quality-report"),
    )


def _release_base() -> dict[str, Any]:
    return {
        "release_decision_id": "release-decision://example/v2",
        "chain_id": "release-chain://example",
        "previous_decision_ref": None,
        "item_id": "item://example",
        "item_version": "v2",
        "components": _components(),
        "release_subject_sha256": HASH,
        "package_manifest_ref": _ref("package-manifest"),
        "package_sha256": HASH,
        "batch_quality_report_ref": None,
        "user_approval_policy_ref": _ref("user-approval-policy"),
        "channel": ReleaseChannel.CANARY,
        "registry": "registry://canary",
        "export_profile": "LH",
        "export_profile_version": "v1",
        "production_attestation_ref": None,
        "idempotency_key": "release-example-v2",
        "actor": "release-authority",
        "decided_at": datetime(2026, 7, 20, tzinfo=UTC),
        "decision_sha256": HASH,
        "audit": _audit(),
    }


def _checkpoint_binding(checkpoint: ApprovalCheckpoint) -> CheckpointDecisionBinding:
    slug = checkpoint.value.casefold().replace("_", "-")
    return CheckpointDecisionBinding(
        checkpoint=checkpoint,
        request_ref=_ref(f"user-approval-request-{slug}"),
        decision_record_ref=_ref(f"user-decision-record-{slug}"),
    )


def test_named_approval_modes_have_exact_checkpoint_sets() -> None:
    _policy(ApprovalMode.NONE, frozenset())
    _policy(ApprovalMode.PLAN_GATES, PLAN_CHECKPOINTS)
    _policy(
        ApprovalMode.PLAN_AND_FINAL,
        ALL_CHECKPOINTS,
        FinalReviewScope.FULL_DATASET,
    )

    with pytest.raises(ValidationError, match="canonical checkpoint set"):
        _policy(ApprovalMode.PLAN_GATES, frozenset({ApprovalCheckpoint.LABEL_PLAN}))
    with pytest.raises(ValidationError, match="selected explicitly"):
        _policy(ApprovalMode.NONE, frozenset(), explicit=False)
    with pytest.raises(ValidationError, match="must match"):
        _policy(
            ApprovalMode.FINAL_ONLY,
            frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            FinalReviewScope.NONE,
        )


def test_disabled_checkpoints_cannot_appear_in_v2_stage_plan() -> None:
    base = _job_base()
    valid = DatasetJobSpecV2(
        **base,
        requested_stages=(
            StageNameV2.TRACE_INDEX,
            StageNameV2.SAFETY,
            StageNameV2.LABEL,
            StageNameV2.TASK_AUTHORING,
            StageNameV2.ATTACHMENT,
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
            StageNameV2.RELEASE,
        ),
        approval_mode=ApprovalMode.NONE,
        enabled_checkpoints=frozenset(),
    )
    assert all("review" not in stage.value for stage in valid.requested_stages)

    with pytest.raises(ValidationError, match="presence must match"):
        DatasetJobSpecV2(
            **base,
            requested_stages=(
                StageNameV2.TRACE_INDEX,
                StageNameV2.LABEL_PLAN,
                StageNameV2.LABEL,
            ),
            approval_mode=ApprovalMode.NONE,
            enabled_checkpoints=frozenset(),
        )


def test_adjustment_requires_new_object_and_directed_invalidation() -> None:
    common: dict[str, Any] = {
        "decision_record_id": "user-decision://example/v2",
        "request_ref": _ref("user-approval-request"),
        "checkpoint": ApprovalCheckpoint.TASK_REWRITE_PLAN,
        "approval_policy_ref": _ref("user-approval-policy"),
        "subject_refs": (_ref("selection-context"),),
        "plan_ref": _ref("task-rewrite-plan"),
        "projection_ref": _ref("task-rewrite-preview"),
        "authenticated_user": "requesting-user",
        "decision": UserDecision.ADJUST,
        "adjustments": (
            TypedAdjustment(
                target_path="rewrite_style",
                operation="SET",
                value="concise-realistic",
                reason="Match the approved authoring style.",
            ),
        ),
        "reason": "Adjust style before batch authoring.",
        "hard_gate_override_requested": False,
        "idempotency_key": "decision-example-v2",
        "decided_at": datetime(2026, 7, 20, tzinfo=UTC),
        "record_sha256": HASH,
    }
    with pytest.raises(ValidationError, match="resulting object"):
        UserDecisionRecord(
            **common,
            resulting_object_ref=None,
            invalidation_scope=InvalidationScope(stages=("task-authoring",)),
        )
    with pytest.raises(ValidationError, match="invalidation scope"):
        UserDecisionRecord(
            **common,
            resulting_object_ref=_ref("task-rewrite-plan-v3"),
            invalidation_scope=None,
        )


def test_environment_acceptance_requires_strategy_and_query_packaging() -> None:
    common: dict[str, Any] = {
        "decision_record_id": "user-decision://environment/v2",
        "request_ref": _ref("user-approval-request"),
        "checkpoint": ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        "approval_policy_ref": _ref("user-approval-policy"),
        "subject_refs": (_ref("task-draft"),),
        "plan_ref": _ref("environment-strategy"),
        "projection_ref": _ref("environment-preview"),
        "authenticated_user": "requesting-user",
        "decision": UserDecision.ACCEPT,
        "reason": "Use a standard environment.",
        "idempotency_key": "environment-decision-v2",
        "decided_at": datetime(2026, 7, 20, tzinfo=UTC),
        "record_sha256": HASH,
    }
    with pytest.raises(ValidationError, match="requires strategy and query packaging"):
        UserDecisionRecord(**common)

    decision = UserDecisionRecord(
        **common,
        environment_decisions=(
            EnvironmentScopeDecision(
                requirement_id="environment-requirement://example",
                strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
            ),
        ),
        query_packaging=QueryPackagingChoice.INCLUDE_QUERY_YAML,
    )
    assert decision.hard_gate_override_requested is False

    with pytest.raises(ValidationError):
        UserDecisionRecord(
            **common,
            environment_decisions=(
                EnvironmentScopeDecision(
                    requirement_id="environment-requirement://example",
                    strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
                ),
            ),
            query_packaging=QueryPackagingChoice.INCLUDE_QUERY_YAML,
            hard_gate_override_requested=True,
        )


def test_release_has_no_synthetic_decision_and_cannot_override_hard_gates() -> None:
    base = _release_base()
    candidate = ReleaseDecisionV2(
        **base,
        automated_quality_passed=False,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        required_checkpoints=PLAN_CHECKPOINTS,
        checkpoint_decisions=(),
        action=ReleaseActionV2.REQUEST_RELEASE,
        state=ReleaseStateV2.CANDIDATE,
    )
    assert candidate.checkpoint_decisions == ()

    with pytest.raises(ValidationError, match="synthetic decisions"):
        ReleaseDecisionV2(
            **base,
            automated_quality_passed=True,
            open_p0_count=0,
            open_p1_count=0,
            unresolved_non_waivable_count=0,
            required_checkpoints=frozenset(),
            checkpoint_decisions=(_checkpoint_binding(ApprovalCheckpoint.FINAL_DATASET_REVIEW),),
            action=ReleaseActionV2.APPROVE,
            state=ReleaseStateV2.APPROVED,
        )

    with pytest.raises(ValidationError, match="automated hard gates"):
        ReleaseDecisionV2(
            **base,
            automated_quality_passed=True,
            open_p0_count=0,
            open_p1_count=0,
            unresolved_non_waivable_count=1,
            required_checkpoints=frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            checkpoint_decisions=(_checkpoint_binding(ApprovalCheckpoint.FINAL_DATASET_REVIEW),),
            action=ReleaseActionV2.APPROVE,
            state=ReleaseStateV2.APPROVED,
        )

    with pytest.raises(ValidationError, match="at most one"):
        ReleaseDecisionV2(
            **base,
            automated_quality_passed=True,
            open_p0_count=0,
            open_p1_count=0,
            unresolved_non_waivable_count=0,
            required_checkpoints=frozenset({ApprovalCheckpoint.FINAL_DATASET_REVIEW}),
            checkpoint_decisions=(
                _checkpoint_binding(ApprovalCheckpoint.FINAL_DATASET_REVIEW),
                _checkpoint_binding(ApprovalCheckpoint.FINAL_DATASET_REVIEW),
            ),
            action=ReleaseActionV2.APPROVE,
            state=ReleaseStateV2.APPROVED,
        )


def test_v2_overlay_manifest_is_current_and_closed() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/export_eval_factory_contracts_v2.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    manifest_path = ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["architecture"] == "v1-base-plus-v2-overlay"
    assert len(manifest["contracts"]) == 415
    assert sum(item["owner"] == "facade" for item in manifest["contracts"]) == 59
    assert sum(item["owner"] == "factory" for item in manifest["contracts"]) == 356
    contracts = {item["python_type"]: item for item in manifest["contracts"]}
    expected_external_evidence_types = {
        "ExternalEvidenceAdmissionReportV2",
        "ExternalEvidencePackageManifestV2",
        "ExternalLabelObservationSummaryV2",
        "ExternalSc010Sc011EvidenceIndexV2",
    }
    assert {
        python_type.rsplit(".", 1)[-1]
        for python_type in contracts
        if python_type.startswith("eval_factory.contracts.external_evidence_v2.")
    } == expected_external_evidence_types
    expected_external_stability_types = {
        "ExternalRealTraceStabilityPolicyV2",
        "ExternalRealTraceStabilityReportV2",
    }
    assert {
        python_type.rsplit(".", 1)[-1]
        for python_type in contracts
        if python_type.startswith("eval_factory.contracts.external_stability_v2.")
    } == expected_external_stability_types
    expected_ai_gateway_types = {
        "GatewayInvocationRequestV2",
        "GatewayInvocationResultV2",
        "GatewayReceiptV2",
        "GatewayUsageV2",
        "ModelCapabilityProfileV2",
        "ModelHealthSnapshotV2",
        "ModelPriceScheduleV2",
        "ModelQualityBaselineV2",
        "ModelRouteCandidateV2",
        "ModelRouteDecisionV2",
        "ModelRouteRequestV2",
        "ModelRoutingPolicyV2",
        "PromptTemplateV2",
        "RAGRequestV2",
        "RAGResultV2",
        "RAGSourcePolicyV2",
    }
    assert {
        python_type.rsplit(".", 1)[-1]
        for python_type in contracts
        if python_type.startswith("eval_factory.contracts.ai_gateway_v2.")
    } == expected_ai_gateway_types
    expected_agent_system_types = {
        "AgentCapabilityV2",
        "AgentDefinitionV2",
        "AgentResultEnvelopeV2",
        "AgentTaskV2",
        "AgentWorkspaceReceiptV2",
        "AttachmentGenerationPlanV2",
        "AttachmentGroupResultV2",
        "AttachmentMockWorkV2",
        "AttachmentQualityAssessmentV2",
        "AttachmentSubgraphResultV2",
        "AttachmentWorkAssignmentV2",
        "CompiledAttachmentGenerationPlanV2",
        "CompiledCriteriaRubricPlanV2",
        "CompiledDatasetBuildPlanV2",
        "CompiledGradingDesignPlanV2",
        "CoreVerticalResultV2",
        "CriteriaRubricGoalV2",
        "CriteriaRubricPlanV2",
        "CriteriaRubricResultV2",
        "DatasetBuildPlanTaskV2",
        "DatasetBuildPlanV2",
        "DatasetDeliveryManifestV2",
        "EvaluationRequirementSpecV2",
        "ExtractedUserPromptV2",
        "FactoryRunCompletionV2",
        "FactoryRunPolicyV2",
        "FactoryRunV2",
        "GradingDesignPlanV2",
        "GradingDesignResultV2",
        "InferredUserIntentV2",
        "IntentClaimV2",
        "JudgeDesignSpecV2",
        "JudgeDesignValidationV2",
        "JudgeTaskMappingV2",
        "PlanDecisionV2",
        "PlanReviewPresentationV2",
        "PlanReviewRequestV2",
        "PlanReviewResultV2",
        "PlanRevisionV2",
        "PlannerAssessmentV2",
        "SolvabilityAssessmentV2",
        "TaskRewriteCandidateV2",
        "TaskRewritePlanV2",
        "TraceCandidateDecisionV2",
    }
    assert {
        python_type.rsplit(".", 1)[-1]
        for python_type in contracts
        if python_type.startswith("eval_factory.contracts.agent_system_v2.")
    } == expected_agent_system_types
    expected_attestation_schemas = {
        "ProductionReadinessAttestationPolicyV2": ("production_readiness_attestation_policy_v2.schema.json"),
        "ProductionReadinessAttestationPrerequisiteV2": (
            "production_readiness_attestation_prerequisite_v2.schema.json"
        ),
        "ProductionReadinessAttestationProjectionV2": (
            "production_readiness_attestation_projection_v2.schema.json"
        ),
        "ProductionReadinessAttestationResultV2": ("production_readiness_attestation_result_v2.schema.json"),
        "ProductionReadinessGateAssessmentV2": ("production_readiness_gate_assessment_v2.schema.json"),
        "ProductionReadinessInvalidationPolicyV2": (
            "production_readiness_invalidation_policy_v2.schema.json"
        ),
        "ProductionReadinessInvalidationRecordV2": (
            "production_readiness_invalidation_record_v2.schema.json"
        ),
        "ProductionReadinessVersionSetV2": ("production_readiness_version_set_v2.schema.json"),
    }
    for type_name, schema_name in expected_attestation_schemas.items():
        contract = contracts[f"eval_factory.contracts.production_attestation_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_readiness_review_schemas = {
        "OperationsSLOAssessmentV2": "operations_s_l_o_assessment_v2.schema.json",
        "OperationsSLOMetricV2": "operations_s_l_o_metric_v2.schema.json",
        "OperationsSLOPolicyV2": "operations_s_l_o_policy_v2.schema.json",
        "ProductionReadinessApprovalRecordV2": ("production_readiness_approval_record_v2.schema.json"),
        "ProductionReadinessDomainReviewV2": ("production_readiness_domain_review_v2.schema.json"),
        "ProductionReadinessPrerequisiteSummaryV2": (
            "production_readiness_prerequisite_summary_v2.schema.json"
        ),
        "ProductionReadinessReviewPolicyV2": ("production_readiness_review_policy_v2.schema.json"),
        "ProductionReadinessReviewReportV2": ("production_readiness_review_report_v2.schema.json"),
    }
    for type_name, schema_name in expected_readiness_review_schemas.items():
        contract = contracts[f"eval_factory.contracts.production_readiness_review_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_release_projection_schemas = {
        "EvaluationItemReleaseSubjectV2": ("evaluation_item_release_subject_v2.schema.json"),
        "ItemReleaseProjectionV2": ("item_release_projection_v2.schema.json"),
        "ReleaseProjectionPolicyV2": ("release_projection_policy_v2.schema.json"),
        "ReleaseProjectionResultV2": ("release_projection_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_release_projection_schemas.items():
        contract = contracts[f"eval_factory.contracts.release_projection_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_release_export_schemas = {
        "LHWorkspaceExportMemberV2": ("l_h_workspace_export_member_v2.schema.json"),
        "LHWorkspaceExportRequestV2": ("l_h_workspace_export_request_v2.schema.json"),
        "LHWorkspaceExportResultV2": ("l_h_workspace_export_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_release_export_schemas.items():
        contract = contracts[f"env_mock_agent.facade.release_export_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "facade"
    expected_release_publication_schemas = {
        "LHExportReceiptV2": "l_h_export_receipt_v2.schema.json",
        "LHReleaseItemManifestV2": ("l_h_release_item_manifest_v2.schema.json"),
        "NonProductionRegistryEntryV2": ("non_production_registry_entry_v2.schema.json"),
        "NonProductionReleaseManifestV2": ("non_production_release_manifest_v2.schema.json"),
        "PublishedItemProjectionV2": ("published_item_projection_v2.schema.json"),
        "ReleasePublicationPolicyV2": ("release_publication_policy_v2.schema.json"),
        "ReleasePublicationResultV2": ("release_publication_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_release_publication_schemas.items():
        contract = contracts[f"eval_factory.contracts.release_publication_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_checkpoint_interaction_schemas = {
        "UserCheckpointDecisionResultV2": ("user_checkpoint_decision_result_v2.schema.json"),
        "UserCheckpointDecisionSubmissionV2": ("user_checkpoint_decision_submission_v2.schema.json"),
        "UserCheckpointInteractionPageV2": ("user_checkpoint_interaction_page_v2.schema.json"),
        "UserCheckpointInteractionPolicyV2": ("user_checkpoint_interaction_policy_v2.schema.json"),
        "UserCheckpointInteractionV2": ("user_checkpoint_interaction_v2.schema.json"),
        "UserCheckpointOpenResultV2": ("user_checkpoint_open_result_v2.schema.json"),
        "UserCheckpointPresentationV2": ("user_checkpoint_presentation_v2.schema.json"),
        "UserCheckpointResumeResultV2": ("user_checkpoint_resume_result_v2.schema.json"),
        "UserCheckpointShowResultV2": ("user_checkpoint_show_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_checkpoint_interaction_schemas.items():
        contract = contracts[f"eval_factory.contracts.checkpoint_interaction_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_user_decision_schemas = {
        "UserApprovalRequestRevisionV2": "user_approval_request_revision_v2.schema.json",
        "UserDecisionAdjustmentEffectV2": "user_decision_adjustment_effect_v2.schema.json",
        "UserDecisionCommitResultV2": "user_decision_commit_result_v2.schema.json",
        "UserDecisionHandlingPolicyV2": "user_decision_handling_policy_v2.schema.json",
    }
    for type_name, schema_name in expected_user_decision_schemas.items():
        contract = contracts[f"eval_factory.contracts.approval_decision_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_user_approval_schemas = {
        "ApprovalRequestGenerationPolicyV2": ("approval_request_generation_policy_v2.schema.json"),
        "FinalDatasetReviewPreviewV2": ("final_dataset_review_preview_v2.schema.json"),
        "UserApprovalRequestCompilationResultV2": ("user_approval_request_compilation_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_user_approval_schemas.items():
        contract = contracts[f"eval_factory.contracts.approval_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_application_schemas = {
        "DirectedRevalidationPlanV2": "directed_revalidation_plan_v2.schema.json",
        "DirectedRevalidationReportV2": "directed_revalidation_report_v2.schema.json",
        "EnvironmentStrategyAdjustmentResultV2": ("environment_strategy_adjustment_result_v2.schema.json"),
        "FinalDatasetAdjustmentResultV2": ("final_dataset_adjustment_result_v2.schema.json"),
        "LabelPlanAdjustmentResultV2": ("label_plan_adjustment_result_v2.schema.json"),
        "RevalidationWorkItemV2": "revalidation_work_item_v2.schema.json",
        "RevalidationWorkResultV2": "revalidation_work_result_v2.schema.json",
        "UserPlanApplicationPolicyV2": ("user_plan_application_policy_v2.schema.json"),
        "UserPlanApplicationV2": "user_plan_application_v2.schema.json",
    }
    for type_name, schema_name in expected_application_schemas.items():
        contract = contracts[f"eval_factory.contracts.approval_application_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_batch_quality_schemas = {
        "BatchQualityPolicyV2": "batch_quality_policy_v2.schema.json",
        "BatchQualityReportV2": "batch_quality_report_v2.schema.json",
        "BatchReviewerFindingV2": "batch_reviewer_finding_v2.schema.json",
        "ItemLineageAuditResultV2": ("item_lineage_audit_result_v2.schema.json"),
        "ItemQualityReportRevisionV2": ("item_quality_report_revision_v2.schema.json"),
        "LineageAuditCheckV2": "lineage_audit_check_v2.schema.json",
    }
    for type_name, schema_name in expected_batch_quality_schemas.items():
        contract = contracts[f"eval_factory.contracts.batch_quality_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_cross_item_facade_schemas = {
        "AttachmentCrossItemSafetyScanLimitsV2": ("attachment_cross_item_safety_scan_limits_v2.schema.json"),
        "AttachmentCrossItemSafetyScanRequestV2": (
            "attachment_cross_item_safety_scan_request_v2.schema.json"
        ),
        "AttachmentCrossItemSafetyScanResultV2": ("attachment_cross_item_safety_scan_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_cross_item_facade_schemas.items():
        contract = contracts[f"env_mock_agent.facade.cross_item_safety_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "facade"
    expected_cross_item_factory_schemas = {
        "AnswerReusePairEvidenceV2": ("answer_reuse_pair_evidence_v2.schema.json"),
        "AttachmentCrossItemSafetyScanEvidenceV2": (
            "attachment_cross_item_safety_scan_evidence_v2.schema.json"
        ),
        "CrossItemSafetyClusterV2": "cross_item_safety_cluster_v2.schema.json",
        "CrossItemSafetyPolicyV2": "cross_item_safety_policy_v2.schema.json",
        "CrossItemSafetyResultV2": "cross_item_safety_result_v2.schema.json",
        "CrossItemVisibleMatchEvidenceV2": ("cross_item_visible_match_evidence_v2.schema.json"),
    }
    for type_name, schema_name in expected_cross_item_factory_schemas.items():
        contract = contracts[f"eval_factory.contracts.cross_item_safety_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_duplicate_facade_schemas = {
        "AttachmentDuplicateFingerprintLimitsV2": ("attachment_duplicate_fingerprint_limits_v2.schema.json"),
        "AttachmentDuplicateFingerprintRequestV2": (
            "attachment_duplicate_fingerprint_request_v2.schema.json"
        ),
        "AttachmentDuplicateFingerprintResultV2": ("attachment_duplicate_fingerprint_result_v2.schema.json"),
    }
    for type_name, schema_name in expected_duplicate_facade_schemas.items():
        contract = contracts[f"env_mock_agent.facade.duplicate_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "facade"
    expected_duplicate_factory_schemas = {
        "AttachmentDuplicateFingerprintV2": ("attachment_duplicate_fingerprint_v2.schema.json"),
        "DuplicateClusterV2": "duplicate_cluster_v2.schema.json",
        "DuplicateDetectionPolicyV2": ("duplicate_detection_policy_v2.schema.json"),
        "DuplicateDetectionResultV2": ("duplicate_detection_result_v2.schema.json"),
        "DuplicatePairEvidenceV2": ("duplicate_pair_evidence_v2.schema.json"),
        "TaskDuplicateFingerprintV2": ("task_duplicate_fingerprint_v2.schema.json"),
    }
    for type_name, schema_name in expected_duplicate_factory_schemas.items():
        contract = contracts[f"eval_factory.contracts.duplicate_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_canary_schemas = {
        "R6CanaryControlPlaneV2": "r6_canary_control_plane_v2.schema.json",
        "R6CanaryDatasetResultV2": "r6_canary_dataset_result_v2.schema.json",
        "R6CanaryExecutionManifestV2": "r6_canary_execution_manifest_v2.schema.json",
        "R6CanarySeedObjectV2": "r6_canary_seed_object_v2.schema.json",
        "R6CanaryTraceBindingV2": "r6_canary_trace_binding_v2.schema.json",
        "SemanticReviewFanoutV2": "semantic_review_fanout_v2.schema.json",
        "SemanticReviewRoundWorkV2": "semantic_review_round_work_v2.schema.json",
    }
    for type_name, schema_name in expected_canary_schemas.items():
        contract = contracts[f"eval_factory.contracts.canary_execution_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_canary_regression_schemas = {
        "CanaryRegressionCaseResultV2": "canary_regression_case_result_v2.schema.json",
        "CanaryRegressionPolicyV2": "canary_regression_policy_v2.schema.json",
        "CanaryRegressionReportV2": "canary_regression_report_v2.schema.json",
        "CanaryRegressionStageSummaryV2": "canary_regression_stage_summary_v2.schema.json",
    }
    for type_name, schema_name in expected_canary_regression_schemas.items():
        contract = contracts[f"eval_factory.contracts.canary_regression_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_real_trace_stability_schemas = {
        "RealTraceStabilityCaseSummaryV2": ("real_trace_stability_case_summary_v2.schema.json"),
        "RealTraceStabilityCategoryCountV2": ("real_trace_stability_category_count_v2.schema.json"),
        "RealTraceStabilityCorpusSummaryV2": ("real_trace_stability_corpus_summary_v2.schema.json"),
        "RealTraceStabilityPolicyV2": ("real_trace_stability_policy_v2.schema.json"),
        "RealTraceStabilityReportV2": ("real_trace_stability_report_v2.schema.json"),
    }
    for type_name, schema_name in expected_real_trace_stability_schemas.items():
        contract = contracts[f"eval_factory.contracts.real_trace_stability_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_scheduler_load_schemas = {
        "SchedulerLoadCaseSummaryV2": "scheduler_load_case_summary_v2.schema.json",
        "SchedulerLoadFaultCountV2": "scheduler_load_fault_count_v2.schema.json",
        "SchedulerLoadPhaseSummaryV2": "scheduler_load_phase_summary_v2.schema.json",
        "SchedulerLoadPolicyV2": "scheduler_load_policy_v2.schema.json",
        "SchedulerLoadReportV2": "scheduler_load_report_v2.schema.json",
    }
    for type_name, schema_name in expected_scheduler_load_schemas.items():
        contract = contracts[f"eval_factory.contracts.scheduler_load_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_concurrency_experiment_schemas = {
        "ConcurrencyExperimentHostSummaryV2": ("concurrency_experiment_host_summary_v2.schema.json"),
        "ConcurrencyExperimentLevelSummaryV2": ("concurrency_experiment_level_summary_v2.schema.json"),
        "ConcurrencyExperimentPercentileSummaryV2": (
            "concurrency_experiment_percentile_summary_v2.schema.json"
        ),
        "ConcurrencyExperimentPolicyV2": ("concurrency_experiment_policy_v2.schema.json"),
        "ConcurrencyExperimentRecoverySummaryV2": ("concurrency_experiment_recovery_summary_v2.schema.json"),
        "ConcurrencyExperimentReportV2": ("concurrency_experiment_report_v2.schema.json"),
        "NonModelResourceRecommendationV2": ("non_model_resource_recommendation_v2.schema.json"),
    }
    for type_name, schema_name in expected_concurrency_experiment_schemas.items():
        contract = contracts[f"eval_factory.contracts.concurrency_experiment_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_orchestration_schemas = {
        "ArtifactGroupFanoutV2": "artifact_group_fanout_v2.schema.json",
        "ResolvedDatasetJobPlanV2": ("resolved_dataset_job_plan_v2.schema.json"),
        "ResolvedJobWorkGraphV2": "resolved_job_work_graph_v2.schema.json",
        "ResolvedStageNodeV2": "resolved_stage_node_v2.schema.json",
        "ResolvedWorkUnitV2": "resolved_work_unit_v2.schema.json",
        "WorkCancellationRecordV2": "work_cancellation_record_v2.schema.json",
        "WorkControlPolicyV2": "work_control_policy_v2.schema.json",
        "WorkDispatchDecisionV2": "work_dispatch_decision_v2.schema.json",
        "WorkLeaseEventV2": "work_lease_event_v2.schema.json",
        "WorkLeaseV2": "work_lease_v2.schema.json",
        "WorkReadinessSnapshotV2": ("work_readiness_snapshot_v2.schema.json"),
        "WorkRetryDecisionV2": "work_retry_decision_v2.schema.json",
    }
    for type_name, schema_name in expected_orchestration_schemas.items():
        contract = contracts[f"eval_factory.contracts.orchestration_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_observability_schemas = {
        "BatchAuditEventV2": "batch_audit_event_v2.schema.json",
        "BatchAuditReportV2": "batch_audit_report_v2.schema.json",
        "BatchMetricsSnapshotV2": "batch_metrics_snapshot_v2.schema.json",
        "ItemMetricsSnapshotV2": "item_metrics_snapshot_v2.schema.json",
        "WorkAttemptMetricsV2": "work_attempt_metrics_v2.schema.json",
    }
    for type_name, schema_name in expected_observability_schemas.items():
        contract = contracts[f"eval_factory.contracts.observability_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_cli_schemas = {
        "PipelineCancellationResultV2": ("pipeline_cancellation_result_v2.schema.json"),
        "PipelineControlConfigV2": ("pipeline_control_config_v2.schema.json"),
        "PipelineCreateResultV2": ("pipeline_create_result_v2.schema.json"),
        "PipelineEventsResultV2": ("pipeline_events_result_v2.schema.json"),
        "PipelineMetricSliceSummaryV2": ("pipeline_metric_slice_summary_v2.schema.json"),
        "PipelineMetricsResultV2": ("pipeline_metrics_result_v2.schema.json"),
        "PipelinePlanResultV2": ("pipeline_plan_result_v2.schema.json"),
        "PipelineResumeResultV2": ("pipeline_resume_result_v2.schema.json"),
        "PipelineStatusResultV2": ("pipeline_status_result_v2.schema.json"),
        "PipelineWorkUnitStatusV2": ("pipeline_work_unit_status_v2.schema.json"),
    }
    for type_name, schema_name in expected_cli_schemas.items():
        contract = contracts[f"eval_factory.contracts.cli_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    telemetry = contracts["env_mock_agent.facade.telemetry_v2.ExecutionTelemetryV2"]
    assert telemetry["schema_path"] == "schemas/overlay/execution_telemetry_v2.schema.json"
    assert telemetry["owner"] == "facade"
    expected_attachment_schemas = {
        "ArtifactBuildContractV2": "artifact_build_contract_v2.schema.json",
        "ArtifactBuildResultV2": "artifact_build_result_v2.schema.json",
        "ArtifactBuildSpecV2": "artifact_build_spec_v2.schema.json",
        "ArtifactEvidenceMatrixV2": "artifact_evidence_matrix_v2.schema.json",
        "ArtifactEvidenceRowV2": "artifact_evidence_row_v2.schema.json",
        "ArtifactEvidenceTargetV2": "artifact_evidence_target_v2.schema.json",
        "ArtifactExecutionBatchV2": "artifact_execution_batch_v2.schema.json",
        "ArtifactExecutionGroupV2": "artifact_execution_group_v2.schema.json",
        "ArtifactExecutionPlanV2": "artifact_execution_plan_v2.schema.json",
        "ArtifactExecutionReceiptV2": "artifact_execution_receipt_v2.schema.json",
        "ArtifactExecutionUnitV2": "artifact_execution_unit_v2.schema.json",
        "ArtifactRoutePlanEntryV2": "artifact_route_plan_entry_v2.schema.json",
        "ArtifactRoutingPlanV2": "artifact_routing_plan_v2.schema.json",
        "ArtifactRoutingPolicyV2": "artifact_routing_policy_v2.schema.json",
        "AttachmentPlanningContextV2": ("attachment_planning_context_v2.schema.json"),
        "AttachmentReconstructionResultV2": ("attachment_reconstruction_result_v2.schema.json"),
        "PromptOnlyDependencyDiscoveryV2": ("prompt_only_dependency_discovery_v2.schema.json"),
        "PromptOnlyDependencyEvidenceBindingV2": ("prompt_only_dependency_evidence_binding_v2.schema.json"),
        "PromptOnlyDependencyPlanningContextV2": ("prompt_only_dependency_planning_context_v2.schema.json"),
        "PublicSourceRetrievalPolicyV2": ("public_source_retrieval_policy_v2.schema.json"),
        "PublicSourceSafetyAssessmentV2": ("public_source_safety_assessment_v2.schema.json"),
        "SourceEvidenceClaimBindingV2": ("source_evidence_claim_binding_v2.schema.json"),
        "SourceEvidenceSetV2": "source_evidence_set_v2.schema.json",
        "SourceEvidenceV2": "source_evidence_v2.schema.json",
    }
    for type_name, schema_name in expected_attachment_schemas.items():
        assert (
            contracts[f"eval_factory.contracts.attachment_v2.{type_name}"]["schema_path"]
            == f"schemas/overlay/{schema_name}"
        )
        assert contracts[f"eval_factory.contracts.attachment_v2.{type_name}"]["owner"] == "factory"
    expected_facade_schemas = {
        "AttachmentExecutionEvidenceGrantV2": (
            "execution_v2",
            "attachment_execution_evidence_grant_v2.schema.json",
        ),
        "AttachmentExecutionRequestV2": (
            "execution_v2",
            "attachment_execution_request_v2.schema.json",
        ),
        "AttachmentExecutionResultV2": (
            "execution_v2",
            "attachment_execution_result_v2.schema.json",
        ),
        "AttachmentExecutionSourceSpanV2": (
            "execution_v2",
            "attachment_execution_source_span_v2.schema.json",
        ),
        "WorldLedgerFactLockV2": (
            "execution_v2",
            "world_ledger_fact_lock_v2.schema.json",
        ),
        "WorldLedgerSnapshotRequestV2": (
            "execution_v2",
            "world_ledger_snapshot_request_v2.schema.json",
        ),
        "WorldLedgerSnapshotV2": (
            "execution_v2",
            "world_ledger_snapshot_v2.schema.json",
        ),
        "PublicSourceFetchRequestV2": (
            "retrieval_v2",
            "public_source_fetch_request_v2.schema.json",
        ),
        "PublicSourceFetchResultV2": (
            "retrieval_v2",
            "public_source_fetch_result_v2.schema.json",
        ),
        "PublicSourceSearchHitV2": (
            "retrieval_v2",
            "public_source_search_hit_v2.schema.json",
        ),
        "PublicSourceSearchRequestV2": (
            "retrieval_v2",
            "public_source_search_request_v2.schema.json",
        ),
        "PublicSourceSearchResultV2": (
            "retrieval_v2",
            "public_source_search_result_v2.schema.json",
        ),
        "AttachmentRouteRequestV2": (
            "routing_v2",
            "attachment_route_request_v2.schema.json",
        ),
        "AttachmentRouteSkipV2": (
            "routing_v2",
            "attachment_route_skip_v2.schema.json",
        ),
        "AttachmentRouteDecisionV2": (
            "routing_v2",
            "attachment_route_decision_v2.schema.json",
        ),
        "AttachmentInventoryMemberV2": (
            "validation_v2",
            "attachment_inventory_member_v2.schema.json",
        ),
        "AttachmentValidationFindingV2": (
            "validation_v2",
            "attachment_validation_finding_v2.schema.json",
        ),
        "AttachmentValidationFingerprintV2": (
            "validation_v2",
            "attachment_validation_fingerprint_v2.schema.json",
        ),
        "AttachmentValidationPiiRuleV2": (
            "validation_v2",
            "attachment_validation_pii_rule_v2.schema.json",
        ),
        "AttachmentValidationRequestV2": (
            "validation_v2",
            "attachment_validation_request_v2.schema.json",
        ),
        "AttachmentValidationResultV2": (
            "validation_v2",
            "attachment_validation_result_v2.schema.json",
        ),
        "AttachmentValidationScanLimitsV2": (
            "validation_v2",
            "attachment_validation_scan_limits_v2.schema.json",
        ),
        "AttachmentRepairRequestV2": (
            "semantic_review_v2",
            "attachment_repair_request_v2.schema.json",
        ),
        "AttachmentRepairResultV2": (
            "semantic_review_v2",
            "attachment_repair_result_v2.schema.json",
        ),
        "AttachmentSemanticFindingPolicyV2": (
            "semantic_review_v2",
            "attachment_semantic_finding_policy_v2.schema.json",
        ),
        "AttachmentSemanticFindingResolutionV2": (
            "semantic_review_v2",
            "attachment_semantic_finding_resolution_v2.schema.json",
        ),
        "AttachmentSemanticReviewFindingV2": (
            "semantic_review_v2",
            "attachment_semantic_review_finding_v2.schema.json",
        ),
        "AttachmentSemanticReviewRequestV2": (
            "semantic_review_v2",
            "attachment_semantic_review_request_v2.schema.json",
        ),
        "AttachmentSemanticReviewResultV2": (
            "semantic_review_v2",
            "attachment_semantic_review_result_v2.schema.json",
        ),
        "SemanticCleanContextAttestationV2": (
            "semantic_review_v2",
            "semantic_clean_context_attestation_v2.schema.json",
        ),
    }
    for type_name, (module_name, schema_name) in expected_facade_schemas.items():
        contract = contracts[f"env_mock_agent.facade.{module_name}.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "facade"
        schema = json.loads((manifest_path.parent / contract["schema_path"]).read_text(encoding="utf-8"))
        assert schema["x-contract-owner"] == "facade"
    expected_validation_schemas = {
        "ArtifactDeterministicValidationResultV2": (
            "artifact_deterministic_validation_result_v2.schema.json"
        ),
        "CandidatePackageInventoryEntryV2": ("candidate_package_inventory_entry_v2.schema.json"),
        "CandidatePackageInventoryV2": ("candidate_package_inventory_v2.schema.json"),
        "DeterministicItemValidationResultV2": ("deterministic_item_validation_result_v2.schema.json"),
        "DeterministicValidationFindingV2": ("deterministic_validation_finding_v2.schema.json"),
    }
    for type_name, schema_name in expected_validation_schemas.items():
        contract = contracts[f"eval_factory.contracts.validation_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_review_schemas = {
        "AttachmentCandidateRevisionV2": ("attachment_candidate_revision_v2.schema.json"),
        "CandidateArtifactVersionV2": ("candidate_artifact_version_v2.schema.json"),
        "CoverageSolvabilityReviewViewV2": ("coverage_solvability_review_view_v2.schema.json"),
        "DeterministicRepairTargetV2": ("deterministic_repair_target_v2.schema.json"),
        "LeakageExecutabilityReviewViewV2": ("leakage_executability_review_view_v2.schema.json"),
        "RealismConsistencyReviewViewV2": ("realism_consistency_review_view_v2.schema.json"),
        "RepairedArtifactBuildResultV2": ("repaired_artifact_build_result_v2.schema.json"),
        "RevisionDeterministicValidationV2": ("revision_deterministic_validation_v2.schema.json"),
        "SemanticFindingResolutionV2": ("semantic_finding_resolution_v2.schema.json"),
        "SemanticReviewFindingV2": ("semantic_review_finding_v2.schema.json"),
        "SemanticReviewPolicyV2": ("semantic_review_policy_v2.schema.json"),
        "SemanticReviewRoundResultV2": ("semantic_review_round_result_v2.schema.json"),
        "SemanticReviewWorkflowResultV2": ("semantic_review_workflow_result_v2.schema.json"),
        "TargetedRepairPlanV2": ("targeted_repair_plan_v2.schema.json"),
    }
    for type_name, schema_name in expected_review_schemas.items():
        contract = contracts[f"eval_factory.contracts.review_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    expected_quality_schemas = {
        "EnvironmentArtifactV2": "environment_artifact_v2.schema.json",
        "EnvironmentSpecV2": "environment_spec_v2.schema.json",
        "FinalPackageManifestV2": "final_package_manifest_v2.schema.json",
        "ItemQualityCompilationResultV2": ("item_quality_compilation_result_v2.schema.json"),
        "PackageMemberProvenanceV2": ("package_member_provenance_v2.schema.json"),
        "ProvenanceManifestV2": "provenance_manifest_v2.schema.json",
        "QualityReportV2": "quality_report_v2.schema.json",
    }
    for type_name, schema_name in expected_quality_schemas.items():
        contract = contracts[f"eval_factory.contracts.quality_v2.{type_name}"]
        assert contract["schema_path"] == f"schemas/overlay/{schema_name}"
        assert contract["owner"] == "factory"
    assert contracts["eval_factory.contracts.task_v2.TaskDraftV2"]["schema_path"] == (
        "schemas/overlay/task_draft_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskRequirementLineageV2"]["schema_path"] == (
        "schemas/overlay/task_requirement_lineage_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskEpisodeV2"]["schema_path"] == (
        "schemas/overlay/task_episode_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskEpisodeSegmentEvidenceBindingV2"]["schema_path"] == (
        "schemas/overlay/task_episode_segment_evidence_binding_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.PromptLeakageFingerprintV2"]["schema_path"] == (
        "schemas/overlay/prompt_leakage_fingerprint_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.PromptLeakageReferenceSetV2"]["schema_path"] == (
        "schemas/overlay/prompt_leakage_reference_set_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskPromptSafetyFindingV2"]["schema_path"] == (
        "schemas/overlay/task_prompt_safety_finding_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskPromptSafetyCheckV2"]["schema_path"] == (
        "schemas/overlay/task_prompt_safety_check_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.TaskPromptSafetyGateV2"]["schema_path"] == (
        "schemas/overlay/task_prompt_safety_gate_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.RubricJudgedObjectV2"]["schema_path"] == (
        "schemas/overlay/rubric_judged_object_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.RubricReachabilityV2"]["schema_path"] == (
        "schemas/overlay/rubric_reachability_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.RubricCriterionV2"]["schema_path"] == (
        "schemas/overlay/rubric_criterion_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.RubricSetV2"]["schema_path"] == (
        "schemas/overlay/rubric_set_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.EvaluatorFailureRuleV2"]["schema_path"] == (
        "schemas/overlay/evaluator_failure_rule_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.EvaluatorBindingV2"]["schema_path"] == (
        "schemas/overlay/evaluator_binding_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.EvaluatorSpecV2"]["schema_path"] == (
        "schemas/overlay/evaluator_spec_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ReferencePolicyV2"]["schema_path"] == (
        "schemas/overlay/reference_policy_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.EvaluatorReferenceGrantV2"]["schema_path"] == (
        "schemas/overlay/evaluator_reference_grant_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ToolRuleV2"]["schema_path"] == (
        "schemas/overlay/tool_rule_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ContestantToolRuleV2"]["schema_path"] == (
        "schemas/overlay/contestant_tool_rule_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ContestantToolPolicyV2"]["schema_path"] == (
        "schemas/overlay/contestant_tool_policy_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ToolPolicyV2"]["schema_path"] == (
        "schemas/overlay/tool_policy_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ProducerAttachmentRequirementV2"]["schema_path"] == (
        "schemas/overlay/producer_attachment_requirement_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ProducerStorageAuthorizationV2"]["schema_path"] == (
        "schemas/overlay/producer_storage_authorization_v2.schema.json"
    )
    assert contracts["eval_factory.contracts.task_v2.ProducerTaskViewV2"]["schema_path"] == (
        "schemas/overlay/producer_task_view_v2.schema.json"
    )
    expected_rewrite_schemas = {
        "R4TaskContractSetV2": "r4_task_contract_set_v2.schema.json",
        "TaskRewritePlanVersionV2": "task_rewrite_plan_version_v2.schema.json",
        "TaskRewriteExamplePreviewV2": "task_rewrite_example_preview_v2.schema.json",
        "TaskRewritePreviewSafetyGateV2": ("task_rewrite_preview_safety_gate_v2.schema.json"),
        "TaskRewritePlanPreviewV2": "task_rewrite_plan_preview_v2.schema.json",
        "TaskContractInvalidationV2": "task_contract_invalidation_v2.schema.json",
        "TaskRewriteApplicationV2": "task_rewrite_application_v2.schema.json",
    }
    for type_name, schema_name in expected_rewrite_schemas.items():
        assert contracts[f"eval_factory.contracts.task_v2.{type_name}"]["schema_path"] == (
            f"schemas/overlay/{schema_name}"
        )
    assert "eval-factory/task-episode/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/artifact-build-spec/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/artifact-evidence-row/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/artifact-evidence-matrix/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/task-draft/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/rubric-criterion/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/rubric-set/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/evaluator-spec/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/reference-policy/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/tool-rule/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/tool-policy/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/producer-task-view/v1" in manifest["supersedes_for_v2_jobs"]
    assert "eval-factory/human-review-record/v1" in manifest["historical_only_for_v2_jobs"]
    for contract in manifest["contracts"]:
        schema = json.loads((manifest_path.parent / contract["schema_path"]).read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        serialized = json.dumps(schema, sort_keys=True)
        assert "human_review_record_refs" not in serialized
        assert "review_quorum_refs" not in serialized
