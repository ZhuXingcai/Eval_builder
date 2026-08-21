from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    DECISIONS_BY_CHECKPOINT,
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentAlternative,
    EnvironmentRequirement,
    EnvironmentStrategy,
    EnvironmentStrategyChoice,
    ExampleKind,
    FinalReviewScope,
    LabelPlan,
    PlanExample,
    QueryPackagingChoice,
    RewriteFidelity,
    TaskRewritePlan,
    UserApprovalPolicy,
    UserApprovalRequest,
)
from eval_factory.contracts.attachment import AttachmentReconstructionResult
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import (
    canonical_json_v2,
    canonical_sha256_v2,
    canonical_value_v2,
)
from eval_factory.contracts.labeling import LabelSpec
from eval_factory.contracts.quality import QualityReport
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.safety import ProvenanceManifest
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EnvironmentSpec,
    EvaluationFailureClass,
    EvaluatorSpec,
    EvidencePriority,
    ProducerTaskView,
    QuerySpec,
    ReferenceMode,
    ReferencePolicy,
    RequirementConflict,
    RequirementLineage,
    RubricCriterion,
    RubricSet,
    RubricVisibility,
    SelectionFirewall,
    TaskDraft,
    TaskDraftStatus,
    ToolPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "evals/golden/eval_factory/evaluation_items/v2"
INDEX_PATH = OUTPUT_ROOT / "manifest.json"
CANARY_PATH = REPO_ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
CONTRACT_PATH = REPO_ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json"
POLICY_PATH = REPO_ROOT / "docs/policies/eval-factory-user-approval-v2.md"
LEGACY_WORKLIST = REPO_ROOT / "evals/golden/eval_factory/evaluation_items/v1/gold-worklist.json"
LEGACY_QUEUE = REPO_ROOT / "evals/golden/eval_factory/evaluation_items/v1/review-queue.json"

FIXED_TIME = datetime(2026, 7, 20, tzinfo=UTC)

PROFILES: dict[str, dict[str, Any]] = {
    "LH_005": {
        "label": "contextual-recovery-after-tool-error",
        "capability": "workspace-design-governance",
        "prompt": (
            "Inspect the current design workspace and determine whether a DESIGN.md file exists. "
            "Explain whether this project should have one, using the observable workspace structure "
            "and existing design documentation as evidence."
        ),
        "intent": "Evaluate whether the active design workspace has sufficient design-system governance.",
        "rewrite_style": "Concise engineering recommendation grounded in workspace evidence.",
        "environment": "The task depends on an application-managed design workspace and active design system.",
        "uncertainties": (),
    },
    "LH_011": {
        "label": "contextual-recovery-after-tool-error",
        "capability": "artifact-set-reconciliation",
        "prompt": (
            "Revise demo2 only. Keep exactly the three leadership-defined scenarios, remove the "
            "unneeded first step, expand the meeting transcript, map every retained input file to a "
            "scenario, and remove unused files and PDF duplicates."
        ),
        "intent": "Reconcile a demo artifact set against three explicit business scenarios.",
        "rewrite_style": "Direct file-maintenance task with explicit scope and deletion boundaries.",
        "environment": "The source task references a Windows workspace and local demo file tree.",
        "uncertainties": (
            "The nested request and response require repair; exact turn boundaries need R1 recovery.",
        ),
    },
    "LH_015": {
        "label": "search-tool-usage",
        "capability": "tool-guided-market-data-demo",
        "prompt": (
            "For demo1, retain three scenarios: report and policy summarization, evidence-based policy "
            "search, and competitor-product data retrieval. Build the local competitor-data web API "
            "and a skill that teaches the target agent to call it, while keeping the questions aligned "
            "with the approved leadership outline."
        ),
        "intent": "Prepare a reproducible three-scenario demo that includes tool-mediated data retrieval.",
        "rewrite_style": "Implementation-ready demo brief with strict scope and evidence attribution.",
        "environment": "The task uses a local web service, a skill file, and a Windows demo workspace.",
        "uncertainties": (),
    },
    "LH_058": {
        "label": "powershell-error-signature",
        "capability": "narrative-dna-extraction",
        "prompt": (
            "Continue the approved narrative-DNA workflow by analyzing the local refined text of "
            "Legend of the Galactic Heroes and producing the narrator DNA artifact under the existing "
            "project conventions."
        ),
        "intent": "Extract a reusable narrator-style contract from an authorized local source.",
        "rewrite_style": "Self-contained continuation that names the source and expected artifact.",
        "environment": "The source is a Windows-local long text file and the trace includes PowerShell use.",
        "uncertainties": ("The short continuation turn depends on prior approved project conventions.",),
    },
    "LH_067": {
        "label": "contextual-recovery-after-tool-error",
        "capability": "durable-project-initialization",
        "prompt": (
            "Initialize the novel project Ash Kings as an epic-fantasy, multi-POV, ultra-long work "
            "using the approved DNA sources. Create the durable project skeleton and indexed references "
            "without skipping prerequisite checks or fabricating missing memory results."
        ),
        "intent": "Initialize a durable creative project from approved constraints and reference indexes.",
        "rewrite_style": "Explicit initialization contract with prerequisites and rollback boundaries.",
        "environment": "The source workflow depends on local DNA files plus memory and document-index services.",
        "uncertainties": (
            "The request contains an invalid Unicode escape and requires audited local repair.",
            "Attachment-read completeness is not yet proven.",
        ),
    },
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _audit(*input_refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=FIXED_TIME,
        created_by="r0-09-reference-fixture-builder",
        governing_versions=(
            VersionBinding(
                component="eval-factory-spec",
                version="approved-v2-2026-07-20",
            ),
            VersionBinding(
                component="cross-stage-contracts",
                version="v2",
                sha256=_file_sha256(CONTRACT_PATH),
            ),
            VersionBinding(
                component="user-approval-policy",
                version="v2",
                sha256=_file_sha256(POLICY_PATH),
            ),
        ),
        input_refs=input_refs,
    )


def _ref(
    value: ContractModel,
    object_type: str,
    object_id: str,
    object_version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=object_version,
        object_sha256=canonical_sha256_v2(value),
    )


def _raw_ref(instance_id: str, raw_sha256: str) -> ObjectRef:
    return ObjectRef(
        object_type="raw-trace",
        object_id=f"raw-traj://{instance_id}",
        object_version="raw",
        object_sha256=raw_sha256,
    )


def _placeholder_ref(instance_id: str, object_type: str, suffix: str) -> ObjectRef:
    seed = f"{instance_id}:{object_type}:{suffix}".encode()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{instance_id}/{suffix}",
        object_version="v2",
        object_sha256=_sha256(seed),
    )


def _load_label(slug: str) -> LabelSpec:
    path = REPO_ROOT / f"specs/002-eval-dataset-factory/labels/v1/{slug}.json"
    return LabelSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _build_fixture(canary: dict[str, Any]) -> dict[str, Any]:
    instance_id = canary["instance_id"]
    raw_sha256 = canary["raw_sha256"]
    profile = PROFILES[instance_id]
    source_ref = _raw_ref(instance_id, raw_sha256)

    span = SourceSpanRef(
        span_id=f"source-span://{instance_id}/candidate-intent",
        source_trace_id=f"source-trace://{instance_id}",
        raw_sha256=raw_sha256,
        approximate=canary["signal_quality"] != "strict_events",
    )
    evidence = EvidenceRef(
        evidence_ref_id=f"evidence-ref://{instance_id}/candidate-intent",
        subject_ref=source_ref,
        source_spans=(span,),
        polarity=EvidencePolarity.POSITIVE,
        capability="candidate-user-intent",
        capability_complete=canary["signal_quality"] == "strict_events",
    )

    label_spec = _load_label(profile["label"])
    label_spec_ref = _ref(
        label_spec,
        "label-spec",
        label_spec.label_spec_id,
        label_spec.label_version,
    )
    label_decision_ref = _placeholder_ref(instance_id, "label-decision", "reference-candidate")
    selection_context_ref = _placeholder_ref(instance_id, "selection-context", "reference-candidate")

    dependency = AttachmentDependency(
        dependency_id=f"attachment-dependency://{instance_id}/workspace",
        description="Candidate input-state workspace needed to execute the reconstructed task.",
        criticality=AttachmentCriticality.REQUIRED,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(evidence,),
    )
    lineage = RequirementLineage(
        requirement_id=f"requirement://{instance_id}/primary-intent",
        statement=profile["intent"],
        evidence_priority=EvidencePriority.DIRECT_OBSERVATION,
        evidence=(evidence,),
        conflict_status=RequirementConflict.NONE,
    )
    task_draft = TaskDraft(
        task_draft_id=f"task-draft://{instance_id}/v2",
        selection_context_ref=selection_context_ref,
        task_episode_refs=(_placeholder_ref(instance_id, "task-episode", "candidate"),),
        visible_prompt=profile["prompt"],
        task_intent=profile["intent"],
        evaluation_claim=(
            "The contestant can complete the scoped task using only the prompt, approved tools, and "
            "reconstructable input-state dependencies."
        ),
        required_capabilities=(profile["capability"],),
        allowed_tools=("file-read", "file-write", "shell"),
        forbidden_outputs=(
            "original agent final answer",
            "pre-completed requested deliverable",
            "grader rules or hidden pass conditions",
        ),
        attachment_dependencies=(dependency,),
        requirement_lineage=(lineage,),
        uncertainties=profile["uncertainties"],
        selection_firewall=SelectionFirewall(
            final_answer_excluded=True,
            completed_deliverable_excluded=True,
            private_reference_excluded=True,
            grader_rules_excluded=True,
            hidden_selection_signals_excluded=True,
            trajectory_specific_steps_excluded=True,
        ),
        status=TaskDraftStatus.CANDIDATE,
        audit=_audit(source_ref, label_decision_ref),
    )
    task_draft_ref = _ref(task_draft, "task-draft", task_draft.task_draft_id)

    query_spec = QuerySpec(
        query_spec_id=f"query-spec://{instance_id}/v2",
        task_draft_ref=task_draft_ref,
        prompt=profile["prompt"],
        attachment_dependency_ids=(dependency.dependency_id,),
        prompt_sha256=_sha256(profile["prompt"].encode()),
        audit=_audit(task_draft_ref),
    )
    query_spec_ref = _ref(query_spec, "query-spec", query_spec.query_spec_id)

    package_sha256 = _sha256(f"{instance_id}:empty-reference-package".encode())
    inventory_ref = _placeholder_ref(instance_id, "package-inventory", "empty-candidate")
    environment_spec = EnvironmentSpec(
        environment_spec_id=f"environment-spec://{instance_id}/v2",
        artifacts=(),
        package_manifest_ref=inventory_ref,
        package_sha256=package_sha256,
        audit=_audit(task_draft_ref),
    )
    environment_spec_ref = _ref(
        environment_spec,
        "environment-spec",
        environment_spec.environment_spec_id,
    )

    rubric_set = RubricSet(
        rubric_set_id=f"rubric-set://{instance_id}/v2",
        query_spec_ref=query_spec_ref,
        criteria=(
            RubricCriterion(
                criterion_id=f"criterion://{instance_id}/task-completion",
                judged_object="contestant-output",
                description=(
                    "Completes the stated task without relying on hidden trace content or producing "
                    "unsupported claims."
                ),
                weight=1.0,
                reachability_evidence=(evidence,),
                visibility=RubricVisibility.EVALUATOR_ONLY,
                evaluator_binding=f"evaluator://{instance_id}/v2",
                approval_status="CANDIDATE",
            ),
        ),
        total_weight=1.0,
        audit=_audit(query_spec_ref),
    )
    rubric_set_ref = _ref(rubric_set, "rubric-set", rubric_set.rubric_set_id)

    evaluator_spec = EvaluatorSpec(
        evaluator_spec_id=f"evaluator-spec://{instance_id}/v2",
        rubric_set_ref=rubric_set_ref,
        evaluator_type="contract-evaluator",
        evaluator_version="candidate-v2",
        input_contract_ref=_placeholder_ref(instance_id, "evaluator-input-contract", "candidate"),
        output_contract_ref=_placeholder_ref(instance_id, "evaluator-output-contract", "candidate"),
        failure_classes=frozenset(EvaluationFailureClass),
        timeout_seconds=300,
        audit=_audit(rubric_set_ref),
    )
    evaluator_spec_ref = _ref(
        evaluator_spec,
        "evaluator-spec",
        evaluator_spec.evaluator_spec_id,
    )

    reference_policy = ReferencePolicy(
        reference_policy_id=f"reference-policy://{instance_id}/v2",
        mode=ReferenceMode.NONE,
        evaluator_access=False,
        human_only_access=False,
        policy_version="reference-policy/v2-candidate",
        audit=_audit(query_spec_ref),
    )
    reference_policy_ref = _ref(
        reference_policy,
        "reference-policy",
        reference_policy.reference_policy_id,
    )

    tool_policy = ToolPolicy(
        tool_policy_id=f"tool-policy://{instance_id}/v2",
        rules=(),
        contestant_projection_ref=_placeholder_ref(
            instance_id,
            "tool-policy-projection",
            "candidate",
        ),
        policy_version="tool-policy/v2-candidate",
        audit=_audit(task_draft_ref),
    )
    tool_policy_ref = _ref(tool_policy, "tool-policy", tool_policy.tool_policy_id)

    provenance_manifest = ProvenanceManifest(
        provenance_manifest_id=f"provenance-manifest://{instance_id}/v2",
        package_sha256=package_sha256,
        inventory_ref=inventory_ref,
        entries=(),
        exact_set_verified=True,
        policy_version="eval-factory-safety/v1",
        audit=_audit(source_ref, environment_spec_ref),
    )
    provenance_manifest_ref = _ref(
        provenance_manifest,
        "provenance-manifest",
        provenance_manifest.provenance_manifest_id,
    )

    quality_report = QualityReport(
        quality_report_id=f"quality-report://{instance_id}/v2",
        item_subject_refs=(
            query_spec_ref,
            environment_spec_ref,
            rubric_set_ref,
            evaluator_spec_ref,
            reference_policy_ref,
            tool_policy_ref,
            provenance_manifest_ref,
        ),
        deterministic_validation_refs=(
            _placeholder_ref(instance_id, "validation-result", "reference-scaffold"),
        ),
        semantic_round_refs=(
            _placeholder_ref(instance_id, "semantic-review-result", "round-1-pending"),
            _placeholder_ref(instance_id, "semantic-review-result", "round-2-pending"),
            _placeholder_ref(instance_id, "semantic-review-result", "round-3-pending"),
        ),
        finding_refs=(),
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        approvable=False,
        audit=_audit(
            query_spec_ref,
            environment_spec_ref,
            provenance_manifest_ref,
        ),
    )
    quality_report_ref = _ref(
        quality_report,
        "quality-report",
        quality_report.quality_report_id,
    )

    producer_view = ProducerTaskView(
        producer_task_view_id=f"producer-task-view://{instance_id}/v2",
        query_instruction=profile["prompt"],
        attachment_dependencies=(dependency,),
        allowed_tools=("file-read", "file-write"),
        safe_evidence_bundle_refs=(_placeholder_ref(instance_id, "evidence-bundle", "candidate-safe-view"),),
        forbidden_outputs=task_draft.forbidden_outputs,
        projection_policy_ref=_placeholder_ref(
            instance_id,
            "projection-policy",
            "attachment-producer-v2",
        ),
        source_task_draft_sha256=canonical_sha256_v2(task_draft),
        audit=_audit(task_draft_ref),
    )
    producer_view_ref = _ref(
        producer_view,
        "producer-task-view",
        producer_view.producer_task_view_id,
    )
    attachment_result = AttachmentReconstructionResult(
        attachment_reconstruction_result_id=f"attachment-result://{instance_id}/v2",
        producer_task_view_ref=producer_view_ref,
        evidence_matrix_ref=_placeholder_ref(instance_id, "artifact-evidence-matrix", "candidate"),
        artifact_result_refs=(_placeholder_ref(instance_id, "artifact-build-result", "pending"),),
        accepted_artifact_refs=(),
        failed_artifact_ids=(),
        resumable_artifact_ids=(),
        environment_spec_ref=environment_spec_ref,
        provenance_manifest_ref=provenance_manifest_ref,
        quality_report_ref=quality_report_ref,
        package_sha256=package_sha256,
        input_state_only=True,
        audit=_audit(producer_view_ref, environment_spec_ref),
    )
    attachment_result_ref = _ref(
        attachment_result,
        "attachment-reconstruction-result",
        attachment_result.attachment_reconstruction_result_id,
    )

    approval_policy = UserApprovalPolicy(
        policy_id=f"user-approval-policy://{instance_id}/v2",
        policy_version="v2-reference-preview",
        mode=ApprovalMode.PLAN_AND_FINAL,
        enabled_checkpoints=ALL_CHECKPOINTS,
        final_review_scope=FinalReviewScope.FULL_DATASET,
        explicit_unattended_choice=False,
        audit=_audit(source_ref),
    )
    approval_policy_ref = _ref(
        approval_policy,
        "user-approval-policy",
        approval_policy.policy_id,
    )

    components = EvaluationItemComponents(
        query_spec_ref=query_spec_ref,
        environment_spec_ref=environment_spec_ref,
        rubric_set_ref=rubric_set_ref,
        evaluator_spec_ref=evaluator_spec_ref,
        reference_policy_ref=reference_policy_ref,
        tool_policy_ref=tool_policy_ref,
        provenance_manifest_ref=provenance_manifest_ref,
        quality_report_ref=quality_report_ref,
    )
    release_decision = ReleaseDecisionV2(
        release_decision_id=f"release-decision://{instance_id}/v2",
        chain_id=f"release-chain://{instance_id}",
        previous_decision_ref=None,
        item_id=f"evaluation-item://{instance_id}",
        item_version="v2",
        components=components,
        release_subject_sha256=_sha256(
            canonical_json_v2(query_spec)
            + canonical_json_v2(environment_spec)
            + canonical_json_v2(rubric_set)
            + canonical_json_v2(quality_report)
        ),
        package_manifest_ref=inventory_ref,
        package_sha256=package_sha256,
        batch_quality_report_ref=None,
        automated_quality_passed=False,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        user_approval_policy_ref=approval_policy_ref,
        required_checkpoints=ALL_CHECKPOINTS,
        checkpoint_decisions=(),
        channel=ReleaseChannel.CANARY,
        registry="registry://reference-fixtures",
        export_profile="LH",
        export_profile_version="v1",
        production_attestation_ref=None,
        action=ReleaseActionV2.REQUEST_RELEASE,
        state=ReleaseStateV2.CANDIDATE,
        idempotency_key=f"reference-release-{instance_id}-v2",
        actor="r0-09-reference-fixture-builder",
        decided_at=FIXED_TIME,
        decision_sha256=_sha256(f"{instance_id}:candidate-release".encode()),
        audit=_audit(quality_report_ref, provenance_manifest_ref),
    )
    release_decision_ref = _ref(
        release_decision,
        "release-decision",
        release_decision.release_decision_id,
    )
    evaluation_item = EvaluationItemV2(
        evaluation_item_id=f"evaluation-item://{instance_id}/v2",
        item_version="v2",
        source_trace_refs=(source_ref,),
        label_decision_refs=(label_decision_ref,),
        task_draft_ref=task_draft_ref,
        attachment_reconstruction_result_ref=attachment_result_ref,
        components=components,
        user_approval_policy_ref=approval_policy_ref,
        user_decision_record_refs=(),
        release_decision_ref=release_decision_ref,
        item_sha256=_sha256(f"{instance_id}:review-ready-not-gold".encode()),
        audit=_audit(source_ref, release_decision_ref),
    )
    evaluation_item_ref = _ref(
        evaluation_item,
        "evaluation-item",
        evaluation_item.evaluation_item_id,
    )

    examples = (
        PlanExample(
            example_id=f"plan-example://{instance_id}/positive",
            kind=ExampleKind.POSITIVE,
            input_summary="The required structured event and capability are directly observed.",
            expected_treatment="Accept only with exact evidence references.",
            evidence_refs=(evidence,),
        ),
        PlanExample(
            example_id=f"plan-example://{instance_id}/negative",
            kind=ExampleKind.NEGATIVE,
            input_summary="The relevant capability is complete and the required event is absent.",
            expected_treatment="Record NO_MATCH with complete negative evidence.",
        ),
        PlanExample(
            example_id=f"plan-example://{instance_id}/ambiguous",
            kind=ExampleKind.AMBIGUOUS,
            input_summary="A text mention exists but execution identity is not proven.",
            expected_treatment="Do not treat the mention as an executed event.",
        ),
        PlanExample(
            example_id=f"plan-example://{instance_id}/abstain",
            kind=ExampleKind.ABSTAIN,
            input_summary="The relevant source range is malformed, truncated, or unavailable.",
            expected_treatment="ABSTAIN rather than infer a negative fact.",
        ),
    )
    clause_refs = tuple(
        _ref(
            predicate,
            "structured-predicate",
            predicate.predicate_id,
            "v1",
        )
        for predicate in (*label_spec.positive_predicates, *label_spec.negative_predicates)
    )
    semantic_refs = (
        ()
        if label_spec.semantic_residual is None
        else (
            _ref(
                label_spec.semantic_residual,
                "semantic-residual-spec",
                label_spec.semantic_residual.residual_id,
                "v1",
            ),
        )
    )
    label_plan = LabelPlan(
        label_plan_id=f"label-plan://{instance_id}/v2",
        label_spec_ref=label_spec_ref,
        intent=label_spec.requirement,
        boundary="Use normalized executed facts; never promote prompt mentions or incomplete negatives.",
        deterministic_clause_refs=clause_refs,
        semantic_clause_refs=semantic_refs,
        examples=examples,
        abstain_rules=(
            "abstain when required capability is incomplete",
            "abstain when evidence identity or task continuity is ambiguous",
        ),
        expected_model_path=(
            () if label_spec.semantic_residual is None else (label_spec.semantic_residual.model_profile,)
        ),
        estimated_model_requests_per_trace=0 if label_spec.semantic_residual is None else 1,
        blind_spots=(
            "Canary traits route examples but are not annotation truth.",
            "Partial parsing cannot prove absence outside recovered ranges.",
        ),
        audit=_audit(source_ref, label_spec_ref),
    )
    label_plan_ref = _ref(label_plan, "label-plan", label_plan.label_plan_id)

    rewrite_plan = TaskRewritePlan(
        task_rewrite_plan_id=f"task-rewrite-plan://{instance_id}/v2",
        selection_context_ref=selection_context_ref,
        target_capability=profile["capability"],
        rewrite_style=profile["rewrite_style"],
        fidelity=RewriteFidelity.CAPABILITY_PRESERVING,
        operational_noise_policy=(
            "Remove retries, runtime chatter, and accidental implementation path while preserving "
            "explicit user constraints and required inputs."
        ),
        examples=(
            PlanExample(
                example_id=f"plan-example://{instance_id}/rewrite",
                kind=ExampleKind.REWRITE,
                input_summary="Multi-turn operational trace with a stable user objective.",
                expected_treatment=profile["prompt"],
                evidence_refs=(evidence,),
            ),
        ),
        forbidden_content_rules=task_draft.forbidden_outputs,
        expected_capability_impact="Preserve the evaluated capability without preserving accidental steps.",
        audit=_audit(source_ref, selection_context_ref),
    )
    rewrite_plan_ref = _ref(
        rewrite_plan,
        "task-rewrite-plan",
        rewrite_plan.task_rewrite_plan_id,
    )

    environment_strategy = EnvironmentStrategy(
        environment_strategy_id=f"environment-strategy://{instance_id}/v2",
        requirements=(
            EnvironmentRequirement(
                requirement_id=f"environment-requirement://{instance_id}/workspace",
                affected_subject_refs=(task_draft_ref,),
                description=profile["environment"],
                requires_special_account=instance_id == "LH_067",
                requires_nonstandard_environment=True,
                evidence_refs=(evidence,),
                alternatives=(
                    EnvironmentAlternative(
                        strategy=EnvironmentStrategyChoice.TRACE_FAITHFUL_MOCK,
                        feasible=True,
                        capability_impact="Highest fidelity; requires safe reconstruction of local dependencies.",
                    ),
                    EnvironmentAlternative(
                        strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
                        feasible=True,
                        capability_impact="Removes local paths or services while preserving the target capability.",
                    ),
                    EnvironmentAlternative(
                        strategy=EnvironmentStrategyChoice.EXCLUDE_TASK,
                        feasible=True,
                        capability_impact="Removes this candidate from the dataset.",
                    ),
                ),
            ),
        ),
        query_packaging_options=frozenset(QueryPackagingChoice),
        recommended_strategy=EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
        recommendation_reason=(
            "R0 reference fixtures should remain executable without private credentials or host-specific paths."
        ),
        audit=_audit(task_draft_ref),
    )
    environment_strategy_ref = _ref(
        environment_strategy,
        "environment-strategy",
        environment_strategy.environment_strategy_id,
    )

    request_specs = (
        (
            ApprovalCheckpoint.LABEL_PLAN,
            (label_spec_ref,),
            label_plan_ref,
            (label_plan_ref,),
            ("label",),
        ),
        (
            ApprovalCheckpoint.TASK_REWRITE_PLAN,
            (selection_context_ref,),
            rewrite_plan_ref,
            (rewrite_plan_ref,),
            ("task-authoring",),
        ),
        (
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
            (task_draft_ref,),
            environment_strategy_ref,
            (environment_strategy_ref,),
            ("task-authoring", "attachment"),
        ),
        (
            ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            (evaluation_item_ref,),
            None,
            (query_spec_ref, quality_report_ref, evaluation_item_ref),
            ("release",),
        ),
    )
    requests: list[UserApprovalRequest] = []
    for checkpoint, subjects, plan_ref, preview_refs, stages in request_specs:
        requests.append(
            UserApprovalRequest(
                request_id=f"user-approval-request://{instance_id}/{checkpoint.value.casefold()}/v2",
                checkpoint=checkpoint,
                requested_by="requesting-user",
                approval_policy_ref=approval_policy_ref,
                subject_refs=subjects,
                plan_ref=plan_ref,
                projection_ref=_placeholder_ref(
                    instance_id,
                    "user-approval-projection",
                    checkpoint.value.casefold(),
                ),
                preview_refs=preview_refs,
                available_decisions=DECISIONS_BY_CHECKPOINT[checkpoint],
                affected_stages=stages,
                idempotency_key=f"{instance_id}-{checkpoint.value.casefold()}-v2",
                audit=_audit(*subjects),
            )
        )

    objects: dict[str, ContractModel] = {
        "task_draft": task_draft,
        "query_spec": query_spec,
        "environment_spec": environment_spec,
        "rubric_set": rubric_set,
        "evaluator_spec": evaluator_spec,
        "reference_policy": reference_policy,
        "tool_policy": tool_policy,
        "provenance_manifest": provenance_manifest,
        "quality_report": quality_report,
        "producer_task_view": producer_view,
        "attachment_reconstruction_result": attachment_result,
        "user_approval_policy": approval_policy,
        "release_decision": release_decision,
        "evaluation_item": evaluation_item,
        "label_plan": label_plan,
        "task_rewrite_plan": rewrite_plan,
        "environment_strategy": environment_strategy,
    }
    for request in requests:
        objects[f"approval_request_{request.checkpoint.value.casefold()}"] = request

    return {
        "schema_version": "eval-factory-reference-fixture/v2",
        "status": "REVIEW_READY_NOT_GOLD",
        "semantic_gold": False,
        "user_approved": False,
        "source": {
            "instance_id": instance_id,
            "source_ref": canary["source_ref"],
            "source_sha256": raw_sha256,
            "signal_quality": canary["signal_quality"],
        },
        "contracts": {
            "path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(CONTRACT_PATH),
        },
        "required_evaluation_components": [
            "QuerySpec",
            "EnvironmentSpec",
            "RubricSet",
            "EvaluatorSpec",
            "ReferencePolicy",
            "ToolPolicy",
            "ProvenanceManifest",
            "QualityReport",
            "ReleaseDecisionV2",
        ],
        "objects": {name: canonical_value_v2(value) for name, value in objects.items()},
        "object_sha256": {name: canonical_sha256_v2(value) for name, value in objects.items()},
        "checkpoint_order": [checkpoint.value for checkpoint in ApprovalCheckpoint],
        "user_decision_records": [],
        "release_assertions": {
            "state": "CANDIDATE",
            "automated_quality_passed": False,
            "required_checkpoints_satisfied": False,
            "production_ready": False,
        },
    }


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    canary_manifest = json.loads(CANARY_PATH.read_text(encoding="utf-8"))
    canaries = {
        item["instance_id"]: item for item in canary_manifest["items"] if item["instance_id"] in PROFILES
    }
    if set(canaries) != set(PROFILES):
        raise ValueError("frozen canary manifest does not contain every v2 fixture source")

    artifacts: dict[str, bytes] = {}
    fixture_entries: list[dict[str, Any]] = []
    for instance_id in sorted(PROFILES):
        relative_path = f"{instance_id}.reference-fixture.json"
        content = _json_bytes(_build_fixture(canaries[instance_id]))
        artifacts[relative_path] = content
        fixture_entries.append(
            {
                "instance_id": instance_id,
                "path": relative_path,
                "sha256": _sha256(content),
                "status": "REVIEW_READY_NOT_GOLD",
            }
        )

    manifest = {
        "schema_version": "eval-factory-reference-fixture-manifest/v2",
        "status": "REVIEW_READY_NOT_GOLD",
        "fixture_count": len(fixture_entries),
        "canary_manifest": {
            "path": str(CANARY_PATH.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(CANARY_PATH),
        },
        "cross_stage_contract": {
            "path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(CONTRACT_PATH),
        },
        "user_approval_policy": {
            "path": str(POLICY_PATH.relative_to(REPO_ROOT)),
            "sha256": _file_sha256(POLICY_PATH),
        },
        "legacy_v1_review_workflow": {
            "status": "SUPERSEDED_NOT_SCHEDULED",
            "superseded_by": "env_mock_agent-ujc.1.13",
            "worklist": {
                "path": str(LEGACY_WORKLIST.relative_to(REPO_ROOT)),
                "sha256": _file_sha256(LEGACY_WORKLIST),
            },
            "review_queue": {
                "path": str(LEGACY_QUEUE.relative_to(REPO_ROOT)),
                "sha256": _file_sha256(LEGACY_QUEUE),
            },
        },
        "fixtures": fixture_entries,
    }
    return artifacts, manifest


def write_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for relative_path, content in artifacts.items():
        (OUTPUT_ROOT / relative_path).write_bytes(content)
    INDEX_PATH.write_bytes(_json_bytes(manifest))


def check_artifacts(artifacts: dict[str, bytes], manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = {OUTPUT_ROOT / path for path in artifacts}
    existing = set(OUTPUT_ROOT.glob("*.reference-fixture.json"))
    if expected != existing:
        errors.append("reference fixture file set drift")
    for relative_path, content in artifacts.items():
        path = OUTPUT_ROOT / relative_path
        if not path.exists() or path.read_bytes() != content:
            errors.append(f"reference fixture drift: {path}")
    expected_manifest = _json_bytes(manifest)
    if not INDEX_PATH.exists() or INDEX_PATH.read_bytes() != expected_manifest:
        errors.append(f"reference fixture manifest drift: {INDEX_PATH}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable R0-09 v2 reference fixtures.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()

    artifacts, manifest = build_artifacts()
    if args.write:
        write_artifacts(artifacts, manifest)
        print(f"wrote {len(artifacts)} reference fixtures and {INDEX_PATH}")
        return
    errors = check_artifacts(artifacts, manifest)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(artifacts)} R0-09 v2 reference fixtures")


if __name__ == "__main__":
    main()
