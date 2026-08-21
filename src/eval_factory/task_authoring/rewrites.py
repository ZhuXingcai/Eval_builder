from __future__ import annotations

import hashlib
import json
import re

from eval_factory.contracts.approval import (
    RewriteFidelity,
    TaskRewritePlan,
    TypedAdjustment,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricSetV2,
    TaskContractInvalidationV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyGateStatusV2,
    TaskPromptSafetyGateV2,
    TaskRewriteApplicationV2,
    TaskRewriteExamplePreviewV2,
    TaskRewritePlanPreviewV2,
    TaskRewritePlanVersionV2,
    TaskRewritePreviewSafetyGateV2,
    TaskRewriteRebuildStageV2,
    ToolPolicyV2,
    contestant_tool_policy_ref,
    evaluator_spec_carried_sha256,
    evaluator_spec_ref,
    producer_storage_authorization_ref,
    producer_task_view_ref,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    reference_policy_carried_sha256,
    reference_policy_ref,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_contract_invalidation_carried_sha256,
    task_contract_invalidation_ref,
    task_draft_carried_sha256,
    task_draft_payload_sha256,
    task_draft_ref,
    task_prompt_safety_gate_ref,
    task_rewrite_application_carried_sha256,
    task_rewrite_application_ref,
    task_rewrite_plan_carried_sha256,
    task_rewrite_plan_preview_carried_sha256,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_version_carried_sha256,
    task_rewrite_plan_version_ref,
    task_rewrite_preview_safety_gate_carried_sha256,
    task_rewrite_preview_safety_gate_ref,
    tool_policy_ref,
)
from eval_factory.provenance.injection import (
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
    TaskRewriteDataBoundaryRequest,
)
from eval_factory.provenance.views import EvidenceViewResult
from eval_factory.task_authoring.producer_models import (
    ProducerTaskViewPolicyError,
)
from eval_factory.task_authoring.producer_view import (
    ProducerTaskViewCompiler,
    validate_current_r4_contracts,
)
from eval_factory.task_authoring.prompt_safety import (
    TaskPromptSafetyCompiler,
    TaskPromptSafetyRequestBuilder,
    validate_prompt_leakage_reference_set_identity,
    validate_task_prompt_safety_gate_identity,
)
from eval_factory.task_authoring.prompt_safety_models import (
    TaskPromptSafetyOutcome,
    TaskPromptSafetyPolicyError,
    TaskPromptSafetyProposal,
    TaskPromptSafetyReason,
    TaskPromptSafetyRequest,
    TaskPromptSafetyResult,
)
from eval_factory.task_authoring.rewrite_models import (
    TASK_REWRITE_POLICY_VERSION,
    TASK_REWRITE_PREVIEW_POLICY_VERSION,
    FakeTaskRewritePromptFixture,
    TaskRewriteAdjustmentResult,
    TaskRewriteApplicationResult,
    TaskRewritePlanCompilationResult,
    TaskRewritePlanCompileOutcome,
    TaskRewritePlanCompileReason,
    TaskRewritePolicyError,
    TaskRewritePromptOutcome,
    TaskRewritePromptProposal,
    TaskRewritePromptReason,
    TaskRewritePromptRequest,
    TaskRewritePromptResult,
    rewrite_payload_sha256,
)
from eval_factory.task_authoring.tool_models import (
    ToolCapabilityCatalog,
    ToolPolicyCompilationOutcome,
)
from eval_factory.task_authoring.tools import (
    ToolPolicyCompiler,
)

_HARD_FORBIDDEN_RULES = (
    "original final answer",
    "completed deliverable",
    "private reference",
    "grader rule",
    "hidden pass condition",
    "secret",
)
_ALLOWED_PRESERVED_TYPES = frozenset(
    {
        "selection-context",
        "task-episode",
        "evidence-bundle",
        "projection-policy",
        "prompt-leakage-reference-set",
        "tool-capability-catalog",
    }
)
_EXAMPLE_PATH = re.compile(r"^examples\[([a-f0-9]{64})\]\.(input_summary|expected_treatment)$")


class R4TaskContractSetCompiler:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def compile(
        self,
        *,
        task_draft: TaskDraftV2,
        task_prompt_safety_gate: TaskPromptSafetyGateV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_storage_authorization: ProducerStorageAuthorizationV2,
        producer_task_view: ProducerTaskViewV2,
        audit: ContractAudit,
    ) -> R4TaskContractSetV2:
        try:
            validate_task_prompt_safety_gate_identity(task_prompt_safety_gate)
            validate_current_r4_contracts(
                task_draft=task_draft,
                rubric_set=rubric_set,
                evaluator_spec=evaluator_spec,
                reference_policy=reference_policy,
                tool_policy=tool_policy,
                contestant_tool_policy=contestant_tool_policy,
                storage_authorization=producer_storage_authorization,
                producer_task_view=producer_task_view,
            )
        except (
            ProducerTaskViewPolicyError,
            TaskPromptSafetyPolicyError,
        ) as exc:
            raise TaskRewritePolicyError(f"current R4 contract chain is invalid: {exc}") from exc
        if (
            task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
            or task_draft.prompt_safety_gate_ref != task_prompt_safety_gate_ref(task_prompt_safety_gate)
        ):
            raise TaskRewritePolicyError("TaskDraft prompt safety gate is stale or mismatched")
        if (
            task_draft.supersedes_task_draft_ref is None
            or task_prompt_safety_gate.source_task_draft_ref != task_draft.supersedes_task_draft_ref
            or task_prompt_safety_gate.status is not TaskPromptSafetyGateStatusV2.PASSED
        ):
            raise TaskRewritePolicyError("prompt safety gate must pass the TaskDraft predecessor")
        contract_set = R4TaskContractSetV2(
            contract_set_id="r4-task-contract-set://pending",
            task_draft_ref=task_draft_ref(task_draft),
            task_prompt_safety_gate_ref=task_prompt_safety_gate_ref(task_prompt_safety_gate),
            rubric_set_ref=rubric_set_ref(rubric_set),
            evaluator_spec_ref=evaluator_spec_ref(evaluator_spec),
            reference_policy_ref=reference_policy_ref(reference_policy),
            tool_policy_ref=tool_policy_ref(tool_policy),
            contestant_tool_policy_ref=contestant_tool_policy_ref(contestant_tool_policy),
            producer_storage_authorization_ref=(
                producer_storage_authorization_ref(producer_storage_authorization)
            ),
            producer_task_view_ref=producer_task_view_ref(producer_task_view),
            contract_set_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    task_draft_ref(task_draft),
                    task_prompt_safety_gate_ref(task_prompt_safety_gate),
                    rubric_set_ref(rubric_set),
                    evaluator_spec_ref(evaluator_spec),
                    reference_policy_ref(reference_policy),
                    tool_policy_ref(tool_policy),
                    contestant_tool_policy_ref(contestant_tool_policy),
                    producer_storage_authorization_ref(producer_storage_authorization),
                    producer_task_view_ref(producer_task_view),
                ),
            ),
        )
        digest = r4_task_contract_set_carried_sha256(contract_set)
        return contract_set.model_copy(
            update={
                "contract_set_id": (f"r4-task-contract-set://sha256/{digest}"),
                "contract_set_sha256": digest,
            }
        )

    def validate_current(
        self,
        *,
        contract_set: R4TaskContractSetV2,
        task_draft: TaskDraftV2,
        task_prompt_safety_gate: TaskPromptSafetyGateV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_storage_authorization: ProducerStorageAuthorizationV2,
        producer_task_view: ProducerTaskViewV2,
    ) -> None:
        _validate_contract_set_identity(contract_set)
        rebuilt = self.compile(
            task_draft=task_draft,
            task_prompt_safety_gate=task_prompt_safety_gate,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_tool_policy=contestant_tool_policy,
            producer_storage_authorization=producer_storage_authorization,
            producer_task_view=producer_task_view,
            audit=contract_set.audit,
        )
        if not _same_except_audit_actor_time(rebuilt, contract_set):
            raise TaskRewritePolicyError("R4 contract set does not match authoritative objects")


class TaskRewritePlanCompiler:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def build_safety_request(
        self,
        *,
        plan: TaskRewritePlan,
        contract_set: R4TaskContractSetV2,
        task_draft: TaskDraftV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        model_profile: str,
        prompt_version: str,
        audit: ContractAudit,
    ) -> TaskPromptSafetyRequest:
        normalized = _normalize_and_validate_plan(
            plan=plan,
            contract_set=contract_set,
            task_draft=task_draft,
        )
        validate_prompt_leakage_reference_set_identity(leakage_reference_set)
        candidate_text = _preview_candidate_text(normalized)
        candidate_draft = _preview_candidate_draft(
            source_task_draft=task_draft,
            candidate_text=candidate_text,
            audit=audit,
        )
        return TaskPromptSafetyRequestBuilder().build(
            task_draft=candidate_draft,
            leakage_reference_set=leakage_reference_set,
            model_profile=model_profile,
            prompt_version=prompt_version,
            audit=audit,
        )

    def compile(
        self,
        *,
        plan: TaskRewritePlan,
        contract_set: R4TaskContractSetV2,
        task_draft: TaskDraftV2,
        task_prompt_safety_gate: TaskPromptSafetyGateV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_storage_authorization: ProducerStorageAuthorizationV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        safety_request: TaskPromptSafetyRequest,
        safety_proposal: TaskPromptSafetyProposal | None,
        audit: ContractAudit,
    ) -> TaskRewritePlanCompilationResult:
        R4TaskContractSetCompiler().validate_current(
            contract_set=contract_set,
            task_draft=task_draft,
            task_prompt_safety_gate=task_prompt_safety_gate,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_tool_policy=contestant_tool_policy,
            producer_storage_authorization=producer_storage_authorization,
            producer_task_view=producer_task_view,
        )
        normalized = _normalize_and_validate_plan(
            plan=plan,
            contract_set=contract_set,
            task_draft=task_draft,
        )
        expected_request = self.build_safety_request(
            plan=normalized,
            contract_set=contract_set,
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            model_profile=safety_request.model_profile,
            prompt_version=safety_request.prompt_version,
            audit=safety_request.audit,
        )
        if expected_request != safety_request:
            raise TaskRewritePolicyError("rewrite preview safety request is stale or mismatched")
        candidate_draft = _preview_candidate_draft(
            source_task_draft=task_draft,
            candidate_text=_preview_candidate_text(normalized),
            audit=safety_request.audit,
        )
        safety = TaskPromptSafetyCompiler().compile(
            request=safety_request,
            proposal=safety_proposal,
            task_draft=candidate_draft,
            leakage_reference_set=leakage_reference_set,
            audit=audit,
        )
        return _plan_compilation_result(
            plan=normalized,
            plan_version=1,
            predecessor=None,
            basis_contract_set_ref=r4_task_contract_set_ref(contract_set),
            safety=safety,
            candidate_draft=candidate_draft,
            audit=audit,
        )


class TaskRewriteAdjustmentCompiler:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def build_safety_request(
        self,
        *,
        source_plan_version: TaskRewritePlanVersionV2,
        source_preview: TaskRewritePlanPreviewV2,
        source_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
        source_contract_set: R4TaskContractSetV2,
        current_application: TaskRewriteApplicationV2 | None,
        adjustments: tuple[TypedAdjustment, ...],
        task_draft: TaskDraftV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        model_profile: str,
        prompt_version: str,
        audit: ContractAudit,
    ) -> TaskPromptSafetyRequest:
        _validate_adjustment_source(
            source_plan_version=source_plan_version,
            source_preview=source_preview,
            source_preview_safety_gate=source_preview_safety_gate,
            source_contract_set=source_contract_set,
            current_application=current_application,
        )
        replacement, _ = _apply_adjustments(
            source_plan_version.plan,
            adjustments,
            audit=audit,
        )
        return TaskRewritePlanCompiler().build_safety_request(
            plan=replacement,
            contract_set=source_contract_set,
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            model_profile=model_profile,
            prompt_version=prompt_version,
            audit=audit,
        )

    def apply(
        self,
        *,
        source_plan_version: TaskRewritePlanVersionV2,
        source_preview: TaskRewritePlanPreviewV2,
        source_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
        source_contract_set: R4TaskContractSetV2,
        current_application: TaskRewriteApplicationV2 | None,
        adjustments: tuple[TypedAdjustment, ...],
        task_draft: TaskDraftV2,
        task_prompt_safety_gate: TaskPromptSafetyGateV2,
        rubric_set: RubricSetV2,
        evaluator_spec: EvaluatorSpecV2,
        reference_policy: ReferencePolicyV2,
        tool_policy: ToolPolicyV2,
        contestant_tool_policy: ContestantToolPolicyV2,
        producer_storage_authorization: ProducerStorageAuthorizationV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        safety_request: TaskPromptSafetyRequest,
        safety_proposal: TaskPromptSafetyProposal | None,
        preserved_object_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> TaskRewriteAdjustmentResult:
        R4TaskContractSetCompiler().validate_current(
            contract_set=source_contract_set,
            task_draft=task_draft,
            task_prompt_safety_gate=task_prompt_safety_gate,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_policy,
            contestant_tool_policy=contestant_tool_policy,
            producer_storage_authorization=producer_storage_authorization,
            producer_task_view=producer_task_view,
        )
        _validate_adjustment_source(
            source_plan_version=source_plan_version,
            source_preview=source_preview,
            source_preview_safety_gate=source_preview_safety_gate,
            source_contract_set=source_contract_set,
            current_application=current_application,
        )
        replacement, changed_paths = _apply_adjustments(
            source_plan_version.plan,
            adjustments,
            audit=audit,
        )
        expected_request = self.build_safety_request(
            source_plan_version=source_plan_version,
            source_preview=source_preview,
            source_preview_safety_gate=source_preview_safety_gate,
            source_contract_set=source_contract_set,
            current_application=current_application,
            adjustments=adjustments,
            task_draft=task_draft,
            leakage_reference_set=leakage_reference_set,
            model_profile=safety_request.model_profile,
            prompt_version=safety_request.prompt_version,
            audit=safety_request.audit,
        )
        if expected_request != safety_request:
            raise TaskRewritePolicyError("adjusted preview safety request is stale or mismatched")
        candidate_draft = _preview_candidate_draft(
            source_task_draft=task_draft,
            candidate_text=_preview_candidate_text(replacement),
            audit=safety_request.audit,
        )
        safety = TaskPromptSafetyCompiler().compile(
            request=safety_request,
            proposal=safety_proposal,
            task_draft=candidate_draft,
            leakage_reference_set=leakage_reference_set,
            audit=audit,
        )
        compiled = _plan_compilation_result(
            plan=replacement,
            plan_version=source_plan_version.plan_version + 1,
            predecessor=task_rewrite_plan_version_ref(source_plan_version),
            basis_contract_set_ref=r4_task_contract_set_ref(source_contract_set),
            safety=safety,
            candidate_draft=candidate_draft,
            audit=audit,
        )
        if compiled.outcome is not TaskRewritePlanCompileOutcome.PREVIEWED:
            return _adjustment_result(
                source_plan_version=source_plan_version,
                compiled=compiled,
                invalidation=None,
                audit=audit,
            )
        assert compiled.plan_version is not None
        assert compiled.preview is not None
        assert compiled.preview_safety_gate is not None
        preserved = _validate_preserved_refs(preserved_object_refs)
        invalidation = _compile_invalidation(
            source_plan_version=source_plan_version,
            replacement_plan_version=compiled.plan_version,
            source_contract_set=source_contract_set,
            changed_paths=changed_paths,
            preserved_object_refs=preserved,
            audit=audit,
        )
        return _adjustment_result(
            source_plan_version=source_plan_version,
            compiled=compiled,
            invalidation=invalidation,
            audit=audit,
        )


class TaskRewritePromptRequestBuilder:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def build(
        self,
        *,
        source_task_draft: TaskDraftV2,
        replacement_plan_version: TaskRewritePlanVersionV2,
        replacement_preview: TaskRewritePlanPreviewV2,
        replacement_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
        model_profile: str,
        prompt_version: str,
        audit: ContractAudit,
    ) -> TaskRewritePromptRequest:
        _validate_rewrite_inputs(
            source_task_draft=source_task_draft,
            replacement_plan_version=replacement_plan_version,
            replacement_preview=replacement_preview,
            replacement_preview_safety_gate=(replacement_preview_safety_gate),
        )
        projection_payload = {
            "source_task_draft_ref": _ref_payload(task_draft_ref(source_task_draft)),
            "replacement_plan_version_ref": _ref_payload(
                task_rewrite_plan_version_ref(replacement_plan_version)
            ),
            "replacement_preview_ref": _ref_payload(task_rewrite_plan_preview_ref(replacement_preview)),
            "source_visible_prompt_sha256": hashlib.sha256(
                source_task_draft.visible_prompt.encode()
            ).hexdigest(),
            "preview_sha256": replacement_preview.preview_sha256,
        }
        projection_hash = rewrite_payload_sha256(projection_payload)
        projection_ref = ObjectRef(
            object_type="task-rewrite-input",
            object_id=f"task-rewrite-input://sha256/{projection_hash}",
            object_version="v2",
            object_sha256=projection_hash,
        )
        segment = PromptBoundarySegment(
            segment_id=(f"prompt-boundary-segment://task-rewrite/{projection_hash}"),
            surface=PromptBoundarySurface.TASK_REWRITE_DATA,
            source_role=PromptBoundarySourceRole.TASK_REWRITE_CONTRACT,
            source_ref=projection_ref,
            content_sha256=projection_hash,
            untrusted_data_marker=True,
            text_preview=None,
        )
        boundary_request = PromptBoundaryEnforcementRequest(
            boundary_id=f"prompt-boundary://task-rewrite/{projection_hash}",
            segments=(segment,),
            audit=_safe_audit(
                audit,
                (
                    task_draft_ref(source_task_draft),
                    task_rewrite_plan_version_ref(replacement_plan_version),
                    task_rewrite_plan_preview_ref(replacement_preview),
                    task_rewrite_preview_safety_gate_ref(replacement_preview_safety_gate),
                    projection_ref,
                ),
            ),
        )
        try:
            boundary = PromptInjectionBoundaryEnforcer().validate_task_rewrite_data_boundary(
                TaskRewriteDataBoundaryRequest(
                    source_task_draft_ref=task_draft_ref(source_task_draft),
                    replacement_plan_version_ref=(task_rewrite_plan_version_ref(replacement_plan_version)),
                    replacement_preview_ref=(task_rewrite_plan_preview_ref(replacement_preview)),
                    rewrite_projection_ref=projection_ref,
                    boundary_request=boundary_request,
                    projection_segment_id=segment.segment_id,
                )
            )
        except PromptInjectionBoundaryPolicyError as exc:
            raise TaskRewritePolicyError("task rewrite prompt boundary is invalid") from exc
        boundary_ref = _prompt_boundary_ref(boundary)
        seed = {
            **projection_payload,
            "prompt_boundary_enforcement_ref": _ref_payload(boundary_ref),
            "source_visible_prompt": source_task_draft.visible_prompt,
            "target_capability": replacement_preview.target_capability,
            "rewrite_style": replacement_preview.rewrite_style,
            "fidelity": replacement_preview.fidelity.value,
            "operational_noise_policy": (replacement_preview.operational_noise_policy),
            "example_summaries": [item.input_summary for item in replacement_preview.examples],
            "example_treatments": [item.expected_treatment for item in replacement_preview.examples],
            "forbidden_content_rules": list(replacement_preview.forbidden_content_rules),
            "expected_capability_impact": (replacement_preview.expected_capability_impact),
            "prompt_requirement_ids": list(source_task_draft.prompt_requirement_ids),
            "model_profile": model_profile,
            "prompt_version": prompt_version,
            "policy_version": self.policy_version,
        }
        digest = rewrite_payload_sha256(seed)
        return TaskRewritePromptRequest(
            request_id=f"task-rewrite-prompt-request://sha256/{digest}",
            source_task_draft_ref=task_draft_ref(source_task_draft),
            replacement_plan_version_ref=(task_rewrite_plan_version_ref(replacement_plan_version)),
            replacement_preview_ref=task_rewrite_plan_preview_ref(replacement_preview),
            replacement_preview_safety_gate_ref=(
                task_rewrite_preview_safety_gate_ref(replacement_preview_safety_gate)
            ),
            prompt_boundary_enforcement_ref=boundary_ref,
            source_visible_prompt=source_task_draft.visible_prompt,
            target_capability=replacement_preview.target_capability,
            rewrite_style=replacement_preview.rewrite_style,
            fidelity=replacement_preview.fidelity.value,
            operational_noise_policy=(replacement_preview.operational_noise_policy),
            example_summaries=tuple(item.input_summary for item in replacement_preview.examples),
            example_treatments=tuple(item.expected_treatment for item in replacement_preview.examples),
            forbidden_content_rules=(replacement_preview.forbidden_content_rules),
            expected_capability_impact=(replacement_preview.expected_capability_impact),
            prompt_requirement_ids=(source_task_draft.prompt_requirement_ids),
            model_profile=model_profile,
            prompt_version=prompt_version,
            untrusted_data_marker=True,
            policy_version=self.policy_version,
            request_sha256=digest,
            audit=_safe_audit(
                audit,
                (
                    task_draft_ref(source_task_draft),
                    task_rewrite_plan_version_ref(replacement_plan_version),
                    task_rewrite_plan_preview_ref(replacement_preview),
                    task_rewrite_preview_safety_gate_ref(replacement_preview_safety_gate),
                    boundary_ref,
                ),
            ),
        )


class FakeTaskRewritePromptRunner:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def run(
        self,
        request: TaskRewritePromptRequest,
        *,
        fixture: FakeTaskRewritePromptFixture,
        audit: ContractAudit,
    ) -> TaskRewritePromptProposal:
        _validate_prompt_request_identity(request)
        payload = {
            "request_ref": _ref_payload(_prompt_request_ref(request)),
            "outcome": fixture.outcome.value,
            "visible_prompt": fixture.visible_prompt,
            "covered_prompt_requirement_ids": list(fixture.covered_prompt_requirement_ids),
            "unresolved_reasons": sorted(item.value for item in fixture.unresolved_reasons),
            "model_profile": request.model_profile,
            "prompt_version": request.prompt_version,
            "policy_version": self.policy_version,
        }
        digest = rewrite_payload_sha256(payload)
        return TaskRewritePromptProposal(
            proposal_id=(f"task-rewrite-prompt-proposal://sha256/{digest}"),
            request_ref=_prompt_request_ref(request),
            outcome=fixture.outcome,
            visible_prompt=fixture.visible_prompt,
            covered_prompt_requirement_ids=(fixture.covered_prompt_requirement_ids),
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            policy_version=self.policy_version,
            proposal_sha256=digest,
            audit=_safe_audit(audit, (_prompt_request_ref(request),)),
        )


class TaskRewritePromptCompiler:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def compile(
        self,
        *,
        request: TaskRewritePromptRequest,
        proposal: TaskRewritePromptProposal | None,
        source_task_draft: TaskDraftV2,
        replacement_plan_version: TaskRewritePlanVersionV2,
        replacement_preview: TaskRewritePlanPreviewV2,
        replacement_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
        audit: ContractAudit,
    ) -> TaskRewritePromptResult:
        _validate_rewrite_inputs(
            source_task_draft=source_task_draft,
            replacement_plan_version=replacement_plan_version,
            replacement_preview=replacement_preview,
            replacement_preview_safety_gate=(replacement_preview_safety_gate),
        )
        _validate_prompt_request_identity(request)
        expected = TaskRewritePromptRequestBuilder().build(
            source_task_draft=source_task_draft,
            replacement_plan_version=replacement_plan_version,
            replacement_preview=replacement_preview,
            replacement_preview_safety_gate=(replacement_preview_safety_gate),
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            audit=request.audit,
        )
        if expected != request:
            raise TaskRewritePolicyError("task rewrite prompt request is stale or mismatched")
        if replacement_preview.fidelity is RewriteFidelity.STANDARD_ENVIRONMENT:
            return _prompt_result(
                request=request,
                proposal=None,
                outcome=TaskRewritePromptOutcome.BLOCKED_CAPABILITY,
                task_draft=None,
                reasons=frozenset({TaskRewritePromptReason.ENVIRONMENT_STRATEGY_REQUIRED}),
                audit=audit,
            )
        if proposal is None:
            return _prompt_result(
                request=request,
                proposal=None,
                outcome=TaskRewritePromptOutcome.BLOCKED_CAPABILITY,
                task_draft=None,
                reasons=frozenset({TaskRewritePromptReason.MODEL_UNAVAILABLE}),
                audit=audit,
            )
        _validate_prompt_proposal_identity(proposal)
        if (
            proposal.request_ref != _prompt_request_ref(request)
            or proposal.model_profile != request.model_profile
            or proposal.prompt_version != request.prompt_version
        ):
            raise TaskRewritePolicyError("task rewrite proposal binding is stale or mismatched")
        if proposal.outcome is not TaskRewritePromptOutcome.REWRITTEN:
            return _prompt_result(
                request=request,
                proposal=proposal,
                outcome=proposal.outcome,
                task_draft=None,
                reasons=proposal.unresolved_reasons,
                audit=audit,
            )
        if set(proposal.covered_prompt_requirement_ids) != set(source_task_draft.prompt_requirement_ids):
            raise TaskRewritePolicyError("rewrite proposal has incomplete requirement coverage")
        if proposal.visible_prompt == source_task_draft.visible_prompt:
            raise TaskRewritePolicyError("rewrite proposal must materially change the visible prompt")
        assert proposal.visible_prompt is not None
        rewritten = _compile_pending_task_draft(
            source=source_task_draft,
            visible_prompt=proposal.visible_prompt,
            model_profile=proposal.model_profile,
            prompt_version=proposal.prompt_version,
            audit=audit,
        )
        return _prompt_result(
            request=request,
            proposal=proposal,
            outcome=TaskRewritePromptOutcome.REWRITTEN,
            task_draft=rewritten,
            reasons=frozenset(),
            audit=audit,
        )


class TaskRewriteApplicationCompiler:
    policy_version = TASK_REWRITE_POLICY_VERSION

    def compile(
        self,
        *,
        source_contract_set: R4TaskContractSetV2,
        source_plan_version: TaskRewritePlanVersionV2,
        replacement_plan_version: TaskRewritePlanVersionV2,
        invalidation: TaskContractInvalidationV2,
        source_task_draft: TaskDraftV2,
        source_task_prompt_safety_gate: TaskPromptSafetyGateV2,
        source_rubric_set: RubricSetV2,
        source_evaluator_spec: EvaluatorSpecV2,
        source_reference_policy: ReferencePolicyV2,
        source_tool_policy: ToolPolicyV2,
        source_contestant_tool_policy: ContestantToolPolicyV2,
        source_storage_authorization: ProducerStorageAuthorizationV2,
        source_producer_task_view: ProducerTaskViewV2,
        rewritten_pending_task_draft: TaskDraftV2,
        rewritten_passed_task_draft: TaskDraftV2,
        rewritten_task_prompt_safety_gate: TaskPromptSafetyGateV2,
        tool_catalog: ToolCapabilityCatalog,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
    ) -> TaskRewriteApplicationResult:
        R4TaskContractSetCompiler().validate_current(
            contract_set=source_contract_set,
            task_draft=source_task_draft,
            task_prompt_safety_gate=source_task_prompt_safety_gate,
            rubric_set=source_rubric_set,
            evaluator_spec=source_evaluator_spec,
            reference_policy=source_reference_policy,
            tool_policy=source_tool_policy,
            contestant_tool_policy=source_contestant_tool_policy,
            producer_storage_authorization=source_storage_authorization,
            producer_task_view=source_producer_task_view,
        )
        _validate_application_inputs(
            source_contract_set=source_contract_set,
            source_plan_version=source_plan_version,
            replacement_plan_version=replacement_plan_version,
            invalidation=invalidation,
            source_task_draft=source_task_draft,
            rewritten_pending_task_draft=rewritten_pending_task_draft,
            rewritten_passed_task_draft=rewritten_passed_task_draft,
            rewritten_task_prompt_safety_gate=(rewritten_task_prompt_safety_gate),
        )
        rubric_set = _rebind_rubric_set(
            source=source_rubric_set,
            task_draft=rewritten_passed_task_draft,
            audit=audit,
        )
        evaluator_spec = _rebind_evaluator_spec(
            source=source_evaluator_spec,
            rubric_set=rubric_set,
            audit=audit,
        )
        reference_policy = _rebind_reference_policy(
            source=source_reference_policy,
            evaluator_spec=evaluator_spec,
            audit=audit,
        )
        tool_result = ToolPolicyCompiler().compile(
            task_draft=rewritten_passed_task_draft,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            catalog=tool_catalog,
            audit=audit,
            previous_tool_policy=source_tool_policy,
        )
        if (
            tool_result.outcome is not ToolPolicyCompilationOutcome.COMPILED
            or tool_result.tool_policy is None
            or tool_result.contestant_projection is None
        ):
            raise TaskRewritePolicyError("replacement ToolPolicy did not compile")
        producer_result = ProducerTaskViewCompiler().compile(
            task_draft=rewritten_passed_task_draft,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_result.tool_policy,
            contestant_tool_policy=tool_result.contestant_projection,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            audit=audit,
            previous_producer_task_view=source_producer_task_view,
        )
        if producer_result.producer_task_view is None or producer_result.storage_authorization is None:
            raise TaskRewritePolicyError("replacement ProducerTaskView did not compile")
        replacement_contract_set = R4TaskContractSetCompiler().compile(
            task_draft=rewritten_passed_task_draft,
            task_prompt_safety_gate=rewritten_task_prompt_safety_gate,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_result.tool_policy,
            contestant_tool_policy=tool_result.contestant_projection,
            producer_storage_authorization=(producer_result.storage_authorization),
            producer_task_view=producer_result.producer_task_view,
            audit=audit,
        )
        source_refs = _contract_set_refs(source_contract_set)
        replacement_refs = _contract_set_refs(replacement_contract_set)
        if any(
            source_ref == replacement_ref
            for source_ref, replacement_ref in zip(
                source_refs,
                replacement_refs,
                strict=True,
            )
        ):
            raise TaskRewritePolicyError(
                "replacement R4 contract set must supersede every invalidated object"
            )
        application = TaskRewriteApplicationV2(
            application_id="task-rewrite-application://pending",
            source_plan_version_ref=task_rewrite_plan_version_ref(source_plan_version),
            replacement_plan_version_ref=task_rewrite_plan_version_ref(replacement_plan_version),
            invalidation_ref=task_contract_invalidation_ref(invalidation),
            source_contract_set_ref=r4_task_contract_set_ref(source_contract_set),
            replacement_contract_set_ref=r4_task_contract_set_ref(replacement_contract_set),
            candidate_state="CANDIDATE_TASK",
            policy_version=self.policy_version,
            application_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    task_rewrite_plan_version_ref(source_plan_version),
                    task_rewrite_plan_version_ref(replacement_plan_version),
                    task_contract_invalidation_ref(invalidation),
                    r4_task_contract_set_ref(source_contract_set),
                    r4_task_contract_set_ref(replacement_contract_set),
                ),
            ),
        )
        application_digest = task_rewrite_application_carried_sha256(application)
        application = application.model_copy(
            update={
                "application_id": (f"task-rewrite-application://sha256/{application_digest}"),
                "application_sha256": application_digest,
            }
        )
        result_payload = {
            "application_ref": _ref_payload(task_rewrite_application_ref(application)),
            "replacement_contract_set_ref": _ref_payload(r4_task_contract_set_ref(replacement_contract_set)),
            "policy_version": self.policy_version,
        }
        result_digest = rewrite_payload_sha256(result_payload)
        return TaskRewriteApplicationResult(
            result_id=(f"task-rewrite-application-result://sha256/{result_digest}"),
            application=application,
            replacement_contract_set=replacement_contract_set,
            rubric_set=rubric_set,
            evaluator_spec=evaluator_spec,
            reference_policy=reference_policy,
            tool_policy=tool_result.tool_policy,
            contestant_tool_policy=tool_result.contestant_projection,
            storage_authorization=producer_result.storage_authorization,
            producer_task_view=producer_result.producer_task_view,
            policy_version=self.policy_version,
            result_sha256=result_digest,
            audit=_safe_audit(
                audit,
                (
                    task_rewrite_application_ref(application),
                    r4_task_contract_set_ref(replacement_contract_set),
                ),
            ),
        )

    def validate_current(
        self,
        *,
        application_result: TaskRewriteApplicationResult,
        source_contract_set: R4TaskContractSetV2,
        source_plan_version: TaskRewritePlanVersionV2,
        replacement_plan_version: TaskRewritePlanVersionV2,
        invalidation: TaskContractInvalidationV2,
        source_task_draft: TaskDraftV2,
        source_task_prompt_safety_gate: TaskPromptSafetyGateV2,
        source_rubric_set: RubricSetV2,
        source_evaluator_spec: EvaluatorSpecV2,
        source_reference_policy: ReferencePolicyV2,
        source_tool_policy: ToolPolicyV2,
        source_contestant_tool_policy: ContestantToolPolicyV2,
        source_storage_authorization: ProducerStorageAuthorizationV2,
        source_producer_task_view: ProducerTaskViewV2,
        rewritten_pending_task_draft: TaskDraftV2,
        rewritten_passed_task_draft: TaskDraftV2,
        rewritten_task_prompt_safety_gate: TaskPromptSafetyGateV2,
        tool_catalog: ToolCapabilityCatalog,
        producer_view_result: EvidenceViewResult,
        producer_evidence_bundle: EvidenceBundle,
    ) -> None:
        rebuilt = self.compile(
            source_contract_set=source_contract_set,
            source_plan_version=source_plan_version,
            replacement_plan_version=replacement_plan_version,
            invalidation=invalidation,
            source_task_draft=source_task_draft,
            source_task_prompt_safety_gate=source_task_prompt_safety_gate,
            source_rubric_set=source_rubric_set,
            source_evaluator_spec=source_evaluator_spec,
            source_reference_policy=source_reference_policy,
            source_tool_policy=source_tool_policy,
            source_contestant_tool_policy=source_contestant_tool_policy,
            source_storage_authorization=source_storage_authorization,
            source_producer_task_view=source_producer_task_view,
            rewritten_pending_task_draft=rewritten_pending_task_draft,
            rewritten_passed_task_draft=rewritten_passed_task_draft,
            rewritten_task_prompt_safety_gate=(rewritten_task_prompt_safety_gate),
            tool_catalog=tool_catalog,
            producer_view_result=producer_view_result,
            producer_evidence_bundle=producer_evidence_bundle,
            audit=application_result.audit,
        )
        if rebuilt.model_dump(
            mode="json",
            exclude={"audit"},
        ) != application_result.model_dump(
            mode="json",
            exclude={"audit"},
        ):
            raise TaskRewritePolicyError(
                "current task rewrite application does not match authoritative inputs"
            )
        if (
            rebuilt.audit.governing_versions != application_result.audit.governing_versions
            or rebuilt.audit.input_refs != application_result.audit.input_refs
        ):
            raise TaskRewritePolicyError("current task rewrite application audit binding is stale")


def _normalize_and_validate_plan(
    *,
    plan: TaskRewritePlan,
    contract_set: R4TaskContractSetV2,
    task_draft: TaskDraftV2,
) -> TaskRewritePlan:
    _validate_contract_set_identity(contract_set)
    if contract_set.task_draft_ref != task_draft_ref(task_draft):
        raise TaskRewritePolicyError("plan contract set TaskDraft ref is stale or mismatched")
    if plan.selection_context_ref != task_draft.selection_context_ref:
        raise TaskRewritePolicyError("plan SelectionContext is stale or mismatched")
    if plan.target_capability != task_draft.evaluation_claim:
        raise TaskRewritePolicyError("plan target capability must equal the TaskDraft evaluation claim")
    example_ids = tuple(item.example_id for item in plan.examples)
    if len(example_ids) != len(set(example_ids)):
        raise TaskRewritePolicyError("rewrite plan example IDs must be unique")
    if any(item.kind.value != "REWRITE" for item in plan.examples):
        raise TaskRewritePolicyError("rewrite plan examples must use REWRITE kind")
    allowed_evidence = {
        item.evidence_ref_id for lineage in task_draft.requirement_lineage for item in lineage.evidence
    } | {
        item.evidence_ref_id
        for dependency in task_draft.attachment_dependencies
        for item in dependency.evidence
    }
    for example in plan.examples:
        if any(evidence.evidence_ref_id not in allowed_evidence for evidence in example.evidence_refs):
            raise TaskRewritePolicyError("rewrite plan example evidence is not authorized")
    rules = set(plan.forbidden_content_rules)
    if not set(task_draft.forbidden_outputs).issubset(rules):
        raise TaskRewritePolicyError("rewrite plan cannot remove TaskDraft forbidden outputs")
    normalized_rules = " ".join(rules).casefold()
    if any(rule not in normalized_rules for rule in _HARD_FORBIDDEN_RULES):
        raise TaskRewritePolicyError("rewrite plan is missing non-waivable forbidden content rules")
    normalized = plan.model_copy(
        update={
            "forbidden_content_rules": tuple(sorted(plan.forbidden_content_rules)),
        }
    )
    digest = task_rewrite_plan_carried_sha256(normalized)
    return normalized.model_copy(
        update={
            "task_rewrite_plan_id": (f"task-rewrite-plan://sha256/{digest}"),
        }
    )


def _preview_payload(plan: TaskRewritePlan) -> dict[str, object]:
    return {
        "target_capability": plan.target_capability,
        "rewrite_style": plan.rewrite_style,
        "fidelity": plan.fidelity.value,
        "operational_noise_policy": plan.operational_noise_policy,
        "examples": [
            {
                "example_id": item.example_id,
                "kind": item.kind.value,
                "input_summary": item.input_summary,
                "expected_treatment": item.expected_treatment,
            }
            for item in plan.examples
        ],
        "forbidden_content_rules": list(plan.forbidden_content_rules),
        "expected_capability_impact": plan.expected_capability_impact,
    }


def _preview_candidate_text(plan: TaskRewritePlan) -> str:
    return json.dumps(
        _preview_payload(plan),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _preview_candidate_draft(
    *,
    source_task_draft: TaskDraftV2,
    candidate_text: str,
    audit: ContractAudit,
) -> TaskDraftV2:
    return _compile_pending_task_draft(
        source=source_task_draft,
        visible_prompt=candidate_text,
        model_profile=source_task_draft.model_profile,
        prompt_version=source_task_draft.prompt_version,
        audit=audit,
    )


def _plan_compilation_result(
    *,
    plan: TaskRewritePlan,
    plan_version: int,
    predecessor: ObjectRef | None,
    basis_contract_set_ref: ObjectRef,
    safety: TaskPromptSafetyResult,
    candidate_draft: TaskDraftV2,
    audit: ContractAudit,
) -> TaskRewritePlanCompilationResult:
    if safety.outcome is not TaskPromptSafetyOutcome.PASSED:
        outcome, reasons = _map_safety_outcome(safety.outcome, safety)
        blocked_gate = (
            _preview_gate_from_safety(
                safety=safety,
                candidate_draft=candidate_draft,
                audit=audit,
            )
            if safety.outcome is TaskPromptSafetyOutcome.BLOCKED
            else None
        )
        return _plan_result(
            outcome=outcome,
            plan_version=None,
            gate=blocked_gate,
            preview=None,
            reasons=reasons,
            audit=audit,
        )
    assert safety.task_prompt_safety_gate is not None
    version = TaskRewritePlanVersionV2(
        task_rewrite_plan_version_id=("task-rewrite-plan-version://pending"),
        plan_version=plan_version,
        supersedes_task_rewrite_plan_version_ref=predecessor,
        plan=plan,
        basis_contract_set_ref=basis_contract_set_ref,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        task_rewrite_plan_version_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            tuple(ref for ref in (predecessor, basis_contract_set_ref) if ref is not None),
        ),
    )
    version_digest = task_rewrite_plan_version_carried_sha256(version)
    version = version.model_copy(
        update={
            "task_rewrite_plan_version_id": (f"task-rewrite-plan-version://sha256/{version_digest}"),
            "task_rewrite_plan_version_sha256": version_digest,
        }
    )
    gate = _preview_gate_from_safety(
        safety=safety,
        candidate_draft=candidate_draft,
        audit=audit,
    )
    preview = TaskRewritePlanPreviewV2(
        preview_id="task-rewrite-plan-preview://pending",
        source_plan_version_ref=task_rewrite_plan_version_ref(version),
        target_capability=plan.target_capability,
        rewrite_style=plan.rewrite_style,
        fidelity=plan.fidelity,
        operational_noise_policy=plan.operational_noise_policy,
        examples=tuple(
            TaskRewriteExamplePreviewV2(
                example_id=item.example_id,
                kind="REWRITE",
                input_summary=item.input_summary,
                expected_treatment=item.expected_treatment,
            )
            for item in plan.examples
        ),
        forbidden_content_rules=plan.forbidden_content_rules,
        expected_capability_impact=plan.expected_capability_impact,
        safety_gate_ref=task_rewrite_preview_safety_gate_ref(gate),
        projection_policy_version=TASK_REWRITE_PREVIEW_POLICY_VERSION,
        preview_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                task_rewrite_plan_version_ref(version),
                task_rewrite_preview_safety_gate_ref(gate),
            ),
        ),
    )
    preview_digest = task_rewrite_plan_preview_carried_sha256(preview)
    preview = preview.model_copy(
        update={
            "preview_id": (f"task-rewrite-plan-preview://sha256/{preview_digest}"),
            "preview_sha256": preview_digest,
        }
    )
    return _plan_result(
        outcome=TaskRewritePlanCompileOutcome.PREVIEWED,
        plan_version=version,
        gate=gate,
        preview=preview,
        reasons=frozenset(),
        audit=audit,
    )


def _preview_gate_from_safety(
    *,
    safety: TaskPromptSafetyResult,
    candidate_draft: TaskDraftV2,
    audit: ContractAudit,
) -> TaskRewritePreviewSafetyGateV2:
    if safety.task_prompt_safety_gate is None:
        raise TaskRewritePolicyError("terminal preview safety result is missing its gate")
    generic_gate = safety.task_prompt_safety_gate
    candidate_hash = hashlib.sha256(candidate_draft.visible_prompt.encode()).hexdigest()
    candidate_ref = ObjectRef(
        object_type="task-rewrite-preview-candidate",
        object_id=(f"task-rewrite-preview-candidate://sha256/{candidate_hash}"),
        object_version="v2",
        object_sha256=candidate_hash,
    )
    gate = TaskRewritePreviewSafetyGateV2(
        gate_id="task-rewrite-preview-safety-gate://pending",
        preview_candidate_ref=candidate_ref,
        leakage_reference_set_ref=(generic_gate.leakage_reference_set_ref),
        preview_content_sha256=candidate_hash,
        status=generic_gate.status,
        checks=generic_gate.checks,
        findings=generic_gate.findings,
        semantic_assessment_ref=generic_gate.semantic_assessment_ref,
        model_profile=generic_gate.model_profile,
        prompt_version=generic_gate.prompt_version,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        gate_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                candidate_ref,
                generic_gate.leakage_reference_set_ref,
                *(
                    (generic_gate.semantic_assessment_ref,)
                    if generic_gate.semantic_assessment_ref is not None
                    else ()
                ),
            ),
        ),
    )
    gate_digest = task_rewrite_preview_safety_gate_carried_sha256(gate)
    gate = gate.model_copy(
        update={
            "gate_id": (f"task-rewrite-preview-safety-gate://sha256/{gate_digest}"),
            "gate_sha256": gate_digest,
        }
    )
    return gate


def _plan_result(
    *,
    outcome: TaskRewritePlanCompileOutcome,
    plan_version: TaskRewritePlanVersionV2 | None,
    gate: TaskRewritePreviewSafetyGateV2 | None,
    preview: TaskRewritePlanPreviewV2 | None,
    reasons: frozenset[TaskRewritePlanCompileReason],
    audit: ContractAudit,
) -> TaskRewritePlanCompilationResult:
    payload = {
        "outcome": outcome.value,
        "plan_version_ref": _maybe_ref(
            task_rewrite_plan_version_ref(plan_version) if plan_version is not None else None
        ),
        "preview_safety_gate_ref": _maybe_ref(
            task_rewrite_preview_safety_gate_ref(gate) if gate is not None else None
        ),
        "preview_ref": _maybe_ref(task_rewrite_plan_preview_ref(preview) if preview is not None else None),
        "unresolved_reasons": sorted(item.value for item in reasons),
        "policy_version": TASK_REWRITE_POLICY_VERSION,
    }
    digest = rewrite_payload_sha256(payload)
    refs = tuple(
        ref
        for ref in (
            (task_rewrite_plan_version_ref(plan_version) if plan_version is not None else None),
            (task_rewrite_preview_safety_gate_ref(gate) if gate is not None else None),
            (task_rewrite_plan_preview_ref(preview) if preview is not None else None),
        )
        if ref is not None
    )
    return TaskRewritePlanCompilationResult(
        result_id=f"task-rewrite-plan-result://sha256/{digest}",
        outcome=outcome,
        plan_version=plan_version,
        preview_safety_gate=gate,
        preview=preview,
        unresolved_reasons=reasons,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _map_safety_outcome(
    outcome: TaskPromptSafetyOutcome,
    safety: TaskPromptSafetyResult,
) -> tuple[
    TaskRewritePlanCompileOutcome,
    frozenset[TaskRewritePlanCompileReason],
]:
    if outcome is TaskPromptSafetyOutcome.BLOCKED:
        return (
            TaskRewritePlanCompileOutcome.BLOCKED_SAFETY,
            frozenset({TaskRewritePlanCompileReason.UNSAFE_PREVIEW}),
        )
    reasons: set[TaskRewritePlanCompileReason] = set()
    for reason in safety.unresolved_reasons:
        if reason is TaskPromptSafetyReason.INCOMPLETE_REFERENCE_COVERAGE:
            reasons.add(TaskRewritePlanCompileReason.INCOMPLETE_SAFETY_COVERAGE)
        elif reason is TaskPromptSafetyReason.MODEL_UNAVAILABLE:
            reasons.add(TaskRewritePlanCompileReason.MODEL_UNAVAILABLE)
        else:
            reasons.add(TaskRewritePlanCompileReason.MISSING_SEMANTIC_ASSESSMENT)
    if not reasons:
        reasons.add(TaskRewritePlanCompileReason.MISSING_SEMANTIC_ASSESSMENT)
    return (
        TaskRewritePlanCompileOutcome.BLOCKED_CAPABILITY,
        frozenset(reasons),
    )


def _apply_adjustments(
    source: TaskRewritePlan,
    adjustments: tuple[TypedAdjustment, ...],
    *,
    audit: ContractAudit,
) -> tuple[TaskRewritePlan, tuple[str, ...]]:
    if not adjustments:
        raise TaskRewritePolicyError("adjustment set cannot be empty")
    paths = tuple(item.target_path for item in adjustments)
    if len(paths) != len(set(paths)):
        raise TaskRewritePolicyError("adjustment target paths must be unique")
    updates: dict[str, object] = {}
    examples = list(source.examples)
    rules = list(source.forbidden_content_rules)
    for adjustment in adjustments:
        path = adjustment.target_path
        value = adjustment.value
        if path in {
            "rewrite_style",
            "operational_noise_policy",
            "expected_capability_impact",
        }:
            if adjustment.operation not in {"SET", "REPLACE"}:
                raise TaskRewritePolicyError(f"{path} supports SET or REPLACE only")
            if not isinstance(value, str) or not value.strip():
                raise TaskRewritePolicyError(f"{path} requires a non-empty string")
            if value == getattr(source, path):
                raise TaskRewritePolicyError(f"{path} adjustment is a no-op")
            updates[path] = value
            continue
        if path == "fidelity":
            if adjustment.operation not in {"SET", "REPLACE"}:
                raise TaskRewritePolicyError("fidelity supports SET or REPLACE only")
            if not isinstance(value, str):
                raise TaskRewritePolicyError("fidelity requires an enum string")
            try:
                fidelity = RewriteFidelity(value)
            except ValueError as exc:
                raise TaskRewritePolicyError("fidelity value is invalid") from exc
            if fidelity is source.fidelity:
                raise TaskRewritePolicyError("fidelity adjustment is a no-op")
            updates[path] = fidelity
            continue
        if path == "forbidden_content_rules":
            if adjustment.operation != "ADD":
                raise TaskRewritePolicyError("forbidden content rules support ADD only")
            if not isinstance(value, str) or not value.strip():
                raise TaskRewritePolicyError("forbidden content rule requires a non-empty string")
            if value in rules:
                raise TaskRewritePolicyError("forbidden content rule already exists")
            rules.append(value)
            continue
        match = _EXAMPLE_PATH.fullmatch(path)
        if match is not None:
            if adjustment.operation not in {"SET", "REPLACE"}:
                raise TaskRewritePolicyError("example fields support SET or REPLACE only")
            if not isinstance(value, str) or not value.strip():
                raise TaskRewritePolicyError("example adjustment requires a non-empty string")
            selector, field = match.groups()
            index = next(
                (
                    position
                    for position, example in enumerate(examples)
                    if _example_selector(example.example_id) == selector
                ),
                None,
            )
            if index is None:
                raise TaskRewritePolicyError("example adjustment selector is unknown")
            if getattr(examples[index], field) == value:
                raise TaskRewritePolicyError("example adjustment is a no-op")
            examples[index] = examples[index].model_copy(update={field: value})
            continue
        raise TaskRewritePolicyError(f"adjustment target path is not allowed: {path}")
    updates["examples"] = tuple(examples)
    updates["forbidden_content_rules"] = tuple(sorted(rules))
    replacement = source.model_copy(
        update={
            **updates,
            "task_rewrite_plan_id": "task-rewrite-plan://pending",
            "audit": audit,
        }
    )
    digest = task_rewrite_plan_carried_sha256(replacement)
    replacement = replacement.model_copy(
        update={"task_rewrite_plan_id": (f"task-rewrite-plan://sha256/{digest}")}
    )
    return replacement, tuple(sorted(paths))


def _example_selector(example_id: str) -> str:
    return hashlib.sha256(example_id.encode()).hexdigest()


def _validate_adjustment_source(
    *,
    source_plan_version: TaskRewritePlanVersionV2,
    source_preview: TaskRewritePlanPreviewV2,
    source_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
    source_contract_set: R4TaskContractSetV2,
    current_application: TaskRewriteApplicationV2 | None,
) -> None:
    _validate_plan_version_identity(source_plan_version)
    _validate_preview_gate_identity(source_preview_safety_gate)
    _validate_preview_identity(source_preview)
    _validate_contract_set_identity(source_contract_set)
    if source_preview.source_plan_version_ref != task_rewrite_plan_version_ref(
        source_plan_version
    ) or source_preview.safety_gate_ref != task_rewrite_preview_safety_gate_ref(source_preview_safety_gate):
        raise TaskRewritePolicyError("adjustment source plan, preview, or contract set is stale")
    if source_plan_version.plan_version > 1:
        if (
            current_application is None
            or current_application.replacement_plan_version_ref
            != task_rewrite_plan_version_ref(source_plan_version)
            or current_application.replacement_contract_set_ref
            != r4_task_contract_set_ref(source_contract_set)
        ):
            raise TaskRewritePolicyError("later adjustment requires the current application binding")
    else:
        if source_plan_version.basis_contract_set_ref != r4_task_contract_set_ref(source_contract_set):
            raise TaskRewritePolicyError("initial plan basis contract set is stale")
        if current_application is not None:
            raise TaskRewritePolicyError("initial plan adjustment cannot carry a current application")


def _compile_invalidation(
    *,
    source_plan_version: TaskRewritePlanVersionV2,
    replacement_plan_version: TaskRewritePlanVersionV2,
    source_contract_set: R4TaskContractSetV2,
    changed_paths: tuple[str, ...],
    preserved_object_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> TaskContractInvalidationV2:
    invalidation = TaskContractInvalidationV2(
        invalidation_id="task-contract-invalidation://pending",
        source_plan_version_ref=task_rewrite_plan_version_ref(source_plan_version),
        replacement_plan_version_ref=task_rewrite_plan_version_ref(replacement_plan_version),
        source_contract_set_ref=r4_task_contract_set_ref(source_contract_set),
        changed_paths=changed_paths,
        invalidated_object_refs=(
            source_contract_set.task_draft_ref,
            source_contract_set.task_prompt_safety_gate_ref,
            source_contract_set.rubric_set_ref,
            source_contract_set.evaluator_spec_ref,
            source_contract_set.reference_policy_ref,
            source_contract_set.tool_policy_ref,
            source_contract_set.contestant_tool_policy_ref,
            source_contract_set.producer_storage_authorization_ref,
            source_contract_set.producer_task_view_ref,
        ),
        preserved_object_refs=preserved_object_refs,
        required_rebuild_stages=tuple(TaskRewriteRebuildStageV2),
        hard_gate_revalidation_required=True,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        invalidation_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                task_rewrite_plan_version_ref(source_plan_version),
                task_rewrite_plan_version_ref(replacement_plan_version),
                r4_task_contract_set_ref(source_contract_set),
                *preserved_object_refs,
            ),
        ),
    )
    digest = task_contract_invalidation_carried_sha256(invalidation)
    return invalidation.model_copy(
        update={
            "invalidation_id": (f"task-contract-invalidation://sha256/{digest}"),
            "invalidation_sha256": digest,
        }
    )


def _adjustment_result(
    *,
    source_plan_version: TaskRewritePlanVersionV2,
    compiled: TaskRewritePlanCompilationResult,
    invalidation: TaskContractInvalidationV2 | None,
    audit: ContractAudit,
) -> TaskRewriteAdjustmentResult:
    payload = {
        "source_plan_version_ref": _ref_payload(task_rewrite_plan_version_ref(source_plan_version)),
        "compiled_result_ref": {
            "object_type": "task-rewrite-plan-result",
            "object_id": compiled.result_id,
            "object_version": TASK_REWRITE_POLICY_VERSION,
            "object_sha256": compiled.result_sha256,
        },
        "invalidation_ref": _maybe_ref(
            task_contract_invalidation_ref(invalidation) if invalidation is not None else None
        ),
        "policy_version": TASK_REWRITE_POLICY_VERSION,
    }
    digest = rewrite_payload_sha256(payload)
    return TaskRewriteAdjustmentResult(
        result_id=(f"task-rewrite-adjustment-result://sha256/{digest}"),
        outcome=compiled.outcome,
        source_plan_version=source_plan_version,
        replacement_plan_version=compiled.plan_version,
        replacement_preview_safety_gate=(compiled.preview_safety_gate),
        replacement_preview=compiled.preview,
        invalidation=invalidation,
        unresolved_reasons=compiled.unresolved_reasons,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    task_rewrite_plan_version_ref(source_plan_version),
                    (
                        task_rewrite_plan_version_ref(compiled.plan_version)
                        if compiled.plan_version is not None
                        else None
                    ),
                    (task_contract_invalidation_ref(invalidation) if invalidation is not None else None),
                )
                if ref is not None
            ),
        ),
    )


def _validate_preserved_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    if not refs:
        raise TaskRewritePolicyError("preserved object inventory cannot be empty")
    normalized = tuple(sorted(set(refs), key=_ref_key))
    if len(normalized) != len(refs):
        raise TaskRewritePolicyError("preserved object refs must be unique")
    if any(ref.object_type not in _ALLOWED_PRESERVED_TYPES for ref in refs):
        raise TaskRewritePolicyError("preserved object inventory contains an invalid type")
    return normalized


def _validate_rewrite_inputs(
    *,
    source_task_draft: TaskDraftV2,
    replacement_plan_version: TaskRewritePlanVersionV2,
    replacement_preview: TaskRewritePlanPreviewV2,
    replacement_preview_safety_gate: TaskRewritePreviewSafetyGateV2,
) -> None:
    _validate_task_draft_identity(source_task_draft)
    _validate_plan_version_identity(replacement_plan_version)
    _validate_preview_gate_identity(replacement_preview_safety_gate)
    _validate_preview_identity(replacement_preview)
    if source_task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED:
        raise TaskRewritePolicyError("task rewrite requires a PASSED source TaskDraft")
    if (
        replacement_preview.source_plan_version_ref != task_rewrite_plan_version_ref(replacement_plan_version)
        or replacement_preview.safety_gate_ref
        != task_rewrite_preview_safety_gate_ref(replacement_preview_safety_gate)
        or replacement_preview_safety_gate.status is not TaskPromptSafetyGateStatusV2.PASSED
    ):
        raise TaskRewritePolicyError("replacement plan preview safety binding is invalid")


def _compile_pending_task_draft(
    *,
    source: TaskDraftV2,
    visible_prompt: str,
    model_profile: str,
    prompt_version: str,
    audit: ContractAudit,
) -> TaskDraftV2:
    source_ref = task_draft_ref(source)
    payload = {
        "task_version": source.task_version + 1,
        "supersedes_task_draft_ref": _ref_payload(source_ref),
        "selection_context_ref": _ref_payload(source.selection_context_ref),
        "task_episode_refs": [_ref_payload(ref) for ref in source.task_episode_refs],
        "visible_prompt": visible_prompt,
        "task_intent": source.task_intent,
        "evaluation_claim": source.evaluation_claim,
        "required_capabilities": list(source.required_capabilities),
        "allowed_tools": list(source.allowed_tools),
        "forbidden_outputs": list(source.forbidden_outputs),
        "attachment_dependencies": [
            item.model_dump(mode="json", exclude_none=False) for item in source.attachment_dependencies
        ],
        "requirement_lineage": [
            item.model_dump(mode="json", exclude_none=False) for item in source.requirement_lineage
        ],
        "prompt_requirement_ids": list(source.prompt_requirement_ids),
        "uncertainties": list(source.uncertainties),
        "prompt_safety_status": (TaskDraftPromptSafetyStatusV2.PENDING.value),
        "prompt_safety_gate_ref": None,
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": TASK_REWRITE_POLICY_VERSION,
    }
    digest = task_draft_payload_sha256(payload)
    return TaskDraftV2(
        task_draft_id=f"task-draft://sha256/{digest}",
        task_version=source.task_version + 1,
        supersedes_task_draft_ref=source_ref,
        selection_context_ref=source.selection_context_ref,
        task_episode_refs=source.task_episode_refs,
        visible_prompt=visible_prompt,
        task_intent=source.task_intent,
        evaluation_claim=source.evaluation_claim,
        required_capabilities=source.required_capabilities,
        allowed_tools=source.allowed_tools,
        forbidden_outputs=source.forbidden_outputs,
        attachment_dependencies=source.attachment_dependencies,
        requirement_lineage=source.requirement_lineage,
        prompt_requirement_ids=source.prompt_requirement_ids,
        uncertainties=source.uncertainties,
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PENDING,
        prompt_safety_gate_ref=None,
        model_profile=model_profile,
        prompt_version=prompt_version,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        task_draft_sha256=digest,
        audit=_safe_audit(audit, (source_ref,)),
    )


def _prompt_request_seed(request: TaskRewritePromptRequest) -> dict[str, object]:
    return {
        "source_task_draft_ref": _ref_payload(request.source_task_draft_ref),
        "replacement_plan_version_ref": _ref_payload(request.replacement_plan_version_ref),
        "replacement_preview_ref": _ref_payload(request.replacement_preview_ref),
        "source_visible_prompt_sha256": hashlib.sha256(request.source_visible_prompt.encode()).hexdigest(),
        "preview_sha256": request.replacement_preview_ref.object_sha256,
        "prompt_boundary_enforcement_ref": _ref_payload(request.prompt_boundary_enforcement_ref),
        "source_visible_prompt": request.source_visible_prompt,
        "target_capability": request.target_capability,
        "rewrite_style": request.rewrite_style,
        "fidelity": request.fidelity,
        "operational_noise_policy": request.operational_noise_policy,
        "example_summaries": list(request.example_summaries),
        "example_treatments": list(request.example_treatments),
        "forbidden_content_rules": list(request.forbidden_content_rules),
        "expected_capability_impact": (request.expected_capability_impact),
        "prompt_requirement_ids": list(request.prompt_requirement_ids),
        "model_profile": request.model_profile,
        "prompt_version": request.prompt_version,
        "policy_version": request.policy_version,
    }


def _validate_prompt_request_identity(
    request: TaskRewritePromptRequest,
) -> None:
    digest = rewrite_payload_sha256(_prompt_request_seed(request))
    if (
        request.request_sha256 != digest
        or request.request_id != f"task-rewrite-prompt-request://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("task rewrite prompt request identity is stale")


def _prompt_request_ref(
    request: TaskRewritePromptRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-prompt-request",
        object_id=request.request_id,
        object_version=TASK_REWRITE_POLICY_VERSION,
        object_sha256=request.request_sha256,
    )


def _validate_prompt_proposal_identity(
    proposal: TaskRewritePromptProposal,
) -> None:
    payload = {
        "request_ref": _ref_payload(proposal.request_ref),
        "outcome": proposal.outcome.value,
        "visible_prompt": proposal.visible_prompt,
        "covered_prompt_requirement_ids": list(proposal.covered_prompt_requirement_ids),
        "unresolved_reasons": sorted(item.value for item in proposal.unresolved_reasons),
        "model_profile": proposal.model_profile,
        "prompt_version": proposal.prompt_version,
        "policy_version": proposal.policy_version,
    }
    digest = rewrite_payload_sha256(payload)
    if (
        proposal.proposal_sha256 != digest
        or proposal.proposal_id != f"task-rewrite-prompt-proposal://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("task rewrite prompt proposal identity is stale")


def _prompt_proposal_ref(
    proposal: TaskRewritePromptProposal,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-rewrite-prompt-proposal",
        object_id=proposal.proposal_id,
        object_version=TASK_REWRITE_POLICY_VERSION,
        object_sha256=proposal.proposal_sha256,
    )


def _prompt_result(
    *,
    request: TaskRewritePromptRequest,
    proposal: TaskRewritePromptProposal | None,
    outcome: TaskRewritePromptOutcome,
    task_draft: TaskDraftV2 | None,
    reasons: frozenset[TaskRewritePromptReason],
    audit: ContractAudit,
) -> TaskRewritePromptResult:
    proposal_ref = _prompt_proposal_ref(proposal) if proposal is not None else None
    payload = {
        "request_ref": _ref_payload(_prompt_request_ref(request)),
        "proposal_ref": _maybe_ref(proposal_ref),
        "outcome": outcome.value,
        "task_draft_ref": _maybe_ref(task_draft_ref(task_draft) if task_draft is not None else None),
        "unresolved_reasons": sorted(item.value for item in reasons),
        "policy_version": TASK_REWRITE_POLICY_VERSION,
    }
    digest = rewrite_payload_sha256(payload)
    return TaskRewritePromptResult(
        result_id=f"task-rewrite-prompt-result://sha256/{digest}",
        request_ref=_prompt_request_ref(request),
        proposal_ref=proposal_ref,
        outcome=outcome,
        task_draft=task_draft,
        unresolved_reasons=reasons,
        policy_version=TASK_REWRITE_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    _prompt_request_ref(request),
                    proposal_ref,
                    (task_draft_ref(task_draft) if task_draft is not None else None),
                )
                if ref is not None
            ),
        ),
    )


def _validate_application_inputs(
    *,
    source_contract_set: R4TaskContractSetV2,
    source_plan_version: TaskRewritePlanVersionV2,
    replacement_plan_version: TaskRewritePlanVersionV2,
    invalidation: TaskContractInvalidationV2,
    source_task_draft: TaskDraftV2,
    rewritten_pending_task_draft: TaskDraftV2,
    rewritten_passed_task_draft: TaskDraftV2,
    rewritten_task_prompt_safety_gate: TaskPromptSafetyGateV2,
) -> None:
    _validate_plan_version_identity(source_plan_version)
    _validate_plan_version_identity(replacement_plan_version)
    _validate_invalidation_identity(invalidation)
    validate_task_prompt_safety_gate_identity(rewritten_task_prompt_safety_gate)
    if (
        replacement_plan_version.plan_version != source_plan_version.plan_version + 1
        or replacement_plan_version.supersedes_task_rewrite_plan_version_ref
        != task_rewrite_plan_version_ref(source_plan_version)
    ):
        raise TaskRewritePolicyError("replacement plan version or predecessor is invalid")
    if (
        invalidation.source_plan_version_ref != task_rewrite_plan_version_ref(source_plan_version)
        or invalidation.replacement_plan_version_ref
        != task_rewrite_plan_version_ref(replacement_plan_version)
        or invalidation.source_contract_set_ref != r4_task_contract_set_ref(source_contract_set)
    ):
        raise TaskRewritePolicyError("invalidation plan or contract-set binding is stale")
    if invalidation.invalidated_object_refs != _contract_set_refs(source_contract_set):
        raise TaskRewritePolicyError("invalidation object inventory is stale or incomplete")
    if (
        rewritten_pending_task_draft.task_version != source_task_draft.task_version + 1
        or rewritten_pending_task_draft.supersedes_task_draft_ref != task_draft_ref(source_task_draft)
        or rewritten_pending_task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PENDING
    ):
        raise TaskRewritePolicyError("rewritten pending TaskDraft predecessor is invalid")
    if (
        rewritten_passed_task_draft.task_version != rewritten_pending_task_draft.task_version + 1
        or rewritten_passed_task_draft.supersedes_task_draft_ref
        != task_draft_ref(rewritten_pending_task_draft)
        or rewritten_passed_task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
        or rewritten_passed_task_draft.prompt_safety_gate_ref
        != task_prompt_safety_gate_ref(rewritten_task_prompt_safety_gate)
        or rewritten_task_prompt_safety_gate.source_task_draft_ref
        != task_draft_ref(rewritten_pending_task_draft)
    ):
        raise TaskRewritePolicyError("rewritten PASSED TaskDraft safety chain is invalid")


def _contract_set_refs(
    contract_set: R4TaskContractSetV2,
) -> tuple[ObjectRef, ...]:
    return (
        contract_set.task_draft_ref,
        contract_set.task_prompt_safety_gate_ref,
        contract_set.rubric_set_ref,
        contract_set.evaluator_spec_ref,
        contract_set.reference_policy_ref,
        contract_set.tool_policy_ref,
        contract_set.contestant_tool_policy_ref,
        contract_set.producer_storage_authorization_ref,
        contract_set.producer_task_view_ref,
    )


def _rebind_rubric_set(
    *,
    source: RubricSetV2,
    task_draft: TaskDraftV2,
    audit: ContractAudit,
) -> RubricSetV2:
    rubric = source.model_copy(
        update={
            "rubric_set_id": "rubric-set://pending",
            "rubric_version": source.rubric_version + 1,
            "supersedes_rubric_set_ref": rubric_set_ref(source),
            "task_draft_ref": task_draft_ref(task_draft),
            "rubric_set_sha256": "0" * 64,
            "audit": _safe_audit(
                audit,
                (rubric_set_ref(source), task_draft_ref(task_draft)),
            ),
        }
    )
    digest = rubric_set_carried_sha256(rubric)
    return rubric.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _rebind_evaluator_spec(
    *,
    source: EvaluatorSpecV2,
    rubric_set: RubricSetV2,
    audit: ContractAudit,
) -> EvaluatorSpecV2:
    spec = source.model_copy(
        update={
            "evaluator_spec_id": "evaluator-spec://pending",
            "evaluator_spec_version": source.evaluator_spec_version + 1,
            "supersedes_evaluator_spec_ref": evaluator_spec_ref(source),
            "rubric_set_ref": rubric_set_ref(rubric_set),
            "evaluator_spec_sha256": "0" * 64,
            "audit": _safe_audit(
                audit,
                (evaluator_spec_ref(source), rubric_set_ref(rubric_set)),
            ),
        }
    )
    digest = evaluator_spec_carried_sha256(spec)
    return spec.model_copy(
        update={
            "evaluator_spec_id": f"evaluator-spec://sha256/{digest}",
            "evaluator_spec_sha256": digest,
        }
    )


def _rebind_reference_policy(
    *,
    source: ReferencePolicyV2,
    evaluator_spec: EvaluatorSpecV2,
    audit: ContractAudit,
) -> ReferencePolicyV2:
    policy = source.model_copy(
        update={
            "reference_policy_id": "reference-policy://pending",
            "reference_policy_version": (source.reference_policy_version + 1),
            "supersedes_reference_policy_ref": (reference_policy_ref(source)),
            "evaluator_spec_ref": evaluator_spec_ref(evaluator_spec),
            "reference_policy_sha256": "0" * 64,
            "audit": _safe_audit(
                audit,
                (
                    reference_policy_ref(source),
                    evaluator_spec_ref(evaluator_spec),
                ),
            ),
        }
    )
    digest = reference_policy_carried_sha256(policy)
    return policy.model_copy(
        update={
            "reference_policy_id": (f"reference-policy://sha256/{digest}"),
            "reference_policy_sha256": digest,
        }
    )


def _validate_contract_set_identity(
    contract_set: R4TaskContractSetV2,
) -> None:
    digest = r4_task_contract_set_carried_sha256(contract_set)
    if (
        contract_set.contract_set_sha256 != digest
        or contract_set.contract_set_id != f"r4-task-contract-set://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("R4 contract set identity is stale or invalid")


def _validate_task_draft_identity(task_draft: TaskDraftV2) -> None:
    digest = task_draft_carried_sha256(task_draft)
    if task_draft.task_draft_sha256 != digest or task_draft.task_draft_id != f"task-draft://sha256/{digest}":
        raise TaskRewritePolicyError("TaskDraft identity is stale or invalid")


def _validate_plan_version_identity(
    version: TaskRewritePlanVersionV2,
) -> None:
    plan_digest = task_rewrite_plan_carried_sha256(version.plan)
    if version.plan.task_rewrite_plan_id != f"task-rewrite-plan://sha256/{plan_digest}":
        raise TaskRewritePolicyError("embedded task rewrite plan identity is stale")
    digest = task_rewrite_plan_version_carried_sha256(version)
    if (
        version.task_rewrite_plan_version_sha256 != digest
        or version.task_rewrite_plan_version_id != f"task-rewrite-plan-version://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("task rewrite plan version identity is stale")


def _validate_preview_gate_identity(
    gate: TaskRewritePreviewSafetyGateV2,
) -> None:
    digest = task_rewrite_preview_safety_gate_carried_sha256(gate)
    if gate.gate_sha256 != digest or gate.gate_id != f"task-rewrite-preview-safety-gate://sha256/{digest}":
        raise TaskRewritePolicyError("rewrite preview safety gate identity is stale")


def _validate_preview_identity(
    preview: TaskRewritePlanPreviewV2,
) -> None:
    digest = task_rewrite_plan_preview_carried_sha256(preview)
    if (
        preview.preview_sha256 != digest
        or preview.preview_id != f"task-rewrite-plan-preview://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("rewrite preview identity is stale")


def _validate_invalidation_identity(
    invalidation: TaskContractInvalidationV2,
) -> None:
    digest = task_contract_invalidation_carried_sha256(invalidation)
    if (
        invalidation.invalidation_sha256 != digest
        or invalidation.invalidation_id != f"task-contract-invalidation://sha256/{digest}"
    ):
        raise TaskRewritePolicyError("task contract invalidation identity is stale")


def _prompt_boundary_ref(
    result: PromptBoundaryEnforcementResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-boundary-enforcement",
        object_id=result.enforcement_id,
        object_version=result.policy_version,
        object_sha256=result.enforcement_sha256,
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _same_except_audit_actor_time(
    left: R4TaskContractSetV2,
    right: R4TaskContractSetV2,
) -> bool:
    if left.model_dump(mode="json", exclude={"audit"}) != right.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return (
        left.audit.governing_versions == right.audit.governing_versions
        and left.audit.input_refs == right.audit.input_refs
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)
