from __future__ import annotations

import hashlib
import json

from eval_factory.contracts.core import ContractAudit, EvidenceRef, Identifier, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.task_v2 import (
    RubricCriterionV2,
    RubricJudgedObjectKindV2,
    RubricJudgedObjectV2,
    RubricReachabilityV2,
    RubricSetV2,
    RubricSourceModeV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
)
from eval_factory.provenance.injection import (
    CandidateTaskContractBoundaryRequest,
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
)
from eval_factory.task_authoring.rubric_models import (
    RUBRIC_AUTHORING_POLICY_VERSION,
    FakeRubricGenerationFixture,
    ImportedRubricDefinition,
    ImportedRubricStatus,
    RubricAttachmentDependencyView,
    RubricAuthoringOutcome,
    RubricAuthoringPolicyError,
    RubricAuthoringReason,
    RubricAuthoringResult,
    RubricCandidateProposal,
    RubricCandidateRequest,
    RubricCriterionSelection,
    RubricTaskRequirementView,
    imported_rubric_definition_sha256,
)


class RubricCandidateRequestBuilder:
    policy_version = RUBRIC_AUTHORING_POLICY_VERSION

    def build(
        self,
        *,
        task_draft: TaskDraftV2,
        allowed_evaluator_bindings: tuple[Identifier, ...],
        audit: ContractAudit,
    ) -> RubricCandidateRequest:
        _validate_task_draft(task_draft)
        evaluator_bindings = _sorted_unique(
            "allowed evaluator binding",
            allowed_evaluator_bindings,
        )
        if not evaluator_bindings:
            raise RubricAuthoringPolicyError("at least one allowed evaluator binding is required")
        task_ref = task_draft_ref(task_draft)
        prompt_requirements = _prompt_requirement_views(task_draft)
        attachment_dependencies = _attachment_dependency_views(task_draft)
        capabilities = tuple(sorted(task_draft.required_capabilities))
        tools = tuple(sorted(task_draft.allowed_tools))
        forbidden_outputs = tuple(sorted(task_draft.forbidden_outputs))
        safe_audit = _safe_audit(audit, (task_ref,))
        projection_seed = _candidate_projection_seed(
            task_draft_ref=task_ref,
            visible_prompt=task_draft.visible_prompt,
            task_intent=task_draft.task_intent,
            evaluation_claim=task_draft.evaluation_claim,
            prompt_requirements=prompt_requirements,
            attachment_dependencies=attachment_dependencies,
            required_capability_ids=capabilities,
            allowed_tool_ids=tools,
            forbidden_outputs=forbidden_outputs,
            allowed_evaluator_bindings=evaluator_bindings,
        )
        projection_ref = _candidate_projection_ref(projection_seed)
        enforcement = _enforce_candidate_projection(
            task_draft_ref=task_ref,
            candidate_projection_ref=projection_ref,
            audit=safe_audit,
        )
        enforcement_ref = _prompt_boundary_ref(enforcement)
        request_seed = _request_seed(
            task_draft_ref=task_ref,
            candidate_task_projection_ref=projection_ref,
            prompt_boundary_enforcement_ref=enforcement_ref,
            visible_prompt=task_draft.visible_prompt,
            task_intent=task_draft.task_intent,
            evaluation_claim=task_draft.evaluation_claim,
            prompt_requirements=prompt_requirements,
            attachment_dependencies=attachment_dependencies,
            required_capability_ids=capabilities,
            allowed_tool_ids=tools,
            forbidden_outputs=forbidden_outputs,
            allowed_evaluator_bindings=evaluator_bindings,
        )
        return RubricCandidateRequest(
            rubric_candidate_request_id=_stable_id(
                "rubric-candidate-request",
                request_seed,
            ),
            task_draft_ref=task_ref,
            candidate_task_projection_ref=projection_ref,
            prompt_boundary_enforcement_ref=enforcement_ref,
            visible_prompt=task_draft.visible_prompt,
            task_intent=task_draft.task_intent,
            evaluation_claim=task_draft.evaluation_claim,
            prompt_requirements=prompt_requirements,
            attachment_dependencies=attachment_dependencies,
            required_capability_ids=capabilities,
            allowed_tool_ids=tools,
            forbidden_outputs=forbidden_outputs,
            allowed_evaluator_bindings=evaluator_bindings,
            policy_version=self.policy_version,
            request_sha256=_stable_hash(request_seed),
            audit=_safe_audit(audit, (task_ref, enforcement_ref)),
        )


class RubricImportAdapter:
    policy_version = RUBRIC_AUTHORING_POLICY_VERSION

    def adapt(
        self,
        request: RubricCandidateRequest,
        *,
        imported: ImportedRubricDefinition,
        audit: ContractAudit,
    ) -> RubricCandidateProposal:
        _validate_request_integrity(request)
        _validate_imported_definition(imported)
        criteria: tuple[RubricCriterionSelection, ...] = ()
        reasons: frozenset[RubricAuthoringReason] = frozenset()
        outcome = RubricAuthoringOutcome.COMPILED
        if imported.status is ImportedRubricStatus.USABLE:
            criteria = _normalize_selections(imported.criteria)
            _validate_selections(request, criteria)
        elif imported.status is ImportedRubricStatus.REJECTED:
            outcome = RubricAuthoringOutcome.ABSTAIN
            reasons = frozenset({RubricAuthoringReason.IMPORT_REJECTED})
        else:
            outcome = RubricAuthoringOutcome.ABSTAIN
            reasons = frozenset({RubricAuthoringReason.AMBIGUOUS_RUBRIC})
        return _build_proposal(
            request=request,
            source_mode=RubricSourceModeV2.IMPORTED,
            outcome=outcome,
            criteria=criteria,
            unresolved_reasons=reasons,
            imported_from_ref=imported.imported_rubric_ref,
            model_profile=None,
            prompt_version=None,
            audit=audit,
        )


class FakeRubricGenerationRunner:
    policy_version = RUBRIC_AUTHORING_POLICY_VERSION

    def run(
        self,
        request: RubricCandidateRequest,
        *,
        fixture: FakeRubricGenerationFixture,
        model_profile: Identifier,
        prompt_version: str,
        audit: ContractAudit,
    ) -> RubricCandidateProposal:
        _validate_request_integrity(request)
        criteria = _normalize_selections(fixture.criteria)
        if criteria:
            _validate_selections(request, criteria)
        return _build_proposal(
            request=request,
            source_mode=RubricSourceModeV2.GENERATED,
            outcome=fixture.outcome,
            criteria=criteria,
            unresolved_reasons=fixture.unresolved_reasons,
            imported_from_ref=None,
            model_profile=model_profile,
            prompt_version=prompt_version,
            audit=audit,
        )


class RubricGeneratedSelectionAdapter:
    policy_version = RUBRIC_AUTHORING_POLICY_VERSION

    def adapt(
        self,
        request: RubricCandidateRequest,
        *,
        outcome: RubricAuthoringOutcome,
        criteria: tuple[RubricCriterionSelection, ...],
        unresolved_reasons: frozenset[RubricAuthoringReason],
        model_profile: Identifier,
        prompt_version: str,
        audit: ContractAudit,
    ) -> RubricCandidateProposal:
        _validate_request_integrity(request)
        normalized = _normalize_selections(criteria)
        if normalized:
            _validate_selections(request, normalized)
        return _build_proposal(
            request=request,
            source_mode=RubricSourceModeV2.GENERATED,
            outcome=outcome,
            criteria=normalized,
            unresolved_reasons=unresolved_reasons,
            imported_from_ref=None,
            model_profile=model_profile,
            prompt_version=prompt_version,
            audit=audit,
        )


class RubricSetCompiler:
    policy_version = RUBRIC_AUTHORING_POLICY_VERSION

    def compile(
        self,
        *,
        request: RubricCandidateRequest,
        proposal: RubricCandidateProposal,
        task_draft: TaskDraftV2,
        audit: ContractAudit,
    ) -> RubricAuthoringResult:
        _validate_task_draft(task_draft)
        _validate_request_integrity(request)
        _validate_proposal_integrity(proposal)
        task_ref = task_draft_ref(task_draft)
        if request.task_draft_ref != task_ref:
            raise RubricAuthoringPolicyError("TaskDraft ref is stale or mismatched")
        expected_request = RubricCandidateRequestBuilder().build(
            task_draft=task_draft,
            allowed_evaluator_bindings=request.allowed_evaluator_bindings,
            audit=request.audit,
        )
        if expected_request != request:
            raise RubricAuthoringPolicyError("request no longer matches authoritative TaskDraft")
        if proposal.request_ref != _request_ref(request):
            raise RubricAuthoringPolicyError("proposal request ref is stale or mismatched")
        if proposal.task_draft_ref != task_ref:
            raise RubricAuthoringPolicyError("proposal TaskDraft ref is stale or mismatched")
        if proposal.outcome is not RubricAuthoringOutcome.COMPILED:
            return _build_result(
                request=request,
                proposal=proposal,
                outcome=proposal.outcome,
                rubric_set=None,
                unresolved_reasons=proposal.unresolved_reasons,
                audit=audit,
            )

        _validate_selections(request, proposal.criteria)
        criteria: list[RubricCriterionV2] = []
        unresolved_reasons: set[RubricAuthoringReason] = set()
        for selection in proposal.criteria:
            criterion, reasons = _compile_criterion(
                selection=selection,
                task_draft=task_draft,
            )
            unresolved_reasons.update(reasons)
            if criterion is not None:
                criteria.append(criterion)
        if unresolved_reasons:
            return _build_result(
                request=request,
                proposal=proposal,
                outcome=RubricAuthoringOutcome.BLOCKED_REACHABILITY,
                rubric_set=None,
                unresolved_reasons=frozenset(unresolved_reasons),
                audit=audit,
            )
        ordered_criteria = tuple(sorted(criteria, key=lambda item: item.criterion_id))
        rubric_set = _compile_rubric_set(
            request=request,
            proposal=proposal,
            criteria=ordered_criteria,
            audit=audit,
        )
        return _build_result(
            request=request,
            proposal=proposal,
            outcome=RubricAuthoringOutcome.COMPILED,
            rubric_set=rubric_set,
            unresolved_reasons=frozenset(),
            audit=audit,
        )


def _compile_criterion(
    *,
    selection: RubricCriterionSelection,
    task_draft: TaskDraftV2,
) -> tuple[RubricCriterionV2 | None, frozenset[RubricAuthoringReason]]:
    requirement_by_id = {
        item.requirement_id: item
        for item in task_draft.requirement_lineage
        if item.requirement_id in set(task_draft.prompt_requirement_ids)
    }
    dependency_by_id = {item.dependency_id: item for item in task_draft.attachment_dependencies}
    reasons: set[RubricAuthoringReason] = set()
    requirements = []
    for requirement_id in selection.prompt_requirement_ids:
        requirement = requirement_by_id.get(requirement_id)
        if requirement is None:
            reasons.add(RubricAuthoringReason.MISSING_PROMPT_REQUIREMENT)
        else:
            requirements.append(requirement)
    dependencies = []
    for dependency_id in selection.attachment_dependency_ids:
        dependency = dependency_by_id.get(dependency_id)
        if dependency is None:
            reasons.add(RubricAuthoringReason.MISSING_ATTACHMENT_DEPENDENCY)
        else:
            dependencies.append(dependency)
    unknown_tools = set(selection.allowed_tool_ids) - set(task_draft.allowed_tools)
    if unknown_tools:
        reasons.add(RubricAuthoringReason.MISSING_ALLOWED_TOOL)
    if (
        selection.judged_object_kind is RubricJudgedObjectKindV2.WORKSPACE_STATE
        and not selection.attachment_dependency_ids
    ):
        reasons.add(RubricAuthoringReason.MISSING_ATTACHMENT_DEPENDENCY)
    if (
        selection.judged_object_kind is RubricJudgedObjectKindV2.TOOL_BEHAVIOR
        and not selection.allowed_tool_ids
    ):
        reasons.add(RubricAuthoringReason.MISSING_ALLOWED_TOOL)
    evidence = _merge_evidence(
        tuple(item for requirement in requirements for item in requirement.evidence)
        + tuple(item for dependency in dependencies for item in dependency.evidence)
    )
    if any(not item.capability_complete for item in evidence):
        reasons.add(RubricAuthoringReason.INCOMPLETE_REACHABILITY_EVIDENCE)
    if reasons:
        return None, frozenset(reasons)

    reachability = RubricReachabilityV2(
        reachability_id="rubric-reachability://pending",
        prompt_requirement_ids=selection.prompt_requirement_ids,
        attachment_dependency_ids=selection.attachment_dependency_ids,
        allowed_tool_ids=selection.allowed_tool_ids,
        evidence=evidence,
        capability_complete=True,
        reachability_sha256="0" * 64,
    )
    reachability_hash = rubric_reachability_carried_sha256(reachability)
    reachability = reachability.model_copy(
        update={
            "reachability_id": f"rubric-reachability://sha256/{reachability_hash}",
            "reachability_sha256": reachability_hash,
        }
    )
    judged_object = RubricJudgedObjectV2(
        judged_object_id=selection.judged_object_id,
        kind=selection.judged_object_kind,
        description=selection.judged_object_description,
    )
    criterion = RubricCriterionV2(
        criterion_id="rubric-criterion://pending",
        judged_object=judged_object,
        description=selection.description,
        weight=selection.weight,
        reachability=reachability,
        visibility=selection.visibility,
        evaluator_binding=selection.evaluator_binding,
        approval_status="CANDIDATE",
        criterion_sha256="0" * 64,
    )
    criterion_hash = rubric_criterion_carried_sha256(criterion)
    return (
        criterion.model_copy(
            update={
                "criterion_id": f"rubric-criterion://sha256/{criterion_hash}",
                "criterion_sha256": criterion_hash,
            }
        ),
        frozenset(),
    )


def _compile_rubric_set(
    *,
    request: RubricCandidateRequest,
    proposal: RubricCandidateProposal,
    criteria: tuple[RubricCriterionV2, ...],
    audit: ContractAudit,
) -> RubricSetV2:
    input_refs = (
        request.task_draft_ref,
        _request_ref(request),
        _proposal_ref(proposal),
        *((proposal.imported_from_ref,) if proposal.imported_from_ref is not None else ()),
    )
    rubric_set = RubricSetV2(
        rubric_set_id="rubric-set://pending",
        rubric_version=1,
        supersedes_rubric_set_ref=None,
        task_draft_ref=request.task_draft_ref,
        source_mode=proposal.source_mode,
        criteria=criteria,
        total_weight=sum(item.weight for item in criteria),
        imported_from_ref=proposal.imported_from_ref,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
        policy_version=RUBRIC_AUTHORING_POLICY_VERSION,
        rubric_set_sha256="0" * 64,
        audit=_safe_audit(audit, input_refs),
    )
    digest = rubric_set_carried_sha256(rubric_set)
    return rubric_set.model_copy(
        update={
            "rubric_set_id": f"rubric-set://sha256/{digest}",
            "rubric_set_sha256": digest,
        }
    )


def _build_result(
    *,
    request: RubricCandidateRequest,
    proposal: RubricCandidateProposal,
    outcome: RubricAuthoringOutcome,
    rubric_set: RubricSetV2 | None,
    unresolved_reasons: frozenset[RubricAuthoringReason],
    audit: ContractAudit,
) -> RubricAuthoringResult:
    request_ref = _request_ref(request)
    proposal_ref = _proposal_ref(proposal)
    seed = {
        "request_ref": _ref_payload(request_ref),
        "proposal_ref": _ref_payload(proposal_ref),
        "outcome": outcome.value,
        "rubric_set_ref": (_ref_payload(rubric_set_ref(rubric_set)) if rubric_set is not None else None),
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "policy_version": RUBRIC_AUTHORING_POLICY_VERSION,
    }
    digest = _stable_hash(seed)
    refs = (
        request_ref,
        proposal_ref,
        *((rubric_set_ref(rubric_set),) if rubric_set is not None else ()),
    )
    return RubricAuthoringResult(
        rubric_authoring_result_id=f"rubric-authoring-result://sha256/{digest}",
        request_ref=request_ref,
        proposal_ref=proposal_ref,
        outcome=outcome,
        rubric_set=rubric_set,
        unresolved_reasons=unresolved_reasons,
        policy_version=RUBRIC_AUTHORING_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _build_proposal(
    *,
    request: RubricCandidateRequest,
    source_mode: RubricSourceModeV2,
    outcome: RubricAuthoringOutcome,
    criteria: tuple[RubricCriterionSelection, ...],
    unresolved_reasons: frozenset[RubricAuthoringReason],
    imported_from_ref: ObjectRef | None,
    model_profile: Identifier | None,
    prompt_version: str | None,
    audit: ContractAudit,
) -> RubricCandidateProposal:
    request_ref = _request_ref(request)
    seed = _proposal_seed(
        request_ref=request_ref,
        task_draft_ref=request.task_draft_ref,
        source_mode=source_mode,
        outcome=outcome,
        criteria=criteria,
        unresolved_reasons=unresolved_reasons,
        imported_from_ref=imported_from_ref,
        model_profile=model_profile,
        prompt_version=prompt_version,
    )
    digest = _stable_hash(seed)
    refs = (
        request_ref,
        request.task_draft_ref,
        *((imported_from_ref,) if imported_from_ref is not None else ()),
    )
    return RubricCandidateProposal(
        rubric_candidate_proposal_id=f"rubric-candidate-proposal://sha256/{digest}",
        request_ref=request_ref,
        task_draft_ref=request.task_draft_ref,
        source_mode=source_mode,
        outcome=outcome,
        criteria=criteria,
        unresolved_reasons=unresolved_reasons,
        imported_from_ref=imported_from_ref,
        model_profile=model_profile,
        prompt_version=prompt_version,
        policy_version=RUBRIC_AUTHORING_POLICY_VERSION,
        proposal_sha256=digest,
        audit=_safe_audit(audit, refs),
    )


def _validate_task_draft(task_draft: TaskDraftV2) -> None:
    expected_hash = task_draft_carried_sha256(task_draft)
    if task_draft.task_draft_sha256 != expected_hash:
        raise RubricAuthoringPolicyError("TaskDraft carried hash is stale or mismatched")
    if task_draft.task_draft_id != f"task-draft://sha256/{expected_hash}":
        raise RubricAuthoringPolicyError("TaskDraft ID is stale or mismatched")
    if task_draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.BLOCKED:
        raise RubricAuthoringPolicyError("BLOCKED TaskDraft cannot author rubrics")


def _validate_request_integrity(request: RubricCandidateRequest) -> None:
    projection_seed = _candidate_projection_seed(
        task_draft_ref=request.task_draft_ref,
        visible_prompt=request.visible_prompt,
        task_intent=request.task_intent,
        evaluation_claim=request.evaluation_claim,
        prompt_requirements=request.prompt_requirements,
        attachment_dependencies=request.attachment_dependencies,
        required_capability_ids=request.required_capability_ids,
        allowed_tool_ids=request.allowed_tool_ids,
        forbidden_outputs=request.forbidden_outputs,
        allowed_evaluator_bindings=request.allowed_evaluator_bindings,
    )
    expected_projection_ref = _candidate_projection_ref(projection_seed)
    if request.candidate_task_projection_ref != expected_projection_ref:
        raise RubricAuthoringPolicyError("request candidate projection is stale or mismatched")
    try:
        expected_enforcement = _enforce_candidate_projection(
            task_draft_ref=request.task_draft_ref,
            candidate_projection_ref=expected_projection_ref,
            audit=request.audit,
        )
    except PromptInjectionBoundaryPolicyError as exc:
        raise RubricAuthoringPolicyError("request prompt boundary is invalid") from exc
    if request.prompt_boundary_enforcement_ref != _prompt_boundary_ref(expected_enforcement):
        raise RubricAuthoringPolicyError("request prompt boundary ref is stale or mismatched")
    seed = _request_seed(
        task_draft_ref=request.task_draft_ref,
        candidate_task_projection_ref=request.candidate_task_projection_ref,
        prompt_boundary_enforcement_ref=request.prompt_boundary_enforcement_ref,
        visible_prompt=request.visible_prompt,
        task_intent=request.task_intent,
        evaluation_claim=request.evaluation_claim,
        prompt_requirements=request.prompt_requirements,
        attachment_dependencies=request.attachment_dependencies,
        required_capability_ids=request.required_capability_ids,
        allowed_tool_ids=request.allowed_tool_ids,
        forbidden_outputs=request.forbidden_outputs,
        allowed_evaluator_bindings=request.allowed_evaluator_bindings,
    )
    digest = _stable_hash(seed)
    if request.request_sha256 != digest:
        raise RubricAuthoringPolicyError("request hash is stale or mismatched")
    if request.rubric_candidate_request_id != (f"rubric-candidate-request://sha256/{digest}"):
        raise RubricAuthoringPolicyError("request ID is stale or mismatched")


def _validate_proposal_integrity(proposal: RubricCandidateProposal) -> None:
    seed = _proposal_seed(
        request_ref=proposal.request_ref,
        task_draft_ref=proposal.task_draft_ref,
        source_mode=proposal.source_mode,
        outcome=proposal.outcome,
        criteria=proposal.criteria,
        unresolved_reasons=proposal.unresolved_reasons,
        imported_from_ref=proposal.imported_from_ref,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
    )
    digest = _stable_hash(seed)
    if proposal.proposal_sha256 != digest:
        raise RubricAuthoringPolicyError("proposal hash is stale or mismatched")
    if proposal.rubric_candidate_proposal_id != (f"rubric-candidate-proposal://sha256/{digest}"):
        raise RubricAuthoringPolicyError("proposal ID is stale or mismatched")


def _validate_imported_definition(imported: ImportedRubricDefinition) -> None:
    if imported.imported_rubric_ref.object_type != "imported-rubric":
        raise RubricAuthoringPolicyError("imported rubric source ref is invalid")

    if imported.definition_sha256 != imported_rubric_definition_sha256(imported):
        raise RubricAuthoringPolicyError("imported rubric definition hash is stale or mismatched")


def _validate_selections(
    request: RubricCandidateRequest,
    criteria: tuple[RubricCriterionSelection, ...],
) -> None:
    requirement_ids = {item.requirement_id for item in request.prompt_requirements}
    dependency_ids = {item.dependency_id for item in request.attachment_dependencies}
    tool_ids = set(request.allowed_tool_ids)
    evaluator_bindings = set(request.allowed_evaluator_bindings)
    for selection in criteria:
        if not set(selection.prompt_requirement_ids).issubset(requirement_ids):
            raise RubricAuthoringPolicyError("criterion selects an unknown prompt requirement")
        if not set(selection.attachment_dependency_ids).issubset(dependency_ids):
            raise RubricAuthoringPolicyError("criterion selects an unknown attachment dependency")
        if not set(selection.allowed_tool_ids).issubset(tool_ids):
            raise RubricAuthoringPolicyError("criterion selects an unknown allowed tool")
        if selection.evaluator_binding not in evaluator_bindings:
            raise RubricAuthoringPolicyError("criterion evaluator binding is not authorized")


def _prompt_requirement_views(
    task_draft: TaskDraftV2,
) -> tuple[RubricTaskRequirementView, ...]:
    prompt_ids = set(task_draft.prompt_requirement_ids)
    return tuple(
        RubricTaskRequirementView(
            requirement_id=item.requirement_id,
            statement=item.statement,
            criticality=item.criticality,
        )
        for item in sorted(
            task_draft.requirement_lineage,
            key=lambda item: item.requirement_id,
        )
        if item.requirement_id in prompt_ids
    )


def _attachment_dependency_views(
    task_draft: TaskDraftV2,
) -> tuple[RubricAttachmentDependencyView, ...]:
    return tuple(
        RubricAttachmentDependencyView(
            dependency_id=item.dependency_id,
            description=item.description,
            criticality=item.criticality,
        )
        for item in sorted(
            task_draft.attachment_dependencies,
            key=lambda item: item.dependency_id,
        )
    )


def _enforce_candidate_projection(
    *,
    task_draft_ref: ObjectRef,
    candidate_projection_ref: ObjectRef,
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    segment = PromptBoundarySegment(
        segment_id=(f"prompt-boundary-segment://rubric-authoring/{candidate_projection_ref.object_sha256}"),
        surface=PromptBoundarySurface.RUBRIC_AUTHORING_DATA,
        source_role=PromptBoundarySourceRole.CANDIDATE_TASK_CONTRACT,
        source_ref=candidate_projection_ref,
        content_sha256=candidate_projection_ref.object_sha256,
        untrusted_data_marker=True,
    )
    boundary_request = PromptBoundaryEnforcementRequest(
        boundary_id=(f"prompt-boundary://rubric-authoring/{candidate_projection_ref.object_sha256}"),
        segments=(segment,),
        audit=audit,
    )
    return PromptInjectionBoundaryEnforcer().validate_candidate_task_contract_boundary(
        CandidateTaskContractBoundaryRequest(
            task_draft_ref=task_draft_ref,
            candidate_projection_ref=candidate_projection_ref,
            boundary_request=boundary_request,
            projection_segment_id=segment.segment_id,
        )
    )


def _candidate_projection_seed(
    *,
    task_draft_ref: ObjectRef,
    visible_prompt: str,
    task_intent: str,
    evaluation_claim: str,
    prompt_requirements: tuple[RubricTaskRequirementView, ...],
    attachment_dependencies: tuple[RubricAttachmentDependencyView, ...],
    required_capability_ids: tuple[Identifier, ...],
    allowed_tool_ids: tuple[Identifier, ...],
    forbidden_outputs: tuple[str, ...],
    allowed_evaluator_bindings: tuple[Identifier, ...],
) -> dict[str, object]:
    return {
        "task_draft_ref": _ref_payload(task_draft_ref),
        "visible_prompt": visible_prompt,
        "task_intent": task_intent,
        "evaluation_claim": evaluation_claim,
        "prompt_requirements": [
            item.model_dump(mode="json", exclude_none=False) for item in prompt_requirements
        ],
        "attachment_dependencies": [
            item.model_dump(mode="json", exclude_none=False) for item in attachment_dependencies
        ],
        "required_capability_ids": list(required_capability_ids),
        "allowed_tool_ids": list(allowed_tool_ids),
        "forbidden_outputs": list(forbidden_outputs),
        "allowed_evaluator_bindings": list(allowed_evaluator_bindings),
        "policy_version": RUBRIC_AUTHORING_POLICY_VERSION,
    }


def _request_seed(
    *,
    task_draft_ref: ObjectRef,
    candidate_task_projection_ref: ObjectRef,
    prompt_boundary_enforcement_ref: ObjectRef,
    visible_prompt: str,
    task_intent: str,
    evaluation_claim: str,
    prompt_requirements: tuple[RubricTaskRequirementView, ...],
    attachment_dependencies: tuple[RubricAttachmentDependencyView, ...],
    required_capability_ids: tuple[Identifier, ...],
    allowed_tool_ids: tuple[Identifier, ...],
    forbidden_outputs: tuple[str, ...],
    allowed_evaluator_bindings: tuple[Identifier, ...],
) -> dict[str, object]:
    return {
        "task_draft_ref": _ref_payload(task_draft_ref),
        "candidate_task_projection_ref": _ref_payload(candidate_task_projection_ref),
        "prompt_boundary_enforcement_ref": _ref_payload(prompt_boundary_enforcement_ref),
        "visible_prompt": visible_prompt,
        "task_intent": task_intent,
        "evaluation_claim": evaluation_claim,
        "prompt_requirements": [
            item.model_dump(mode="json", exclude_none=False) for item in prompt_requirements
        ],
        "attachment_dependencies": [
            item.model_dump(mode="json", exclude_none=False) for item in attachment_dependencies
        ],
        "required_capability_ids": list(required_capability_ids),
        "allowed_tool_ids": list(allowed_tool_ids),
        "forbidden_outputs": list(forbidden_outputs),
        "allowed_evaluator_bindings": list(allowed_evaluator_bindings),
        "policy_version": RUBRIC_AUTHORING_POLICY_VERSION,
    }


def _proposal_seed(
    *,
    request_ref: ObjectRef,
    task_draft_ref: ObjectRef,
    source_mode: RubricSourceModeV2,
    outcome: RubricAuthoringOutcome,
    criteria: tuple[RubricCriterionSelection, ...],
    unresolved_reasons: frozenset[RubricAuthoringReason],
    imported_from_ref: ObjectRef | None,
    model_profile: Identifier | None,
    prompt_version: str | None,
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "task_draft_ref": _ref_payload(task_draft_ref),
        "source_mode": source_mode.value,
        "outcome": outcome.value,
        "criteria": [item.model_dump(mode="json", exclude_none=False) for item in criteria],
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "imported_from_ref": _maybe_ref_payload(imported_from_ref),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": RUBRIC_AUTHORING_POLICY_VERSION,
    }


def _normalize_selections(
    criteria: tuple[RubricCriterionSelection, ...],
) -> tuple[RubricCriterionSelection, ...]:
    normalized = tuple(
        item.model_copy(
            update={
                "prompt_requirement_ids": tuple(sorted(item.prompt_requirement_ids)),
                "attachment_dependency_ids": tuple(sorted(item.attachment_dependency_ids)),
                "allowed_tool_ids": tuple(sorted(item.allowed_tool_ids)),
            }
        )
        for item in criteria
    )
    return tuple(sorted(normalized, key=lambda item: item.selection_id))


def _merge_evidence(evidence: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    by_id: dict[str, EvidenceRef] = {}
    for item in evidence:
        current = by_id.get(item.evidence_ref_id)
        if current is not None and current != item:
            raise RubricAuthoringPolicyError("duplicate reachability evidence ID has conflicting values")
        by_id[item.evidence_ref_id] = item
    return tuple(by_id[key] for key in sorted(by_id))


def _candidate_projection_ref(seed: dict[str, object]) -> ObjectRef:
    digest = _stable_hash(seed)
    return ObjectRef(
        object_type="task-draft-rubric-input",
        object_id=f"task-draft-rubric-input://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _request_ref(request: RubricCandidateRequest) -> ObjectRef:
    return ObjectRef(
        object_type="rubric-candidate-request",
        object_id=request.rubric_candidate_request_id,
        object_version="r4-05",
        object_sha256=request.request_sha256,
    )


def _proposal_ref(proposal: RubricCandidateProposal) -> ObjectRef:
    return ObjectRef(
        object_type="rubric-candidate-proposal",
        object_id=proposal.rubric_candidate_proposal_id,
        object_version="r4-05",
        object_sha256=proposal.proposal_sha256,
    )


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
    input_refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    ordered_refs = tuple(
        sorted(
            input_refs,
            key=lambda item: (
                item.object_type,
                item.object_id,
                item.object_version,
                item.object_sha256,
            ),
        )
    )
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=ordered_refs,
    )


def _sorted_unique(
    label: str,
    values: tuple[Identifier, ...],
) -> tuple[Identifier, ...]:
    if len(values) != len(set(values)):
        raise RubricAuthoringPolicyError(f"{label} IDs must be unique")
    return tuple(sorted(values))


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _maybe_ref_payload(ref: ObjectRef | None) -> dict[str, object] | None:
    if ref is None:
        return None
    return _ref_payload(ref)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(namespace: str, payload: object) -> str:
    return f"{namespace}://sha256/{_stable_hash(payload)}"
