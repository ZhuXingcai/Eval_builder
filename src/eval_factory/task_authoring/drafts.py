from __future__ import annotations

import hashlib
import json

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.labeling_v2 import (
    SelectionContextV2,
    selection_context_ref,
)
from eval_factory.contracts.safety import EvidenceBundle, ProjectionPolicy
from eval_factory.contracts.task import (
    AttachmentDependency,
    RequirementConflict,
)
from eval_factory.contracts.task_v2 import (
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskEpisodeV2,
    TaskRequirementLineageV2,
    task_draft_payload_sha256,
    task_draft_ref,
    task_episode_ref,
)
from eval_factory.provenance.bundles import (
    EVIDENCE_COMPILATION_POLICY_VERSION,
)
from eval_factory.provenance.injection import (
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
)
from eval_factory.provenance.views import (
    EVIDENCE_VIEW_POLICY_VERSION,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewResult,
)
from eval_factory.task_authoring.draft_models import (
    TASK_DRAFT_AUTHORING_POLICY_VERSION,
    TASK_DRAFT_EVIDENCE_PURPOSE,
    FakeTaskDraftAuthoringFixture,
    TaskDraftAttachmentFixture,
    TaskDraftAuthoringOutcome,
    TaskDraftAuthoringPolicyError,
    TaskDraftAuthoringProposal,
    TaskDraftAuthoringReason,
    TaskDraftAuthoringRequest,
    TaskDraftAuthoringResult,
    TaskDraftContentFixture,
    TaskDraftEvidenceView,
    TaskDraftRequirementFixture,
    validate_outcome_shape,
)
from eval_factory.task_authoring.models import (
    TASK_EPISODE_CONSUMER_STAGE,
    TASK_EPISODE_EVIDENCE_PURPOSE,
    is_safe_task_authoring_ref,
)


class TaskDraftAuthoringRequestBuilder:
    policy_version = TASK_DRAFT_AUTHORING_POLICY_VERSION

    def build(
        self,
        *,
        selection_context: SelectionContextV2,
        task_episodes: tuple[TaskEpisodeV2, ...],
        selection_evidence_bundle: EvidenceBundle,
        authoring_view_result: EvidenceViewResult,
        authoring_evidence_bundle: EvidenceBundle,
        allowed_capability_ids: tuple[Identifier, ...],
        allowed_tool_ids: tuple[Identifier, ...],
        required_forbidden_outputs: tuple[str, ...],
        abstain_conditions: tuple[str, ...],
        model_profile: Identifier,
        prompt_version: str,
        audit: ContractAudit,
    ) -> TaskDraftAuthoringRequest:
        _validate_authoritative_inputs(
            selection_context=selection_context,
            task_episodes=task_episodes,
            selection_evidence_bundle=selection_evidence_bundle,
            authoring_view_result=authoring_view_result,
            authoring_evidence_bundle=authoring_evidence_bundle,
        )
        evidence_views = _task_draft_evidence_views(
            authoring_view_result,
            authoring_evidence_bundle,
        )
        enforcement = _enforce_prompt_as_data(
            authoring_view_result=authoring_view_result,
            authoring_evidence_bundle=authoring_evidence_bundle,
            audit=audit,
        )
        episode_refs = _sort_refs(tuple(task_episode_ref(item) for item in task_episodes))
        capabilities = tuple(sorted(allowed_capability_ids))
        tools = tuple(sorted(allowed_tool_ids))
        forbidden_outputs = tuple(sorted(required_forbidden_outputs))
        conditions = tuple(sorted(abstain_conditions))
        seed = _request_seed(
            selection_context_ref=selection_context_ref(selection_context),
            trace_envelope_ref=task_episodes[0].trace_envelope_ref,
            task_episode_refs=episode_refs,
            selection_evidence_bundle_ref=_evidence_bundle_ref(selection_evidence_bundle),
            authoring_evidence_bundle_ref=_evidence_bundle_ref(authoring_evidence_bundle),
            authoring_projection_policy_ref=authoring_evidence_bundle.projection_policy_ref,
            prompt_boundary_enforcement_ref=_prompt_boundary_ref(enforcement),
            evidence_views=evidence_views,
            allowed_capability_ids=capabilities,
            allowed_tool_ids=tools,
            required_forbidden_outputs=forbidden_outputs,
            abstain_conditions=conditions,
            model_profile=model_profile,
            prompt_version=prompt_version,
            returned_characters=authoring_evidence_bundle.returned_characters,
            max_characters=authoring_evidence_bundle.max_characters,
        )
        return TaskDraftAuthoringRequest(
            semantic_authoring_request_id=_stable_id(
                "task-draft-authoring-request",
                seed,
            ),
            selection_context_ref=selection_context_ref(selection_context),
            trace_envelope_ref=task_episodes[0].trace_envelope_ref,
            task_episode_refs=episode_refs,
            selection_evidence_bundle_ref=_evidence_bundle_ref(selection_evidence_bundle),
            authoring_evidence_bundle_ref=_evidence_bundle_ref(authoring_evidence_bundle),
            authoring_projection_policy_ref=authoring_evidence_bundle.projection_policy_ref,
            prompt_boundary_enforcement_ref=_prompt_boundary_ref(enforcement),
            evidence_views=evidence_views,
            allowed_capability_ids=capabilities,
            allowed_tool_ids=tools,
            required_forbidden_outputs=forbidden_outputs,
            abstain_conditions=conditions,
            model_profile=model_profile,
            prompt_version=prompt_version,
            returned_characters=authoring_evidence_bundle.returned_characters,
            max_characters=authoring_evidence_bundle.max_characters,
            request_sha256=_stable_hash(seed),
            audit=_sanitized_audit(
                audit,
                selection_context=selection_context,
                task_episodes=task_episodes,
                selection_evidence_bundle=selection_evidence_bundle,
                authoring_evidence_bundle=authoring_evidence_bundle,
                prompt_boundary_enforcement=enforcement,
            ),
        )


class FakeTaskDraftAuthoringRunner:
    policy_version = TASK_DRAFT_AUTHORING_POLICY_VERSION

    def run(
        self,
        request: TaskDraftAuthoringRequest,
        *,
        fixture: FakeTaskDraftAuthoringFixture,
        audit: ContractAudit,
    ) -> TaskDraftAuthoringProposal:
        _validate_request_integrity(request)
        _validate_fixture(fixture)
        content = fixture.content
        if content is not None:
            _validate_semantic_content(request, content)
            content = _normalize_content(content)
        seed = _proposal_seed(
            request_ref=_request_ref(request),
            selection_context_ref=request.selection_context_ref,
            task_episode_refs=request.task_episode_refs,
            selection_evidence_bundle_ref=request.selection_evidence_bundle_ref,
            authoring_evidence_bundle_ref=request.authoring_evidence_bundle_ref,
            outcome=fixture.outcome,
            content=content,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
        )
        return TaskDraftAuthoringProposal(
            semantic_authoring_proposal_id=_stable_id(
                "task-draft-authoring-proposal",
                seed,
            ),
            request_ref=_request_ref(request),
            selection_context_ref=request.selection_context_ref,
            task_episode_refs=request.task_episode_refs,
            selection_evidence_bundle_ref=request.selection_evidence_bundle_ref,
            authoring_evidence_bundle_ref=request.authoring_evidence_bundle_ref,
            outcome=fixture.outcome,
            content=content,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            proposal_sha256=_stable_hash(seed),
            audit=_safe_artifact_audit(
                audit,
                (
                    _request_ref(request),
                    request.selection_context_ref,
                    *request.task_episode_refs,
                    request.selection_evidence_bundle_ref,
                    request.authoring_evidence_bundle_ref,
                ),
            ),
        )


class TaskDraftCompiler:
    policy_version = TASK_DRAFT_AUTHORING_POLICY_VERSION

    def compile(
        self,
        *,
        request: TaskDraftAuthoringRequest,
        proposal: TaskDraftAuthoringProposal,
        selection_context: SelectionContextV2,
        task_episodes: tuple[TaskEpisodeV2, ...],
        selection_evidence_bundle: EvidenceBundle,
        authoring_view_result: EvidenceViewResult,
        authoring_evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
    ) -> TaskDraftAuthoringResult:
        _validate_request_integrity(request)
        _validate_proposal_integrity(proposal)
        if selection_context_ref(selection_context) != request.selection_context_ref:
            raise TaskDraftAuthoringPolicyError("SelectionContext ref is stale or mismatched")
        expected_episode_refs = _sort_refs(tuple(task_episode_ref(item) for item in task_episodes))
        if expected_episode_refs != request.task_episode_refs:
            raise TaskDraftAuthoringPolicyError("TaskEpisode refs are stale or mismatched")
        if _evidence_bundle_ref(selection_evidence_bundle) != request.selection_evidence_bundle_ref:
            raise TaskDraftAuthoringPolicyError("selection evidence bundle ref is stale or mismatched")
        if _evidence_bundle_ref(authoring_evidence_bundle) != request.authoring_evidence_bundle_ref:
            raise TaskDraftAuthoringPolicyError("authoring evidence bundle ref is stale or mismatched")

        expected_request = TaskDraftAuthoringRequestBuilder().build(
            selection_context=selection_context,
            task_episodes=task_episodes,
            selection_evidence_bundle=selection_evidence_bundle,
            authoring_view_result=authoring_view_result,
            authoring_evidence_bundle=authoring_evidence_bundle,
            allowed_capability_ids=request.allowed_capability_ids,
            allowed_tool_ids=request.allowed_tool_ids,
            required_forbidden_outputs=request.required_forbidden_outputs,
            abstain_conditions=request.abstain_conditions,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            audit=request.audit,
        )
        if expected_request != request:
            raise TaskDraftAuthoringPolicyError("request no longer matches authoritative task-draft inputs")
        _validate_proposal_binding(request, proposal)

        task_draft: TaskDraftV2 | None = None
        if proposal.outcome is TaskDraftAuthoringOutcome.DRAFTED:
            task_draft = _compile_task_draft(
                request=request,
                proposal=proposal,
                selection_context=selection_context,
                task_episodes=task_episodes,
                selection_evidence_bundle=selection_evidence_bundle,
                authoring_evidence_bundle=authoring_evidence_bundle,
                audit=audit,
            )
        seed = {
            "request_ref": _ref_payload(_request_ref(request)),
            "proposal_ref": _ref_payload(_proposal_ref(proposal)),
            "outcome": proposal.outcome.value,
            "task_draft_ref": (None if task_draft is None else _ref_payload(task_draft_ref(task_draft))),
            "unresolved_reasons": sorted(item.value for item in proposal.unresolved_reasons),
            "policy_version": TASK_DRAFT_AUTHORING_POLICY_VERSION,
        }
        return TaskDraftAuthoringResult(
            task_draft_authoring_result_id=_stable_id(
                "task-draft-authoring-result",
                seed,
            ),
            request_ref=_request_ref(request),
            proposal_ref=_proposal_ref(proposal),
            outcome=proposal.outcome,
            task_draft=task_draft,
            unresolved_reasons=proposal.unresolved_reasons,
            result_sha256=_stable_hash(seed),
            audit=_safe_artifact_audit(
                audit,
                (
                    _request_ref(request),
                    _proposal_ref(proposal),
                    *((task_draft_ref(task_draft),) if task_draft is not None else ()),
                ),
            ),
        )


def _validate_authoritative_inputs(
    *,
    selection_context: SelectionContextV2,
    task_episodes: tuple[TaskEpisodeV2, ...],
    selection_evidence_bundle: EvidenceBundle,
    authoring_view_result: EvidenceViewResult,
    authoring_evidence_bundle: EvidenceBundle,
) -> None:
    if not task_episodes:
        raise TaskDraftAuthoringPolicyError("task-draft authoring requires TaskEpisodes")
    _validate_selection_context_identity(selection_context)
    _validate_bundle(
        selection_evidence_bundle,
        purpose=TASK_EPISODE_EVIDENCE_PURPOSE,
        require_evidence=True,
    )
    selection_bundle_ref = _evidence_bundle_ref(selection_evidence_bundle)
    if selection_context.safe_evidence_bundle_ref != selection_bundle_ref:
        raise TaskDraftAuthoringPolicyError("SelectionContext selection evidence bundle mismatch")
    context_ref = selection_context_ref(selection_context)
    episode_ids: set[str] = set()
    trace_ref = task_episodes[0].trace_envelope_ref
    for episode in task_episodes:
        _validate_task_episode_identity(episode)
        if episode.task_episode_id in episode_ids:
            raise TaskDraftAuthoringPolicyError("duplicate TaskEpisode identity")
        episode_ids.add(episode.task_episode_id)
        if episode.selection_context_ref != context_ref:
            raise TaskDraftAuthoringPolicyError("TaskEpisode SelectionContext mismatch")
        if episode.evidence_bundle_ref != selection_bundle_ref:
            raise TaskDraftAuthoringPolicyError("TaskEpisode selection evidence bundle mismatch")
        if episode.trace_envelope_ref != trace_ref:
            raise TaskDraftAuthoringPolicyError("TaskEpisode trace envelopes do not match")
    if trace_ref.object_id != selection_evidence_bundle.trace_ir_version_id:
        raise TaskDraftAuthoringPolicyError("TaskEpisode and selection evidence trace mismatch")

    if authoring_view_result.principal_type is not EvidenceViewPrincipalType.TASK_AUTHOR:
        raise TaskDraftAuthoringPolicyError("authoring view must use task-author principal")
    if authoring_view_result.purpose is not EvidenceViewPurpose.TASK_AUTHORING:
        raise TaskDraftAuthoringPolicyError("authoring view must use task-authoring purpose")
    if authoring_view_result.policy_version != EVIDENCE_VIEW_POLICY_VERSION:
        raise TaskDraftAuthoringPolicyError("authoring view policy is stale or invalid")
    if authoring_view_result.returned_characters > authoring_view_result.max_characters:
        raise TaskDraftAuthoringPolicyError("authoring view exceeds authorized budget")
    _validate_authoring_view_identity(authoring_view_result)
    _validate_bundle(
        authoring_evidence_bundle,
        purpose=TASK_DRAFT_EVIDENCE_PURPOSE,
        require_evidence=True,
    )
    if (
        authoring_evidence_bundle.trace_ir_version_id != selection_evidence_bundle.trace_ir_version_id
        or authoring_evidence_bundle.source_trace_id != selection_evidence_bundle.source_trace_id
    ):
        raise TaskDraftAuthoringPolicyError("authoring and selection evidence trace mismatch")
    expected_policy_ref = _projection_policy_ref(authoring_view_result.projection_policy)
    if authoring_evidence_bundle.projection_policy_ref != expected_policy_ref:
        raise TaskDraftAuthoringPolicyError("authoring view and bundle projection policy mismatch")
    view_refs = {_ref_key(item.projected_ref) for item in authoring_view_result.included_items}
    bundle_refs = {_ref_key(item.subject_ref) for item in authoring_evidence_bundle.evidence}
    if view_refs != bundle_refs:
        raise TaskDraftAuthoringPolicyError("authoring view and bundle projected subjects mismatch")


def _validate_authoring_view_identity(
    view_result: EvidenceViewResult,
) -> None:
    policy = view_result.projection_policy
    policy_seed = {
        "principal_type": policy.principal_type,
        "purpose": policy.purpose,
        "recursive_allow_fields": list(policy.recursive_allow_fields),
        "denied_object_types": list(policy.denied_object_types),
        "source_schema_versions": list(policy.source_schema_versions),
        "policy_version": policy.policy_version,
    }
    if policy.projection_policy_id != _stable_id(
        "projection-policy",
        policy_seed,
    ):
        raise TaskDraftAuthoringPolicyError("authoring projection policy identity is stale or invalid")

    returned_characters = 0
    projection_ids: set[str] = set()
    for item in view_result.included_items:
        if not is_safe_task_authoring_ref(item.source_ref) or not is_safe_task_authoring_ref(
            item.projected_ref
        ):
            raise TaskDraftAuthoringPolicyError("unsafe authoring projection reference")
        projected_seed = {
            "source_ref": _ref_payload(item.source_ref),
            "mode": item.projection_mode.value,
            "content": item.content,
            "structure_fields": [
                field.model_dump(mode="json", exclude_none=False)
                for field in sorted(
                    item.structure_fields,
                    key=lambda field: field.key,
                )
            ],
            "external_uri": item.external_uri,
            "policy_version": view_result.policy_version,
        }
        digest = _stable_hash(projected_seed)
        expected_ref = ObjectRef(
            object_type=f"{item.source_ref.object_type}-projection",
            object_id=f"evidence-view://sha256/{digest}",
            object_version=view_result.policy_version,
            object_sha256=digest,
        )
        if item.projected_ref != expected_ref:
            raise TaskDraftAuthoringPolicyError("authoring projection identity is stale or invalid")
        if item.child_decision.subject_ref != expected_ref or item.child_decision.subject_sha256 != digest:
            raise TaskDraftAuthoringPolicyError("authoring projection decision is stale or invalid")
        item_seed = {
            "source_ref": _ref_payload(item.source_ref),
            "projected_ref": _ref_payload(expected_ref),
            "mode": item.projection_mode.value,
            "child_decision_id": item.child_decision.provenance_decision_id,
            "policy_version": view_result.policy_version,
        }
        expected_item_id = _stable_id(
            "evidence-projection-item",
            item_seed,
        )
        if item.projection_item_id != expected_item_id:
            raise TaskDraftAuthoringPolicyError("authoring projection item identity is stale or invalid")
        if expected_item_id in projection_ids:
            raise TaskDraftAuthoringPolicyError("duplicate authoring projection identity")
        projection_ids.add(expected_item_id)
        returned_characters += len(item.content or "")
        returned_characters += len(item.external_uri or "")
        returned_characters += sum(
            len(field.value) if isinstance(field.value, str) else 0 for field in item.structure_fields
        )
    if returned_characters != view_result.returned_characters:
        raise TaskDraftAuthoringPolicyError("authoring view returned-character count is stale or invalid")


def _validate_bundle(
    bundle: EvidenceBundle,
    *,
    purpose: str,
    require_evidence: bool,
) -> None:
    if bundle.consumer_stage != TASK_EPISODE_CONSUMER_STAGE:
        raise TaskDraftAuthoringPolicyError("unexpected evidence bundle consumer stage")
    if bundle.purpose != purpose:
        raise TaskDraftAuthoringPolicyError("unexpected evidence bundle purpose")
    if bundle.projection_policy_ref.object_type != "projection-policy" or not is_safe_task_authoring_ref(
        bundle.projection_policy_ref
    ):
        raise TaskDraftAuthoringPolicyError("invalid evidence bundle projection policy")
    if bundle.model_dump(mode="python")["tainted_content_included"] is not False:
        raise TaskDraftAuthoringPolicyError("tainted evidence bundle cannot enter task authoring")
    if bundle.returned_characters > bundle.max_characters:
        raise TaskDraftAuthoringPolicyError("evidence bundle exceeds authorized budget")
    if require_evidence and not bundle.evidence:
        raise TaskDraftAuthoringPolicyError("task-draft authoring requires safe evidence")
    evidence_ids: set[str] = set()
    for evidence in bundle.evidence:
        if evidence.evidence_ref_id in evidence_ids:
            raise TaskDraftAuthoringPolicyError("duplicate evidence bundle identity")
        evidence_ids.add(evidence.evidence_ref_id)
        if not is_safe_task_authoring_ref(evidence.subject_ref):
            raise TaskDraftAuthoringPolicyError("unsafe evidence bundle reference")
        if any(span.source_trace_id != bundle.source_trace_id for span in evidence.source_spans):
            raise TaskDraftAuthoringPolicyError("evidence source trace does not match bundle source trace")
    expected_hash = _stable_hash(_bundle_seed(bundle))
    if (
        bundle.bundle_sha256 != expected_hash
        or bundle.evidence_bundle_id != f"evidence-bundle://sha256/{expected_hash}"
    ):
        raise TaskDraftAuthoringPolicyError("evidence bundle identity is stale or invalid")


def _task_draft_evidence_views(
    view_result: EvidenceViewResult,
    bundle: EvidenceBundle,
) -> tuple[TaskDraftEvidenceView, ...]:
    item_by_ref = {_ref_key(item.projected_ref): item for item in view_result.included_items}
    values: list[TaskDraftEvidenceView] = []
    for evidence in sorted(
        bundle.evidence,
        key=lambda item: item.evidence_ref_id,
    ):
        item = item_by_ref.get(_ref_key(evidence.subject_ref))
        if item is None:
            raise TaskDraftAuthoringPolicyError("authoring evidence is missing its safe projection")
        values.append(
            TaskDraftEvidenceView(
                evidence_ref_id=evidence.evidence_ref_id,
                subject_ref=evidence.subject_ref,
                projection_mode=item.projection_mode,
                content=item.content,
                structure_fields=item.structure_fields,
                external_uri=item.external_uri,
                capability=evidence.capability,
                capability_complete=evidence.capability_complete,
                untrusted_data_marker=True,
            )
        )
    return tuple(values)


def _enforce_prompt_as_data(
    *,
    authoring_view_result: EvidenceViewResult,
    authoring_evidence_bundle: EvidenceBundle,
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    item_by_ref = {_ref_key(item.projected_ref): item for item in authoring_view_result.included_items}
    bundle_ref = _evidence_bundle_ref(authoring_evidence_bundle)
    segments: list[PromptBoundarySegment] = []
    for evidence in sorted(
        authoring_evidence_bundle.evidence,
        key=lambda item: item.evidence_ref_id,
    ):
        item = item_by_ref[_ref_key(evidence.subject_ref)]
        preview = item.content
        if preview is not None and len(preview) > 2000:
            preview = None
        segments.append(
            PromptBoundarySegment(
                segment_id=f"prompt-boundary-segment://task-draft/{_stable_hash(evidence.evidence_ref_id)}",
                surface=PromptBoundarySurface.EVIDENCE_DATA,
                source_role=PromptBoundarySourceRole.SAFE_EVIDENCE_PROJECTION,
                source_ref=item.projected_ref,
                decision=item.child_decision,
                projection_policy_ref=authoring_evidence_bundle.projection_policy_ref,
                evidence_bundle_ref=bundle_ref,
                content_sha256=item.projected_ref.object_sha256,
                untrusted_data_marker=True,
                text_preview=preview,
            )
        )
    boundary_seed = {
        "segments": [
            {
                "segment_id": item.segment_id,
                "source_ref": _ref_payload(item.source_ref),
            }
            for item in segments
        ],
        "projection_policy_ref": _ref_payload(authoring_evidence_bundle.projection_policy_ref),
        "evidence_bundle_ref": _ref_payload(bundle_ref),
        "policy_version": TASK_DRAFT_AUTHORING_POLICY_VERSION,
    }
    try:
        return PromptInjectionBoundaryEnforcer().enforce(
            PromptBoundaryEnforcementRequest(
                boundary_id=_stable_id(
                    "prompt-boundary",
                    boundary_seed,
                ),
                segments=tuple(segments),
                approved_projection_policy_refs=(authoring_evidence_bundle.projection_policy_ref,),
                approved_evidence_bundle_refs=(bundle_ref,),
                collect_blocked_metadata=False,
                audit=audit,
            )
        )
    except PromptInjectionBoundaryPolicyError as exc:
        raise TaskDraftAuthoringPolicyError("task-author evidence failed prompt-as-data enforcement") from exc


def _validate_fixture(fixture: FakeTaskDraftAuthoringFixture) -> None:
    try:
        validate_outcome_shape(
            fixture.outcome,
            fixture.content,
            fixture.unresolved_reasons,
        )
    except ValueError as exc:
        raise TaskDraftAuthoringPolicyError(str(exc)) from exc
    if not fixture.model_available:
        if fixture.outcome is not TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY:
            raise TaskDraftAuthoringPolicyError("unavailable model must produce BLOCKED_CAPABILITY")
        if TaskDraftAuthoringReason.MODEL_UNAVAILABLE not in fixture.unresolved_reasons:
            raise TaskDraftAuthoringPolicyError("unavailable model requires MODEL_UNAVAILABLE reason")


def _validate_semantic_content(
    request: TaskDraftAuthoringRequest,
    content: TaskDraftContentFixture,
) -> None:
    evidence_ids = {item.evidence_ref_id for item in request.evidence_views}
    episode_ids = {ref.object_id for ref in request.task_episode_refs}
    for requirement in content.requirements:
        if any(item not in evidence_ids for item in requirement.evidence_ref_ids):
            raise TaskDraftAuthoringPolicyError("requirement selects unknown evidence")
        if any(item not in episode_ids for item in requirement.task_episode_ids):
            raise TaskDraftAuthoringPolicyError("requirement selects unknown episode")
        if requirement.conflict_status is RequirementConflict.UNRESOLVED:
            raise TaskDraftAuthoringPolicyError("drafted proposal contains unresolved requirement conflict")
    for dependency in content.attachment_dependencies:
        if any(item not in evidence_ids for item in dependency.evidence_ref_ids):
            raise TaskDraftAuthoringPolicyError("attachment selects unknown evidence")
    if not set(content.required_capability_ids).issubset(set(request.allowed_capability_ids)):
        raise TaskDraftAuthoringPolicyError("proposal selects unauthorized capability")
    if not set(content.allowed_tool_ids).issubset(set(request.allowed_tool_ids)):
        raise TaskDraftAuthoringPolicyError("proposal selects unauthorized tool")
    if not set(request.required_forbidden_outputs).issubset(set(content.forbidden_outputs)):
        raise TaskDraftAuthoringPolicyError("proposal is missing required forbidden output")
    requirement_ids = {item.requirement_id for item in content.requirements}
    if not set(content.prompt_requirement_ids).issubset(requirement_ids):
        raise TaskDraftAuthoringPolicyError("prompt requirement is not present in requirement lineage")
    critical_ids = {
        item.requirement_id for item in content.requirements if item.criticality.value == "CRITICAL"
    }
    if not critical_ids.issubset(set(content.prompt_requirement_ids)):
        raise TaskDraftAuthoringPolicyError("critical prompt requirement is missing")


def _normalize_content(
    content: TaskDraftContentFixture,
) -> TaskDraftContentFixture:
    return TaskDraftContentFixture(
        visible_prompt=content.visible_prompt,
        task_intent=content.task_intent,
        evaluation_claim=content.evaluation_claim,
        required_capability_ids=tuple(sorted(content.required_capability_ids)),
        allowed_tool_ids=tuple(sorted(content.allowed_tool_ids)),
        forbidden_outputs=tuple(sorted(content.forbidden_outputs)),
        attachment_dependencies=tuple(
            sorted(
                (
                    TaskDraftAttachmentFixture(
                        dependency_id=item.dependency_id,
                        description=item.description,
                        criticality=item.criticality,
                        evidence_priority=item.evidence_priority,
                        evidence_ref_ids=tuple(sorted(item.evidence_ref_ids)),
                    )
                    for item in content.attachment_dependencies
                ),
                key=lambda item: item.dependency_id,
            )
        ),
        requirements=tuple(
            sorted(
                (
                    TaskDraftRequirementFixture(
                        requirement_id=item.requirement_id,
                        statement=item.statement,
                        criticality=item.criticality,
                        evidence_priority=item.evidence_priority,
                        evidence_ref_ids=tuple(sorted(item.evidence_ref_ids)),
                        task_episode_ids=tuple(sorted(item.task_episode_ids)),
                        conflict_status=item.conflict_status,
                    )
                    for item in content.requirements
                ),
                key=lambda item: item.requirement_id,
            )
        ),
        prompt_requirement_ids=tuple(sorted(content.prompt_requirement_ids)),
        uncertainties=tuple(sorted(content.uncertainties)),
    )


def _compile_task_draft(
    *,
    request: TaskDraftAuthoringRequest,
    proposal: TaskDraftAuthoringProposal,
    selection_context: SelectionContextV2,
    task_episodes: tuple[TaskEpisodeV2, ...],
    selection_evidence_bundle: EvidenceBundle,
    authoring_evidence_bundle: EvidenceBundle,
    audit: ContractAudit,
) -> TaskDraftV2:
    content = proposal.content
    if content is None:
        raise TaskDraftAuthoringPolicyError("drafted proposal is missing task content")
    _validate_semantic_content(request, content)
    evidence_by_id = {item.evidence_ref_id: item for item in authoring_evidence_bundle.evidence}
    episode_ref_by_id = {item.task_episode_id: task_episode_ref(item) for item in task_episodes}
    lineage = tuple(
        TaskRequirementLineageV2(
            requirement_id=item.requirement_id,
            statement=item.statement,
            criticality=item.criticality,
            evidence_priority=item.evidence_priority,
            evidence=tuple(evidence_by_id[evidence_id] for evidence_id in item.evidence_ref_ids),
            task_episode_refs=tuple(episode_ref_by_id[episode_id] for episode_id in item.task_episode_ids),
            conflict_status=item.conflict_status,
        )
        for item in content.requirements
    )
    dependencies = tuple(
        AttachmentDependency(
            dependency_id=item.dependency_id,
            description=item.description,
            criticality=item.criticality,
            evidence_priority=item.evidence_priority,
            evidence=tuple(evidence_by_id[evidence_id] for evidence_id in item.evidence_ref_ids),
        )
        for item in content.attachment_dependencies
    )
    episode_refs = _sort_refs(tuple(task_episode_ref(item) for item in task_episodes))
    draft_seed = {
        "task_version": 1,
        "supersedes_task_draft_ref": None,
        "selection_context_ref": _ref_payload(selection_context_ref(selection_context)),
        "task_episode_refs": [_ref_payload(ref) for ref in episode_refs],
        "visible_prompt": content.visible_prompt,
        "task_intent": content.task_intent,
        "evaluation_claim": content.evaluation_claim,
        "required_capabilities": list(content.required_capability_ids),
        "allowed_tools": list(content.allowed_tool_ids),
        "forbidden_outputs": list(content.forbidden_outputs),
        "attachment_dependencies": [
            item.model_dump(mode="json", exclude_none=False) for item in dependencies
        ],
        "requirement_lineage": [item.model_dump(mode="json", exclude_none=False) for item in lineage],
        "prompt_requirement_ids": list(content.prompt_requirement_ids),
        "uncertainties": list(content.uncertainties),
        "prompt_safety_status": TaskDraftPromptSafetyStatusV2.PENDING.value,
        "prompt_safety_gate_ref": None,
        "model_profile": proposal.model_profile,
        "prompt_version": proposal.prompt_version,
        "policy_version": TASK_DRAFT_AUTHORING_POLICY_VERSION,
    }
    draft_hash = task_draft_payload_sha256(draft_seed)
    return TaskDraftV2(
        task_draft_id=f"task-draft://sha256/{draft_hash}",
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=selection_context_ref(selection_context),
        task_episode_refs=episode_refs,
        visible_prompt=content.visible_prompt,
        task_intent=content.task_intent,
        evaluation_claim=content.evaluation_claim,
        required_capabilities=content.required_capability_ids,
        allowed_tools=content.allowed_tool_ids,
        forbidden_outputs=content.forbidden_outputs,
        attachment_dependencies=dependencies,
        requirement_lineage=lineage,
        prompt_requirement_ids=content.prompt_requirement_ids,
        uncertainties=content.uncertainties,
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PENDING,
        prompt_safety_gate_ref=None,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
        policy_version=TASK_DRAFT_AUTHORING_POLICY_VERSION,
        task_draft_sha256=draft_hash,
        audit=_sanitized_audit(
            audit,
            selection_context=selection_context,
            task_episodes=task_episodes,
            selection_evidence_bundle=selection_evidence_bundle,
            authoring_evidence_bundle=authoring_evidence_bundle,
            prompt_boundary_enforcement_ref=request.prompt_boundary_enforcement_ref,
        ),
    )


def _validate_request_integrity(request: TaskDraftAuthoringRequest) -> None:
    seed = _request_seed(
        selection_context_ref=request.selection_context_ref,
        trace_envelope_ref=request.trace_envelope_ref,
        task_episode_refs=request.task_episode_refs,
        selection_evidence_bundle_ref=request.selection_evidence_bundle_ref,
        authoring_evidence_bundle_ref=request.authoring_evidence_bundle_ref,
        authoring_projection_policy_ref=request.authoring_projection_policy_ref,
        prompt_boundary_enforcement_ref=request.prompt_boundary_enforcement_ref,
        evidence_views=request.evidence_views,
        allowed_capability_ids=request.allowed_capability_ids,
        allowed_tool_ids=request.allowed_tool_ids,
        required_forbidden_outputs=request.required_forbidden_outputs,
        abstain_conditions=request.abstain_conditions,
        model_profile=request.model_profile,
        prompt_version=request.prompt_version,
        returned_characters=request.returned_characters,
        max_characters=request.max_characters,
    )
    if request.request_sha256 != _stable_hash(seed):
        raise TaskDraftAuthoringPolicyError("request hash is stale or invalid")
    if request.semantic_authoring_request_id != _stable_id(
        "task-draft-authoring-request",
        seed,
    ):
        raise TaskDraftAuthoringPolicyError("request identity is stale or invalid")


def _validate_proposal_integrity(
    proposal: TaskDraftAuthoringProposal,
) -> None:
    seed = _proposal_seed(
        request_ref=proposal.request_ref,
        selection_context_ref=proposal.selection_context_ref,
        task_episode_refs=proposal.task_episode_refs,
        selection_evidence_bundle_ref=proposal.selection_evidence_bundle_ref,
        authoring_evidence_bundle_ref=proposal.authoring_evidence_bundle_ref,
        outcome=proposal.outcome,
        content=proposal.content,
        unresolved_reasons=proposal.unresolved_reasons,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
    )
    if proposal.proposal_sha256 != _stable_hash(seed):
        raise TaskDraftAuthoringPolicyError("proposal hash is stale or invalid")
    if proposal.semantic_authoring_proposal_id != _stable_id(
        "task-draft-authoring-proposal",
        seed,
    ):
        raise TaskDraftAuthoringPolicyError("proposal identity is stale or invalid")


def _validate_proposal_binding(
    request: TaskDraftAuthoringRequest,
    proposal: TaskDraftAuthoringProposal,
) -> None:
    if proposal.request_ref != _request_ref(request):
        raise TaskDraftAuthoringPolicyError("proposal request ref is stale or mismatched")
    if proposal.selection_context_ref != request.selection_context_ref:
        raise TaskDraftAuthoringPolicyError("proposal SelectionContext ref is mismatched")
    if proposal.task_episode_refs != request.task_episode_refs:
        raise TaskDraftAuthoringPolicyError("proposal TaskEpisode refs are mismatched")
    if proposal.selection_evidence_bundle_ref != request.selection_evidence_bundle_ref:
        raise TaskDraftAuthoringPolicyError("proposal selection evidence bundle ref is mismatched")
    if proposal.authoring_evidence_bundle_ref != request.authoring_evidence_bundle_ref:
        raise TaskDraftAuthoringPolicyError("proposal authoring evidence bundle ref is mismatched")
    if proposal.model_profile != request.model_profile:
        raise TaskDraftAuthoringPolicyError("proposal model profile is mismatched")
    if proposal.prompt_version != request.prompt_version:
        raise TaskDraftAuthoringPolicyError("proposal prompt version is mismatched")


def _request_seed(
    *,
    selection_context_ref: ObjectRef,
    trace_envelope_ref: ObjectRef,
    task_episode_refs: tuple[ObjectRef, ...],
    selection_evidence_bundle_ref: ObjectRef,
    authoring_evidence_bundle_ref: ObjectRef,
    authoring_projection_policy_ref: ObjectRef,
    prompt_boundary_enforcement_ref: ObjectRef,
    evidence_views: tuple[TaskDraftEvidenceView, ...],
    allowed_capability_ids: tuple[str, ...],
    allowed_tool_ids: tuple[str, ...],
    required_forbidden_outputs: tuple[str, ...],
    abstain_conditions: tuple[str, ...],
    model_profile: str,
    prompt_version: str,
    returned_characters: int,
    max_characters: int,
) -> dict[str, object]:
    return {
        "selection_context_ref": _ref_payload(selection_context_ref),
        "trace_envelope_ref": _ref_payload(trace_envelope_ref),
        "task_episode_refs": [_ref_payload(ref) for ref in task_episode_refs],
        "selection_evidence_bundle_ref": _ref_payload(selection_evidence_bundle_ref),
        "authoring_evidence_bundle_ref": _ref_payload(authoring_evidence_bundle_ref),
        "authoring_projection_policy_ref": _ref_payload(authoring_projection_policy_ref),
        "prompt_boundary_enforcement_ref": _ref_payload(prompt_boundary_enforcement_ref),
        "evidence_views": [item.model_dump(mode="json", exclude_none=False) for item in evidence_views],
        "allowed_capability_ids": list(allowed_capability_ids),
        "allowed_tool_ids": list(allowed_tool_ids),
        "required_forbidden_outputs": list(required_forbidden_outputs),
        "abstain_conditions": list(abstain_conditions),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "returned_characters": returned_characters,
        "max_characters": max_characters,
        "policy_version": TASK_DRAFT_AUTHORING_POLICY_VERSION,
    }


def _proposal_seed(
    *,
    request_ref: ObjectRef,
    selection_context_ref: ObjectRef,
    task_episode_refs: tuple[ObjectRef, ...],
    selection_evidence_bundle_ref: ObjectRef,
    authoring_evidence_bundle_ref: ObjectRef,
    outcome: TaskDraftAuthoringOutcome,
    content: TaskDraftContentFixture | None,
    unresolved_reasons: frozenset[TaskDraftAuthoringReason],
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "selection_context_ref": _ref_payload(selection_context_ref),
        "task_episode_refs": [_ref_payload(ref) for ref in task_episode_refs],
        "selection_evidence_bundle_ref": _ref_payload(selection_evidence_bundle_ref),
        "authoring_evidence_bundle_ref": _ref_payload(authoring_evidence_bundle_ref),
        "outcome": outcome.value,
        "content": (None if content is None else content.model_dump(mode="json", exclude_none=False)),
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": TASK_DRAFT_AUTHORING_POLICY_VERSION,
    }


def _bundle_seed(bundle: EvidenceBundle) -> dict[str, object]:
    return {
        "source_trace_id": bundle.source_trace_id,
        "trace_ir_version_id": bundle.trace_ir_version_id,
        "consumer_stage": bundle.consumer_stage,
        "purpose": bundle.purpose,
        "projection_policy_ref": _ref_payload(bundle.projection_policy_ref),
        "evidence": [item.model_dump(mode="json", exclude_none=False) for item in bundle.evidence],
        "excluded_subject_refs": [_ref_payload(ref) for ref in bundle.excluded_subject_refs],
        "returned_characters": bundle.returned_characters,
        "max_characters": bundle.max_characters,
        "policy_version": EVIDENCE_COMPILATION_POLICY_VERSION,
    }


def _validate_selection_context_identity(
    context: SelectionContextV2,
) -> None:
    if context.selection_context_id != f"selection-context://sha256/{context.selection_context_sha256}":
        raise TaskDraftAuthoringPolicyError("SelectionContext identity is stale or invalid")


def _validate_task_episode_identity(episode: TaskEpisodeV2) -> None:
    if episode.task_episode_id != f"task-episode://sha256/{episode.task_episode_sha256}":
        raise TaskDraftAuthoringPolicyError("TaskEpisode identity is stale or invalid")


def _sanitized_audit(
    audit: ContractAudit,
    *,
    selection_context: SelectionContextV2,
    task_episodes: tuple[TaskEpisodeV2, ...],
    selection_evidence_bundle: EvidenceBundle,
    authoring_evidence_bundle: EvidenceBundle,
    prompt_boundary_enforcement: PromptBoundaryEnforcementResult | None = None,
    prompt_boundary_enforcement_ref: ObjectRef | None = None,
) -> ContractAudit:
    boundary_ref = prompt_boundary_enforcement_ref
    if boundary_ref is None and prompt_boundary_enforcement is not None:
        boundary_ref = _prompt_boundary_ref(prompt_boundary_enforcement)
    refs = (
        selection_context_ref(selection_context),
        *(task_episode_ref(item) for item in task_episodes),
        _evidence_bundle_ref(selection_evidence_bundle),
        _evidence_bundle_ref(authoring_evidence_bundle),
        authoring_evidence_bundle.projection_policy_ref,
        *((boundary_ref,) if boundary_ref is not None else ()),
    )
    unique = {_ref_key(ref): ref for ref in refs}
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(unique[key] for key in sorted(unique)),
    )


def _safe_artifact_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    unique = {_ref_key(ref): ref for ref in refs}
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(unique[key] for key in sorted(unique)),
    )


def _evidence_bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return ObjectRef(
        object_type="evidence-bundle",
        object_id=bundle.evidence_bundle_id,
        object_version="v1",
        object_sha256=bundle.bundle_sha256,
    )


def _projection_policy_ref(policy: ProjectionPolicy) -> ObjectRef:
    return ObjectRef(
        object_type="projection-policy",
        object_id=policy.projection_policy_id,
        object_version=policy.policy_version,
        object_sha256=policy.canonical_sha256(),
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


def _request_ref(request: TaskDraftAuthoringRequest) -> ObjectRef:
    return ObjectRef(
        object_type="task-draft-authoring-request",
        object_id=request.semantic_authoring_request_id,
        object_version=request.policy_version,
        object_sha256=request.request_sha256,
    )


def _proposal_ref(proposal: TaskDraftAuthoringProposal) -> ObjectRef:
    return ObjectRef(
        object_type="task-draft-authoring-proposal",
        object_id=proposal.semantic_authoring_proposal_id,
        object_version=proposal.policy_version,
        object_sha256=proposal.proposal_sha256,
    )


def _sort_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(refs, key=_ref_key))


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
