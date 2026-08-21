from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import PurePosixPath

from eval_factory.attachment_planning.bridge import AttachmentPlanningBridge
from eval_factory.attachment_planning.dependency_models import (
    PROMPT_ONLY_DEPENDENCY_PLANNING_POLICY_VERSION,
    PROMPT_ONLY_DEPENDENCY_POLICY_VERSION,
    FakePromptOnlyDependencyFixture,
    PromptOnlyCandidateRole,
    PromptOnlyDependencyAssessment,
    PromptOnlyDependencyDiscoveryProposal,
    PromptOnlyDependencyDiscoveryRequest,
    PromptOnlyDependencyDiscoveryResult,
    PromptOnlyDependencyOutcome,
    PromptOnlyDependencyPolicyError,
    PromptOnlyDependencyReason,
    PromptOnlyDependencyView,
    PromptOnlyPathFact,
    PromptOnlyPathFactSource,
)
from eval_factory.attachment_planning.models import (
    ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
    AttachmentPlanningPolicyError,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    AttachmentPlanningContextV2,
    PromptOnlyDependencyDiscoveryV2,
    PromptOnlyDependencyEvidenceBindingV2,
    PromptOnlyDependencyPlanningContextV2,
    artifact_evidence_target_carried_sha256,
    artifact_evidence_target_ref,
    attachment_planning_context_carried_sha256,
    attachment_planning_context_ref,
    prompt_only_dependency_discovery_carried_sha256,
    prompt_only_dependency_discovery_ref,
    prompt_only_dependency_planning_context_carried_sha256,
    prompt_only_dependency_planning_context_ref,
    validate_artifact_evidence_target_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task import EvidencePriority
from eval_factory.contracts.task_v2 import (
    ProducerAttachmentRequirementV2,
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    producer_task_view_carried_sha256,
    producer_task_view_ref,
    task_draft_carried_sha256,
    task_draft_ref,
)
from eval_factory.provenance.injection import (
    PromptBoundaryEnforcementRequest,
    PromptBoundaryEnforcementResult,
    PromptBoundarySegment,
    PromptBoundarySourceRole,
    PromptBoundarySurface,
    PromptInjectionBoundaryEnforcer,
    PromptInjectionBoundaryPolicyError,
    PromptOnlyDependencyDataBoundaryRequest,
)

_PATH_PATTERN = re.compile(
    r"(?<![\w./\\:-])"
    r"(?:[\w.-]+[\\/])*[\w.-]+"
    r"\.(?:txt|md|csv|json|ya?ml|xlsx|docx|pdf|zip|py|js|ts|html|css)"
    r"(?![\w-])",
    flags=re.IGNORECASE | re.UNICODE,
)
_EXTENSION_TO_MEDIA_TYPE = {
    ".css": "text/css",
    ".csv": "text/csv",
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".py": "text/x-python",
    ".ts": "text/typescript",
    ".txt": "text/plain",
    ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".zip": "application/zip",
}
_MEDIA_TYPE_TO_EXTENSION = {
    media_type: extension for extension, media_type in _EXTENSION_TO_MEDIA_TYPE.items()
}


class PromptOnlyDependencyPlanningProjector:
    policy_version = PROMPT_ONLY_DEPENDENCY_PLANNING_POLICY_VERSION

    def compile(
        self,
        *,
        task_draft: TaskDraftV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        attachment_planning_context: AttachmentPlanningContextV2,
        audit: ContractAudit,
    ) -> PromptOnlyDependencyPlanningContextV2:
        try:
            AttachmentPlanningBridge().validate_current(
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=evidence_bundle,
                attachment_planning_context=attachment_planning_context,
            )
        except AttachmentPlanningPolicyError as exc:
            raise PromptOnlyDependencyPolicyError(
                "prompt-only dependency source identity is stale or invalid"
            ) from exc
        draft_digest = task_draft_carried_sha256(task_draft)
        if (
            task_draft.task_draft_sha256 != draft_digest
            or task_draft.task_draft_id != f"task-draft://sha256/{draft_digest}"
        ):
            raise PromptOnlyDependencyPolicyError("prompt-only dependency TaskDraft identity is stale")
        if (
            task_draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
            or task_draft.prompt_safety_gate_ref is None
        ):
            raise PromptOnlyDependencyPolicyError(
                "prompt-only dependency planning requires a PASSED TaskDraft"
            )
        if (
            task_draft.task_draft_sha256 != producer_task_view.source_task_draft_sha256
            or task_draft.task_draft_sha256 != attachment_planning_context.source_task_draft_sha256
        ):
            raise PromptOnlyDependencyPolicyError(
                "prompt-only dependency TaskDraft binding is stale or mismatched"
            )
        draft_dependencies = {item.dependency_id: item for item in task_draft.attachment_dependencies}
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        if set(draft_dependencies) != set(requirements):
            raise PromptOnlyDependencyPolicyError("prompt-only dependency inventory is mismatched")
        bindings: list[PromptOnlyDependencyEvidenceBindingV2] = []
        evidence_subject_refs: list[ObjectRef] = []
        for dependency_id in sorted(requirements):
            dependency = draft_dependencies[dependency_id]
            requirement = requirements[dependency_id]
            if (
                dependency.description != requirement.description
                or dependency.criticality is not requirement.criticality
            ):
                raise PromptOnlyDependencyPolicyError("prompt-only dependency declaration is mismatched")
            evidence = tuple(
                sorted(
                    dependency.evidence,
                    key=lambda item: item.evidence_ref_id,
                )
            )
            if any(
                item.polarity is not EvidencePolarity.POSITIVE
                or not item.capability_complete
                or any(
                    span.source_trace_id != attachment_planning_context.source_trace_id
                    for span in item.source_spans
                )
                for item in evidence
            ):
                raise PromptOnlyDependencyPolicyError(
                    "prompt-only dependency evidence is unsafe or mismatched"
                )
            try:
                binding = PromptOnlyDependencyEvidenceBindingV2(
                    attachment_dependency_id=dependency_id,
                    evidence_priority=dependency.evidence_priority,
                    evidence=evidence,
                )
            except ValueError as exc:
                raise PromptOnlyDependencyPolicyError("prompt-only dependency evidence is invalid") from exc
            bindings.append(binding)
            evidence_subject_refs.extend(item.subject_ref for item in evidence)
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        view_ref = producer_task_view_ref(producer_task_view)
        value = PromptOnlyDependencyPlanningContextV2(
            prompt_only_dependency_planning_context_id=("prompt-only-dependency-planning-context://pending"),
            attachment_planning_context_ref=context_ref,
            producer_task_view_ref=view_ref,
            task_prompt_safety_gate_ref=task_draft.prompt_safety_gate_ref,
            source_task_draft_sha256=task_draft.task_draft_sha256,
            source_trace_id=attachment_planning_context.source_trace_id,
            dependency_evidence_bindings=tuple(bindings),
            policy_version=self.policy_version,
            prompt_only_dependency_planning_context_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    context_ref,
                    view_ref,
                    task_draft_ref(task_draft),
                    task_draft.prompt_safety_gate_ref,
                    *evidence_subject_refs,
                ),
            ),
        )
        digest = prompt_only_dependency_planning_context_carried_sha256(value)
        return value.model_copy(
            update={
                "prompt_only_dependency_planning_context_id": (
                    f"prompt-only-dependency-planning-context://sha256/{digest}"
                ),
                "prompt_only_dependency_planning_context_sha256": digest,
            }
        )

    def validate_current(
        self,
        *,
        task_draft: TaskDraftV2,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        attachment_planning_context: AttachmentPlanningContextV2,
        dependency_planning_context: PromptOnlyDependencyPlanningContextV2,
    ) -> None:
        digest = prompt_only_dependency_planning_context_carried_sha256(dependency_planning_context)
        if (
            dependency_planning_context.prompt_only_dependency_planning_context_sha256 != digest
            or dependency_planning_context.prompt_only_dependency_planning_context_id
            != f"prompt-only-dependency-planning-context://sha256/{digest}"
        ):
            raise PromptOnlyDependencyPolicyError("current prompt-only dependency planning context is stale")
        rebuilt = self.compile(
            task_draft=task_draft,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
            attachment_planning_context=attachment_planning_context,
            audit=dependency_planning_context.audit,
        )
        if not _same_planning_context_except_audit_actor_time(
            rebuilt,
            dependency_planning_context,
        ):
            raise PromptOnlyDependencyPolicyError(
                "current prompt-only dependency planning context does not match authoritative inputs"
            )


class PromptOnlyDependencyRequestBuilder:
    policy_version = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION

    def build(
        self,
        *,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        dependency_planning_context: PromptOnlyDependencyPlanningContextV2,
        existing_targets: tuple[ArtifactEvidenceTargetV2, ...],
        model_profile: Identifier,
        prompt_version: str,
        abstain_conditions: tuple[str, ...],
        audit: ContractAudit,
    ) -> PromptOnlyDependencyDiscoveryRequest:
        _validate_source_context(
            attachment_planning_context,
            producer_task_view,
            dependency_planning_context,
        )
        requirements = {item.dependency_id: item for item in producer_task_view.attachment_requirements}
        bindings = {
            item.attachment_dependency_id: item
            for item in dependency_planning_context.dependency_evidence_bindings
        }
        if set(requirements) != set(bindings):
            raise PromptOnlyDependencyPolicyError("prompt-only dependency evidence coverage is incomplete")
        existing_by_dependency = _validate_existing_targets(
            existing_targets,
            attachment_planning_context,
            requirements,
            bindings,
        )
        dependency_views = tuple(
            PromptOnlyDependencyView(
                dependency_id=dependency_id,
                description=requirements[dependency_id].description,
                criticality=requirements[dependency_id].criticality,
                evidence_priority=bindings[dependency_id].evidence_priority,
                explicit_path_facts=_path_facts(
                    text=requirements[dependency_id].description,
                    source=(PromptOnlyPathFactSource.REQUIREMENT_DESCRIPTION),
                    dependency_id=dependency_id,
                ),
                existing_target_ref=(
                    artifact_evidence_target_ref(existing_by_dependency[dependency_id])
                    if dependency_id in existing_by_dependency
                    else None
                ),
            )
            for dependency_id in sorted(requirements)
        )
        forbidden_outputs = tuple(sorted(producer_task_view.forbidden_outputs))
        forbidden_facts = tuple(
            sorted(
                (
                    fact
                    for text in forbidden_outputs
                    for fact in _path_facts(
                        text=text,
                        source=PromptOnlyPathFactSource.FORBIDDEN_OUTPUT,
                        dependency_id=None,
                    )
                ),
                key=lambda item: item.fact_id,
            )
        )
        conditions = tuple(sorted(abstain_conditions))
        context_ref = attachment_planning_context_ref(attachment_planning_context)
        view_ref = producer_task_view_ref(producer_task_view)
        planning_ref = prompt_only_dependency_planning_context_ref(dependency_planning_context)
        projection_payload = _projection_payload(
            context_ref=context_ref,
            view_ref=view_ref,
            planning_ref=planning_ref,
            query_instruction=producer_task_view.query_instruction,
            dependencies=dependency_views,
            forbidden_outputs=forbidden_outputs,
            forbidden_path_facts=forbidden_facts,
            model_profile=model_profile,
            prompt_version=prompt_version,
            abstain_conditions=conditions,
        )
        projection_ref = _projection_ref(projection_payload)
        boundary = _enforce_prompt_only_dependency_boundary(
            context_ref=context_ref,
            view_ref=view_ref,
            planning_ref=planning_ref,
            projection_ref=projection_ref,
            audit=audit,
        )
        boundary_ref = _prompt_boundary_ref(boundary)
        seed = _request_seed(
            context_ref=context_ref,
            view_ref=view_ref,
            planning_ref=planning_ref,
            query_instruction=producer_task_view.query_instruction,
            dependencies=dependency_views,
            forbidden_outputs=forbidden_outputs,
            forbidden_path_facts=forbidden_facts,
            prompt_boundary_ref=boundary_ref,
            model_profile=model_profile,
            prompt_version=prompt_version,
            abstain_conditions=conditions,
        )
        digest = _payload_sha256(seed)
        return PromptOnlyDependencyDiscoveryRequest(
            request_id=f"prompt-only-dependency-request://sha256/{digest}",
            attachment_planning_context_ref=context_ref,
            producer_task_view_ref=view_ref,
            dependency_planning_context_ref=planning_ref,
            query_instruction=producer_task_view.query_instruction,
            dependencies=dependency_views,
            forbidden_outputs=forbidden_outputs,
            forbidden_path_facts=forbidden_facts,
            prompt_boundary_enforcement_ref=boundary_ref,
            model_profile=model_profile,
            prompt_version=prompt_version,
            abstain_conditions=conditions,
            policy_version=self.policy_version,
            request_sha256=digest,
            audit=_safe_audit(
                audit,
                (
                    context_ref,
                    view_ref,
                    planning_ref,
                    boundary_ref,
                    *tuple(artifact_evidence_target_ref(item) for item in existing_targets),
                ),
            ),
        )


class FakePromptOnlyDependencyRunner:
    policy_version = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION

    def run(
        self,
        request: PromptOnlyDependencyDiscoveryRequest,
        *,
        fixture: FakePromptOnlyDependencyFixture,
        audit: ContractAudit,
    ) -> PromptOnlyDependencyDiscoveryProposal:
        _validate_request_identity(request)
        try:
            fixture = FakePromptOnlyDependencyFixture.model_validate(fixture.model_dump(mode="python"))
        except ValueError as exc:
            raise PromptOnlyDependencyPolicyError("prompt-only dependency fixture is invalid") from exc
        unresolved = {item.dependency_id for item in request.dependencies if item.existing_target_ref is None}
        assessment_ids = {item.dependency_id for item in fixture.assessments}
        if not assessment_ids.issubset(unresolved):
            raise PromptOnlyDependencyPolicyError(
                "semantic assessment selects an unknown or resolved dependency"
            )
        if fixture.outcome is PromptOnlyDependencyOutcome.RESOLVED:
            semantic_required = {
                item.dependency_id
                for item in request.dependencies
                if item.existing_target_ref is None and not _one_explicit_complete_fact(item)
            }
            if assessment_ids != semantic_required:
                raise PromptOnlyDependencyPolicyError(
                    "semantic assessments do not exactly cover residual dependencies"
                )
        seed = _proposal_seed(
            request_ref=_request_ref(request),
            outcome=fixture.outcome,
            assessments=tuple(
                sorted(
                    fixture.assessments,
                    key=lambda item: item.dependency_id,
                )
            ),
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
        )
        digest = _payload_sha256(seed)
        return PromptOnlyDependencyDiscoveryProposal(
            proposal_id=(f"prompt-only-dependency-proposal://sha256/{digest}"),
            request_ref=_request_ref(request),
            outcome=fixture.outcome,
            assessments=tuple(
                sorted(
                    fixture.assessments,
                    key=lambda item: item.dependency_id,
                )
            ),
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            policy_version=self.policy_version,
            proposal_sha256=digest,
            audit=_safe_audit(audit, (_request_ref(request),)),
        )


class PromptOnlyDependencyCompiler:
    policy_version = PROMPT_ONLY_DEPENDENCY_POLICY_VERSION

    def compile(
        self,
        *,
        request: PromptOnlyDependencyDiscoveryRequest,
        proposal: PromptOnlyDependencyDiscoveryProposal | None,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        dependency_planning_context: PromptOnlyDependencyPlanningContextV2,
        existing_targets: tuple[ArtifactEvidenceTargetV2, ...],
        audit: ContractAudit,
    ) -> PromptOnlyDependencyDiscoveryResult:
        _validate_request_identity(request)
        _validate_source_context(
            attachment_planning_context,
            producer_task_view,
            dependency_planning_context,
        )
        expected = PromptOnlyDependencyRequestBuilder().build(
            attachment_planning_context=attachment_planning_context,
            producer_task_view=producer_task_view,
            dependency_planning_context=dependency_planning_context,
            existing_targets=existing_targets,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            abstain_conditions=request.abstain_conditions,
            audit=request.audit,
        )
        if not _same_request_except_audit_actor_time(expected, request):
            raise PromptOnlyDependencyPolicyError("prompt-only dependency request is stale or mismatched")
        if proposal is not None:
            _validate_proposal_identity(proposal)
            if (
                proposal.request_ref != _request_ref(request)
                or proposal.model_profile != request.model_profile
                or proposal.prompt_version != request.prompt_version
            ):
                raise PromptOnlyDependencyPolicyError(
                    "prompt-only dependency proposal is stale or mismatched"
                )

        if not request.dependencies:
            if proposal is not None and proposal.outcome is not PromptOnlyDependencyOutcome.NOT_REQUIRED:
                raise PromptOnlyDependencyPolicyError(
                    "semantic proposal is not allowed without attachment dependencies"
                )
            return _result(
                request=request,
                proposal=proposal,
                outcome=PromptOnlyDependencyOutcome.NOT_REQUIRED,
                discovery=None,
                unresolved_dependency_ids=(),
                reasons=frozenset(),
                audit=audit,
            )

        unresolved_views = tuple(item for item in request.dependencies if item.existing_target_ref is None)
        binding_by_dependency = {
            item.attachment_dependency_id: item
            for item in dependency_planning_context.dependency_evidence_bindings
        }
        requirement_by_dependency = {
            item.dependency_id: item for item in producer_task_view.attachment_requirements
        }
        existing_by_dependency = _validate_existing_targets(
            existing_targets,
            attachment_planning_context,
            requirement_by_dependency,
            binding_by_dependency,
        )
        forbidden_dependency_ids = {
            view.dependency_id
            for view in unresolved_views
            if any(
                _matches_forbidden_path(
                    fact.logical_path,
                    request.forbidden_path_facts,
                )
                for fact in view.explicit_path_facts
            )
        }
        forbidden_dependency_ids.update(
            dependency_id
            for dependency_id, target in existing_by_dependency.items()
            if _matches_forbidden_path(
                target.logical_path,
                request.forbidden_path_facts,
            )
        )
        if forbidden_dependency_ids:
            return _result(
                request=request,
                proposal=proposal,
                outcome=PromptOnlyDependencyOutcome.BLOCKED_SAFETY,
                discovery=None,
                unresolved_dependency_ids=tuple(sorted(forbidden_dependency_ids)),
                reasons=frozenset({PromptOnlyDependencyReason.FORBIDDEN_OUTPUT_MATCH}),
                audit=audit,
            )

        if proposal is not None:
            _validate_proposal_scope(request, proposal)

        assessments = {} if proposal is None else {item.dependency_id: item for item in proposal.assessments}
        requested_output_ids = tuple(
            sorted(
                dependency_id
                for dependency_id, assessment in assessments.items()
                if assessment.candidate_role
                in {
                    PromptOnlyCandidateRole.REQUESTED_OUTPUT,
                    PromptOnlyCandidateRole.COMPLETED_DELIVERABLE,
                }
            )
        )
        if requested_output_ids:
            return _result(
                request=request,
                proposal=proposal,
                outcome=PromptOnlyDependencyOutcome.BLOCKED_SAFETY,
                discovery=None,
                unresolved_dependency_ids=requested_output_ids,
                reasons=frozenset({PromptOnlyDependencyReason.REQUESTED_OUTPUT_PROPOSED}),
                audit=audit,
            )

        direct_missing = tuple(
            item.dependency_id
            for item in unresolved_views
            if item.evidence_priority is EvidencePriority.DIRECT_OBSERVATION
        )
        if direct_missing:
            return _result(
                request=request,
                proposal=proposal,
                outcome=(PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY),
                discovery=None,
                unresolved_dependency_ids=tuple(sorted(direct_missing)),
                reasons=frozenset({PromptOnlyDependencyReason.DIRECT_TARGET_REQUIRED}),
                audit=audit,
            )

        ambiguous = tuple(
            item.dependency_id
            for item in unresolved_views
            if len({fact.logical_path for fact in item.explicit_path_facts}) > 1
        )
        if ambiguous:
            return _result(
                request=request,
                proposal=proposal,
                outcome=PromptOnlyDependencyOutcome.ABSTAIN,
                discovery=None,
                unresolved_dependency_ids=tuple(sorted(ambiguous)),
                reasons=frozenset({PromptOnlyDependencyReason.AMBIGUOUS_EXPLICIT_FACTS}),
                audit=audit,
            )

        if proposal is not None and proposal.outcome is not (PromptOnlyDependencyOutcome.RESOLVED):
            return _result(
                request=request,
                proposal=proposal,
                outcome=proposal.outcome,
                discovery=None,
                unresolved_dependency_ids=tuple(sorted(item.dependency_id for item in unresolved_views)),
                reasons=proposal.unresolved_reasons,
                audit=audit,
            )

        discovered: list[ArtifactEvidenceTargetV2] = []
        deterministic_ids: list[str] = []
        semantic_ids: list[str] = []

        for view in unresolved_views:
            explicit_fact = view.explicit_path_facts[0] if len(view.explicit_path_facts) == 1 else None
            assessment = assessments.get(view.dependency_id)
            if explicit_fact is not None:
                path = explicit_fact.logical_path
                media_type = explicit_fact.media_type
                deterministic_ids.append(view.dependency_id)
            else:
                if assessment is None:
                    return _result(
                        request=request,
                        proposal=proposal,
                        outcome=(PromptOnlyDependencyOutcome.BLOCKED_CAPABILITY),
                        discovery=None,
                        unresolved_dependency_ids=(view.dependency_id,),
                        reasons=frozenset({PromptOnlyDependencyReason.MEDIA_TYPE_UNRESOLVED}),
                        audit=audit,
                    )
                role_outcome = _role_outcome(assessment)
                if role_outcome is not None:
                    outcome, reason = role_outcome
                    return _result(
                        request=request,
                        proposal=proposal,
                        outcome=outcome,
                        discovery=None,
                        unresolved_dependency_ids=(view.dependency_id,),
                        reasons=frozenset({reason}),
                        audit=audit,
                    )
                media_type = assessment.proposed_media_type
                if media_type is None:
                    return _result(
                        request=request,
                        proposal=proposal,
                        outcome=PromptOnlyDependencyOutcome.ABSTAIN,
                        discovery=None,
                        unresolved_dependency_ids=(view.dependency_id,),
                        reasons=frozenset({PromptOnlyDependencyReason.MEDIA_TYPE_UNRESOLVED}),
                        audit=audit,
                    )
                path = _neutral_path(view.dependency_id, media_type)
                semantic_ids.append(view.dependency_id)
            if media_type is None:
                return _result(
                    request=request,
                    proposal=proposal,
                    outcome=PromptOnlyDependencyOutcome.ABSTAIN,
                    discovery=None,
                    unresolved_dependency_ids=(view.dependency_id,),
                    reasons=frozenset({PromptOnlyDependencyReason.MEDIA_TYPE_UNRESOLVED}),
                    audit=audit,
                )
            if _matches_forbidden_path(
                path,
                request.forbidden_path_facts,
            ):
                return _result(
                    request=request,
                    proposal=proposal,
                    outcome=PromptOnlyDependencyOutcome.BLOCKED_SAFETY,
                    discovery=None,
                    unresolved_dependency_ids=(view.dependency_id,),
                    reasons=frozenset({PromptOnlyDependencyReason.FORBIDDEN_OUTPUT_MATCH}),
                    audit=audit,
                )
            discovered.append(
                _compile_target(
                    context=attachment_planning_context,
                    requirement=requirement_by_dependency[view.dependency_id],
                    binding=binding_by_dependency[view.dependency_id],
                    logical_path=path,
                    media_type=media_type,
                    planning_ref=prompt_only_dependency_planning_context_ref(dependency_planning_context),
                    audit=audit,
                )
            )

        all_targets = tuple(
            sorted(
                (*existing_by_dependency.values(), *discovered),
                key=lambda item: (item.logical_path, item.artifact_id),
            )
        )
        dependency_ids = {item.attachment_dependency_id for item in all_targets}
        if dependency_ids != set(requirement_by_dependency):
            raise PromptOnlyDependencyPolicyError(
                "prompt-only dependency targets do not exactly cover requirements"
            )
        _require_unique(
            "prompt-only artifact IDs",
            tuple(item.artifact_id for item in all_targets),
        )
        _require_unique(
            "prompt-only logical paths",
            tuple(item.logical_path for item in all_targets),
        )
        existing_refs = tuple(
            sorted(
                (artifact_evidence_target_ref(item) for item in existing_by_dependency.values()),
                key=_ref_key,
            )
        )
        discovered_refs = tuple(
            sorted(
                (artifact_evidence_target_ref(item) for item in discovered),
                key=_ref_key,
            )
        )
        discovery = PromptOnlyDependencyDiscoveryV2(
            prompt_only_dependency_discovery_id=("prompt-only-dependency-discovery://pending"),
            attachment_planning_context_ref=(attachment_planning_context_ref(attachment_planning_context)),
            producer_task_view_ref=producer_task_view_ref(producer_task_view),
            dependency_planning_context_ref=(
                prompt_only_dependency_planning_context_ref(dependency_planning_context)
            ),
            prompt_boundary_enforcement_ref=(request.prompt_boundary_enforcement_ref),
            targets=all_targets,
            existing_target_refs=existing_refs,
            discovered_target_refs=discovered_refs,
            deterministic_dependency_ids=tuple(sorted(deterministic_ids)),
            semantic_dependency_ids=tuple(sorted(semantic_ids)),
            semantic_evaluated=bool(semantic_ids),
            model_profile=(request.model_profile if semantic_ids else None),
            prompt_version=(request.prompt_version if semantic_ids else None),
            policy_version=self.policy_version,
            prompt_only_dependency_discovery_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    request.attachment_planning_context_ref,
                    request.producer_task_view_ref,
                    request.dependency_planning_context_ref,
                    request.prompt_boundary_enforcement_ref,
                    *existing_refs,
                    *discovered_refs,
                    *((_proposal_ref(proposal),) if proposal else ()),
                ),
            ),
        )
        digest = prompt_only_dependency_discovery_carried_sha256(discovery)
        discovery = discovery.model_copy(
            update={
                "prompt_only_dependency_discovery_id": (
                    f"prompt-only-dependency-discovery://sha256/{digest}"
                ),
                "prompt_only_dependency_discovery_sha256": digest,
            }
        )
        return _result(
            request=request,
            proposal=proposal,
            outcome=PromptOnlyDependencyOutcome.RESOLVED,
            discovery=discovery,
            unresolved_dependency_ids=(),
            reasons=frozenset(),
            audit=audit,
        )

    def validate_current(
        self,
        *,
        request: PromptOnlyDependencyDiscoveryRequest,
        proposal: PromptOnlyDependencyDiscoveryProposal | None,
        attachment_planning_context: AttachmentPlanningContextV2,
        producer_task_view: ProducerTaskViewV2,
        dependency_planning_context: PromptOnlyDependencyPlanningContextV2,
        existing_targets: tuple[ArtifactEvidenceTargetV2, ...],
        discovery: PromptOnlyDependencyDiscoveryV2,
    ) -> None:
        try:
            digest = prompt_only_dependency_discovery_carried_sha256(discovery)
            if (
                discovery.prompt_only_dependency_discovery_sha256 != digest
                or discovery.prompt_only_dependency_discovery_id
                != f"prompt-only-dependency-discovery://sha256/{digest}"
            ):
                raise PromptOnlyDependencyPolicyError("discovery identity is stale")
            for target in discovery.targets:
                validate_artifact_evidence_target_identity(target)
            rebuilt = self.compile(
                request=request,
                proposal=proposal,
                attachment_planning_context=(attachment_planning_context),
                producer_task_view=producer_task_view,
                dependency_planning_context=(dependency_planning_context),
                existing_targets=existing_targets,
                audit=discovery.audit,
            )
            if (
                rebuilt.outcome is not PromptOnlyDependencyOutcome.RESOLVED
                or rebuilt.discovery is None
                or not _same_discovery_except_audit_actor_time(
                    rebuilt.discovery,
                    discovery,
                )
            ):
                raise PromptOnlyDependencyPolicyError("discovery does not match authoritative inputs")
        except (PromptOnlyDependencyPolicyError, ValueError) as exc:
            raise PromptOnlyDependencyPolicyError(
                f"current prompt-only dependency validation failed: {exc}"
            ) from exc


def _validate_source_context(
    context: AttachmentPlanningContextV2,
    view: ProducerTaskViewV2,
    planning: PromptOnlyDependencyPlanningContextV2,
) -> None:
    context_digest = attachment_planning_context_carried_sha256(context)
    if (
        context.attachment_planning_context_sha256 != context_digest
        or context.attachment_planning_context_id != f"attachment-planning-context://sha256/{context_digest}"
    ):
        raise PromptOnlyDependencyPolicyError("attachment planning context identity is stale")
    view_digest = producer_task_view_carried_sha256(view)
    if (
        view.producer_task_view_sha256 != view_digest
        or view.producer_task_view_id != f"producer-task-view://sha256/{view_digest}"
    ):
        raise PromptOnlyDependencyPolicyError("producer task view identity is stale")
    planning_digest = prompt_only_dependency_planning_context_carried_sha256(planning)
    if (
        planning.prompt_only_dependency_planning_context_sha256 != planning_digest
        or planning.prompt_only_dependency_planning_context_id
        != f"prompt-only-dependency-planning-context://sha256/{planning_digest}"
    ):
        raise PromptOnlyDependencyPolicyError("dependency planning context identity is stale")
    if (
        planning.attachment_planning_context_ref != attachment_planning_context_ref(context)
        or planning.producer_task_view_ref != producer_task_view_ref(view)
        or planning.source_task_draft_sha256 != context.source_task_draft_sha256
        or planning.source_task_draft_sha256 != view.source_task_draft_sha256
        or planning.source_trace_id != context.source_trace_id
    ):
        raise PromptOnlyDependencyPolicyError("dependency planning source graph is mismatched")
    if planning.policy_version != PROMPT_ONLY_DEPENDENCY_PLANNING_POLICY_VERSION:
        raise PromptOnlyDependencyPolicyError("dependency planning policy is stale or unsupported")


def _validate_existing_targets(
    targets: tuple[ArtifactEvidenceTargetV2, ...],
    context: AttachmentPlanningContextV2,
    requirements: Mapping[str, ProducerAttachmentRequirementV2],
    bindings: Mapping[str, PromptOnlyDependencyEvidenceBindingV2],
) -> dict[str, ArtifactEvidenceTargetV2]:
    values: dict[str, ArtifactEvidenceTargetV2] = {}
    artifact_ids: set[str] = set()
    paths: set[str] = set()
    context_ref = attachment_planning_context_ref(context)
    for target in targets:
        try:
            validate_artifact_evidence_target_identity(target)
        except ValueError as exc:
            raise PromptOnlyDependencyPolicyError("existing artifact target identity is stale") from exc
        if (
            target.attachment_planning_context_ref != context_ref
            or target.attachment_dependency_id not in requirements
        ):
            raise PromptOnlyDependencyPolicyError("existing artifact target source is mismatched")
        requirement = requirements[target.attachment_dependency_id]
        if target.criticality is not requirement.criticality:
            raise PromptOnlyDependencyPolicyError("existing artifact target criticality is mismatched")
        if target.policy_version != ARTIFACT_EVIDENCE_MODE_POLICY_VERSION:
            raise PromptOnlyDependencyPolicyError("existing artifact target policy is stale or unsupported")
        binding = bindings.get(target.attachment_dependency_id)
        if binding is None or target.requirement_evidence != binding.evidence:
            raise PromptOnlyDependencyPolicyError("existing artifact target evidence is stale or mismatched")
        if target.attachment_dependency_id in values:
            raise PromptOnlyDependencyPolicyError("duplicate existing artifact target dependency")
        if target.artifact_id in artifact_ids:
            raise PromptOnlyDependencyPolicyError("duplicate existing artifact target ID")
        if target.logical_path in paths:
            raise PromptOnlyDependencyPolicyError("duplicate existing artifact target path")
        values[target.attachment_dependency_id] = target
        artifact_ids.add(target.artifact_id)
        paths.add(target.logical_path)
    return values


def _path_facts(
    *,
    text: str,
    source: PromptOnlyPathFactSource,
    dependency_id: str | None,
) -> tuple[PromptOnlyPathFact, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    text_sha = hashlib.sha256(normalized.encode()).hexdigest()
    values: dict[tuple[str, int, int], PromptOnlyPathFact] = {}
    for match in _PATH_PATTERN.finditer(normalized):
        candidate = match.group(0).replace("\\", "/")
        if not _safe_relative_path(candidate):
            continue
        extension = PurePosixPath(candidate).suffix.casefold()
        payload = {
            "attachment_dependency_id": dependency_id,
            "source": source.value,
            "logical_path": candidate,
            "media_type": _EXTENSION_TO_MEDIA_TYPE.get(extension),
            "source_text_sha256": text_sha,
            "char_start": match.start(),
            "char_end": match.end(),
        }
        digest = _payload_sha256(payload)
        fact = PromptOnlyPathFact(
            fact_id=f"prompt-only-path-fact://sha256/{digest}",
            attachment_dependency_id=dependency_id,
            source=source,
            logical_path=candidate,
            media_type=_EXTENSION_TO_MEDIA_TYPE.get(extension),
            source_text_sha256=text_sha,
            char_start=match.start(),
            char_end=match.end(),
        )
        values[(candidate, match.start(), match.end())] = fact
    return tuple(sorted(values.values(), key=lambda item: item.fact_id))


def _safe_relative_path(value: str) -> bool:
    if (
        not value
        or value.startswith(("/", "\\"))
        or "://" in value
        or re.match(r"^[A-Za-z]:", value)
        or any(ord(char) < 32 for char in value)
    ):
        return False
    parts = value.replace("\\", "/").split("/")
    return len(value) <= 512 and all(part not in {"", ".", ".."} for part in parts)


def _one_explicit_complete_fact(view: PromptOnlyDependencyView) -> bool:
    return len(view.explicit_path_facts) == 1 and view.explicit_path_facts[0].media_type is not None


def _neutral_path(dependency_id: str, media_type: str) -> str:
    extension = _MEDIA_TYPE_TO_EXTENSION.get(media_type)
    if extension is None:
        raise PromptOnlyDependencyPolicyError("semantic media type has no stable prompt-only extension")
    digest = hashlib.sha256(dependency_id.encode()).hexdigest()[:16]
    return f"inputs/prompt-only/{digest}{extension}"


def _matches_forbidden_path(
    value: str,
    facts: tuple[PromptOnlyPathFact, ...],
) -> bool:
    candidate = PurePosixPath(value.casefold())
    candidate_keys = {
        str(candidate),
        candidate.name,
        candidate.stem,
    }
    for fact in facts:
        forbidden = PurePosixPath(fact.logical_path.casefold())
        if candidate_keys & {
            str(forbidden),
            forbidden.name,
            forbidden.stem,
        }:
            return True
    return False


def _role_outcome(
    assessment: PromptOnlyDependencyAssessment,
) -> (
    tuple[
        PromptOnlyDependencyOutcome,
        PromptOnlyDependencyReason,
    ]
    | None
):
    if assessment.candidate_role is (PromptOnlyCandidateRole.INPUT_DEPENDENCY):
        return None
    if assessment.candidate_role is PromptOnlyCandidateRole.AMBIGUOUS:
        return (
            PromptOnlyDependencyOutcome.ABSTAIN,
            PromptOnlyDependencyReason.AMBIGUOUS_INPUT_OUTPUT_ROLE,
        )
    return (
        PromptOnlyDependencyOutcome.BLOCKED_SAFETY,
        PromptOnlyDependencyReason.REQUESTED_OUTPUT_PROPOSED,
    )


def _compile_target(
    *,
    context: AttachmentPlanningContextV2,
    requirement: ProducerAttachmentRequirementV2,
    binding: PromptOnlyDependencyEvidenceBindingV2,
    logical_path: str,
    media_type: str,
    planning_ref: ObjectRef,
    audit: ContractAudit,
) -> ArtifactEvidenceTargetV2:
    artifact_seed = {
        "attachment_planning_context_ref": _ref_payload(attachment_planning_context_ref(context)),
        "attachment_dependency_id": requirement.dependency_id,
        "logical_path": logical_path,
        "media_type": media_type,
    }
    artifact_digest = _payload_sha256(artifact_seed)
    target = ArtifactEvidenceTargetV2(
        artifact_evidence_target_id="artifact-evidence-target://pending",
        attachment_planning_context_ref=(attachment_planning_context_ref(context)),
        attachment_dependency_id=requirement.dependency_id,
        artifact_id=f"artifact://prompt-only/sha256/{artifact_digest}",
        logical_path=logical_path,
        media_type=media_type,
        criticality=requirement.criticality,
        requirement_evidence=binding.evidence,
        candidate_source_refs=(),
        policy_version=ARTIFACT_EVIDENCE_MODE_POLICY_VERSION,
        artifact_evidence_target_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                attachment_planning_context_ref(context),
                planning_ref,
                *tuple(item.subject_ref for item in binding.evidence),
            ),
        ),
    )
    digest = artifact_evidence_target_carried_sha256(target)
    return target.model_copy(
        update={
            "artifact_evidence_target_id": (f"artifact-evidence-target://sha256/{digest}"),
            "artifact_evidence_target_sha256": digest,
        }
    )


def _result(
    *,
    request: PromptOnlyDependencyDiscoveryRequest,
    proposal: PromptOnlyDependencyDiscoveryProposal | None,
    outcome: PromptOnlyDependencyOutcome,
    discovery: PromptOnlyDependencyDiscoveryV2 | None,
    unresolved_dependency_ids: tuple[str, ...],
    reasons: frozenset[PromptOnlyDependencyReason],
    audit: ContractAudit,
) -> PromptOnlyDependencyDiscoveryResult:
    request_ref = _request_ref(request)
    proposal_ref = _proposal_ref(proposal) if proposal is not None else None
    discovery_ref = prompt_only_dependency_discovery_ref(discovery) if discovery is not None else None
    payload = {
        "request_ref": _ref_payload(request_ref),
        "proposal_ref": (_ref_payload(proposal_ref) if proposal_ref is not None else None),
        "outcome": outcome.value,
        "discovery_ref": (_ref_payload(discovery_ref) if discovery_ref is not None else None),
        "unresolved_dependency_ids": list(unresolved_dependency_ids),
        "reasons": sorted(item.value for item in reasons),
        "policy_version": PROMPT_ONLY_DEPENDENCY_POLICY_VERSION,
    }
    digest = _payload_sha256(payload)
    return PromptOnlyDependencyDiscoveryResult(
        result_id=f"prompt-only-dependency-result://sha256/{digest}",
        request_ref=request_ref,
        proposal_ref=proposal_ref,
        outcome=outcome,
        discovery=discovery,
        unresolved_dependency_ids=unresolved_dependency_ids,
        reasons=reasons,
        policy_version=PROMPT_ONLY_DEPENDENCY_POLICY_VERSION,
        result_sha256=digest,
        audit=_safe_audit(
            audit,
            tuple(
                ref
                for ref in (
                    request_ref,
                    proposal_ref,
                    discovery_ref,
                )
                if ref is not None
            ),
        ),
    )


def _validate_request_identity(
    request: PromptOnlyDependencyDiscoveryRequest,
) -> None:
    digest = _payload_sha256(
        _request_seed(
            context_ref=request.attachment_planning_context_ref,
            view_ref=request.producer_task_view_ref,
            planning_ref=request.dependency_planning_context_ref,
            query_instruction=request.query_instruction,
            dependencies=request.dependencies,
            forbidden_outputs=request.forbidden_outputs,
            forbidden_path_facts=request.forbidden_path_facts,
            prompt_boundary_ref=request.prompt_boundary_enforcement_ref,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            abstain_conditions=request.abstain_conditions,
        )
    )
    if (
        request.request_sha256 != digest
        or request.request_id != f"prompt-only-dependency-request://sha256/{digest}"
    ):
        raise PromptOnlyDependencyPolicyError("prompt-only dependency request identity is stale")


def _validate_proposal_identity(
    proposal: PromptOnlyDependencyDiscoveryProposal,
) -> None:
    digest = _payload_sha256(
        _proposal_seed(
            request_ref=proposal.request_ref,
            outcome=proposal.outcome,
            assessments=proposal.assessments,
            unresolved_reasons=proposal.unresolved_reasons,
            model_profile=proposal.model_profile,
            prompt_version=proposal.prompt_version,
        )
    )
    if (
        proposal.proposal_sha256 != digest
        or proposal.proposal_id != f"prompt-only-dependency-proposal://sha256/{digest}"
    ):
        raise PromptOnlyDependencyPolicyError("prompt-only dependency proposal identity is stale")


def _validate_proposal_scope(
    request: PromptOnlyDependencyDiscoveryRequest,
    proposal: PromptOnlyDependencyDiscoveryProposal,
) -> None:
    unresolved = {item.dependency_id for item in request.dependencies if item.existing_target_ref is None}
    assessment_ids = {item.dependency_id for item in proposal.assessments}
    if not assessment_ids.issubset(unresolved):
        raise PromptOnlyDependencyPolicyError("semantic proposal selects an unknown or resolved dependency")
    semantic_required = {
        item.dependency_id
        for item in request.dependencies
        if item.existing_target_ref is None and not _one_explicit_complete_fact(item)
    }
    if not semantic_required:
        raise PromptOnlyDependencyPolicyError(
            "semantic proposal is not allowed without residual dependencies"
        )
    if proposal.outcome is PromptOnlyDependencyOutcome.NOT_REQUIRED:
        raise PromptOnlyDependencyPolicyError("NOT_REQUIRED proposal is invalid when dependencies exist")
    if proposal.outcome is PromptOnlyDependencyOutcome.RESOLVED and assessment_ids != semantic_required:
        raise PromptOnlyDependencyPolicyError(
            "semantic proposal does not exactly cover residual dependencies"
        )


def _same_request_except_audit_actor_time(
    expected: PromptOnlyDependencyDiscoveryRequest,
    observed: PromptOnlyDependencyDiscoveryRequest,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return _same_audit_lineage(expected.audit, observed.audit)


def _same_discovery_except_audit_actor_time(
    expected: PromptOnlyDependencyDiscoveryV2,
    observed: PromptOnlyDependencyDiscoveryV2,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit", "targets"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit", "targets"},
    ):
        return False
    if len(expected.targets) != len(observed.targets):
        return False
    for expected_target, observed_target in zip(
        expected.targets,
        observed.targets,
        strict=True,
    ):
        if expected_target.model_dump(
            mode="json",
            exclude={"audit"},
        ) != observed_target.model_dump(
            mode="json",
            exclude={"audit"},
        ):
            return False
        if not _same_audit_lineage(
            expected_target.audit,
            observed_target.audit,
        ):
            return False
    return _same_audit_lineage(expected.audit, observed.audit)


def _same_planning_context_except_audit_actor_time(
    expected: PromptOnlyDependencyPlanningContextV2,
    observed: PromptOnlyDependencyPlanningContextV2,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return _same_audit_lineage(expected.audit, observed.audit)


def _same_audit_lineage(
    expected: ContractAudit,
    observed: ContractAudit,
) -> bool:
    return (
        expected.governing_versions == observed.governing_versions
        and expected.input_refs == observed.input_refs
    )


def _projection_payload(
    *,
    context_ref: ObjectRef,
    view_ref: ObjectRef,
    planning_ref: ObjectRef,
    query_instruction: str,
    dependencies: tuple[PromptOnlyDependencyView, ...],
    forbidden_outputs: tuple[str, ...],
    forbidden_path_facts: tuple[PromptOnlyPathFact, ...],
    model_profile: str,
    prompt_version: str,
    abstain_conditions: tuple[str, ...],
) -> dict[str, object]:
    return {
        "attachment_planning_context_ref": _ref_payload(context_ref),
        "producer_task_view_ref": _ref_payload(view_ref),
        "dependency_planning_context_ref": _ref_payload(planning_ref),
        "query_instruction": query_instruction,
        "dependencies": [item.model_dump(mode="json", exclude_none=False) for item in dependencies],
        "forbidden_outputs": list(forbidden_outputs),
        "forbidden_path_facts": [
            item.model_dump(mode="json", exclude_none=False) for item in forbidden_path_facts
        ],
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "abstain_conditions": list(abstain_conditions),
        "policy_version": PROMPT_ONLY_DEPENDENCY_POLICY_VERSION,
    }


def _request_seed(
    *,
    context_ref: ObjectRef,
    view_ref: ObjectRef,
    planning_ref: ObjectRef,
    query_instruction: str,
    dependencies: tuple[PromptOnlyDependencyView, ...],
    forbidden_outputs: tuple[str, ...],
    forbidden_path_facts: tuple[PromptOnlyPathFact, ...],
    prompt_boundary_ref: ObjectRef,
    model_profile: str,
    prompt_version: str,
    abstain_conditions: tuple[str, ...],
) -> dict[str, object]:
    return {
        **_projection_payload(
            context_ref=context_ref,
            view_ref=view_ref,
            planning_ref=planning_ref,
            query_instruction=query_instruction,
            dependencies=dependencies,
            forbidden_outputs=forbidden_outputs,
            forbidden_path_facts=forbidden_path_facts,
            model_profile=model_profile,
            prompt_version=prompt_version,
            abstain_conditions=abstain_conditions,
        ),
        "prompt_boundary_enforcement_ref": _ref_payload(prompt_boundary_ref),
    }


def _proposal_seed(
    *,
    request_ref: ObjectRef,
    outcome: PromptOnlyDependencyOutcome,
    assessments: tuple[PromptOnlyDependencyAssessment, ...],
    unresolved_reasons: frozenset[PromptOnlyDependencyReason],
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "outcome": outcome.value,
        "assessments": [item.model_dump(mode="json", exclude_none=False) for item in assessments],
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": PROMPT_ONLY_DEPENDENCY_POLICY_VERSION,
    }


def _projection_ref(payload: dict[str, object]) -> ObjectRef:
    digest = _payload_sha256(payload)
    return ObjectRef(
        object_type="prompt-only-dependency-input",
        object_id=f"prompt-only-dependency-input://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _enforce_prompt_only_dependency_boundary(
    *,
    context_ref: ObjectRef,
    view_ref: ObjectRef,
    planning_ref: ObjectRef,
    projection_ref: ObjectRef,
    audit: ContractAudit,
) -> PromptBoundaryEnforcementResult:
    segment = PromptBoundarySegment(
        segment_id=("prompt-boundary-segment://prompt-only-dependency/projection"),
        surface=PromptBoundarySurface.DEPENDENCY_DISCOVERY_DATA,
        source_role=(PromptBoundarySourceRole.PROMPT_ONLY_DEPENDENCY_CONTRACT),
        source_ref=projection_ref,
        content_sha256=projection_ref.object_sha256,
        untrusted_data_marker=True,
        text_preview=None,
    )
    try:
        return PromptInjectionBoundaryEnforcer().validate_prompt_only_dependency_data_boundary(
            PromptOnlyDependencyDataBoundaryRequest(
                attachment_planning_context_ref=context_ref,
                producer_task_view_ref=view_ref,
                dependency_planning_context_ref=planning_ref,
                discovery_projection_ref=projection_ref,
                boundary_request=PromptBoundaryEnforcementRequest(
                    boundary_id=("prompt-boundary://prompt-only-dependency/r5-03"),
                    segments=(segment,),
                    approved_projection_policy_refs=(),
                    approved_evidence_bundle_refs=(),
                    collect_blocked_metadata=False,
                    audit=_safe_audit(
                        audit,
                        (
                            context_ref,
                            view_ref,
                            planning_ref,
                            projection_ref,
                        ),
                    ),
                ),
                projection_segment_id=segment.segment_id,
            )
        )
    except PromptInjectionBoundaryPolicyError as exc:
        raise PromptOnlyDependencyPolicyError(
            "prompt-only dependency input failed prompt-as-data enforcement"
        ) from exc


def _prompt_boundary_ref(
    result: PromptBoundaryEnforcementResult,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-boundary-enforcement",
        object_id=result.enforcement_id,
        object_version=result.policy_version,
        object_sha256=result.enforcement_sha256,
    )


def _request_ref(
    request: PromptOnlyDependencyDiscoveryRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-only-dependency-request",
        object_id=request.request_id,
        object_version=request.policy_version,
        object_sha256=request.request_sha256,
    )


def _proposal_ref(
    proposal: PromptOnlyDependencyDiscoveryProposal,
) -> ObjectRef:
    return ObjectRef(
        object_type="prompt-only-dependency-proposal",
        object_id=proposal.proposal_id,
        object_version=proposal.policy_version,
        object_sha256=proposal.proposal_sha256,
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


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise PromptOnlyDependencyPolicyError(f"{label} must be unique")


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
