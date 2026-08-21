from __future__ import annotations

import hashlib
import json

from eval_factory.contracts.core import (
    ContractAudit,
    EvidenceRef,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.labeling_v2 import (
    SelectionContextV2,
    selection_context_ref,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    TaskEpisodeSegmentEvidenceBindingV2,
    TaskEpisodeV2,
    task_episode_ref,
)
from eval_factory.contracts.trace import InteractionSegment
from eval_factory.task_authoring.models import (
    TASK_EPISODE_CONSUMER_STAGE,
    TASK_EPISODE_EVIDENCE_PURPOSE,
    TASK_EPISODE_GROUPING_POLICY_VERSION,
    FakeTaskEpisodeGroupingFixture,
    TaskEpisodeGroupingOutcome,
    TaskEpisodeGroupingPolicyError,
    TaskEpisodeGroupingProposal,
    TaskEpisodeGroupingRequest,
    TaskEpisodeGroupingResult,
    TaskEpisodeGroupProposal,
    TaskEpisodeSegmentSelection,
    TaskEpisodeSegmentView,
    TaskEpisodeUnresolvedReason,
    ensure_safe_task_authoring_ref,
    require_unique,
    validate_outcome_shape,
)
from eval_factory.trace.indexing.models import object_ref_for_segment


class TaskEpisodeGroupingRequestBuilder:
    policy_version = TASK_EPISODE_GROUPING_POLICY_VERSION

    def build(
        self,
        *,
        selection_context: SelectionContextV2,
        trace_envelope_ref: ObjectRef,
        segments: tuple[InteractionSegment, ...],
        evidence_bundle: EvidenceBundle,
        model_profile: Identifier,
        prompt_version: str,
        abstain_conditions: tuple[str, ...],
        audit: ContractAudit,
    ) -> TaskEpisodeGroupingRequest:
        _validate_builder_inputs(
            selection_context=selection_context,
            trace_envelope_ref=trace_envelope_ref,
            segments=segments,
            evidence_bundle=evidence_bundle,
        )
        segment_views = tuple(
            TaskEpisodeSegmentView(
                segment_id=segment.segment_id,
                boundary_method=segment.boundary_method,
                sequence_start=segment.sequence_start,
                sequence_end=segment.sequence_end,
                segment_sha256=segment.canonical_sha256(),
            )
            for segment in _sort_segments(segments)
        )
        evidence_refs = tuple(sorted(evidence_bundle.evidence, key=lambda item: item.evidence_ref_id))
        conditions = tuple(sorted(abstain_conditions))
        seed = _request_seed(
            selection_context_ref=selection_context_ref(selection_context),
            trace_envelope_ref=trace_envelope_ref,
            evidence_bundle_ref=_evidence_bundle_ref(evidence_bundle),
            projection_policy_ref=evidence_bundle.projection_policy_ref,
            segments=segment_views,
            evidence_refs=evidence_refs,
            abstain_conditions=conditions,
            model_profile=model_profile,
            prompt_version=prompt_version,
            returned_characters=evidence_bundle.returned_characters,
            max_characters=evidence_bundle.max_characters,
        )
        return TaskEpisodeGroupingRequest(
            semantic_grouping_request_id=_stable_id("task-episode-grouping-request", seed),
            selection_context_ref=selection_context_ref(selection_context),
            trace_envelope_ref=trace_envelope_ref,
            evidence_bundle_ref=_evidence_bundle_ref(evidence_bundle),
            projection_policy_ref=evidence_bundle.projection_policy_ref,
            segments=segment_views,
            evidence_refs=evidence_refs,
            abstain_conditions=conditions,
            model_profile=model_profile,
            prompt_version=prompt_version,
            returned_characters=evidence_bundle.returned_characters,
            max_characters=evidence_bundle.max_characters,
            request_sha256=_stable_hash(seed),
            audit=audit,
        )


class FakeTaskEpisodeGroupingRunner:
    policy_version = TASK_EPISODE_GROUPING_POLICY_VERSION

    def run(
        self,
        request: TaskEpisodeGroupingRequest,
        *,
        fixture: FakeTaskEpisodeGroupingFixture,
        audit: ContractAudit,
    ) -> TaskEpisodeGroupingProposal:
        _validate_request_integrity(request)
        _validate_fixture_shape(fixture)
        segments = {item.segment_id: item for item in request.segments}
        evidence_ids = {item.evidence_ref_id for item in request.evidence_refs}
        segment_order = {item.segment_id: index for index, item in enumerate(request.segments)}
        groups: list[TaskEpisodeGroupProposal] = []
        semantic_group_keys: set[tuple[object, ...]] = set()

        for fixture_group in fixture.groups:
            _validate_rationale_ref(fixture_group.rationale_ref)
            selections: list[TaskEpisodeSegmentSelection] = []
            for selection in fixture_group.selections:
                if selection.segment_id not in segments:
                    raise TaskEpisodeGroupingPolicyError(
                        f"selected segment is not present in request: {selection.segment_id}"
                    )
                for evidence_ref_id in selection.evidence_ref_ids:
                    if evidence_ref_id not in evidence_ids:
                        raise TaskEpisodeGroupingPolicyError(
                            f"selected evidence is not present in request: {evidence_ref_id}"
                        )
                selections.append(
                    TaskEpisodeSegmentSelection(
                        segment_id=selection.segment_id,
                        evidence_ref_ids=tuple(sorted(selection.evidence_ref_ids)),
                    )
                )
            normalized = tuple(
                sorted(selections, key=lambda item: (segment_order[item.segment_id], item.segment_id))
            )
            semantic_key = (
                tuple((item.segment_id, item.evidence_ref_ids) for item in normalized),
                _ref_key(fixture_group.rationale_ref),
            )
            if semantic_key in semantic_group_keys:
                raise TaskEpisodeGroupingPolicyError("duplicate semantic task-episode group")
            semantic_group_keys.add(semantic_key)
            groups.append(
                TaskEpisodeGroupProposal(
                    group_id=fixture_group.group_id,
                    selections=normalized,
                    rationale_ref=fixture_group.rationale_ref,
                )
            )

        normalized_groups = tuple(sorted(groups, key=_group_sort_key))
        seed = _proposal_seed(
            request_ref=_request_ref(request),
            selection_context_ref=request.selection_context_ref,
            trace_envelope_ref=request.trace_envelope_ref,
            evidence_bundle_ref=request.evidence_bundle_ref,
            outcome=fixture.outcome,
            groups=normalized_groups,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
        )
        return TaskEpisodeGroupingProposal(
            semantic_grouping_proposal_id=_stable_id("task-episode-grouping-proposal", seed),
            request_ref=_request_ref(request),
            selection_context_ref=request.selection_context_ref,
            trace_envelope_ref=request.trace_envelope_ref,
            evidence_bundle_ref=request.evidence_bundle_ref,
            outcome=fixture.outcome,
            groups=normalized_groups,
            unresolved_reasons=fixture.unresolved_reasons,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            proposal_sha256=_stable_hash(seed),
            audit=audit,
        )


class TaskEpisodeCompiler:
    policy_version = TASK_EPISODE_GROUPING_POLICY_VERSION

    def compile(
        self,
        *,
        request: TaskEpisodeGroupingRequest,
        proposal: TaskEpisodeGroupingProposal,
        selection_context: SelectionContextV2,
        segments: tuple[InteractionSegment, ...],
        evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
    ) -> TaskEpisodeGroupingResult:
        _validate_request_integrity(request)
        _validate_proposal_integrity(proposal)
        if selection_context_ref(selection_context) != request.selection_context_ref:
            raise TaskEpisodeGroupingPolicyError("SelectionContext ref is stale or mismatched")
        if _evidence_bundle_ref(evidence_bundle) != request.evidence_bundle_ref:
            raise TaskEpisodeGroupingPolicyError("evidence bundle ref is stale or mismatched")
        expected_request = TaskEpisodeGroupingRequestBuilder().build(
            selection_context=selection_context,
            trace_envelope_ref=request.trace_envelope_ref,
            segments=segments,
            evidence_bundle=evidence_bundle,
            model_profile=request.model_profile,
            prompt_version=request.prompt_version,
            abstain_conditions=request.abstain_conditions,
            audit=request.audit,
        )
        if expected_request != request:
            raise TaskEpisodeGroupingPolicyError(
                "request no longer matches authoritative task-episode inputs"
            )
        _validate_proposal_binding(request, proposal)

        episodes: tuple[TaskEpisodeV2, ...] = ()
        if proposal.outcome is TaskEpisodeGroupingOutcome.GROUPED:
            episodes = _compile_episodes(
                proposal=proposal,
                selection_context=selection_context,
                trace_envelope_ref=request.trace_envelope_ref,
                segments=segments,
                evidence_bundle=evidence_bundle,
                audit=audit,
            )
        seed = _result_seed(
            request_ref=_request_ref(request),
            proposal_ref=_proposal_ref(proposal),
            outcome=proposal.outcome,
            episodes=episodes,
            unresolved_reasons=proposal.unresolved_reasons,
        )
        return TaskEpisodeGroupingResult(
            task_episode_grouping_result_id=_stable_id("task-episode-grouping-result", seed),
            request_ref=_request_ref(request),
            proposal_ref=_proposal_ref(proposal),
            outcome=proposal.outcome,
            episodes=episodes,
            unresolved_reasons=proposal.unresolved_reasons,
            result_sha256=_stable_hash(seed),
            audit=audit,
        )


def _validate_builder_inputs(
    *,
    selection_context: SelectionContextV2,
    trace_envelope_ref: ObjectRef,
    segments: tuple[InteractionSegment, ...],
    evidence_bundle: EvidenceBundle,
) -> None:
    _require_ref_type(trace_envelope_ref, "trace-envelope", "trace_envelope_ref")
    ensure_safe_task_authoring_ref(trace_envelope_ref, "trace envelope")
    for ref in (
        *selection_context.approved_label_decision_refs,
        selection_context.safe_evidence_bundle_ref,
        *selection_context.task_authoring_note_refs,
        selection_context.projection_policy_ref,
    ):
        ensure_safe_task_authoring_ref(ref, "SelectionContext")
    if not segments:
        raise TaskEpisodeGroupingPolicyError("task-episode grouping requires at least one segment")
    require_unique("segment", (item.segment_id for item in segments))
    for segment in segments:
        if segment.trace_ir_version_id != trace_envelope_ref.object_id:
            raise TaskEpisodeGroupingPolicyError("segment and trace envelope trace mismatch")
        if segment.sequence_end < segment.sequence_start:
            raise TaskEpisodeGroupingPolicyError("segment sequence range is invalid")

    expected_bundle_ref = _evidence_bundle_ref(evidence_bundle)
    if selection_context.safe_evidence_bundle_ref != expected_bundle_ref:
        raise TaskEpisodeGroupingPolicyError(
            "SelectionContext evidence bundle ref does not match supplied evidence bundle"
        )
    if selection_context.projection_policy_ref != evidence_bundle.projection_policy_ref:
        raise TaskEpisodeGroupingPolicyError(
            "SelectionContext projection policy does not match evidence bundle"
        )
    if evidence_bundle.trace_ir_version_id != trace_envelope_ref.object_id:
        raise TaskEpisodeGroupingPolicyError("evidence bundle and trace envelope trace mismatch")
    if evidence_bundle.consumer_stage != TASK_EPISODE_CONSUMER_STAGE:
        raise TaskEpisodeGroupingPolicyError("unexpected evidence bundle consumer stage")
    if evidence_bundle.purpose != TASK_EPISODE_EVIDENCE_PURPOSE:
        raise TaskEpisodeGroupingPolicyError("unexpected evidence bundle purpose")
    if evidence_bundle.projection_policy_ref.object_type != "projection-policy":
        raise TaskEpisodeGroupingPolicyError("invalid evidence bundle projection policy")
    # Recheck serialized input because model_copy can bypass Literal validation.
    if evidence_bundle.model_dump(mode="python")["tainted_content_included"] is not False:
        raise TaskEpisodeGroupingPolicyError("tainted evidence bundle cannot enter task authoring")
    if evidence_bundle.returned_characters > evidence_bundle.max_characters:
        raise TaskEpisodeGroupingPolicyError("evidence bundle exceeds task-episode grouping budget")
    if not evidence_bundle.evidence:
        raise TaskEpisodeGroupingPolicyError("task-episode grouping requires safe evidence")
    require_unique("evidence ref", (item.evidence_ref_id for item in evidence_bundle.evidence))
    for evidence in evidence_bundle.evidence:
        ensure_safe_task_authoring_ref(evidence.subject_ref, "evidence")
        if any(span.source_trace_id != evidence_bundle.source_trace_id for span in evidence.source_spans):
            raise TaskEpisodeGroupingPolicyError(
                "evidence source trace does not match evidence bundle source trace"
            )


def _validate_fixture_shape(fixture: FakeTaskEpisodeGroupingFixture) -> None:
    try:
        validate_outcome_shape(fixture.outcome, fixture.groups, fixture.unresolved_reasons)
    except ValueError as exc:
        raise TaskEpisodeGroupingPolicyError(str(exc)) from exc
    require_unique("group", (item.group_id for item in fixture.groups))
    if not fixture.model_available:
        if fixture.outcome is not TaskEpisodeGroupingOutcome.BLOCKED_CAPABILITY:
            raise TaskEpisodeGroupingPolicyError("unavailable model must produce BLOCKED_CAPABILITY")
        if TaskEpisodeUnresolvedReason.MODEL_UNAVAILABLE not in fixture.unresolved_reasons:
            raise TaskEpisodeGroupingPolicyError("unavailable model requires MODEL_UNAVAILABLE reason")


def _validate_proposal_binding(
    request: TaskEpisodeGroupingRequest,
    proposal: TaskEpisodeGroupingProposal,
) -> None:
    if proposal.request_ref != _request_ref(request):
        raise TaskEpisodeGroupingPolicyError("proposal request ref is stale or mismatched")
    if proposal.selection_context_ref != request.selection_context_ref:
        raise TaskEpisodeGroupingPolicyError("proposal SelectionContext ref is mismatched")
    if proposal.trace_envelope_ref != request.trace_envelope_ref:
        raise TaskEpisodeGroupingPolicyError("proposal trace envelope ref is mismatched")
    if proposal.evidence_bundle_ref != request.evidence_bundle_ref:
        raise TaskEpisodeGroupingPolicyError("proposal evidence bundle ref is mismatched")
    if proposal.model_profile != request.model_profile:
        raise TaskEpisodeGroupingPolicyError("proposal model profile is mismatched")
    if proposal.prompt_version != request.prompt_version:
        raise TaskEpisodeGroupingPolicyError("proposal prompt version is mismatched")


def _compile_episodes(
    *,
    proposal: TaskEpisodeGroupingProposal,
    selection_context: SelectionContextV2,
    trace_envelope_ref: ObjectRef,
    segments: tuple[InteractionSegment, ...],
    evidence_bundle: EvidenceBundle,
    audit: ContractAudit,
) -> tuple[TaskEpisodeV2, ...]:
    by_id = {item.segment_id: item for item in segments}
    segment_order = {item.segment_id: index for index, item in enumerate(_sort_segments(segments))}
    evidence_ids = {item.evidence_ref_id for item in evidence_bundle.evidence}
    episodes: list[TaskEpisodeV2] = []
    for group in proposal.groups:
        segment_refs: list[ObjectRef] = []
        bindings: list[TaskEpisodeSegmentEvidenceBindingV2] = []
        selections = sorted(
            group.selections,
            key=lambda item: (segment_order.get(item.segment_id, len(segment_order)), item.segment_id),
        )
        for selection in selections:
            segment = by_id.get(selection.segment_id)
            if segment is None:
                raise TaskEpisodeGroupingPolicyError(
                    f"proposal segment is not authoritative: {selection.segment_id}"
                )
            if any(item not in evidence_ids for item in selection.evidence_ref_ids):
                raise TaskEpisodeGroupingPolicyError("proposal evidence binding is not authoritative")
            segment_ref = object_ref_for_segment(segment)
            segment_refs.append(segment_ref)
            bindings.append(
                TaskEpisodeSegmentEvidenceBindingV2(
                    segment_ref=segment_ref,
                    evidence_ref_ids=tuple(sorted(selection.evidence_ref_ids)),
                )
            )
        seed = _episode_seed(
            selection_context_ref=selection_context_ref(selection_context),
            trace_envelope_ref=trace_envelope_ref,
            segment_refs=tuple(segment_refs),
            bindings=tuple(bindings),
            rationale_ref=group.rationale_ref,
            evidence_bundle_ref=_evidence_bundle_ref(evidence_bundle),
            model_profile=proposal.model_profile,
            prompt_version=proposal.prompt_version,
        )
        episode_hash = _stable_hash(seed)
        episodes.append(
            TaskEpisodeV2(
                task_episode_id=f"task-episode://sha256/{episode_hash}",
                selection_context_ref=selection_context_ref(selection_context),
                trace_envelope_ref=trace_envelope_ref,
                segment_refs=tuple(segment_refs),
                segment_evidence_bindings=tuple(bindings),
                rationale_ref=group.rationale_ref,
                evidence_bundle_ref=_evidence_bundle_ref(evidence_bundle),
                model_profile=proposal.model_profile,
                prompt_version=proposal.prompt_version,
                policy_version=TASK_EPISODE_GROUPING_POLICY_VERSION,
                task_episode_sha256=episode_hash,
                audit=audit,
            )
        )
    ordered = tuple(sorted(episodes, key=lambda item: item.task_episode_id))
    require_unique("task episode", (item.task_episode_id for item in ordered))
    return ordered


def _validate_request_integrity(request: TaskEpisodeGroupingRequest) -> None:
    seed = _request_seed(
        selection_context_ref=request.selection_context_ref,
        trace_envelope_ref=request.trace_envelope_ref,
        evidence_bundle_ref=request.evidence_bundle_ref,
        projection_policy_ref=request.projection_policy_ref,
        segments=request.segments,
        evidence_refs=request.evidence_refs,
        abstain_conditions=request.abstain_conditions,
        model_profile=request.model_profile,
        prompt_version=request.prompt_version,
        returned_characters=request.returned_characters,
        max_characters=request.max_characters,
    )
    if request.request_sha256 != _stable_hash(seed):
        raise TaskEpisodeGroupingPolicyError("request hash is stale or mismatched")
    if request.semantic_grouping_request_id != _stable_id("task-episode-grouping-request", seed):
        raise TaskEpisodeGroupingPolicyError("request ID is stale or mismatched")


def _validate_proposal_integrity(proposal: TaskEpisodeGroupingProposal) -> None:
    seed = _proposal_seed(
        request_ref=proposal.request_ref,
        selection_context_ref=proposal.selection_context_ref,
        trace_envelope_ref=proposal.trace_envelope_ref,
        evidence_bundle_ref=proposal.evidence_bundle_ref,
        outcome=proposal.outcome,
        groups=proposal.groups,
        unresolved_reasons=proposal.unresolved_reasons,
        model_profile=proposal.model_profile,
        prompt_version=proposal.prompt_version,
    )
    if proposal.proposal_sha256 != _stable_hash(seed):
        raise TaskEpisodeGroupingPolicyError("proposal hash is stale or mismatched")
    if proposal.semantic_grouping_proposal_id != _stable_id(
        "task-episode-grouping-proposal",
        seed,
    ):
        raise TaskEpisodeGroupingPolicyError("proposal ID is stale or mismatched")


def _request_seed(
    *,
    selection_context_ref: ObjectRef,
    trace_envelope_ref: ObjectRef,
    evidence_bundle_ref: ObjectRef,
    projection_policy_ref: ObjectRef,
    segments: tuple[TaskEpisodeSegmentView, ...],
    evidence_refs: tuple[EvidenceRef, ...],
    abstain_conditions: tuple[str, ...],
    model_profile: str,
    prompt_version: str,
    returned_characters: int,
    max_characters: int,
) -> dict[str, object]:
    return {
        "selection_context_ref": _ref_payload(selection_context_ref),
        "trace_envelope_ref": _ref_payload(trace_envelope_ref),
        "evidence_bundle_ref": _ref_payload(evidence_bundle_ref),
        "projection_policy_ref": _ref_payload(projection_policy_ref),
        "segments": [item.model_dump(mode="json", exclude_none=False) for item in segments],
        "evidence_refs": [item.model_dump(mode="json", exclude_none=False) for item in evidence_refs],
        "abstain_conditions": list(abstain_conditions),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "returned_characters": returned_characters,
        "max_characters": max_characters,
        "policy_version": TASK_EPISODE_GROUPING_POLICY_VERSION,
    }


def _proposal_seed(
    *,
    request_ref: ObjectRef,
    selection_context_ref: ObjectRef,
    trace_envelope_ref: ObjectRef,
    evidence_bundle_ref: ObjectRef,
    outcome: TaskEpisodeGroupingOutcome,
    groups: tuple[TaskEpisodeGroupProposal, ...],
    unresolved_reasons: frozenset[TaskEpisodeUnresolvedReason],
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "selection_context_ref": _ref_payload(selection_context_ref),
        "trace_envelope_ref": _ref_payload(trace_envelope_ref),
        "evidence_bundle_ref": _ref_payload(evidence_bundle_ref),
        "outcome": outcome.value,
        "groups": [item.model_dump(mode="json", exclude_none=False) for item in groups],
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": TASK_EPISODE_GROUPING_POLICY_VERSION,
    }


def _episode_seed(
    *,
    selection_context_ref: ObjectRef,
    trace_envelope_ref: ObjectRef,
    segment_refs: tuple[ObjectRef, ...],
    bindings: tuple[TaskEpisodeSegmentEvidenceBindingV2, ...],
    rationale_ref: ObjectRef,
    evidence_bundle_ref: ObjectRef,
    model_profile: str,
    prompt_version: str,
) -> dict[str, object]:
    return {
        "selection_context_ref": _ref_payload(selection_context_ref),
        "trace_envelope_ref": _ref_payload(trace_envelope_ref),
        "segment_refs": [_ref_payload(item) for item in segment_refs],
        "segment_evidence_bindings": [item.model_dump(mode="json", exclude_none=False) for item in bindings],
        "rationale_ref": _ref_payload(rationale_ref),
        "evidence_bundle_ref": _ref_payload(evidence_bundle_ref),
        "model_profile": model_profile,
        "prompt_version": prompt_version,
        "policy_version": TASK_EPISODE_GROUPING_POLICY_VERSION,
    }


def _result_seed(
    *,
    request_ref: ObjectRef,
    proposal_ref: ObjectRef,
    outcome: TaskEpisodeGroupingOutcome,
    episodes: tuple[TaskEpisodeV2, ...],
    unresolved_reasons: frozenset[TaskEpisodeUnresolvedReason],
) -> dict[str, object]:
    return {
        "request_ref": _ref_payload(request_ref),
        "proposal_ref": _ref_payload(proposal_ref),
        "outcome": outcome.value,
        "episode_refs": [_ref_payload(task_episode_ref(item)) for item in episodes],
        "unresolved_reasons": sorted(item.value for item in unresolved_reasons),
        "policy_version": TASK_EPISODE_GROUPING_POLICY_VERSION,
    }


def _evidence_bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return ObjectRef(
        object_type="evidence-bundle",
        object_id=bundle.evidence_bundle_id,
        object_version="v1",
        object_sha256=bundle.bundle_sha256,
    )


def _request_ref(request: TaskEpisodeGroupingRequest) -> ObjectRef:
    return ObjectRef(
        object_type="task-episode-grouping-request",
        object_id=request.semantic_grouping_request_id,
        object_version=request.policy_version,
        object_sha256=request.request_sha256,
    )


def _proposal_ref(proposal: TaskEpisodeGroupingProposal) -> ObjectRef:
    return ObjectRef(
        object_type="task-episode-grouping-proposal",
        object_id=proposal.semantic_grouping_proposal_id,
        object_version=proposal.policy_version,
        object_sha256=proposal.proposal_sha256,
    )


def _sort_segments(
    segments: tuple[InteractionSegment, ...],
) -> tuple[InteractionSegment, ...]:
    return tuple(
        sorted(
            segments,
            key=lambda item: (
                item.sequence_start,
                item.sequence_end,
                item.boundary_method,
                item.segment_id,
            ),
        )
    )


def _group_sort_key(group: TaskEpisodeGroupProposal) -> tuple[object, ...]:
    return (
        tuple(item.segment_id for item in group.selections),
        _ref_key(group.rationale_ref),
        group.group_id,
    )


def _validate_rationale_ref(ref: ObjectRef) -> None:
    if ref.object_type != "task-episode-rationale":
        raise TaskEpisodeGroupingPolicyError("rationale ref must reference task-episode-rationale")
    ensure_safe_task_authoring_ref(ref, "rationale")


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


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
