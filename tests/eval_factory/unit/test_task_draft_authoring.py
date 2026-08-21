from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    SelectionContextV2,
    TaskDraftPromptSafetyStatusV2,
    TaskEpisodeSegmentEvidenceBindingV2,
    TaskEpisodeV2,
    selection_context_ref,
    task_episode_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    EvidencePriority,
    RequirementConflict,
)
from eval_factory.provenance import (
    EvidenceBundleCompiler,
    EvidenceBundleCompileRequest,
    EvidenceViewEngine,
    EvidenceViewPrincipal,
    EvidenceViewPrincipalType,
    EvidenceViewPurpose,
    EvidenceViewRequest,
    EvidenceViewSubject,
    ProjectionEvidenceBinding,
)
from eval_factory.task_authoring import (
    FakeTaskDraftAuthoringFixture,
    FakeTaskDraftAuthoringRunner,
    TaskDraftAttachmentFixture,
    TaskDraftAuthoringOutcome,
    TaskDraftAuthoringPolicyError,
    TaskDraftAuthoringReason,
    TaskDraftAuthoringRequest,
    TaskDraftAuthoringRequestBuilder,
    TaskDraftCompiler,
    TaskDraftContentFixture,
    TaskDraftRequirementFixture,
)

ROOT = Path(__file__).resolve().parents[3]
HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
SOURCE_TRACE_ID = "source-trace://task-draft-authoring"
TRACE_ID = "trace-ir://task-draft-authoring/v1"
REQUIRED_FORBIDDEN_OUTPUTS = (
    "original agent final answer",
    "pre-completed requested deliverable",
    "grader rules or hidden pass conditions",
)


def _audit(
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="task-draft-authoring-test",
        governing_versions=(VersionBinding(component="task-draft-authoring", version="r4-03"),),
        input_refs=input_refs,
    )


def _ref(
    object_type: str,
    object_id: str,
    *,
    digest: str = HASH,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=version,
        object_sha256=digest,
    )


def _source_span(suffix: str = "intent") -> SourceSpanRef:
    return SourceSpanRef(
        span_id=f"source-span://task-draft/{suffix}",
        source_trace_id=SOURCE_TRACE_ID,
        raw_sha256=HASH,
    )


def _decision(subject: ObjectRef) -> ProvenanceDecision:
    return ProvenanceDecision(
        provenance_decision_id="provenance-decision://task-draft/safe",
        subject_ref=subject,
        origin_class=OriginClass.PREEXISTING_WORKSPACE_INPUT,
        visibility=Visibility.STAGE_PROJECTION,
        disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        rule_ids=("task-draft-safe-view/v1",),
        source_event_refs=(_ref("trace-event", "trace-event://task-draft/intent"),),
        confidence=1.0,
        review_required=False,
        policy_version="provenance-decision-table/r2-01-v1",
        subject_sha256=subject.object_sha256,
        audit=_audit(),
    )


def _authoring_view(*, text: str = "The workspace contains design source files but no DESIGN.md."):
    subject = _ref(
        "file-version",
        "file-version://task-draft/input-workspace",
        digest=OTHER_HASH,
    )
    return EvidenceViewEngine().project(
        EvidenceViewRequest(
            principal=EvidenceViewPrincipal(
                principal_id="principal://task-author/r4-03",
                principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
                allowed_purposes=frozenset({EvidenceViewPurpose.TASK_AUTHORING}),
                max_subjects=10,
                max_characters=2000,
            ),
            purpose=EvidenceViewPurpose.TASK_AUTHORING,
            subjects=(
                EvidenceViewSubject(
                    subject_ref=subject,
                    decision=_decision(subject),
                    projection_text=text,
                ),
            ),
            max_characters=2000,
            audit=_audit(),
        )
    )


def _bundle(
    view_result,
    *,
    purpose: str,
    trace_ir_version_id: str = TRACE_ID,
):
    item = view_result.included_items[0]
    return (
        EvidenceBundleCompiler()
        .compile(
            EvidenceBundleCompileRequest(
                source_trace_id=SOURCE_TRACE_ID,
                trace_ir_version_id=trace_ir_version_id,
                consumer_stage="task-authoring",
                purpose=purpose,
                view_result=view_result,
                bindings=(
                    ProjectionEvidenceBinding(
                        projection_item_id=item.projection_item_id,
                        source_spans=(_source_span(purpose),),
                        polarity=EvidencePolarity.POSITIVE,
                        capability="task-intent-evidence",
                        capability_complete=True,
                    ),
                ),
                max_characters=2000,
                audit=_audit(),
            )
        )
        .evidence_bundle
    )


def _bundle_ref(bundle) -> ObjectRef:
    return _ref(
        "evidence-bundle",
        bundle.evidence_bundle_id,
        digest=bundle.bundle_sha256,
    )


def _selection_context(selection_bundle) -> SelectionContextV2:
    return SelectionContextV2(
        selection_context_id="selection-context://sha256/" + HASH,
        candidate_id="candidate://sha256/" + OTHER_HASH,
        approved_label_decision_refs=(
            _ref(
                "label-decision",
                "label-decision://sha256/" + THIRD_HASH,
                digest=THIRD_HASH,
                version="label-decision-merge/r3-04-v1",
            ),
        ),
        safe_evidence_bundle_ref=_bundle_ref(selection_bundle),
        task_authoring_note_refs=(),
        excluded_signal_hashes=(THIRD_HASH,),
        projection_policy_ref=selection_bundle.projection_policy_ref,
        policy_version="selection-context-firewall/r4-02-v1",
        selection_context_sha256=HASH,
        audit=_audit(),
    )


def _episode(
    context: SelectionContextV2,
    selection_bundle,
    suffix: str = "primary",
) -> TaskEpisodeV2:
    episode_hash = HASH if suffix == "primary" else OTHER_HASH
    segment_ref = _ref(
        "interaction-segment",
        f"interaction-segment://task-draft/{suffix}",
        digest=episode_hash,
        version="deterministic-interaction-segments/v1",
    )
    return TaskEpisodeV2(
        task_episode_id=f"task-episode://sha256/{episode_hash}",
        selection_context_ref=selection_context_ref(context),
        trace_envelope_ref=_ref(
            "trace-envelope",
            TRACE_ID,
            version="stored-manifest/v1",
        ),
        segment_refs=(segment_ref,),
        segment_evidence_bindings=(
            TaskEpisodeSegmentEvidenceBindingV2(
                segment_ref=segment_ref,
                evidence_ref_ids=(selection_bundle.evidence[0].evidence_ref_id,),
            ),
        ),
        rationale_ref=_ref(
            "task-episode-rationale",
            f"task-episode-rationale://task-draft/{suffix}",
        ),
        evidence_bundle_ref=_bundle_ref(selection_bundle),
        model_profile="internal-task-episode-grouper-v1",
        prompt_version="task-episode-grouping-prompt/v1",
        policy_version="task-episode-grouping/r4-01-v1",
        task_episode_sha256=episode_hash,
        audit=_audit(),
    )


def _content(
    evidence_ref_id: str,
    episode_id: str,
) -> TaskDraftContentFixture:
    return TaskDraftContentFixture(
        visible_prompt=(
            "Inspect the current workspace and determine whether a DESIGN.md file exists. "
            "Explain whether this project should have one using observable input-state evidence."
        ),
        task_intent="Evaluate workspace design-system governance.",
        evaluation_claim=(
            "The contestant can assess the workspace using only the prompt, approved tools, "
            "and input-state evidence."
        ),
        required_capability_ids=("workspace-analysis",),
        allowed_tool_ids=("file-read",),
        forbidden_outputs=REQUIRED_FORBIDDEN_OUTPUTS,
        attachment_dependencies=(
            TaskDraftAttachmentFixture(
                dependency_id="attachment-dependency://task-draft/workspace",
                description="Input-state workspace required for inspection.",
                criticality=AttachmentCriticality.REQUIRED,
                evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
                evidence_ref_ids=(evidence_ref_id,),
            ),
        ),
        requirements=(
            TaskDraftRequirementFixture(
                requirement_id="requirement://task-draft/design-inspection",
                statement="Inspect the workspace for DESIGN.md and explain the design need.",
                criticality=AttachmentCriticality.CRITICAL,
                evidence_priority=EvidencePriority.DIRECT_OBSERVATION,
                evidence_ref_ids=(evidence_ref_id,),
                task_episode_ids=(episode_id,),
                conflict_status=RequirementConflict.NONE,
            ),
        ),
        prompt_requirement_ids=("requirement://task-draft/design-inspection",),
        uncertainties=(),
    )


def _inputs(*, reverse: bool = False):
    view = _authoring_view()
    selection_bundle = _bundle(view, purpose="task-episode-grouping")
    authoring_bundle = _bundle(view, purpose="task-draft-authoring")
    context = _selection_context(selection_bundle)
    first = _episode(context, selection_bundle, "primary")
    second = _episode(context, selection_bundle, "secondary")
    episodes = (second, first) if reverse else (first, second)
    return context, episodes, selection_bundle, view, authoring_bundle


def _scenario(
    *,
    reverse: bool = False,
    audit: ContractAudit | None = None,
):
    active_audit = audit or _audit()
    context, episodes, selection_bundle, view, authoring_bundle = _inputs(reverse=reverse)
    request = TaskDraftAuthoringRequestBuilder().build(
        selection_context=context,
        task_episodes=episodes,
        selection_evidence_bundle=selection_bundle,
        authoring_view_result=view,
        authoring_evidence_bundle=authoring_bundle,
        allowed_capability_ids=("workspace-analysis", "text-reasoning"),
        allowed_tool_ids=("file-read", "file-list"),
        required_forbidden_outputs=REQUIRED_FORBIDDEN_OUTPUTS,
        abstain_conditions=("ambiguous task intent", "missing safe evidence"),
        model_profile="internal-task-author-v1",
        prompt_version="task-draft-authoring-prompt/v1",
        audit=active_audit,
    )
    primary_episode = next(item for item in episodes if item.task_episode_id.endswith(HASH))
    fixture = FakeTaskDraftAuthoringFixture(
        fixture_id="task-draft-fixture://gold/design-governance",
        outcome=TaskDraftAuthoringOutcome.DRAFTED,
        content=_content(
            authoring_bundle.evidence[0].evidence_ref_id,
            primary_episode.task_episode_id,
        ),
        unresolved_reasons=frozenset(),
        model_available=True,
    )
    proposal = FakeTaskDraftAuthoringRunner().run(
        request,
        fixture=fixture,
        audit=active_audit,
    )
    result = TaskDraftCompiler().compile(
        request=request,
        proposal=proposal,
        selection_context=context,
        task_episodes=episodes,
        selection_evidence_bundle=selection_bundle,
        authoring_view_result=view,
        authoring_evidence_bundle=authoring_bundle,
        audit=active_audit,
    )
    return (
        request,
        proposal,
        result,
        context,
        episodes,
        selection_bundle,
        view,
        authoring_bundle,
    )


def test_gold_task_intent_compiles_exact_pending_task_draft_v2() -> None:
    request, proposal, result, context, episodes, _, _, authoring_bundle = _scenario(reverse=True)

    assert result.outcome is TaskDraftAuthoringOutcome.DRAFTED
    assert not result.unresolved_reasons
    assert result.task_draft is not None
    draft = result.task_draft
    assert draft.task_version == 1
    assert draft.supersedes_task_draft_ref is None
    assert draft.prompt_safety_status is TaskDraftPromptSafetyStatusV2.PENDING
    assert draft.prompt_safety_gate_ref is None
    assert draft.selection_context_ref == selection_context_ref(context)
    assert draft.task_episode_refs == tuple(
        sorted(
            (task_episode_ref(item) for item in episodes),
            key=lambda ref: ref.object_id,
        )
    )
    assert draft.visible_prompt == proposal.content.visible_prompt
    assert draft.task_intent == "Evaluate workspace design-system governance."
    assert draft.required_capabilities == ("workspace-analysis",)
    assert draft.allowed_tools == ("file-read",)
    assert draft.forbidden_outputs == tuple(sorted(REQUIRED_FORBIDDEN_OUTPUTS))
    assert draft.prompt_requirement_ids == ("requirement://task-draft/design-inspection",)
    lineage = draft.requirement_lineage[0]
    assert lineage.evidence == (authoring_bundle.evidence[0],)
    assert lineage.task_episode_refs[0].object_id.endswith(HASH)
    assert draft.attachment_dependencies[0].evidence == (authoring_bundle.evidence[0],)
    assert draft.task_draft_id.startswith("task-draft://sha256/")
    assert draft.task_draft_sha256 == draft.task_draft_id.rsplit("/", 1)[-1]
    assert request.model_profile == proposal.model_profile == draft.model_profile


def test_request_contains_only_bounded_untrusted_safe_projection_data() -> None:
    request, _, _, context, _, _, view, _ = _scenario()
    serialized = request.model_dump_json()

    assert request.evidence_views[0].untrusted_data_marker is True
    assert request.prompt_boundary_enforcement_ref.object_type == ("prompt-boundary-enforcement")
    assert request.evidence_views[0].content == (
        "The workspace contains design source files but no DESIGN.md."
    )
    assert view.included_items[0].source_ref.object_id not in serialized
    assert view.included_items[0].child_decision.provenance_decision_id not in serialized
    assert context.approved_label_decision_refs[0].object_id not in serialized
    assert context.excluded_signal_hashes[0] not in serialized
    for marker in (
        "raw-trace",
        "private-reference",
        "final-output",
        "grader-rule",
        "hidden-selection-signal",
        "quarantine",
        "answer-bearing",
        "configured-pii",
    ):
        assert marker not in serialized


def test_request_proposal_result_and_draft_audits_discard_caller_refs() -> None:
    caller_ref = _ref(
        "private-reference",
        "private-reference://caller-must-not-flow",
    )
    request, proposal, result = _scenario(audit=_audit(input_refs=(caller_ref,)))[:3]
    assert result.task_draft is not None

    for artifact in (request, proposal, result, result.task_draft):
        assert caller_ref not in artifact.audit.input_refs
        assert all(ref.object_type != "private-reference" for ref in artifact.audit.input_refs)


def test_all_requirements_dependencies_and_critical_prompt_ids_are_bound() -> None:
    _, _, result, _, episodes, _, _, authoring_bundle = _scenario()
    assert result.task_draft is not None
    draft = result.task_draft

    evidence_ids = {item.evidence_ref_id for item in authoring_bundle.evidence}
    episode_refs = {task_episode_ref(item) for item in episodes}
    for requirement in draft.requirement_lineage:
        assert {item.evidence_ref_id for item in requirement.evidence} <= (evidence_ids)
        assert set(requirement.task_episode_refs) <= episode_refs
        if requirement.criticality is AttachmentCriticality.CRITICAL:
            assert requirement.requirement_id in draft.prompt_requirement_ids
    for dependency in draft.attachment_dependencies:
        assert {item.evidence_ref_id for item in dependency.evidence} <= (evidence_ids)


@pytest.mark.parametrize(
    ("fixture", "outcome", "reason"),
    (
        (
            FakeTaskDraftAuthoringFixture(
                fixture_id="task-draft-fixture://abstain",
                outcome=TaskDraftAuthoringOutcome.ABSTAIN,
                content=None,
                unresolved_reasons=frozenset({TaskDraftAuthoringReason.AMBIGUOUS_TASK_INTENT}),
            ),
            TaskDraftAuthoringOutcome.ABSTAIN,
            TaskDraftAuthoringReason.AMBIGUOUS_TASK_INTENT,
        ),
        (
            FakeTaskDraftAuthoringFixture(
                fixture_id="task-draft-fixture://blocked",
                outcome=TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY,
                content=None,
                unresolved_reasons=frozenset({TaskDraftAuthoringReason.MISSING_SAFE_AUTHORING_EVIDENCE}),
            ),
            TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY,
            TaskDraftAuthoringReason.MISSING_SAFE_AUTHORING_EVIDENCE,
        ),
        (
            FakeTaskDraftAuthoringFixture(
                fixture_id="task-draft-fixture://model-unavailable",
                outcome=TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY,
                content=None,
                unresolved_reasons=frozenset({TaskDraftAuthoringReason.MODEL_UNAVAILABLE}),
                model_available=False,
            ),
            TaskDraftAuthoringOutcome.BLOCKED_CAPABILITY,
            TaskDraftAuthoringReason.MODEL_UNAVAILABLE,
        ),
    ),
)
def test_typed_no_output_outcomes_never_fabricate_task_draft(
    fixture: FakeTaskDraftAuthoringFixture,
    outcome: TaskDraftAuthoringOutcome,
    reason: TaskDraftAuthoringReason,
) -> None:
    context, episodes, selection_bundle, view, authoring_bundle = _inputs()
    request = TaskDraftAuthoringRequestBuilder().build(
        selection_context=context,
        task_episodes=episodes,
        selection_evidence_bundle=selection_bundle,
        authoring_view_result=view,
        authoring_evidence_bundle=authoring_bundle,
        allowed_capability_ids=("workspace-analysis",),
        allowed_tool_ids=("file-read",),
        required_forbidden_outputs=REQUIRED_FORBIDDEN_OUTPUTS,
        abstain_conditions=("ambiguous",),
        model_profile="internal-task-author-v1",
        prompt_version="task-draft-authoring-prompt/v1",
        audit=_audit(),
    )
    proposal = FakeTaskDraftAuthoringRunner().run(
        request,
        fixture=fixture,
        audit=_audit(),
    )
    result = TaskDraftCompiler().compile(
        request=request,
        proposal=proposal,
        selection_context=context,
        task_episodes=episodes,
        selection_evidence_bundle=selection_bundle,
        authoring_view_result=view,
        authoring_evidence_bundle=authoring_bundle,
        audit=_audit(),
    )

    assert result.outcome is outcome
    assert reason in result.unresolved_reasons
    assert result.task_draft is None


def test_request_rejects_wrong_principal_purpose_empty_taint_and_trace() -> None:
    context, episodes, selection_bundle, view, authoring_bundle = _inputs()
    builder = TaskDraftAuthoringRequestBuilder()
    kwargs = {
        "selection_context": context,
        "task_episodes": episodes,
        "selection_evidence_bundle": selection_bundle,
        "authoring_view_result": view,
        "authoring_evidence_bundle": authoring_bundle,
        "allowed_capability_ids": ("workspace-analysis",),
        "allowed_tool_ids": ("file-read",),
        "required_forbidden_outputs": REQUIRED_FORBIDDEN_OUTPUTS,
        "abstain_conditions": ("ambiguous",),
        "model_profile": "internal-task-author-v1",
        "prompt_version": "task-draft-authoring-prompt/v1",
        "audit": _audit(),
    }

    with pytest.raises(TaskDraftAuthoringPolicyError, match="task-author"):
        builder.build(
            **{
                **kwargs,
                "authoring_view_result": view.model_copy(
                    update={"principal_type": EvidenceViewPrincipalType.DEFAULT_SAFE}
                ),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="purpose"):
        builder.build(
            **{
                **kwargs,
                "authoring_evidence_bundle": authoring_bundle.model_copy(
                    update={"purpose": "task-episode-grouping"}
                ),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="safe evidence"):
        builder.build(
            **{
                **kwargs,
                "authoring_evidence_bundle": authoring_bundle.model_copy(update={"evidence": ()}),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="tainted"):
        builder.build(
            **{
                **kwargs,
                "authoring_evidence_bundle": authoring_bundle.model_copy(
                    update={"tainted_content_included": True}
                ),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="trace"):
        builder.build(
            **{
                **kwargs,
                "authoring_evidence_bundle": _bundle(
                    view,
                    purpose="task-draft-authoring",
                    trace_ir_version_id="trace-ir://other/v1",
                ),
            }
        )
    tampered_item = view.included_items[0].model_copy(
        update={"content": "tampered safe-looking authoring content"}
    )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="projection identity"):
        builder.build(
            **{
                **kwargs,
                "authoring_view_result": view.model_copy(update={"included_items": (tampered_item,)}),
            }
        )


def test_runner_rejects_unknown_evidence_episode_capability_tool_and_forbidden_output() -> None:
    request, _, _, _, episodes, _, _, authoring_bundle = _scenario()
    valid = _content(
        authoring_bundle.evidence[0].evidence_ref_id,
        episodes[0].task_episode_id,
    )
    runner = FakeTaskDraftAuthoringRunner()

    updates = (
        (
            {
                "requirements": (
                    valid.requirements[0].model_copy(
                        update={"evidence_ref_ids": ("evidence-ref://missing",)}
                    ),
                )
            },
            "evidence",
        ),
        (
            {
                "requirements": (
                    valid.requirements[0].model_copy(
                        update={"task_episode_ids": ("task-episode://missing",)}
                    ),
                )
            },
            "episode",
        ),
        ({"required_capability_ids": ("unknown-capability",)}, "capability"),
        ({"allowed_tool_ids": ("shell-admin",)}, "tool"),
        ({"forbidden_outputs": ("only-one",)}, "forbidden"),
    )
    for update, message in updates:
        fixture = FakeTaskDraftAuthoringFixture(
            fixture_id=f"task-draft-fixture://invalid/{message.replace(' ', '-')}",
            outcome=TaskDraftAuthoringOutcome.DRAFTED,
            content=valid.model_copy(update=update),
            unresolved_reasons=frozenset(),
        )
        with pytest.raises(TaskDraftAuthoringPolicyError, match=message):
            runner.run(request, fixture=fixture, audit=_audit())

    with pytest.raises(ValidationError, match="prompt requirement"):
        FakeTaskDraftAuthoringFixture(
            fixture_id="task-draft-fixture://invalid/prompt-requirement",
            outcome=TaskDraftAuthoringOutcome.DRAFTED,
            content=valid.model_copy(
                update={
                    "prompt_requirement_ids": ("requirement://missing",),
                }
            ),
            unresolved_reasons=frozenset(),
        )


def test_compiler_rejects_stale_request_proposal_context_episode_and_bundle() -> None:
    (
        request,
        proposal,
        _,
        context,
        episodes,
        selection_bundle,
        view,
        authoring_bundle,
    ) = _scenario()
    compiler = TaskDraftCompiler()
    kwargs = {
        "request": request,
        "proposal": proposal,
        "selection_context": context,
        "task_episodes": episodes,
        "selection_evidence_bundle": selection_bundle,
        "authoring_view_result": view,
        "authoring_evidence_bundle": authoring_bundle,
        "audit": _audit(),
    }

    with pytest.raises(TaskDraftAuthoringPolicyError, match="request"):
        compiler.compile(
            **{
                **kwargs,
                "request": request.model_copy(update={"request_sha256": OTHER_HASH}),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="proposal"):
        compiler.compile(
            **{
                **kwargs,
                "proposal": proposal.model_copy(update={"proposal_sha256": OTHER_HASH}),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="SelectionContext"):
        compiler.compile(
            **{
                **kwargs,
                "selection_context": context.model_copy(update={"selection_context_sha256": OTHER_HASH}),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="TaskEpisode"):
        compiler.compile(
            **{
                **kwargs,
                "task_episodes": (
                    episodes[0].model_copy(update={"task_episode_sha256": THIRD_HASH}),
                    episodes[1],
                ),
            }
        )
    with pytest.raises(TaskDraftAuthoringPolicyError, match="selection evidence"):
        compiler.compile(
            **{
                **kwargs,
                "selection_evidence_bundle": selection_bundle.model_copy(
                    update={"bundle_sha256": OTHER_HASH}
                ),
            }
        )


def test_strict_semantic_models_reject_authoritative_and_raw_fields() -> None:
    request, _, _, _, _, _, _, authoring_bundle = _scenario()
    content = _content(
        authoring_bundle.evidence[0].evidence_ref_id,
        request.task_episode_refs[0].object_id,
    )

    with pytest.raises(ValidationError):
        TaskDraftContentFixture.model_validate(
            {
                **content.model_dump(mode="json"),
                "task_draft_id": "task-draft://forbidden",
            }
        )
    with pytest.raises(ValidationError):
        TaskDraftRequirementFixture.model_validate(
            {
                **content.requirements[0].model_dump(mode="json"),
                "evidence_refs": [],
            }
        )
    with pytest.raises(ValidationError):
        FakeTaskDraftAuthoringFixture.model_validate(
            {
                "fixture_id": "task-draft-fixture://raw",
                "outcome": "DRAFTED",
                "content": {
                    **content.model_dump(mode="json"),
                    "raw_trace_text": "forbidden",
                },
                "unresolved_reasons": [],
                "model_available": True,
            }
        )
    with pytest.raises(ValidationError):
        TaskDraftAuthoringRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "approved_label_decision_refs": [],
            }
        )


def test_task_draft_authoring_is_stable_across_audit_and_input_order() -> None:
    first = _scenario(
        reverse=False,
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
    )
    second = _scenario(
        reverse=True,
        audit=_audit(datetime(2026, 7, 27, tzinfo=UTC)),
    )
    first_request, first_proposal, first_result = first[:3]
    second_request, second_proposal, second_result = second[:3]
    assert first_result.task_draft is not None
    assert second_result.task_draft is not None

    assert first_request.semantic_authoring_request_id == (second_request.semantic_authoring_request_id)
    assert first_request.request_sha256 == second_request.request_sha256
    assert first_proposal.semantic_authoring_proposal_id == (second_proposal.semantic_authoring_proposal_id)
    assert first_proposal.proposal_sha256 == second_proposal.proposal_sha256
    assert first_result.task_draft.task_draft_id == (second_result.task_draft.task_draft_id)
    assert first_result.task_draft.task_draft_sha256 == (second_result.task_draft.task_draft_sha256)
    assert first_result.task_draft_authoring_result_id == (second_result.task_draft_authoring_result_id)
    assert first_result.result_sha256 == second_result.result_sha256
    assert first_result.canonical_sha256() != second_result.canonical_sha256()


def test_task_draft_authoring_is_stable_across_python_hash_seed(
    tmp_path: Path,
) -> None:
    script = tmp_path / "check_task_draft_seed.py"
    test_file = ROOT / "tests/eval_factory/unit/test_task_draft_authoring.py"
    script.write_text(
        f"""
import runpy
namespace = runpy.run_path({str(test_file)!r}, run_name='task_draft_fixture')
request, proposal, result = namespace['_scenario'](reverse=True)[:3]
draft = result.task_draft
assert draft is not None
print(request.semantic_authoring_request_id)
print(request.request_sha256)
print(proposal.semantic_authoring_proposal_id)
print(proposal.proposal_sha256)
print(draft.task_draft_id)
print(draft.task_draft_sha256)
print(result.task_draft_authoring_result_id)
print(result.result_sha256)
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        process = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"},
        )
        assert process.returncode == 0, process.stderr
        outputs.append(process.stdout)

    assert outputs[0] == outputs[1]
