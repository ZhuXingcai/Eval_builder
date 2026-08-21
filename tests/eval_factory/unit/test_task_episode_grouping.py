from __future__ import annotations

import os
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import SelectionContextV2
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.trace import InteractionSegment
from eval_factory.task_authoring import (
    FakeTaskEpisodeGroupingFixture,
    FakeTaskEpisodeGroupingRunner,
    TaskEpisodeCompiler,
    TaskEpisodeGroupFixture,
    TaskEpisodeGroupingOutcome,
    TaskEpisodeGroupingPolicyError,
    TaskEpisodeGroupingRequest,
    TaskEpisodeGroupingRequestBuilder,
    TaskEpisodeSegmentSelection,
    TaskEpisodeUnresolvedReason,
)
from eval_factory.trace.indexing.models import object_ref_for_segment

HASH = "a" * 64
BUNDLE_HASH = "b" * 64
CONTEXT_HASH = "c" * 64
EXCLUDED_SIGNAL_HASH = "d" * 64
ROOT = Path(__file__).resolve().parents[3]
SOURCE_TRACE_ID = "source-trace://task-episode"
TRACE_IR_VERSION_ID = "trace-ir://task-episode/v1"


def _audit(created_at: datetime = datetime(2026, 7, 25, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="task-episode-test",
        governing_versions=(VersionBinding(component="task-episode-grouping", version="r4-01"),),
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


def _trace_ref() -> ObjectRef:
    return _ref(
        "trace-envelope",
        TRACE_IR_VERSION_ID,
        version="stored-manifest/v1",
    )


def _segment(
    suffix: str,
    *,
    start: int,
    end: int,
    trace_ir_version_id: str = TRACE_IR_VERSION_ID,
) -> InteractionSegment:
    return InteractionSegment(
        segment_id=f"interaction-segment://{suffix}",
        trace_ir_version_id=trace_ir_version_id,
        boundary_method="context" if start else "user_turn",
        sequence_start=start,
        sequence_end=end,
        member_event_refs=(
            _ref(
                "trace-event",
                f"trace-event://{suffix}/{start}",
                digest=(f"{start + 1:x}" * 64)[:64],
            ),
        ),
    )


def _evidence(
    suffix: str,
    *,
    subject_ref: ObjectRef | None = None,
    source_trace_id: str = SOURCE_TRACE_ID,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://{suffix}",
        subject_ref=subject_ref
        or _ref(
            "file-version-projection",
            f"evidence-view://task-episode/{suffix}",
        ),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://task-episode/{suffix}",
                source_trace_id=source_trace_id,
                raw_sha256=HASH,
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="task-episode-grouping",
        capability_complete=True,
    )


def _bundle(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
    consumer_stage: str = "task-authoring",
    purpose: str = "task-episode-grouping",
    trace_ir_version_id: str = TRACE_IR_VERSION_ID,
    returned_characters: int = 120,
    max_characters: int = 1000,
    audit: ContractAudit | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_bundle_id="evidence-bundle://task-episode/safe",
        source_trace_id=SOURCE_TRACE_ID,
        trace_ir_version_id=trace_ir_version_id,
        consumer_stage=consumer_stage,
        purpose=purpose,
        projection_policy_ref=_ref(
            "projection-policy",
            "projection-policy://task-episode/safe",
        ),
        evidence=(_evidence("one"), _evidence("two")) if evidence is None else evidence,
        excluded_subject_refs=(),
        returned_characters=returned_characters,
        max_characters=max_characters,
        tainted_content_included=False,
        bundle_sha256=BUNDLE_HASH,
        audit=audit or _audit(),
    )


def _bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return _ref(
        "evidence-bundle",
        bundle.evidence_bundle_id,
        digest=bundle.bundle_sha256,
    )


def _selection_context(
    bundle: EvidenceBundle,
    *,
    audit: ContractAudit | None = None,
    selection_context_sha256: str = CONTEXT_HASH,
) -> SelectionContextV2:
    return SelectionContextV2(
        selection_context_id="selection-context://task-episode/candidate",
        candidate_id="candidate://task-episode/candidate",
        approved_label_decision_refs=(
            _ref(
                "label-decision",
                "label-decision://task-episode/approved",
                version="label-decision-merge/r3-04-v1",
            ),
        ),
        safe_evidence_bundle_ref=_bundle_ref(bundle),
        task_authoring_note_refs=(),
        excluded_signal_hashes=(EXCLUDED_SIGNAL_HASH,),
        projection_policy_ref=bundle.projection_policy_ref,
        policy_version="selection-context/r3-01-v1",
        selection_context_sha256=selection_context_sha256,
        audit=audit or _audit(),
    )


def _group(
    group_id: str = "task-episode-group://primary",
    *,
    segment_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
    rationale_ref: ObjectRef | None = None,
) -> TaskEpisodeGroupFixture:
    return TaskEpisodeGroupFixture(
        group_id=group_id,
        selections=tuple(
            TaskEpisodeSegmentSelection(
                segment_id=segment_id,
                evidence_ref_ids=(evidence_id,),
            )
            for segment_id, evidence_id in zip(segment_ids, evidence_ids, strict=True)
        ),
        rationale_ref=rationale_ref
        or _ref(
            "task-episode-rationale",
            f"task-episode-rationale://{group_id.rsplit('/', 1)[-1]}",
        ),
    )


def _scenario(
    *,
    audit: ContractAudit | None = None,
    reverse_inputs: bool = False,
):
    active_audit = audit or _audit()
    first = _segment("one", start=0, end=0)
    second = _segment("two", start=1, end=3)
    segments = (second, first) if reverse_inputs else (first, second)
    evidence = (_evidence("two"), _evidence("one")) if reverse_inputs else None
    bundle = _bundle(evidence=evidence, audit=active_audit)
    context = _selection_context(bundle, audit=active_audit)
    request = TaskEpisodeGroupingRequestBuilder().build(
        selection_context=context,
        trace_envelope_ref=_trace_ref(),
        segments=segments,
        evidence_bundle=bundle,
        model_profile="internal-task-episode-grouper-v1",
        prompt_version="task-episode-grouping-prompt/v1",
        abstain_conditions=("missing safe evidence", "ambiguous task boundary"),
        audit=active_audit,
    )
    fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://grouped",
        outcome=TaskEpisodeGroupingOutcome.GROUPED,
        groups=(
            _group(
                segment_ids=(second.segment_id, first.segment_id)
                if reverse_inputs
                else (first.segment_id, second.segment_id),
                evidence_ids=("evidence-ref://two", "evidence-ref://one")
                if reverse_inputs
                else ("evidence-ref://one", "evidence-ref://two"),
            ),
        ),
        unresolved_reasons=frozenset(),
        model_available=True,
    )
    proposal = FakeTaskEpisodeGroupingRunner().run(
        request,
        fixture=fixture,
        audit=active_audit,
    )
    result = TaskEpisodeCompiler().compile(
        request=request,
        proposal=proposal,
        selection_context=context,
        segments=segments,
        evidence_bundle=bundle,
        audit=active_audit,
    )
    return request, proposal, result, (first, second), bundle, context


def test_grouped_proposal_emits_exact_task_episode_v2_lineage() -> None:
    request, proposal, result, segments, bundle, context = _scenario(reverse_inputs=True)

    assert result.outcome is TaskEpisodeGroupingOutcome.GROUPED
    assert not result.unresolved_reasons
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.segment_refs == tuple(object_ref_for_segment(item) for item in segments)
    assert tuple(binding.segment_ref for binding in episode.segment_evidence_bindings) == (
        episode.segment_refs
    )
    assert episode.segment_evidence_bindings[0].evidence_ref_ids == ("evidence-ref://one",)
    assert episode.segment_evidence_bindings[1].evidence_ref_ids == ("evidence-ref://two",)
    assert episode.selection_context_ref.object_id == context.selection_context_id
    assert episode.selection_context_ref.object_sha256 == context.selection_context_sha256
    assert episode.evidence_bundle_ref == _bundle_ref(bundle)
    assert episode.model_profile == request.model_profile == proposal.model_profile
    assert episode.prompt_version == request.prompt_version == proposal.prompt_version
    assert episode.task_episode_id.startswith("task-episode://sha256/")
    assert len(episode.task_episode_sha256) == 64

    serialized = request.model_dump_json()
    assert "label-decision" not in serialized
    assert EXCLUDED_SIGNAL_HASH not in serialized
    assert "raw_trace" not in serialized
    assert "final-output" not in serialized


def test_overlapping_episode_annotations_do_not_mutate_base_segments() -> None:
    _, _, _, segments, bundle, context = _scenario()
    original = tuple(item.canonical_json() for item in segments)
    request = TaskEpisodeGroupingRequestBuilder().build(
        selection_context=context,
        trace_envelope_ref=_trace_ref(),
        segments=segments,
        evidence_bundle=bundle,
        model_profile="internal-task-episode-grouper-v1",
        prompt_version="task-episode-grouping-prompt/v1",
        abstain_conditions=("ambiguous task boundary",),
        audit=_audit(),
    )
    fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://overlap",
        outcome=TaskEpisodeGroupingOutcome.GROUPED,
        groups=(
            _group(
                "task-episode-group://first",
                segment_ids=(segments[0].segment_id, segments[1].segment_id),
                evidence_ids=("evidence-ref://one", "evidence-ref://two"),
            ),
            _group(
                "task-episode-group://second",
                segment_ids=(segments[1].segment_id,),
                evidence_ids=("evidence-ref://two",),
            ),
        ),
        unresolved_reasons=frozenset(),
    )
    proposal = FakeTaskEpisodeGroupingRunner().run(request, fixture=fixture, audit=_audit())
    result = TaskEpisodeCompiler().compile(
        request=request,
        proposal=proposal,
        selection_context=context,
        segments=segments,
        evidence_bundle=bundle,
        audit=_audit(),
    )

    assert len(result.episodes) == 2
    assert result.episodes[1].segment_refs[0] in result.episodes[0].segment_refs
    assert tuple(item.canonical_json() for item in segments) == original


@pytest.mark.parametrize(
    ("bundle_update", "message"),
    (
        ({"consumer_stage": "semantic-labeler"}, "consumer stage"),
        ({"purpose": "labeling"}, "purpose"),
        ({"trace_ir_version_id": "trace-ir://other/v1"}, "trace"),
        ({"returned_characters": 1001}, "budget"),
        ({"evidence": ()}, "safe evidence"),
    ),
)
def test_request_rejects_invalid_evidence_bundle_boundary(
    bundle_update: dict[str, object],
    message: str,
) -> None:
    bundle = _bundle().model_copy(update=bundle_update)
    context = _selection_context(bundle)

    with pytest.raises(TaskEpisodeGroupingPolicyError, match=message):
        TaskEpisodeGroupingRequestBuilder().build(
            selection_context=context,
            trace_envelope_ref=_trace_ref(),
            segments=(_segment("one", start=0, end=0),),
            evidence_bundle=bundle,
            model_profile="internal-task-episode-grouper-v1",
            prompt_version="task-episode-grouping-prompt/v1",
            abstain_conditions=("ambiguous",),
            audit=_audit(),
        )


def test_request_rejects_tainted_span_or_selection_context_mismatch() -> None:
    bundle = _bundle()
    context = _selection_context(bundle)
    builder = TaskEpisodeGroupingRequestBuilder()
    base = {
        "selection_context": context,
        "trace_envelope_ref": _trace_ref(),
        "segments": (_segment("one", start=0, end=0),),
        "evidence_bundle": bundle,
        "model_profile": "internal-task-episode-grouper-v1",
        "prompt_version": "task-episode-grouping-prompt/v1",
        "abstain_conditions": ("ambiguous",),
        "audit": _audit(),
    }

    with pytest.raises(TaskEpisodeGroupingPolicyError, match="tainted"):
        builder.build(
            **{
                **base,
                "evidence_bundle": bundle.model_copy(update={"tainted_content_included": True}),
            }
        )
    wrong_span = _bundle(evidence=(_evidence("one", source_trace_id="source-trace://other"),))
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="source trace"):
        builder.build(
            **{
                **base,
                "selection_context": _selection_context(wrong_span),
                "evidence_bundle": wrong_span,
            }
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="evidence bundle"):
        builder.build(
            **{
                **base,
                "selection_context": context.model_copy(
                    update={"safe_evidence_bundle_ref": _ref("evidence-bundle", "evidence-bundle://other")}
                ),
            }
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="projection policy"):
        builder.build(
            **{
                **base,
                "selection_context": context.model_copy(
                    update={
                        "projection_policy_ref": _ref(
                            "projection-policy",
                            "projection-policy://other",
                        )
                    }
                ),
            }
        )


@pytest.mark.parametrize(
    "object_type",
    (
        "raw-trace",
        "final-output",
        "completed-deliverable",
        "private-reference",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "quarantine",
        "answer-bearing",
        "secret",
        "configured-pii",
    ),
)
def test_request_and_runner_reject_unsafe_evidence_or_rationale(object_type: str) -> None:
    unsafe = _ref(object_type, f"{object_type}://unsafe")
    unsafe_bundle = _bundle(evidence=(_evidence("unsafe", subject_ref=unsafe),))

    with pytest.raises(TaskEpisodeGroupingPolicyError, match="unsafe"):
        TaskEpisodeGroupingRequestBuilder().build(
            selection_context=_selection_context(unsafe_bundle),
            trace_envelope_ref=_trace_ref(),
            segments=(_segment("one", start=0, end=0),),
            evidence_bundle=unsafe_bundle,
            model_profile="internal-task-episode-grouper-v1",
            prompt_version="task-episode-grouping-prompt/v1",
            abstain_conditions=("ambiguous",),
            audit=_audit(),
        )

    request, _, _, segments, _, _ = _scenario()
    fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id=f"task-episode-fixture://unsafe/{object_type}",
        outcome=TaskEpisodeGroupingOutcome.GROUPED,
        groups=(
            _group(
                segment_ids=(segments[0].segment_id,),
                evidence_ids=("evidence-ref://one",),
                rationale_ref=unsafe,
            ),
        ),
        unresolved_reasons=frozenset(),
    )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="rationale"):
        FakeTaskEpisodeGroupingRunner().run(request, fixture=fixture, audit=_audit())


def test_unknown_duplicate_and_cross_trace_segments_fail_closed() -> None:
    bundle = _bundle()
    context = _selection_context(bundle)
    first = _segment("one", start=0, end=0)
    builder = TaskEpisodeGroupingRequestBuilder()

    with pytest.raises(TaskEpisodeGroupingPolicyError, match="duplicate segment"):
        builder.build(
            selection_context=context,
            trace_envelope_ref=_trace_ref(),
            segments=(first, first),
            evidence_bundle=bundle,
            model_profile="internal-task-episode-grouper-v1",
            prompt_version="task-episode-grouping-prompt/v1",
            abstain_conditions=("ambiguous",),
            audit=_audit(),
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="trace"):
        builder.build(
            selection_context=context,
            trace_envelope_ref=_trace_ref(),
            segments=(_segment("other", start=1, end=1, trace_ir_version_id="trace-ir://other/v1"),),
            evidence_bundle=bundle,
            model_profile="internal-task-episode-grouper-v1",
            prompt_version="task-episode-grouping-prompt/v1",
            abstain_conditions=("ambiguous",),
            audit=_audit(),
        )

    request, _, _, _, _, _ = _scenario()
    unknown = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://unknown-segment",
        outcome=TaskEpisodeGroupingOutcome.GROUPED,
        groups=(
            _group(
                segment_ids=("interaction-segment://missing",),
                evidence_ids=("evidence-ref://one",),
            ),
        ),
        unresolved_reasons=frozenset(),
    )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="selected segment"):
        FakeTaskEpisodeGroupingRunner().run(request, fixture=unknown, audit=_audit())


def test_empty_duplicate_or_unknown_evidence_bindings_fail_closed() -> None:
    request, _, _, segments, _, _ = _scenario()

    with pytest.raises(ValidationError):
        TaskEpisodeGroupFixture(
            group_id="task-episode-group://empty-binding",
            selections=(),
            rationale_ref=_ref(
                "task-episode-rationale",
                "task-episode-rationale://empty-binding",
            ),
        )

    valid = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://valid-binding",
        outcome=TaskEpisodeGroupingOutcome.GROUPED,
        groups=(
            TaskEpisodeGroupFixture(
                group_id="task-episode-group://valid-binding",
                selections=(
                    TaskEpisodeSegmentSelection(
                        segment_id=segments[0].segment_id,
                        evidence_ref_ids=("evidence-ref://one",),
                    ),
                ),
                rationale_ref=_ref(
                    "task-episode-rationale",
                    "task-episode-rationale://valid-binding",
                ),
            ),
        ),
        unresolved_reasons=frozenset(),
    )
    proposal = FakeTaskEpisodeGroupingRunner().run(request, fixture=valid, audit=_audit())
    assert proposal.groups[0].selections[0].segment_id == segments[0].segment_id

    unknown_evidence = valid.model_copy(
        update={
            "groups": (
                _group(
                    segment_ids=(segments[0].segment_id,),
                    evidence_ids=("evidence-ref://missing",),
                ),
            )
        }
    )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="selected evidence"):
        FakeTaskEpisodeGroupingRunner().run(request, fixture=unknown_evidence, audit=_audit())

    with pytest.raises(ValidationError, match="segment selections must be unique"):
        TaskEpisodeGroupFixture(
            group_id="task-episode-group://duplicate-binding",
            selections=(
                TaskEpisodeSegmentSelection(
                    segment_id=segments[0].segment_id,
                    evidence_ref_ids=("evidence-ref://one",),
                ),
                TaskEpisodeSegmentSelection(
                    segment_id=segments[0].segment_id,
                    evidence_ref_ids=("evidence-ref://one",),
                ),
            ),
            rationale_ref=_ref(
                "task-episode-rationale",
                "task-episode-rationale://duplicate-binding",
            ),
        )

    duplicate_groups = valid.model_copy(update={"groups": (valid.groups[0], valid.groups[0])})
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="duplicate group"):
        FakeTaskEpisodeGroupingRunner().run(
            request,
            fixture=duplicate_groups,
            audit=_audit(),
        )


def test_abstain_and_blocked_capability_emit_no_fallback_episode() -> None:
    request, _, _, segments, bundle, context = _scenario()

    abstain_fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://abstain",
        outcome=TaskEpisodeGroupingOutcome.ABSTAIN,
        groups=(),
        unresolved_reasons=frozenset({TaskEpisodeUnresolvedReason.AMBIGUOUS_TASK_BOUNDARY}),
    )
    abstain_proposal = FakeTaskEpisodeGroupingRunner().run(
        request,
        fixture=abstain_fixture,
        audit=_audit(),
    )
    abstain = TaskEpisodeCompiler().compile(
        request=request,
        proposal=abstain_proposal,
        selection_context=context,
        segments=segments,
        evidence_bundle=bundle,
        audit=_audit(),
    )
    assert abstain.outcome is TaskEpisodeGroupingOutcome.ABSTAIN
    assert abstain.episodes == ()

    blocked_fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://model-unavailable",
        outcome=TaskEpisodeGroupingOutcome.BLOCKED_CAPABILITY,
        groups=(),
        unresolved_reasons=frozenset({TaskEpisodeUnresolvedReason.MODEL_UNAVAILABLE}),
        model_available=False,
    )
    blocked_proposal = FakeTaskEpisodeGroupingRunner().run(
        request,
        fixture=blocked_fixture,
        audit=_audit(),
    )
    blocked = TaskEpisodeCompiler().compile(
        request=request,
        proposal=blocked_proposal,
        selection_context=context,
        segments=segments,
        evidence_bundle=bundle,
        audit=_audit(),
    )
    assert blocked.outcome is TaskEpisodeGroupingOutcome.BLOCKED_CAPABILITY
    assert blocked.episodes == ()
    assert blocked.unresolved_reasons == frozenset({TaskEpisodeUnresolvedReason.MODEL_UNAVAILABLE})


def test_blocked_capability_requires_capability_reason() -> None:
    request, _, _, _, _, _ = _scenario()
    fixture = FakeTaskEpisodeGroupingFixture(
        fixture_id="task-episode-fixture://invalid-block",
        outcome=TaskEpisodeGroupingOutcome.BLOCKED_CAPABILITY,
        groups=(),
        unresolved_reasons=frozenset({TaskEpisodeUnresolvedReason.AMBIGUOUS_TASK_BOUNDARY}),
    )

    with pytest.raises(TaskEpisodeGroupingPolicyError, match="capability"):
        FakeTaskEpisodeGroupingRunner().run(request, fixture=fixture, audit=_audit())


def test_stale_request_proposal_context_or_bundle_fails_closed() -> None:
    request, proposal, _, segments, bundle, context = _scenario()
    compiler = TaskEpisodeCompiler()

    with pytest.raises(TaskEpisodeGroupingPolicyError, match="request"):
        compiler.compile(
            request=request.model_copy(update={"request_sha256": "d" * 64}),
            proposal=proposal,
            selection_context=context,
            segments=segments,
            evidence_bundle=bundle,
            audit=_audit(),
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="proposal"):
        compiler.compile(
            request=request,
            proposal=proposal.model_copy(update={"proposal_sha256": "d" * 64}),
            selection_context=context,
            segments=segments,
            evidence_bundle=bundle,
            audit=_audit(),
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="SelectionContext"):
        compiler.compile(
            request=request,
            proposal=proposal,
            selection_context=context.model_copy(update={"selection_context_sha256": "d" * 64}),
            segments=segments,
            evidence_bundle=bundle,
            audit=_audit(),
        )
    with pytest.raises(TaskEpisodeGroupingPolicyError, match="evidence bundle"):
        compiler.compile(
            request=request,
            proposal=proposal,
            selection_context=context,
            segments=segments,
            evidence_bundle=bundle.model_copy(update={"bundle_sha256": "d" * 64}),
            audit=_audit(),
        )


def test_models_reject_hidden_or_segment_mutation_fields() -> None:
    request, proposal, _, _, _, _ = _scenario()

    with pytest.raises(ValidationError):
        TaskEpisodeGroupingRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "approved_label_decision_refs": [],
            }
        )
    with pytest.raises(ValidationError):
        TaskEpisodeGroupingRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "excluded_signal_hashes": [],
            }
        )
    with pytest.raises(ValidationError):
        type(proposal.groups[0]).model_validate(
            {
                **proposal.groups[0].model_dump(mode="json"),
                "sequence_start": 999,
                "member_event_refs": [],
            }
        )


def test_grouping_identity_ignores_audit_time_input_order_and_hash_seed(tmp_path: Path) -> None:
    first = _scenario(audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)))
    second = _scenario(
        audit=_audit(datetime(2026, 7, 26, tzinfo=UTC)),
        reverse_inputs=True,
    )

    assert first[0].semantic_grouping_request_id == second[0].semantic_grouping_request_id
    assert first[0].request_sha256 == second[0].request_sha256
    assert first[1].semantic_grouping_proposal_id == second[1].semantic_grouping_proposal_id
    assert first[1].proposal_sha256 == second[1].proposal_sha256
    assert first[2].task_episode_grouping_result_id == second[2].task_episode_grouping_result_id
    assert first[2].result_sha256 == second[2].result_sha256
    assert first[2].episodes[0].task_episode_id == second[2].episodes[0].task_episode_id
    assert first[2].episodes[0].task_episode_sha256 == second[2].episodes[0].task_episode_sha256

    script = tmp_path / "check_task_episode_seed.py"
    script.write_text(
        f"""
import runpy
namespace = runpy.run_path({str(Path(__file__))!r})
request, proposal, result, *_ = namespace["_scenario"](reverse_inputs=True)
print(request.semantic_grouping_request_id)
print(request.request_sha256)
print(proposal.semantic_grouping_proposal_id)
print(proposal.proposal_sha256)
print(result.task_episode_grouping_result_id)
print(result.result_sha256)
print(result.episodes[0].task_episode_id)
print(result.episodes[0].task_episode_sha256)
""",
        encoding="utf-8",
    )
    run_one = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": "1"},
        check=True,
        text=True,
        capture_output=True,
    )
    run_two = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": "2"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert run_one.stdout == run_two.stdout


def test_test_module_can_be_loaded_without_running_pytest() -> None:
    namespace = runpy.run_path(str(Path(__file__)))
    assert "_scenario" in namespace
