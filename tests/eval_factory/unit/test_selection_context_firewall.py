from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
    label_decision_ref,
    selection_context_ref,
)
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
from eval_factory.labeling import (
    LABEL_DECISION_MERGE_POLICY_VERSION,
    LabelBatchProgressTracker,
    LabelCalibrationObservation,
)
from eval_factory.provenance.bundles import EVIDENCE_COMPILATION_POLICY_VERSION
from eval_factory.task_authoring import (
    SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
    SelectionContextFirewall,
    SelectionContextFirewallOutcome,
    SelectionContextFirewallPolicyError,
    SelectionContextFirewallReason,
    SelectionContextFirewallRequest,
    SelectionContextFirewallResult,
    TaskEpisodeGroupingRequestBuilder,
)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_TRACE_ID = "source-trace://selection-firewall"
TRACE_IR_VERSION_ID = "trace-ir://selection-firewall/v1"
HASH = "a" * 64
OTHER_HASH = "b" * 64


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit(
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
    *,
    input_refs: tuple[ObjectRef, ...] = (),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="selection-firewall-test",
        governing_versions=(VersionBinding(component="selection-context-firewall", version="r4-02"),),
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


def _trace_ref(trace_id: str = TRACE_IR_VERSION_ID) -> ObjectRef:
    return _ref(
        "trace-envelope",
        trace_id,
        digest=_hash(trace_id),
        version="stored-manifest/v1",
    )


def _evidence(
    suffix: str,
    *,
    subject_ref: ObjectRef | None = None,
    source_trace_id: str = SOURCE_TRACE_ID,
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE,
    complete: bool = True,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://selection-firewall/{suffix}",
        subject_ref=subject_ref
        or _ref(
            "file-version-projection",
            f"evidence-view://selection-firewall/{suffix}",
            digest=_hash(f"subject:{suffix}"),
        ),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://selection-firewall/{suffix}",
                source_trace_id=source_trace_id,
                raw_sha256=_hash(f"span:{suffix}"),
            ),
        ),
        polarity=polarity,
        capability="selection-firewall",
        capability_complete=complete,
    )


def _decision(
    suffix: str,
    *,
    decision: LabelDecisionValueV2 = LabelDecisionValueV2.MATCH,
    execution_status: LabelExecutionStatus = LabelExecutionStatus.FINAL,
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset(),
    semantic: bool = False,
    trace_id: str = TRACE_IR_VERSION_ID,
    label_spec_id: str | None = None,
    evidence_subject_ref: ObjectRef | None = None,
) -> LabelDecisionV2:
    decision_hash = _hash(f"decision:{suffix}")
    evidence = _evidence(
        f"decision-{suffix}",
        subject_ref=evidence_subject_ref,
        polarity=(
            EvidencePolarity.NEGATIVE
            if decision is LabelDecisionValueV2.NO_MATCH
            else EvidencePolarity.POSITIVE
        ),
    )
    positive = (evidence,) if decision is LabelDecisionValueV2.MATCH and not semantic else ()
    negative = (evidence,) if decision is LabelDecisionValueV2.NO_MATCH else ()
    semantic_evidence = (evidence,) if decision is LabelDecisionValueV2.MATCH and semantic else ()
    return LabelDecisionV2(
        label_decision_id=f"label-decision://sha256/{decision_hash}",
        label_spec_ref=_ref(
            "label-spec",
            label_spec_id or f"label-spec://selection-firewall/{suffix}",
            digest=_hash(f"label-spec:{label_spec_id or suffix}"),
            version="v2",
        ),
        trace_envelope_ref=_trace_ref(trace_id),
        decision=decision,
        execution_status=execution_status,
        positive_evidence=positive,
        negative_evidence=negative,
        semantic_evidence=semantic_evidence,
        structured_capability_complete=True,
        confidence=0.91,
        rule_version="structured-labeling/r3-02-v1",
        model_profile="internal-semantic-labeler-v1" if semantic else None,
        prompt_version="semantic-label/r3-03-v1" if semantic else None,
        unresolved_reasons=unresolved_reasons,
        policy_version=LABEL_DECISION_MERGE_POLICY_VERSION,
        decision_sha256=decision_hash,
        audit=_audit(),
    )


def _bundle(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
    consumer_stage: str = "task-authoring",
    purpose: str = "task-episode-grouping",
    source_trace_id: str = SOURCE_TRACE_ID,
    trace_ir_version_id: str = TRACE_IR_VERSION_ID,
    returned_characters: int = 120,
    max_characters: int = 1000,
) -> EvidenceBundle:
    evidence_items = (_evidence("bundle-one"), _evidence("bundle-two")) if evidence is None else evidence
    policy_ref = _ref(
        "projection-policy",
        "projection-policy://selection-firewall/safe",
        digest=_hash("projection-policy"),
        version="evidence-views/r2-05-v1",
    )
    seed = {
        "source_trace_id": source_trace_id,
        "trace_ir_version_id": trace_ir_version_id,
        "consumer_stage": consumer_stage,
        "purpose": purpose,
        "projection_policy_ref": policy_ref.model_dump(mode="json", exclude_none=False),
        "evidence": [item.model_dump(mode="json", exclude_none=False) for item in evidence_items],
        "excluded_subject_refs": [],
        "returned_characters": returned_characters,
        "max_characters": max_characters,
        "policy_version": EVIDENCE_COMPILATION_POLICY_VERSION,
    }
    bundle_hash = _stable_hash(seed)
    return EvidenceBundle(
        evidence_bundle_id=f"evidence-bundle://sha256/{bundle_hash}",
        source_trace_id=source_trace_id,
        trace_ir_version_id=trace_ir_version_id,
        consumer_stage=consumer_stage,
        purpose=purpose,
        projection_policy_ref=policy_ref,
        evidence=evidence_items,
        excluded_subject_refs=(),
        returned_characters=returned_characters,
        max_characters=max_characters,
        tainted_content_included=False,
        bundle_sha256=bundle_hash,
        audit=_audit(),
    )


def _bundle_ref(bundle: EvidenceBundle) -> ObjectRef:
    return _ref(
        "evidence-bundle",
        bundle.evidence_bundle_id,
        digest=bundle.bundle_sha256,
    )


def _request(
    *,
    decisions: tuple[LabelDecisionV2, ...] | None = None,
    bundle: EvidenceBundle | None = None,
    notes: tuple[ObjectRef, ...] | None = None,
    restricted: tuple[ObjectRef, ...] | None = None,
    audit: ContractAudit | None = None,
) -> SelectionContextFirewallRequest:
    active_bundle = bundle or _bundle()
    return SelectionContextFirewallRequest(
        decisions=(_decision("one"), _decision("two")) if decisions is None else decisions,
        evidence_bundle=active_bundle,
        task_authoring_note_refs=(
            ((active_bundle.evidence[0].subject_ref,) if active_bundle.evidence else ())
            if notes is None
            else notes
        ),
        restricted_signal_refs=(
            (
                _ref(
                    "hidden-selection-signal",
                    "hidden-selection-signal://error-signature/powershell-paired-error",
                    digest=_hash("restricted-signal"),
                ),
            )
            if restricted is None
            else restricted
        ),
        audit=audit
        or _audit(
            input_refs=(
                _ref(
                    "private-reference",
                    "private-reference://caller-must-not-flow",
                    digest=OTHER_HASH,
                ),
            )
        ),
    )


def _allowed_result(
    *,
    reverse_inputs: bool = False,
    created_at: datetime = datetime(2026, 7, 26, tzinfo=UTC),
) -> SelectionContextFirewallResult:
    bundle = _bundle()
    decisions = (_decision("one"), _decision("two", semantic=True))
    notes = (bundle.evidence[0].subject_ref, bundle.evidence[1].subject_ref)
    restricted = (
        _ref(
            "hidden-selection-signal",
            "hidden-selection-signal://error-signature/powershell-paired-error",
            digest=_hash("restricted-signal"),
        ),
        _ref(
            "label-threshold",
            "label-threshold://restricted/confidence",
            digest=decisions[0].label_spec_ref.object_sha256,
        ),
    )
    if reverse_inputs:
        decisions = tuple(reversed(decisions))
        notes = tuple(reversed(notes))
        restricted = tuple(reversed(restricted))
    return SelectionContextFirewall().apply(
        _request(
            decisions=decisions,
            bundle=bundle,
            notes=notes,
            restricted=restricted,
            audit=_audit(
                created_at,
                input_refs=(
                    _ref(
                        "private-reference",
                        "private-reference://caller-must-not-flow",
                        digest=OTHER_HASH,
                    ),
                ),
            ),
        )
    )


def test_final_match_decisions_emit_exact_safe_selection_context() -> None:
    request = _request()
    result = SelectionContextFirewall().apply(request)

    assert result.outcome is SelectionContextFirewallOutcome.ALLOWED
    assert result.blocked_reasons == frozenset()
    assert result.selection_context is not None
    context = result.selection_context
    expected_decision_refs = tuple(
        sorted(
            (label_decision_ref(item) for item in request.decisions),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )
    assert context.approved_label_decision_refs == expected_decision_refs
    assert context.safe_evidence_bundle_ref == _bundle_ref(request.evidence_bundle)
    assert context.task_authoring_note_refs == request.task_authoring_note_refs
    assert context.projection_policy_ref == request.evidence_bundle.projection_policy_ref
    assert context.policy_version == SELECTION_CONTEXT_FIREWALL_POLICY_VERSION
    assert context.candidate_id.startswith("candidate://sha256/")
    assert context.selection_context_id.startswith("selection-context://sha256/")
    assert context.selection_context_sha256 == context.selection_context_id.rsplit("/", 1)[-1]
    assert selection_context_ref(context).object_sha256 == context.selection_context_sha256
    assert result.selection_context_firewall_result_id.startswith(
        "selection-context-firewall-result://sha256/"
    )

    expected_hashes = {
        *(item.decision_sha256 for item in request.decisions),
        *(item.label_spec_ref.object_sha256 for item in request.decisions),
        *(
            evidence.subject_ref.object_sha256
            for item in request.decisions
            for evidence in (
                *item.positive_evidence,
                *item.negative_evidence,
                *item.semantic_evidence,
            )
        ),
        *(item.object_sha256 for item in request.restricted_signal_refs),
    }
    assert context.excluded_signal_hashes == tuple(sorted(expected_hashes))
    assert result.excluded_signal_count == len(expected_hashes)


def test_semantic_match_is_admitted_without_disclosing_semantic_metadata() -> None:
    decision = _decision("semantic", semantic=True)
    result = SelectionContextFirewall().apply(_request(decisions=(decision,)))

    assert result.outcome is SelectionContextFirewallOutcome.ALLOWED
    serialized = result.model_dump_json()
    assert decision.label_spec_ref.object_id not in serialized
    assert decision.semantic_evidence[0].evidence_ref_id not in serialized
    assert decision.semantic_evidence[0].subject_ref.object_id not in serialized
    assert decision.model_profile not in serialized
    assert decision.prompt_version not in serialized
    assert str(decision.confidence) not in serialized


def test_restricted_values_and_caller_audit_refs_never_leave_firewall() -> None:
    request = _request()
    result = SelectionContextFirewall().apply(request)
    assert result.selection_context is not None

    serialized = result.model_dump_json()
    for restricted in request.restricted_signal_refs:
        assert restricted.object_type not in serialized
        assert restricted.object_id not in serialized
    assert request.audit.input_refs[0].object_type not in serialized
    assert request.audit.input_refs[0].object_id not in serialized
    for decision in request.decisions:
        assert decision.label_spec_ref.object_id not in serialized
        for evidence in (
            *decision.positive_evidence,
            *decision.negative_evidence,
            *decision.semantic_evidence,
        ):
            assert evidence.evidence_ref_id not in serialized
            assert evidence.subject_ref.object_id not in serialized

    expected_audit_refs = {
        *(label_decision_ref(item) for item in request.decisions),
        _bundle_ref(request.evidence_bundle),
        request.evidence_bundle.projection_policy_ref,
        *request.task_authoring_note_refs,
    }
    assert set(result.audit.input_refs) == expected_audit_refs
    assert set(result.selection_context.audit.input_refs) == expected_audit_refs


@pytest.mark.parametrize(
    ("decisions", "reason"),
    (
        ((), SelectionContextFirewallReason.NO_DECISIONS),
        (
            (_decision("no-match", decision=LabelDecisionValueV2.NO_MATCH),),
            SelectionContextFirewallReason.DECISION_NOT_MATCH,
        ),
        (
            (
                _decision(
                    "abstain",
                    decision=LabelDecisionValueV2.ABSTAIN,
                    execution_status=LabelExecutionStatus.UNRESOLVED,
                    unresolved_reasons=frozenset({LabelUnresolvedReason.AMBIGUOUS_EVIDENCE}),
                ),
            ),
            SelectionContextFirewallReason.DECISION_NOT_MATCH,
        ),
        (
            (
                _decision(
                    "non-final",
                    execution_status=LabelExecutionStatus.REVIEW_REQUIRED,
                ),
            ),
            SelectionContextFirewallReason.DECISION_NOT_FINAL,
        ),
        (
            (
                _decision(
                    "unresolved",
                    decision=LabelDecisionValueV2.ABSTAIN,
                    execution_status=LabelExecutionStatus.UNRESOLVED,
                    unresolved_reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
                ),
            ),
            SelectionContextFirewallReason.DECISION_UNRESOLVED,
        ),
    ),
)
def test_ineligible_decisions_return_typed_no_context_result(
    decisions: tuple[LabelDecisionV2, ...],
    reason: SelectionContextFirewallReason,
) -> None:
    result = SelectionContextFirewall().apply(_request(decisions=decisions))

    assert result.outcome is SelectionContextFirewallOutcome.BLOCKED_DECISION
    assert result.selection_context is None
    assert reason in result.blocked_reasons
    assert len(result.input_sha256) == 64
    assert result.result_sha256 == result.selection_context_firewall_result_id.rsplit("/", 1)[-1]


def test_empty_safe_evidence_returns_typed_blocked_evidence() -> None:
    result = SelectionContextFirewall().apply(_request(bundle=_bundle(evidence=())))

    assert result.outcome is SelectionContextFirewallOutcome.BLOCKED_EVIDENCE
    assert result.selection_context is None
    assert result.blocked_reasons == frozenset({SelectionContextFirewallReason.MISSING_SAFE_EVIDENCE})


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda decisions: (decisions[0].model_copy(update={"policy_version": "stale-policy"}),),
            "decision policy",
        ),
        (
            lambda decisions: (
                decisions[0].model_copy(update={"label_decision_id": "label-decision://not-opaque"}),
            ),
            "decision identity",
        ),
        (
            lambda decisions: (decisions[0], decisions[0]),
            "duplicate decision",
        ),
        (
            lambda decisions: (
                decisions[0],
                _decision(
                    "duplicate-label",
                    label_spec_id=decisions[0].label_spec_ref.object_id,
                ),
            ),
            "duplicate label",
        ),
        (
            lambda decisions: (
                decisions[0].model_copy(update={"trace_envelope_ref": _trace_ref("trace-ir://other/v1")}),
            ),
            "trace",
        ),
        (
            lambda decisions: (
                _decision(
                    "unsafe",
                    evidence_subject_ref=_ref(
                        "final-output",
                        "final-output://must-fail",
                    ),
                ),
            ),
            "unsafe decision",
        ),
    ),
)
def test_corrupt_or_unsafe_decisions_fail_closed(
    mutate,
    message: str,
) -> None:
    decisions = (_decision("one"), _decision("two"))

    with pytest.raises(SelectionContextFirewallPolicyError, match=message):
        SelectionContextFirewall().apply(_request(decisions=mutate(decisions)))


@pytest.mark.parametrize(
    ("bundle_update", "message"),
    (
        ({"consumer_stage": "semantic-labeler"}, "consumer stage"),
        ({"purpose": "labeling"}, "purpose"),
        (
            {
                "projection_policy_ref": _ref(
                    "raw-trace",
                    "raw-trace://unsafe-policy",
                )
            },
            "projection policy",
        ),
        ({"tainted_content_included": True}, "tainted"),
        ({"returned_characters": 1001}, "budget"),
        (
            {
                "evidence": (
                    _evidence(
                        "wrong-span",
                        source_trace_id="source-trace://other",
                    ),
                )
            },
            "source trace",
        ),
        (
            {
                "evidence": (
                    _evidence(
                        "unsafe",
                        subject_ref=_ref(
                            "private-reference",
                            "private-reference://unsafe-evidence",
                        ),
                    ),
                )
            },
            "unsafe evidence",
        ),
    ),
)
def test_invalid_bundle_boundaries_fail_closed(
    bundle_update: dict[str, object],
    message: str,
) -> None:
    bundle = _bundle().model_copy(update=bundle_update)
    request = _request().model_copy(update={"evidence_bundle": bundle})

    with pytest.raises(SelectionContextFirewallPolicyError, match=message):
        SelectionContextFirewall().apply(request)


def test_stale_bundle_identity_fails_closed() -> None:
    bundle = _bundle()
    mutated = bundle.model_copy(
        update={
            "evidence": (
                bundle.evidence[0].model_copy(update={"capability": "silently-mutated-capability"}),
                bundle.evidence[1],
            )
        }
    )

    with pytest.raises(SelectionContextFirewallPolicyError, match="bundle identity"):
        SelectionContextFirewall().apply(_request(bundle=mutated))


def test_notes_must_be_unique_exact_safe_bundle_subjects() -> None:
    bundle = _bundle()
    authorized = bundle.evidence[0].subject_ref

    with pytest.raises(SelectionContextFirewallPolicyError, match="duplicate note"):
        SelectionContextFirewall().apply(_request(bundle=bundle, notes=(authorized, authorized)))
    with pytest.raises(SelectionContextFirewallPolicyError, match="authorized"):
        SelectionContextFirewall().apply(
            _request(
                bundle=bundle,
                notes=(
                    _ref(
                        "file-version-projection",
                        "evidence-view://safe-looking-but-not-authorized",
                    ),
                ),
            )
        )
    with pytest.raises(SelectionContextFirewallPolicyError, match="unsafe note"):
        SelectionContextFirewall().apply(
            _request(
                bundle=bundle,
                notes=(
                    _ref(
                        "grader-rule",
                        "grader-rule://unsafe-note",
                    ),
                ),
            )
        )


def test_shared_carried_ref_authorities_are_used_by_r3_and_r4() -> None:
    decision = _decision("shared")
    expected_decision_ref = label_decision_ref(decision)

    assert LabelBatchProgressTracker.decision_ref(decision) == expected_decision_ref
    assert LabelCalibrationObservation.from_decision(decision).decision_ref == (expected_decision_ref)

    firewall_result = SelectionContextFirewall().apply(_request(decisions=(decision,)))
    assert firewall_result.selection_context is not None
    context = firewall_result.selection_context
    segment = InteractionSegment(
        segment_id="interaction-segment://selection-firewall/one",
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        boundary_method="user_turn",
        sequence_start=0,
        sequence_end=0,
        member_event_refs=(
            _ref(
                "trace-event",
                "trace-event://selection-firewall/one",
            ),
        ),
    )
    request = TaskEpisodeGroupingRequestBuilder().build(
        selection_context=context,
        trace_envelope_ref=_trace_ref(),
        segments=(segment,),
        evidence_bundle=_request(decisions=(decision,)).evidence_bundle,
        model_profile="internal-task-episode-grouper-v1",
        prompt_version="task-episode-grouping-prompt/v1",
        abstain_conditions=("ambiguous task boundary",),
        audit=_audit(),
    )
    assert request.selection_context_ref == selection_context_ref(context)
    serialized = request.model_dump_json()
    assert decision.label_decision_id not in serialized
    assert decision.decision_sha256 not in serialized
    assert context.excluded_signal_hashes[0] not in serialized


def test_allowed_identity_is_stable_across_audit_time_and_input_order() -> None:
    first = _allowed_result(
        reverse_inputs=False,
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
    )
    second = _allowed_result(
        reverse_inputs=True,
        created_at=datetime(2026, 7, 27, tzinfo=UTC),
    )
    assert first.selection_context is not None
    assert second.selection_context is not None

    assert first.input_sha256 == second.input_sha256
    assert first.selection_context.candidate_id == second.selection_context.candidate_id
    assert first.selection_context.selection_context_id == second.selection_context.selection_context_id
    assert (
        first.selection_context.selection_context_sha256 == second.selection_context.selection_context_sha256
    )
    assert first.selection_context_firewall_result_id == second.selection_context_firewall_result_id
    assert first.result_sha256 == second.result_sha256
    assert first.canonical_sha256() != second.canonical_sha256()


def test_blocked_results_bind_distinct_exact_inputs() -> None:
    no_decisions = SelectionContextFirewall().apply(_request(decisions=()))
    no_match = SelectionContextFirewall().apply(
        _request(decisions=(_decision("no-match", decision=LabelDecisionValueV2.NO_MATCH),))
    )

    assert no_decisions.selection_context is None
    assert no_match.selection_context is None
    assert no_decisions.input_sha256 != no_match.input_sha256
    for result in (no_decisions, no_match):
        serialized = result.model_dump_json()
        assert "label-spec://selection-firewall" not in serialized
        assert "evidence-ref://selection-firewall/decision" not in serialized
        assert "hidden-selection-signal://error-signature" not in serialized


def test_firewall_models_reject_unknown_hidden_fields() -> None:
    request = _request()
    with pytest.raises(ValidationError):
        SelectionContextFirewallRequest.model_validate(
            {
                **request.model_dump(mode="json"),
                "hidden_signal_text": "must not be accepted",
            }
        )

    result = SelectionContextFirewall().apply(request)
    with pytest.raises(ValidationError):
        SelectionContextFirewallResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "raw_trace_text": "must not be accepted",
            }
        )


def test_firewall_identity_is_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_selection_firewall_seed.py"
    test_file = ROOT / "tests/eval_factory/unit/test_selection_context_firewall.py"
    script.write_text(
        f"""
import runpy
namespace = runpy.run_path({str(test_file)!r}, run_name='selection_firewall_fixture')
result = namespace['_allowed_result'](reverse_inputs=True)
context = result.selection_context
assert context is not None
print(result.input_sha256)
print(context.candidate_id)
print(context.selection_context_id)
print(context.selection_context_sha256)
print(result.selection_context_firewall_result_id)
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
