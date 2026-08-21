from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
)
from eval_factory.labeling import (
    LABEL_BATCH_POLICY_VERSION,
    LabelBatchExporter,
    LabelBatchPlan,
    LabelBatchPolicyError,
    LabelBatchProgressTracker,
    LabelPairStatus,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
ROOT = Path(__file__).resolve().parents[3]


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="label-batch-test",
        governing_versions=(VersionBinding(component="label-batch", version="r3-05"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _trace_ref(index: int = 1) -> ObjectRef:
    return _ref(
        "trace-envelope",
        f"trace-envelope://batch-test/{index}",
        [HASH, OTHER_HASH, THIRD_HASH][index - 1],
    )


def _label_ref(index: int = 1) -> ObjectRef:
    return _ref(
        "label-spec",
        f"label-spec://batch-test/{index}",
        [THIRD_HASH, HASH, OTHER_HASH][index - 1],
    )


def _span() -> SourceSpanRef:
    return SourceSpanRef(
        span_id="source-span://label-batch-test",
        source_trace_id="source-trace://label-batch-test",
        raw_sha256=HASH,
    )


def _evidence(evidence_id: str, *, polarity: EvidencePolarity = EvidencePolarity.POSITIVE) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=evidence_id,
        subject_ref=_ref("trace-event", f"trace-event://{evidence_id.rsplit('/', 1)[-1]}"),
        source_spans=(_span(),),
        polarity=polarity,
        capability="label-batch-evidence",
        capability_complete=True,
    )


def _plan(
    *,
    traces: tuple[ObjectRef, ...] = (_trace_ref(),),
    labels: tuple[ObjectRef, ...] = (_label_ref(),),
    audit: ContractAudit | None = None,
) -> LabelBatchPlan:
    return LabelBatchPlan.build(
        trace_envelope_refs=traces,
        label_spec_refs=labels,
        audit=audit or _audit(),
    )


def _decision(
    trace_ref: ObjectRef,
    label_ref: ObjectRef,
    *,
    decision_id: str = "label-decision://batch-test/1",
    decision: LabelDecisionValueV2 = LabelDecisionValueV2.MATCH,
    evidence: tuple[EvidenceRef, ...] = (_evidence("evidence-ref://batch/positive"),),
    unresolved: frozenset[LabelUnresolvedReason] = frozenset(),
    confidence: float = 0.91,
    model_profile: str | None = None,
    prompt_version: str | None = None,
    decision_sha256: str = OTHER_HASH,
    audit: ContractAudit | None = None,
) -> LabelDecisionV2:
    return LabelDecisionV2(
        label_decision_id=decision_id,
        label_spec_ref=label_ref,
        trace_envelope_ref=trace_ref,
        decision=decision,
        execution_status=(
            LabelExecutionStatus.UNRESOLVED
            if decision is LabelDecisionValueV2.ABSTAIN
            else LabelExecutionStatus.FINAL
        ),
        positive_evidence=evidence if decision is LabelDecisionValueV2.MATCH else (),
        negative_evidence=(evidence if decision is LabelDecisionValueV2.NO_MATCH else ()),
        semantic_evidence=(),
        structured_capability_complete=True,
        confidence=confidence,
        rule_version="label-decision-merge/r3-04-v1",
        model_profile=model_profile,
        prompt_version=prompt_version,
        unresolved_reasons=unresolved,
        policy_version="label-decision-merge/r3-04-v1",
        decision_sha256=decision_sha256,
        audit=audit or _audit(),
    )


def _completed_checkpoint():
    tracker = LabelBatchProgressTracker()
    plan = _plan()
    checkpoint = tracker.initialize(plan, audit=_audit())
    pair = checkpoint.pairs[0]
    decision = _decision(pair.trace_envelope_ref, pair.label_spec_ref)
    checkpoint = tracker.record_decision(
        checkpoint,
        pair_id=pair.pair_id,
        decision=decision,
        audit=_audit(),
    )
    return checkpoint, decision


def test_batch_plan_expands_deterministic_pair_ids() -> None:
    plan = _plan(
        traces=(_trace_ref(2), _trace_ref(1)),
        labels=(_label_ref(2), _label_ref(1)),
        audit=_audit(),
    )
    same_plan = _plan(
        traces=(_trace_ref(2), _trace_ref(1)),
        labels=(_label_ref(2), _label_ref(1)),
        audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
    )

    checkpoint = LabelBatchProgressTracker().initialize(plan, audit=_audit())
    same_checkpoint = LabelBatchProgressTracker().initialize(
        same_plan,
        audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
    )

    assert plan.batch_id == same_plan.batch_id
    assert checkpoint.checkpoint_id == same_checkpoint.checkpoint_id
    assert checkpoint.checkpoint_sha256 == same_checkpoint.checkpoint_sha256
    assert [pair.status for pair in checkpoint.pairs] == [LabelPairStatus.PENDING] * 4
    assert [pair.pair_id for pair in checkpoint.pairs] == sorted(pair.pair_id for pair in checkpoint.pairs)
    assert len({pair.pair_id for pair in checkpoint.pairs}) == 4


def test_progress_summary_and_completion_records_decision_ref() -> None:
    tracker = LabelBatchProgressTracker()
    checkpoint = tracker.initialize(_plan(), audit=_audit())
    pair = checkpoint.pairs[0]
    running = tracker.record_started(checkpoint, pair_id=pair.pair_id, audit=_audit())
    decision = _decision(pair.trace_envelope_ref, pair.label_spec_ref)

    completed = tracker.record_decision(
        running,
        pair_id=pair.pair_id,
        decision=decision,
        audit=_audit(),
    )

    summary = completed.summary
    assert summary.total == 1
    assert summary.succeeded == 1
    assert summary.pending == 0
    assert completed.pairs[0].status is LabelPairStatus.SUCCEEDED
    assert completed.pairs[0].label_decision_ref == LabelBatchProgressTracker.decision_ref(decision)
    assert completed.pairs[0].attempt_count == 1


def test_resume_skips_completed_pairs_and_returns_only_incomplete_work() -> None:
    tracker = LabelBatchProgressTracker()
    checkpoint = tracker.initialize(
        _plan(traces=(_trace_ref(1), _trace_ref(2)), labels=(_label_ref(1),)),
        audit=_audit(),
    )
    first, second = checkpoint.pairs
    decision = _decision(first.trace_envelope_ref, first.label_spec_ref)
    checkpoint = tracker.record_decision(
        checkpoint,
        pair_id=first.pair_id,
        decision=decision,
        audit=_audit(),
    )

    resume = tracker.resume_work(checkpoint)

    assert [pair.pair_id for pair in resume.executable_pairs] == [second.pair_id]
    assert resume.summary.resumed_completed == 1
    assert resume.summary.executable == 1


def test_failed_pair_does_not_block_unrelated_pairs() -> None:
    tracker = LabelBatchProgressTracker()
    checkpoint = tracker.initialize(
        _plan(traces=(_trace_ref(1), _trace_ref(2)), labels=(_label_ref(1),)),
        audit=_audit(),
    )
    first, second = checkpoint.pairs
    failed = tracker.record_failure(
        checkpoint,
        pair_id=first.pair_id,
        failure_code="temporary-model-error",
        retryable=True,
        audit=_audit(),
    )

    resume = tracker.resume_work(failed)

    assert failed.summary.failed == 1
    assert {pair.pair_id for pair in resume.executable_pairs} == {first.pair_id, second.pair_id}


def test_mismatched_decision_refs_fail_closed() -> None:
    tracker = LabelBatchProgressTracker()
    checkpoint = tracker.initialize(_plan(), audit=_audit())
    pair = checkpoint.pairs[0]
    decision = _decision(
        _trace_ref(2),
        pair.label_spec_ref,
        decision_id="label-decision://batch-test/mismatch",
    )

    with pytest.raises(LabelBatchPolicyError, match="trace envelope"):
        tracker.record_decision(
            checkpoint,
            pair_id=pair.pair_id,
            decision=decision,
            audit=_audit(),
        )


def test_missing_export_decision_fails_closed(tmp_path: Path) -> None:
    checkpoint, decision = _completed_checkpoint()
    stale_decision = _decision(
        decision.trace_envelope_ref,
        decision.label_spec_ref,
        decision_id=decision.label_decision_id,
        confidence=0.92,
        decision_sha256=THIRD_HASH,
    )

    with pytest.raises(LabelBatchPolicyError, match="stale or missing label decision"):
        LabelBatchExporter().write_jsonl(
            checkpoint,
            decisions=(stale_decision,),
            path=tmp_path / "labels.jsonl",
            audit=_audit(),
        )


def test_jsonl_export_is_deterministic_sorted_and_content_free(tmp_path: Path) -> None:
    checkpoint, decision = _completed_checkpoint()
    exporter = LabelBatchExporter()
    path_one = tmp_path / "labels-one.jsonl"
    path_two = tmp_path / "labels-two.jsonl"

    manifest_one = exporter.write_jsonl(checkpoint, decisions=(decision,), path=path_one, audit=_audit())
    manifest_two = exporter.write_jsonl(
        checkpoint,
        decisions=(decision,),
        path=path_two,
        audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
    )

    lines = path_one.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    forbidden = (
        "raw-trace",
        "content_blob",
        "private-reference",
        "quarantine",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "answer-bearing",
    )

    assert path_one.read_bytes() == path_two.read_bytes()
    assert manifest_one.content_sha256 == manifest_two.content_sha256
    assert manifest_one.manifest_sha256 == manifest_two.manifest_sha256
    assert [row["row_id"] for row in rows] == sorted(row["row_id"] for row in rows)
    assert rows[0]["decision"] == "MATCH"
    assert rows[0]["structured_capability_complete"] is True
    assert rows[0]["positive_evidence_ref_ids"] == ["evidence-ref://batch/positive"]
    assert rows[0]["batch_policy_version"] == LABEL_BATCH_POLICY_VERSION
    assert all(term not in json.dumps(rows, sort_keys=True).lower() for term in forbidden)


def test_parquet_export_matches_jsonl_row_ids(tmp_path: Path) -> None:
    checkpoint, decision = _completed_checkpoint()
    exporter = LabelBatchExporter()
    jsonl_path = tmp_path / "labels.jsonl"
    parquet_path = tmp_path / "labels.parquet"

    jsonl_manifest = exporter.write_jsonl(
        checkpoint,
        decisions=(decision,),
        path=jsonl_path,
        audit=_audit(),
    )
    parquet_manifest = exporter.write_parquet(
        checkpoint,
        decisions=(decision,),
        path=parquet_path,
        audit=_audit(),
    )

    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    parquet_rows = pd.read_parquet(parquet_path).to_dict(orient="records")

    assert parquet_manifest.row_count == jsonl_manifest.row_count == 1
    assert parquet_manifest.row_ids == jsonl_manifest.row_ids
    assert [row["row_id"] for row in parquet_rows] == [row["row_id"] for row in jsonl_rows]


def test_checkpoint_and_export_ids_ignore_audit_time_and_hash_seed(tmp_path: Path) -> None:
    checkpoint, decision = _completed_checkpoint()
    manifest = LabelBatchExporter().write_jsonl(
        checkpoint,
        decisions=(decision,),
        path=tmp_path / "labels.jsonl",
        audit=_audit(),
    )
    script = tmp_path / "check_label_batch_seed.py"
    script.write_text(
        f"""
from datetime import UTC, datetime
from pathlib import Path
from eval_factory.contracts.core import ContractAudit, EvidencePolarity, EvidenceRef, ObjectRef, SourceSpanRef, VersionBinding
from eval_factory.contracts.labeling_v2 import LabelDecisionV2, LabelDecisionValueV2, LabelExecutionStatus
from eval_factory.labeling import LabelBatchExporter, LabelBatchPlan, LabelBatchProgressTracker

HASH = '{HASH}'
OTHER_HASH = '{OTHER_HASH}'
THIRD_HASH = '{THIRD_HASH}'
audit = ContractAudit(created_at=datetime(2026, 7, 25, tzinfo=UTC), created_by='seed-check', governing_versions=(VersionBinding(component='label-batch', version='r3-05'),))
trace_ref = ObjectRef(object_type='trace-envelope', object_id='trace-envelope://batch-test/1', object_version='v1', object_sha256=HASH)
label_ref = ObjectRef(object_type='label-spec', object_id='label-spec://batch-test/1', object_version='v1', object_sha256=THIRD_HASH)
span = SourceSpanRef(span_id='source-span://label-batch-test', source_trace_id='source-trace://label-batch-test', raw_sha256=HASH)
evidence = EvidenceRef(evidence_ref_id='evidence-ref://batch/positive', subject_ref=ObjectRef(object_type='trace-event', object_id='trace-event://positive', object_version='v1', object_sha256=HASH), source_spans=(span,), polarity=EvidencePolarity.POSITIVE, capability='label-batch-evidence', capability_complete=True)
plan = LabelBatchPlan.build(trace_envelope_refs=(trace_ref,), label_spec_refs=(label_ref,), audit=audit)
tracker = LabelBatchProgressTracker()
checkpoint = tracker.initialize(plan, audit=audit)
pair = checkpoint.pairs[0]
decision = LabelDecisionV2(label_decision_id='label-decision://batch-test/1', label_spec_ref=label_ref, trace_envelope_ref=trace_ref, decision=LabelDecisionValueV2.MATCH, execution_status=LabelExecutionStatus.FINAL, positive_evidence=(evidence,), negative_evidence=(), semantic_evidence=(), structured_capability_complete=True, confidence=0.91, rule_version='label-decision-merge/r3-04-v1', model_profile=None, prompt_version=None, unresolved_reasons=frozenset(), policy_version='label-decision-merge/r3-04-v1', decision_sha256=OTHER_HASH, audit=audit)
checkpoint = tracker.record_decision(checkpoint, pair_id=pair.pair_id, decision=decision, audit=audit)
manifest = LabelBatchExporter().write_jsonl(checkpoint, decisions=(decision,), path=Path('{(tmp_path / "seed-labels.jsonl").as_posix()}'), audit=audit)
print(plan.batch_id)
print(checkpoint.checkpoint_id)
print(checkpoint.checkpoint_sha256)
print(manifest.manifest_id)
print(manifest.manifest_sha256)
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

    assert checkpoint.checkpoint_sha256
    assert manifest.manifest_sha256
    assert run_one.stdout == run_two.stdout
