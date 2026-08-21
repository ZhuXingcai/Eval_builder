from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_plan_review import _audit, _ref, _service
from typer.testing import CliRunner

from eval_factory.agent_system.core_vertical import CoreVerticalExecution
from eval_factory.agent_system.output import (
    CoreOutputAssembler,
    CoreOutputConflictError,
    CoreOutputIntegrityError,
)
from eval_factory.cli import app
from eval_factory.contracts.agent_system_v2 import (
    CoreVerticalResultV2,
    ExtractedUserPromptV2,
    FactoryCompletionOutcomeV2,
    InferredUserIntentV2,
    IntentClaimV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ObjectRef, SourceSpanRef

HASH = "a" * 64
RUN_ID = "factory-run://plan-review"
REQUIREMENT_ID = "evaluation-requirement-spec://plan-review"


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(*values: ObjectRef) -> tuple[ObjectRef, ...]:
    return tuple(sorted(values, key=_ref_key))


def _execution(requirement_ref: ObjectRef) -> CoreVerticalExecution:
    prompt = ExtractedUserPromptV2.create(
        extracted_prompt_id="extracted-user-prompt://candidate",
        trace_ref=_ref("trace-ir", "candidate"),
        interaction_segment_ref=_ref("interaction-segment", "candidate"),
        content_ref=_ref("user-prompt-content", "candidate"),
        source_spans=(
            SourceSpanRef(
                span_id="source-span://candidate",
                source_trace_id="source-trace://candidate",
                raw_sha256=HASH,
            ),
        ),
        context_segment_refs=(),
        audit=_audit(),
    )
    intent = InferredUserIntentV2.create(
        inferred_intent_id="inferred-user-intent://candidate",
        extracted_prompt_ref=prompt.to_ref(),
        claims=(
            IntentClaimV2(
                claim_id="intent-claim://candidate",
                summary="Complete the source-grounded requested task.",
                evidence_refs=(prompt.to_ref(),),
                confidence_basis_points=9000,
                uncertain=False,
            ),
        ),
        unresolved_requirements=(),
        abstained=False,
        audit=_audit(),
    )
    rewrite = TaskRewriteCandidateV2.create(
        candidate_id="task-rewrite-candidate://candidate",
        extracted_prompt_ref=prompt.to_ref(),
        inferred_intent_ref=intent.to_ref(),
        rewrite_plan_ref=_ref("task-rewrite-plan", "candidate"),
        rewritten_prompt_ref=_ref("rewritten-prompt-content", "candidate"),
        evidence_refs=_sorted_refs(prompt.to_ref(), intent.to_ref()),
        audit=_audit(),
    )
    candidate = TraceCandidateDecisionV2.create(
        decision_id="trace-candidate-decision://candidate",
        source_trace_id="source-trace://candidate",
        source_ref=_ref("trace-source", "candidate"),
        disposition=TraceCandidateDispositionV2.CANDIDATE,
        reason_codes=(),
        cleaned_trace_ref=_ref("trace-ir", "candidate"),
        audit=_audit(),
    )
    non_candidate = TraceCandidateDecisionV2.create(
        decision_id="trace-candidate-decision://non-candidate",
        source_trace_id="source-trace://non-candidate",
        source_ref=_ref("trace-source", "non-candidate"),
        disposition=TraceCandidateDispositionV2.NON_CANDIDATE,
        reason_codes=("USER_PROMPT_MISSING",),
        cleaned_trace_ref=None,
        audit=_audit(),
    )
    result = CoreVerticalResultV2.create(
        result_id="core-vertical-result://output-test",
        manifest_ref=_ref("trace-manifest"),
        requirement_spec_ref=requirement_ref,
        candidate_decision_refs=(candidate.to_ref(),),
        non_candidate_decision_refs=(non_candidate.to_ref(),),
        blocked_decision_refs=(),
        extracted_prompt_refs=(prompt.to_ref(),),
        inferred_intent_refs=(intent.to_ref(),),
        rewrite_candidate_refs=(rewrite.to_ref(),),
        route_decision_refs=_sorted_refs(
            _ref("model-route-decision", "extraction"),
            _ref("model-route-decision", "planning"),
            _ref("model-route-decision", "rewrite"),
        ),
        source_count=2,
        audit=_audit(),
    )
    return CoreVerticalExecution(
        result=result,
        decisions=(candidate, non_candidate),
        extracted_prompts=(prompt,),
        inferred_intents=(intent,),
        rewrite_candidates=(rewrite,),
    )


def _inputs(tmp_path: Path):
    _, store = _service(tmp_path)
    requirement = store.get_requirement(REQUIREMENT_ID)
    execution = _execution(requirement.to_ref())
    return store, requirement, execution


def test_core_output_is_atomic_content_free_and_exactly_replayed(
    tmp_path: Path,
) -> None:
    store, requirement, execution = _inputs(tmp_path)
    output_root = tmp_path / "core-output"
    assembler = CoreOutputAssembler(store=store, root=output_root)

    first = assembler.assemble(
        run_id=RUN_ID,
        requirement=requirement,
        execution=execution,
        audit=_audit(),
        idempotency_key="assemble-output",
    )
    replay = assembler.assemble(
        run_id=RUN_ID,
        requirement=requirement,
        execution=execution,
        audit=_audit("replay"),
        idempotency_key="assemble-output",
    )

    assert first.reused is False
    assert replay.reused is True
    assert replay.view == first.view
    assert replay.bundle_path == first.bundle_path
    assert first.view.completion.outcome is FactoryCompletionOutcomeV2.INCOMPLETE
    assert first.view.completion.reason_codes == ("CORE_VERTICAL_ONLY",)
    assert first.view.delivery_manifest.item_count == 1
    assert tuple(value.relative_path for value in first.view.inventory.files) == (
        "README.md",
        "audit-refs.json",
        "completion.json",
        "extracted-user-prompts.jsonl",
        "inferred-user-intents.jsonl",
        "manifest.json",
        "requirement-summary.json",
        "task-rewrite-candidates.jsonl",
        "trace-candidates.jsonl",
        "unresolved-items.json",
    )
    rendered = b"".join(
        (first.bundle_path / value.relative_path).read_bytes() for value in first.view.inventory.files
    )
    assert b"prompt_text" not in rendered
    assert b"model_output_body" not in rendered
    current_run = store.get_run(RUN_ID)
    assert current_run.delivery_manifest_ref is None
    assert first.view.delivery_manifest.to_ref() in current_run.result_refs

    before = tuple(sorted((output_root / "cas/sha256").glob("*/*")))
    with pytest.raises(CoreOutputConflictError, match="already has"):
        assembler.assemble(
            run_id=RUN_ID,
            requirement=requirement,
            execution=execution,
            audit=_audit(),
            idempotency_key="different-key",
        )
    assert tuple(sorted((output_root / "cas/sha256").glob("*/*"))) == before


class _CrashAt:
    def __init__(self, point: str) -> None:
        self.point = point
        self.fired = False

    def __call__(self, point: str) -> None:
        if point == self.point and not self.fired:
            self.fired = True
            raise RuntimeError(f"injected core output crash: {point}")


@pytest.mark.parametrize(
    "fault_point",
    (
        "after_rename",
        "after_completion",
        "after_manifest",
        "after_inventory",
        "after_output_head",
        "after_run_head",
        "after_outbox",
        "after_idempotency",
    ),
)
def test_core_output_crash_replay_has_one_authority(
    tmp_path: Path,
    fault_point: str,
) -> None:
    store, requirement, execution = _inputs(tmp_path)
    output_root = tmp_path / "core-output"
    crashing = CoreOutputAssembler(
        store=store,
        root=output_root,
        fault_injector=_CrashAt(fault_point),
    )
    with pytest.raises(RuntimeError, match="injected core output crash"):
        crashing.assemble(
            run_id=RUN_ID,
            requirement=requirement,
            execution=execution,
            audit=_audit(),
            idempotency_key="assemble-output",
        )

    recovered = CoreOutputAssembler(store=store, root=output_root).assemble(
        run_id=RUN_ID,
        requirement=requirement,
        execution=execution,
        audit=_audit(),
        idempotency_key="assemble-output",
    )
    assert recovered.reused is True
    connection = store._connect()
    try:
        assert connection.execute("SELECT COUNT(*) FROM core_output_current_heads").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM core_output_inventories").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM dataset_delivery_manifests").fetchone()[0] == 1
    finally:
        connection.close()


def test_core_output_detects_physical_and_projection_drift(tmp_path: Path) -> None:
    store, requirement, execution = _inputs(tmp_path)
    assembler = CoreOutputAssembler(store=store, root=tmp_path / "core-output")
    written = assembler.assemble(
        run_id=RUN_ID,
        requirement=requirement,
        execution=execution,
        audit=_audit(),
        idempotency_key="assemble-output",
    )
    (written.bundle_path / "README.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(CoreOutputIntegrityError, match="inventory differs"):
        assembler.get(RUN_ID)

    connection = store._connect()
    try:
        connection.execute(
            "UPDATE core_output_inventories SET bundle_sha256 = ?",
            ("b" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(CoreOutputIntegrityError):
        assembler.get(RUN_ID)


def test_agent_output_cli_returns_content_free_authority(tmp_path: Path) -> None:
    store, requirement, execution = _inputs(tmp_path)
    output_root = tmp_path / "core-output"
    CoreOutputAssembler(store=store, root=output_root).assemble(
        run_id=RUN_ID,
        requirement=requirement,
        execution=execution,
        audit=_audit(),
        idempotency_key="assemble-output",
    )

    result = CliRunner().invoke(
        app,
        [
            "agent",
            "output",
            "--factory-store",
            str(store.path),
            "--output-root",
            str(output_root),
            "--run",
            RUN_ID,
        ],
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["delivery_manifest"]["item_count"] == 1
    assert "bundle_path" not in payload
