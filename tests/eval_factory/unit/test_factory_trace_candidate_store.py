from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_factory_core_material import (
    _execution,
    _prepared,
    _ref,
    _request,
)

from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlIntegrityError,
)
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparation,
)
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import ExtractedUserPromptV2


@pytest.mark.asyncio
async def test_trace_candidate_store_replays_and_resolves_owner_material(
    tmp_path: Path,
) -> None:
    store, _core_materials = await _prepared(tmp_path)
    execution = _execution()
    preparation = TraceCandidatePreparation(
        decision=execution.decisions[0],
        extracted_prompt=execution.extracted_prompts[0],
        inferred_intent=execution.inferred_intents[0],
        rewrite_candidate=execution.rewrite_candidates[0],
        route_refs=execution.result.route_decision_refs,
    )
    candidates = FactoryTraceCandidateMaterialStore(store)
    source_ref = preparation.decision.source_ref
    trace_ref = _ref("trace-envelope")

    first = candidates.commit(
        run_id=_request().dataset_run_id,
        trace_source_ref=source_ref,
        trace_envelope_ref=trace_ref,
        preparation=preparation,
        audit=preparation.decision.audit,
        idempotency_key="commit-trace-candidate",
    )
    replay = candidates.commit(
        run_id=_request().dataset_run_id,
        trace_source_ref=source_ref,
        trace_envelope_ref=trace_ref,
        preparation=preparation,
        audit=preparation.decision.audit,
        idempotency_key="commit-trace-candidate",
    )
    second_key = candidates.commit(
        run_id=_request().dataset_run_id,
        trace_source_ref=source_ref,
        trace_envelope_ref=trace_ref,
        preparation=preparation,
        audit=preparation.decision.audit,
        idempotency_key="commit-trace-candidate-replay",
    )

    assert replay == second_key == first
    assert (
        candidates.get(
            run_id=_request().dataset_run_id,
            trace_source_ref=source_ref,
        )
        == first
    )
    assert (
        candidates.load(
            preparation.extracted_prompt.to_ref(),
            ExtractedUserPromptV2,
        )
        == preparation.extracted_prompt
    )
    assert (
        candidates.load(
            _ref("missing-trace-candidate-material"),
            ExtractedUserPromptV2,
        )
        is None
    )


@pytest.mark.asyncio
async def test_trace_candidate_store_rejects_conflict_and_missing_head(
    tmp_path: Path,
) -> None:
    store, _core_materials = await _prepared(tmp_path)
    execution = _execution()
    preparation = TraceCandidatePreparation(
        decision=execution.decisions[0],
        extracted_prompt=execution.extracted_prompts[0],
        inferred_intent=execution.inferred_intents[0],
        rewrite_candidate=execution.rewrite_candidates[0],
        route_refs=execution.result.route_decision_refs,
    )
    candidates = FactoryTraceCandidateMaterialStore(store)
    source_ref = preparation.decision.source_ref
    candidates.commit(
        run_id=_request().dataset_run_id,
        trace_source_ref=source_ref,
        trace_envelope_ref=_ref("trace-envelope"),
        preparation=preparation,
        audit=preparation.decision.audit,
        idempotency_key="commit-trace-candidate",
    )
    changed = TraceCandidatePreparation(
        decision=preparation.decision,
        extracted_prompt=preparation.extracted_prompt,
        inferred_intent=preparation.inferred_intent,
        rewrite_candidate=preparation.rewrite_candidate,
        route_refs=(_ref("model-route-decision", "changed"),),
    )
    with pytest.raises(
        FactoryControlConflictError,
        match="conflicts",
    ):
        candidates.commit(
            run_id=_request().dataset_run_id,
            trace_source_ref=source_ref,
            trace_envelope_ref=_ref("trace-envelope"),
            preparation=changed,
            audit=preparation.decision.audit,
            idempotency_key="changed-trace-candidate",
        )

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM factory_trace_candidate_current_heads",
        )
        connection.commit()
    with pytest.raises(
        FactoryControlIntegrityError,
        match="current head",
    ):
        candidates.get(
            run_id=_request().dataset_run_id,
            trace_source_ref=source_ref,
        )
