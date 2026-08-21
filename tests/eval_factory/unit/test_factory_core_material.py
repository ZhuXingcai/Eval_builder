from __future__ import annotations

import sqlite3

import pytest
from test_factory_dataset_runtime import (
    USER,
    _audit,
    _policy,
    _ref,
    _request,
    _requirement,
    _runtime,
)

from eval_factory.agent_system.core_material import (
    FactoryCoreMaterialError,
    FactoryCoreMaterialStore,
)
from eval_factory.agent_system.core_vertical import (
    CoreVerticalExecution,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
)
from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    CoreVerticalResultV2,
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    IntentClaimV2,
    PlanDecisionKindV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import SourceSpanRef

HASH = "a" * 64


async def _prepared(tmp_path):
    runtime, reviews, _provider, store = _runtime(tmp_path)
    waiting = await runtime.advance(
        request=_request(),
        policy=_policy(),
        requirement=_requirement(),
        audit=_audit(),
    )
    review_id = waiting.pending_review_refs[0].object_id
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by=USER,
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-core-material",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by=USER,
            idempotency_key="resume-core-material",
        ),
        audit=_audit(),
    )
    return store, FactoryCoreMaterialStore(store)


def _execution(*, route_suffix: str = "base") -> CoreVerticalExecution:
    decision = TraceCandidateDecisionV2.create(
        decision_id="trace-candidate-decision://material",
        source_trace_id="source-trace://material",
        source_ref=_ref("trace-source"),
        disposition=TraceCandidateDispositionV2.CANDIDATE,
        reason_codes=(),
        cleaned_trace_ref=_ref("trace-ir"),
        audit=_audit(),
    )
    prompt = ExtractedUserPromptV2.create(
        extracted_prompt_id="extracted-user-prompt://material",
        trace_ref=decision.cleaned_trace_ref,
        interaction_segment_ref=_ref("interaction-segment"),
        content_ref=_ref("user-prompt-content"),
        source_spans=(
            SourceSpanRef(
                span_id="source-span://material",
                source_trace_id=decision.source_trace_id,
                raw_sha256=HASH,
            ),
        ),
        context_segment_refs=(),
        audit=_audit(),
    )
    intent = InferredUserIntentV2.create(
        inferred_intent_id="inferred-user-intent://material",
        extracted_prompt_ref=prompt.to_ref(),
        claims=(
            IntentClaimV2(
                claim_id="intent-claim://material",
                summary="Build one safe candidate.",
                evidence_refs=(prompt.to_ref(),),
                confidence_basis_points=9_000,
                uncertain=False,
            ),
        ),
        unresolved_requirements=(),
        abstained=False,
        audit=_audit(),
    )
    rewrite = TaskRewriteCandidateV2.create(
        candidate_id="task-rewrite-candidate://material",
        extracted_prompt_ref=prompt.to_ref(),
        inferred_intent_ref=intent.to_ref(),
        rewrite_plan_ref=_ref("task-rewrite-plan"),
        rewritten_prompt_ref=_ref("rewritten-prompt-content"),
        evidence_refs=(prompt.to_ref(), intent.to_ref()),
        audit=_audit(),
    )
    result = CoreVerticalResultV2.create(
        result_id=f"core-vertical-result://material/{route_suffix}",
        manifest_ref=_request().manifest_ref,
        requirement_spec_ref=_requirement().to_ref(),
        candidate_decision_refs=(decision.to_ref(),),
        non_candidate_decision_refs=(),
        blocked_decision_refs=(),
        extracted_prompt_refs=(prompt.to_ref(),),
        inferred_intent_refs=(intent.to_ref(),),
        rewrite_candidate_refs=(rewrite.to_ref(),),
        route_decision_refs=(
            _ref("model-route-decision", f"{route_suffix}-a"),
            _ref("model-route-decision", f"{route_suffix}-b"),
            _ref("model-route-decision", f"{route_suffix}-c"),
        ),
        source_count=1,
        audit=_audit(),
    )
    return CoreVerticalExecution(
        result=result,
        decisions=(decision,),
        extracted_prompts=(prompt,),
        inferred_intents=(intent,),
        rewrite_candidates=(rewrite,),
    )


@pytest.mark.asyncio
async def test_core_material_commit_replay_and_rebuild(tmp_path) -> None:
    store, materials = await _prepared(tmp_path)
    execution = _execution()
    compiled_ref = store.get_plan(_requirement().run_id)[1].to_ref()

    committed = materials.commit(
        run_id=_requirement().run_id,
        compiled_plan_ref=compiled_ref,
        execution=execution,
        idempotency_key="core-material",
    )
    replay = materials.commit(
        run_id=_requirement().run_id,
        compiled_plan_ref=compiled_ref,
        execution=execution,
        idempotency_key="core-material",
    )
    assert replay == committed == execution

    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM factory_core_current_heads")
    with pytest.raises(
        FactoryControlIntegrityError,
        match="current head",
    ):
        materials.get(_requirement().run_id)
    assert materials.rebuild_current_heads() == 1
    assert materials.get(_requirement().run_id) == execution


@pytest.mark.asyncio
async def test_core_material_conflict_and_wrong_plan_are_closed(
    tmp_path,
) -> None:
    store, materials = await _prepared(tmp_path)
    execution = _execution()
    compiled_ref = store.get_plan(_requirement().run_id)[1].to_ref()
    materials.commit(
        run_id=_requirement().run_id,
        compiled_plan_ref=compiled_ref,
        execution=execution,
        idempotency_key="core-material",
    )

    with pytest.raises(
        FactoryControlConflictError,
        match="authority already exists",
    ):
        materials.commit(
            run_id=_requirement().run_id,
            compiled_plan_ref=compiled_ref,
            execution=execution,
            idempotency_key="different-key",
        )
    with pytest.raises(
        FactoryControlConflictError,
        match="idempotency",
    ):
        materials.commit(
            run_id=_requirement().run_id,
            compiled_plan_ref=compiled_ref,
            execution=_execution(route_suffix="changed"),
            idempotency_key="core-material",
        )
    with pytest.raises(
        FactoryCoreMaterialError,
        match="compiled plan",
    ):
        materials.commit(
            run_id=_requirement().run_id,
            compiled_plan_ref=_ref(
                "compiled-dataset-build-plan",
                "wrong",
            ),
            execution=_execution(route_suffix="wrong-plan"),
            idempotency_key="wrong-plan",
        )


@pytest.mark.asyncio
async def test_core_material_rejects_inventory_and_immutable_drift(
    tmp_path,
) -> None:
    store, materials = await _prepared(tmp_path)
    execution = _execution()
    incomplete = CoreVerticalExecution(
        result=execution.result,
        decisions=execution.decisions,
        extracted_prompts=execution.extracted_prompts,
        inferred_intents=execution.inferred_intents,
        rewrite_candidates=(),
    )
    with pytest.raises(
        FactoryCoreMaterialError,
        match="inventory",
    ):
        materials.commit(
            run_id=_requirement().run_id,
            compiled_plan_ref=store.get_plan(_requirement().run_id)[1].to_ref(),
            execution=incomplete,
            idempotency_key="incomplete",
        )

    materials.commit(
        run_id=_requirement().run_id,
        compiled_plan_ref=store.get_plan(_requirement().run_id)[1].to_ref(),
        execution=execution,
        idempotency_key="valid",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE factory_core_materials
            SET record_json = '{}'
            WHERE object_id = ?
            """,
            (execution.result.object_id,),
        )
    with pytest.raises(
        FactoryControlIntegrityError,
        match="schema",
    ):
        materials.get(_requirement().run_id)


def test_core_material_missing_read_is_typed(tmp_path) -> None:
    store = FactoryControlStore(tmp_path / "empty.sqlite3")
    materials = FactoryCoreMaterialStore(store)
    with pytest.raises(
        FactoryControlNotFoundError,
        match="not found",
    ):
        materials.get("factory-run://missing")
