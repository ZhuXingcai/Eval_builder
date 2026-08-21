from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from test_factory_core_material import _execution
from test_factory_graph_bootstrap import _factory_inputs
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparation,
)
from eval_factory.contracts.agent_system_v2 import (
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.packs.generic_agent_trace.candidate_aggregate import (
    GenericAgentCandidateAggregate,
)


def test_candidate_aggregate_rebuilds_core_fan_in() -> None:
    _policy, requirement, request = _factory_inputs()
    execution = _execution()
    source_ref = execution.decisions[0].source_ref
    preparation = TraceCandidatePreparation(
        decision=execution.decisions[0],
        extracted_prompt=execution.extracted_prompts[0],
        inferred_intent=execution.inferred_intents[0],
        rewrite_candidate=execution.rewrite_candidates[0],
        route_refs=(
            ref("model-route-decision", "aggregate-a", version="v2"),
            ref("model-route-decision", "aggregate-b", version="v2"),
            ref("model-route-decision", "aggregate-c", version="v2"),
        ),
    )
    material = SimpleNamespace(
        authority=SimpleNamespace(
            to_ref=lambda: ref(
                "factory-trace-candidate-authority",
                "aggregate",
                version="v1",
            ),
        ),
        preparation=preparation,
    )
    planning_route_ref = ref(
        "model-route-decision",
        "aggregate-planning",
        version="v2",
    )
    aggregate = GenericAgentCandidateAggregate(
        request=request,
        requirement=requirement,
        factory_store=cast(
            Any,
            SimpleNamespace(
                get_planning_authority=lambda run_id: SimpleNamespace(
                    route_decision_ref=planning_route_ref,
                ),
            ),
        ),
        candidate_store=cast(
            Any,
            SimpleNamespace(get=lambda **kwargs: material),
        ),
        source_refs=(source_ref,),
    )

    result = aggregate.build(audit=audit())
    replay = aggregate.build(audit=audit())

    assert replay == result
    assert result.source_count == 1
    assert result.candidate_decision_refs == (preparation.decision.to_ref(),)
    assert result.extracted_prompt_refs == (preparation.extracted_prompt.to_ref(),)
    assert planning_route_ref in result.route_decision_refs


def test_candidate_aggregate_accepts_deterministic_non_candidate() -> None:
    _policy, requirement, request = _factory_inputs()
    execution = _execution()
    source_ref = execution.decisions[0].source_ref
    decision = TraceCandidateDecisionV2.create(
        decision_id="trace-candidate-decision://aggregate/non-candidate",
        source_trace_id=execution.decisions[0].source_trace_id,
        source_ref=source_ref,
        disposition=TraceCandidateDispositionV2.NON_CANDIDATE,
        reason_codes=("USER_PROMPT_MISSING",),
        cleaned_trace_ref=None,
        audit=audit(),
    )
    preparation = TraceCandidatePreparation(
        decision=decision,
        extracted_prompt=None,
        inferred_intent=None,
        rewrite_candidate=None,
        route_refs=(),
    )
    material = SimpleNamespace(
        authority=SimpleNamespace(
            to_ref=lambda: ref(
                "factory-trace-candidate-authority",
                "aggregate-non-candidate",
                version="v1",
            ),
        ),
        preparation=preparation,
    )
    planning_route_ref = ref(
        "model-route-decision",
        "aggregate-planning",
        version="v2",
    )
    aggregate = GenericAgentCandidateAggregate(
        request=request,
        requirement=requirement,
        factory_store=cast(
            Any,
            SimpleNamespace(
                get_planning_authority=lambda run_id: SimpleNamespace(
                    route_decision_ref=planning_route_ref,
                ),
            ),
        ),
        candidate_store=cast(
            Any,
            SimpleNamespace(get=lambda **kwargs: material),
        ),
        source_refs=(source_ref,),
    )

    result = aggregate.build(audit=audit())

    assert result.candidate_decision_refs == ()
    assert result.non_candidate_decision_refs == (decision.to_ref(),)
    assert result.route_decision_refs == (planning_route_ref,)
