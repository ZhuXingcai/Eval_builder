from __future__ import annotations

from collections import Counter
from typing import Any, cast

import pytest
from pydantic import BaseModel
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness import ArtifactEnvelopeV1, ArtifactModalityV1
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs import build_generic_agent_evaluation_blueprint
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    RequirementPlanningCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentCapabilityRequestPreparation,
    GenericAgentCapabilityRequestSource,
    GenericAgentFactoryWorkflow,
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamTaskSpec,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_pack_v1_0,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
    GenericAgentTaskScopeV1,
)
from eval_factory.team import (
    TeamCapabilityRunner,
    TeamSnapshot,
    TeamStore,
    TeamTaskWork,
)


def _sessions() -> dict[str, ObjectRef]:
    return {
        role: ref("harness-session", f"materializer-{role}")
        for role in (
            "control",
            "coordinator",
            "quality",
            "requirement",
            "task",
            "trace",
        )
    }


def _materialize(
    source_refs: tuple[ObjectRef, ...] | None = None,
):
    registration = team_fixture().registration
    sources = (
        source_refs
        if source_refs is not None
        else sorted_refs(
            (
                ref("trace-source", "sensitive-source-a", version="v2"),
                ref("trace-source", "sensitive-source-b", version="v2"),
            )
        )
    )
    return GenericAgentTaskGraphMaterializer().materialize(
        registration=registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "task-materializer",
            version="v2",
        ),
        source_refs=sources,
        independent_session_refs=_sessions(),
        team_id="team-task-materializer",
        team_incarnation_id="team-task-materializer-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )


def test_materializer_expands_capability_types_into_task_instances() -> None:
    materialized = _materialize()
    graph = materialized.authority.graph
    counts = Counter(value.capability_id for value in materialized.task_instances)

    assert len(graph.tasks) == len(materialized.task_instances) == 23
    assert counts == {
        "capability.requirement-planning": 1,
        "capability.trace-ingestion": 2,
        "capability.task-authoring": 2,
        "capability.attachment-reconstruction": 2,
        "capability.criteria-rubric": 2,
        "capability.grading-design": 2,
        "capability.quality-review": 2,
        "capability.batch-quality": 1,
        "capability.plan-review": 8,
        "capability.delivery": 1,
    }
    plan_kinds = Counter(
        value.plan_kind
        for value in materialized.task_instances
        if value.capability_id == "capability.plan-review"
    )
    assert plan_kinds == {
        PlanKindV2.GLOBAL_BUILD: 1,
        PlanKindV2.ATTACHMENT_GENERATION: 2,
        PlanKindV2.CRITERIA_RUBRIC: 2,
        PlanKindV2.GRADING_DESIGN: 2,
        PlanKindV2.FINAL_DELIVERY: 1,
    }
    assert sum(value.scope is GenericAgentTaskScopeV1.SOURCE for value in materialized.task_instances) == 18


def test_materializer_keeps_source_branches_isolated_and_fans_in() -> None:
    materialized = _materialize()
    tasks = {value.task_id: value for value in materialized.authority.graph.tasks}
    by_scope = {}
    for instance in materialized.task_instances:
        if instance.scope_ref is not None:
            by_scope.setdefault(instance.scope_ref, {})[
                instance.capability_id,
                instance.plan_kind,
            ] = tasks[instance.task_id]

    for scoped in by_scope.values():
        trace = scoped["capability.trace-ingestion", None]
        author = scoped["capability.task-authoring", None]
        attachment_review = scoped[
            "capability.plan-review",
            PlanKindV2.ATTACHMENT_GENERATION,
        ]
        attachment = scoped["capability.attachment-reconstruction", None]
        quality = scoped["capability.quality-review", None]
        criteria_review = scoped[
            "capability.plan-review",
            PlanKindV2.CRITERIA_RUBRIC,
        ]
        criteria = scoped["capability.criteria-rubric", None]
        grading_review = scoped[
            "capability.plan-review",
            PlanKindV2.GRADING_DESIGN,
        ]
        grading = scoped["capability.grading-design", None]
        assert author.dependency_task_ids == (trace.task_id,)
        assert set(attachment.dependency_task_ids) == {
            author.task_id,
            attachment_review.task_id,
        }
        assert set(quality.dependency_task_ids) == {
            author.task_id,
            attachment.task_id,
        }
        assert criteria_review.dependency_task_ids == (quality.task_id,)
        assert set(criteria.dependency_task_ids) == {
            author.task_id,
            attachment.task_id,
            criteria_review.task_id,
        }
        assert grading_review.dependency_task_ids == (criteria.task_id,)
        assert set(grading.dependency_task_ids) == {
            criteria.task_id,
            grading_review.task_id,
        }

    batch = next(value for value in tasks.values() if value.task_kind == "batch-quality")
    delivery = next(value for value in tasks.values() if value.task_kind == "delivery")
    assert len(batch.dependency_task_ids) == 8
    assert len(batch.input_artifact_head_ids) == 8
    assert len(delivery.input_artifact_head_ids) == 17
    assert set(batch.dependency_task_ids).issubset(delivery.dependency_task_ids)


def test_materializer_preserves_taskless_coordinator_and_hides_source_ids() -> None:
    materialized = _materialize()
    grants = {value.member_id: value for value in materialized.authority.authority.grants}

    assert grants["team-task-materializer.member-coordinator"].task_ids == ()
    assert len(grants["team-task-materializer.member-control"].task_ids) == 9
    rendered_ids = "\n".join(value.task_id for value in materialized.authority.graph.tasks)
    assert "sensitive-source" not in rendered_ids
    for task in materialized.authority.graph.tasks:
        assert materialized.instance(task.task_id).task_id == task.task_id


def test_workflow_validates_materialized_task_instances(tmp_path) -> None:
    base = team_fixture()
    sources = sorted_refs(
        (
            ref("trace-source", "workflow-source-a", version="v2"),
            ref("trace-source", "workflow-source-b", version="v2"),
        )
    )
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=base.registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "workflow-materialized",
            version="v2",
        ),
        source_refs=sources,
        independent_session_refs=_sessions(),
        team_id="team-workflow-materialized",
        team_incarnation_id="team-workflow-materialized-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    authority = materialized.authority
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    requirement_definition = next(
        value
        for value in base.registration.capability_definitions
        if value.capability_id == "capability.requirement-planning"
    )
    envelope = ArtifactEnvelopeV1.create(
        artifact_id="materialized-requirement-input",
        subject_ref=authority.team.goal_ref,
        schema_ref=requirement_definition.request_schema_ref,
        content_ref=authority.team.goal_ref,
        media_type="application/json",
        modality=ArtifactModalityV1.DOCUMENT,
        domain_tags=("generic-agent-trace",),
        semantic_role="evaluation-requirement",
        purpose="evaluation-data-production",
        classification="INTERNAL",
        lineage_refs=(),
        producer_capability_ref=requirement_definition.to_ref(),
        producer_task_ref=None,
        validation_refs=(),
        revision=1,
        predecessor_envelope_ref=None,
        audit=audit(),
    )
    store.seed_artifact_head(
        team_ref=authority.team.to_ref(),
        head_id="team-workflow-materialized.head-evaluation-requirement",
        envelope=envelope,
        audit=audit(),
        idempotency_key="seed-requirement",
    )
    current = store.get_snapshot(authority.team.team_id)
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=base.registration,
        requirement_ref=current.team.goal_ref,
        team_ref=current.team.to_ref(),
        roster_ref=current.roster.to_ref(),
        task_graph_ref=current.graph.to_ref(),
        authority_ref=current.authority.to_ref(),
        audit=audit(),
    )
    registry, runtime, _providers = capability_runtime(base)
    workflow = GenericAgentFactoryWorkflow(
        blueprint=blueprint,
        store=store,
        runner=TeamCapabilityRunner(
            store=store,
            runtime=runtime,
            registry=registry,
        ),
        request_source=cast(GenericAgentCapabilityRequestSource, cast(Any, object())),
    )

    assert workflow.validate_team(authority.team.team_id) == current


@pytest.mark.asyncio
async def test_workflow_prepares_request_before_loading_and_dispatch(
    tmp_path,
) -> None:
    base = team_fixture()
    materialized = _materialize(
        (ref("trace-source", "prepare-source", version="v2"),),
    )
    authority = materialized.authority
    store = TeamStore(tmp_path / "team.sqlite3")
    store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    requirement_definition = next(
        value
        for value in base.registration.capability_definitions
        if value.capability_id == "capability.requirement-planning"
    )
    store.seed_artifact_head(
        team_ref=authority.team.to_ref(),
        head_id="team-task-materializer.head-evaluation-requirement",
        envelope=ArtifactEnvelopeV1.create(
            artifact_id="prepared-requirement-input",
            subject_ref=authority.team.goal_ref,
            schema_ref=requirement_definition.request_schema_ref,
            content_ref=authority.team.goal_ref,
            media_type="application/json",
            modality=ArtifactModalityV1.DOCUMENT,
            domain_tags=("generic-agent-trace",),
            semantic_role="evaluation-requirement",
            purpose="evaluation-data-production",
            classification="INTERNAL",
            lineage_refs=(),
            producer_capability_ref=requirement_definition.to_ref(),
            producer_task_ref=None,
            validation_refs=(),
            revision=1,
            predecessor_envelope_ref=None,
            audit=audit(),
        ),
        audit=audit(),
        idempotency_key="seed-requirement",
    )
    current = store.get_snapshot(authority.team.team_id)
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=base.registration,
        requirement_ref=current.team.goal_ref,
        team_ref=current.team.to_ref(),
        roster_ref=current.roster.to_ref(),
        task_graph_ref=current.graph.to_ref(),
        authority_ref=current.authority.to_ref(),
        audit=audit(),
    )
    calls: list[str] = []

    class _Preparation:
        def prepare(
            self,
            *,
            snapshot: TeamSnapshot,
            work: TeamTaskWork,
            spec: GenericAgentTeamTaskSpec,
            audit: object,
        ) -> None:
            del snapshot, spec, audit
            calls.append(f"prepare:{work.task.task_id}")

    class _Source:
        def build(
            self,
            *,
            snapshot: TeamSnapshot,
            work: TeamTaskWork,
            spec: GenericAgentTeamTaskSpec,
        ) -> BaseModel:
            del snapshot, spec
            calls.append(f"load:{work.task.task_id}")
            return RequirementPlanningCapabilityRequestV1.create(
                plan_ref=ref(
                    "dataset-build-plan",
                    "prepared",
                    version="v2",
                ),
                policy_ref=ref(
                    "factory-run-policy",
                    "prepared",
                    version="v2",
                ),
                audit=audit(),
            )

    registry, runtime, _providers = capability_runtime(base)
    workflow = GenericAgentFactoryWorkflow(
        blueprint=blueprint,
        store=store,
        runner=TeamCapabilityRunner(
            store=store,
            runtime=runtime,
            registry=registry,
        ),
        request_source=_Source(),
        request_preparation=cast(
            GenericAgentCapabilityRequestPreparation,
            _Preparation(),
        ),
    )

    result = await workflow.dispatch_ready(
        authority.team.team_id,
        audit=audit(),
        limit=1,
        capability_ids=frozenset(
            {"capability.requirement-planning"},
        ),
    )

    task_id = next(
        value.task_id
        for value in materialized.task_instances
        if value.capability_id == "capability.requirement-planning"
    )
    assert result.attempted_task_refs
    assert calls == [f"prepare:{task_id}", f"load:{task_id}"]


def test_materializer_rejects_noncanonical_or_legacy_inputs() -> None:
    source_a = ref("trace-source", "source-a", version="v2")
    source_b = ref("trace-source", "source-b", version="v2")
    with pytest.raises(ValueError, match="sorted and unique"):
        _materialize((source_b, source_a))
    with pytest.raises(ValueError, match="between 1 and 1000"):
        _materialize(())
    with pytest.raises(
        GenericAgentFactoryWorkflowError,
        match="exact current",
    ):
        GenericAgentTaskGraphMaterializer().materialize(
            registration=build_generic_agent_trace_pack_v1_0(audit=audit()),
            requirement_ref=ref(
                "evaluation-requirement-spec",
                "legacy",
                version="v2",
            ),
            source_refs=(source_a,),
            independent_session_refs=_sessions(),
            team_id="team-legacy-materializer",
            team_incarnation_id="team-legacy-materializer-incarnation-1",
            max_model_requests=10,
            max_model_tokens=10_000,
            max_cost_micro_usd=1_000_000,
            audit=audit(),
        )
