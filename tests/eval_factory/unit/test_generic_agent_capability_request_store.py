from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    RequirementPlanningCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentTeamTaskSpec,
)
from eval_factory.packs.generic_agent_trace.request_preparation import (
    GenericAgentTaskRequestPreparer,
)
from eval_factory.packs.generic_agent_trace.request_store import (
    GenericAgentCapabilityRequestConflictError,
    GenericAgentCapabilityRequestIntegrityError,
    GenericAgentCapabilityRequestNotFoundError,
    GenericAgentCapabilityRequestStore,
    StoredGenericAgentCapabilityRequestSource,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterializer,
)
from eval_factory.team import TeamStore


def _fixture(tmp_path: Path):
    base = team_fixture()
    materialized = GenericAgentTaskGraphMaterializer().materialize(
        registration=base.registration,
        requirement_ref=ref(
            "evaluation-requirement-spec",
            "request-store",
            version="v2",
        ),
        source_refs=(ref("trace-source", "request-store", version="v2"),),
        independent_session_refs={
            role: ref("harness-session", f"request-store-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-request-store",
        team_incarnation_id="team-request-store-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    authority = materialized.authority
    team_store = TeamStore(tmp_path / "team.sqlite3")
    team_store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    task = next(value for value in authority.graph.tasks if value.task_kind == "requirement-planning")
    definition = next(
        value
        for value in base.registration.capability_definitions
        if value.capability_id == "capability.requirement-planning"
    )
    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    request_store = GenericAgentCapabilityRequestStore(
        tmp_path / "requests.sqlite3",
        private_store=private_store,
    )
    return (
        base,
        materialized,
        team_store,
        task,
        definition,
        request_store,
    )


def _request(suffix: str) -> RequirementPlanningCapabilityRequestV1:
    return RequirementPlanningCapabilityRequestV1.create(
        plan_ref=ref("dataset-build-plan", suffix, version="v2"),
        policy_ref=ref("factory-run-policy", "request-store", version="v2"),
        audit=audit(),
    )


def test_request_store_persists_replays_and_rebuilds_successors(
    tmp_path: Path,
) -> None:
    base, _materialized, team_store, task, definition, store = _fixture(
        tmp_path,
    )
    sources = sorted_refs(
        (
            ref("factory-run", "request-store", version="v2"),
            ref("dataset-build-plan", "v1", version="v2"),
        )
    )
    first = store.commit(
        task_ref=task.to_ref(),
        capability_definition_ref=definition.to_ref(),
        request=_request("v1"),
        source_authority_refs=sources,
        audit=audit(),
        idempotency_key="request-v1",
    )
    replay = store.commit(
        task_ref=task.to_ref(),
        capability_definition_ref=definition.to_ref(),
        request=_request("v1"),
        source_authority_refs=sources,
        audit=audit(),
        idempotency_key="request-v1",
    )
    same_authority = store.commit(
        task_ref=task.to_ref(),
        capability_definition_ref=definition.to_ref(),
        request=_request("v1"),
        source_authority_refs=sources,
        audit=audit(),
        idempotency_key="request-v1-alias",
    )
    successor_sources = sorted_refs(
        (
            ref("factory-run", "request-store", version="v2"),
            ref("dataset-build-plan", "v2", version="v2"),
        )
    )
    successor = store.commit(
        task_ref=task.to_ref(),
        capability_definition_ref=definition.to_ref(),
        request=_request("v2"),
        source_authority_refs=successor_sources,
        audit=audit(),
        idempotency_key="request-v2",
    )

    assert replay == same_authority == first
    assert successor.binding_version == 2
    assert successor.predecessor_binding_ref == first.to_ref()
    assert store.get(task.to_ref()) == successor
    with pytest.raises(
        GenericAgentCapabilityRequestIntegrityError,
        match="stale",
    ):
        store.load_request(first, RequirementPlanningCapabilityRequestV1)

    registry, _runtime, _providers = capability_runtime(base)
    source = StoredGenericAgentCapabilityRequestSource(
        store=GenericAgentCapabilityRequestStore(
            store.path,
            private_store=store.private_store,
        ),
        registry=registry,
    )
    work = team_store.get_task_work(authority_id := "team-request-store", task.task_id)
    built = source.build(
        snapshot=team_store.get_snapshot(authority_id),
        work=work,
        spec=GenericAgentTeamTaskSpec(
            capability_id="capability.requirement-planning",
            dependency_capability_ids=(),
            input_schema_refs=(definition.request_schema_ref,),
            output_schema_refs=(definition.result_schema_ref,),
        ),
    )
    assert built == _request("v2")

    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM capability_request_heads")
        connection.commit()
    with pytest.raises(GenericAgentCapabilityRequestNotFoundError):
        store.get(task.to_ref())
    assert store.rebuild_current_heads() == (task.object_id,)
    assert store.get(task.to_ref()) == successor


def test_request_store_rejects_changed_idempotency_command(
    tmp_path: Path,
) -> None:
    _base, _materialized, _team_store, task, definition, store = _fixture(
        tmp_path,
    )
    store.commit(
        task_ref=task.to_ref(),
        capability_definition_ref=definition.to_ref(),
        request=_request("v1"),
        source_authority_refs=(),
        audit=audit(),
        idempotency_key="same-key",
    )

    with pytest.raises(
        GenericAgentCapabilityRequestConflictError,
        match="another command",
    ):
        store.commit(
            task_ref=task.to_ref(),
            capability_definition_ref=definition.to_ref(),
            request=_request("v2"),
            source_authority_refs=(),
            audit=audit(),
            idempotency_key="same-key",
        )


@pytest.mark.parametrize(
    "fault_point",
    (
        "after_binding",
        "after_head",
        "after_outbox",
        "after_idempotency",
    ),
)
def test_request_store_faults_roll_back_sqlite_authority(
    tmp_path: Path,
    fault_point: str,
) -> None:
    base, _materialized, _team_store, task, definition, _store = _fixture(
        tmp_path / "base",
    )
    pending = {fault_point}

    def inject(point: str) -> None:
        if point in pending:
            pending.remove(point)
            raise RuntimeError(f"injected request fault: {point}")

    private_store = FactoryPrivateObjectStore(tmp_path / "private")
    store = GenericAgentCapabilityRequestStore(
        tmp_path / "requests.sqlite3",
        private_store=private_store,
        fault_injector=inject,
    )
    with pytest.raises(RuntimeError, match=fault_point):
        store.commit(
            task_ref=task.to_ref(),
            capability_definition_ref=definition.to_ref(),
            request=_request("fault"),
            source_authority_refs=(),
            audit=audit(),
            idempotency_key=f"fault-{fault_point}",
        )
    assert not pending
    with pytest.raises(GenericAgentCapabilityRequestNotFoundError):
        store.get(task.to_ref())
    with sqlite3.connect(store.path) as connection:
        for table in (
            "capability_request_bindings",
            "capability_request_heads",
            "capability_request_outbox",
            "capability_request_idempotency",
        ):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table}",
            ).fetchone() == (0,)
    del base


def test_task_request_preparer_validates_provider_model(
    tmp_path: Path,
) -> None:
    base, materialized, _team_store, task, _definition, store = _fixture(
        tmp_path,
    )
    registry, _runtime, _providers = capability_runtime(base)
    preparer = GenericAgentTaskRequestPreparer(
        materialization=materialized,
        registry=registry,
        store=store,
    )

    binding = preparer.commit(
        task_id=task.task_id,
        request=_request("prepared"),
        source_authority_refs=(),
        audit=audit(),
        idempotency_key="prepared-request",
    )

    assert binding.task_ref == task.to_ref()
    with pytest.raises(
        Exception,
        match="request model differs",
    ):
        review_task = next(
            value for value in materialized.authority.graph.tasks if value.task_kind == "plan-review-global"
        )
        preparer.commit(
            task_id=review_task.task_id,
            request=_request("wrong-model"),
            source_authority_refs=(),
            audit=audit(),
            idempotency_key="wrong-model",
        )
