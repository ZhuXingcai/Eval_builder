from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.graph_bootstrap import (
    FactoryDatasetGraphBootstrap,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness.graph_models import (
    HarnessGraphExecutionBindingV1,
)
from eval_factory.harness.session_store import HarnessSessionStore
from eval_factory.packs import build_generic_agent_evaluation_blueprint
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflow,
    materialize_generic_agent_team,
)
from eval_factory.team import TeamCapabilityRunner, TeamStore


def _factory_inputs() -> tuple[
    FactoryRunPolicyV2,
    EvaluationRequirementSpecV2,
    FactoryDatasetRunRequestV2,
]:
    policy = FactoryRunPolicyV2.create(
        policy_id="policy.graph-bootstrap",
        allowed_task_kinds=("dataset-production",),
        max_transitions=64,
        max_plan_revisions=4,
        max_agent_attempts=3,
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.graph-bootstrap",
        run_id="factory-run.graph-bootstrap",
        source_ref=ref(
            "evaluation-requirement-source",
            "graph-bootstrap",
            version="v2",
        ),
        goals=("Build evaluation data.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=audit(),
    )
    request = FactoryDatasetRunRequestV2.create(
        dataset_run_id=requirement.run_id,
        requirement_spec_ref=requirement.to_ref(),
        manifest_ref=ref("trace-manifest", "graph-bootstrap", version="v2"),
        source_authorization_ref=ref(
            "trace-source-authorization",
            "graph-bootstrap",
            version="v2",
        ),
        factory_policy_ref=policy.to_ref(),
        pipeline_policy_refs=(ref("trace-index-policy", "graph-bootstrap", version="v2"),),
        gateway_registry_refs=(ref("agent-registry", "graph-bootstrap", version="v2"),),
        output_target_ref=ref(
            "candidate-output-target",
            "graph-bootstrap",
            version="v2",
        ),
        idempotency_key="graph-bootstrap",
        max_transitions=64,
        audit=audit(),
    )
    return policy, requirement, request


def _fixture(tmp_path: Path) -> dict[str, Any]:
    policy, requirement, request = _factory_inputs()
    factory_store = FactoryControlStore(tmp_path / "factory.sqlite3")
    run = factory_store.create_run(
        policy=policy,
        requirement=requirement,
        idempotency_key="create-run",
    )
    factory_store.commit_dataset_request(
        request,
        idempotency_key="commit-request",
    )
    dataset_runtime = FactoryDatasetRuntime(
        store=factory_store,
        planner=cast(Any, object()),
        compiler=cast(Any, object()),
        plan_reviews=cast(Any, SimpleNamespace()),
        requested_by="user://graph-bootstrap",
    )

    base = team_fixture()
    authority = materialize_generic_agent_team(
        registration=base.registration,
        requirement_ref=requirement.to_ref(),
        independent_session_refs={
            role: ref("harness-session", f"graph-bootstrap-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        team_id="team-graph-bootstrap",
        team_incarnation_id="team-graph-bootstrap-incarnation-1",
        max_model_requests=100,
        max_model_tokens=1_000_000,
        max_cost_micro_usd=100_000_000,
        audit=audit(),
    )
    team_store = TeamStore(tmp_path / "team.sqlite3")
    team_store.create_team(
        team=authority.team,
        roster=authority.roster,
        graph=authority.graph,
        authority=authority.authority,
        idempotency_key="create-team",
    )
    checkpoint = team_store.create_checkpoint(
        authority.team.team_id,
        audit=audit(),
        idempotency_key="checkpoint",
    )
    snapshot = team_store.get_snapshot(authority.team.team_id)
    blueprint = build_generic_agent_evaluation_blueprint(
        registration=base.registration,
        requirement_ref=requirement.to_ref(),
        team_ref=snapshot.team.to_ref(),
        roster_ref=snapshot.roster.to_ref(),
        task_graph_ref=snapshot.graph.to_ref(),
        authority_ref=snapshot.authority.to_ref(),
        audit=audit(),
    )
    registry, capability_runtime_value, _providers = capability_runtime(base)
    workflow = GenericAgentFactoryWorkflow(
        blueprint=blueprint,
        store=team_store,
        runner=TeamCapabilityRunner(
            store=team_store,
            runtime=capability_runtime_value,
            registry=registry,
        ),
        request_source=cast(Any, object()),
    )
    binding = HarnessGraphExecutionBindingV1.create(
        binding_id="binding-graph-bootstrap",
        session_ref=ref("harness-session", "graph-bootstrap"),
        requirement_ref=requirement.to_ref(),
        requirement_policy_ref=ref(
            "harness-requirement-policy",
            "graph-bootstrap",
        ),
        factory_run_ref=run.to_ref(),
        factory_request_ref=request.to_ref(),
        factory_policy_ref=policy.to_ref(),
        expected_factory_run_version=run.run_version,
        pack_manifest_ref=base.registration.manifest.to_ref(),
        composition_ref=base.registration.composition.to_ref(),
        blueprint_ref=blueprint.to_ref(),
        team_ref=snapshot.team.to_ref(),
        roster_ref=snapshot.roster.to_ref(),
        task_graph_ref=snapshot.graph.to_ref(),
        execution_authority_ref=snapshot.authority.to_ref(),
        team_checkpoint_ref=checkpoint.to_ref(),
        expected_team_version=snapshot.team.team_version,
        expected_task_graph_revision=snapshot.graph.revision,
        expected_authority_version=snapshot.authority.authority_version,
        blackboard_head_refs=(),
        thread_id="thread-graph-bootstrap",
        max_transitions=64,
        audit=audit(),
    )
    return {
        "dataset_runtime": dataset_runtime,
        "workflow": workflow,
        "output_reader": CandidateDatasetOutputAssembler(
            store=factory_store,
            root=tmp_path / "candidate-output",
        ),
        "session_store": HarnessSessionStore(
            tmp_path / "harness.sqlite3",
        ),
        "team_store": team_store,
        "registration": base.registration,
        "blueprint": blueprint,
        "binding": binding,
        "request": request,
        "policy": policy,
        "requirement": requirement,
        "journal_path": tmp_path / "journal.sqlite3",
        "checkpoint_path": tmp_path / "checkpoint.sqlite3",
        "audit": audit(),
    }


def test_graph_bootstrap_composes_default_runtime(tmp_path: Path) -> None:
    values = _fixture(tmp_path)

    result = FactoryDatasetGraphBootstrap.build(**values)

    assert result.binding == values["binding"]
    assert result.runtime.journal is result.journal
    assert result.runtime.checkpoint_path == values["checkpoint_path"]
    assert cast(Any, result.runtime.commands).workflow is values["workflow"]


def test_graph_bootstrap_creates_binding_from_current_owner_authority(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)

    binding = FactoryDatasetGraphBootstrap.create_binding(
        binding_id="binding-created-from-current-authority",
        thread_id="thread-created-from-current-authority",
        session_ref=values["binding"].session_ref,
        requirement_policy_ref=(values["binding"].requirement_policy_ref),
        dataset_runtime=values["dataset_runtime"],
        team_store=values["team_store"],
        registration=values["registration"],
        blueprint=values["blueprint"],
        request=values["request"],
        policy=values["policy"],
        requirement=values["requirement"],
        audit=values["audit"],
    )

    snapshot = values["team_store"].get_snapshot_at(binding.team_ref)
    assert (
        binding.factory_run_ref
        == values["dataset_runtime"].store.get_run(values["request"].dataset_run_id).to_ref()
    )
    assert binding.team_ref == snapshot.team.to_ref()
    assert binding.execution_authority_ref == snapshot.authority.to_ref()
    assert binding.blackboard_head_refs == ()


def test_graph_binding_creation_requires_checkpoint_and_exact_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    monkeypatch.setattr(
        values["team_store"],
        "current_checkpoint",
        lambda team_id: None,
    )
    with pytest.raises(ValueError, match="current Team checkpoint"):
        FactoryDatasetGraphBootstrap.create_binding(
            binding_id="binding-without-checkpoint",
            thread_id="thread-without-checkpoint",
            session_ref=values["binding"].session_ref,
            requirement_policy_ref=(values["binding"].requirement_policy_ref),
            dataset_runtime=values["dataset_runtime"],
            team_store=values["team_store"],
            registration=values["registration"],
            blueprint=values["blueprint"],
            request=values["request"],
            policy=values["policy"],
            requirement=values["requirement"],
            audit=values["audit"],
        )

    values = _fixture(tmp_path / "drift")
    changed = values["blueprint"].model_copy(
        update={
            "requirement_ref": ref(
                "evaluation-requirement-spec",
                "drifted",
                version="v2",
            ),
        },
    )
    with pytest.raises(ValueError, match="current workflow authority"):
        FactoryDatasetGraphBootstrap.create_binding(
            binding_id="binding-with-drift",
            thread_id="thread-with-drift",
            session_ref=values["binding"].session_ref,
            requirement_policy_ref=(values["binding"].requirement_policy_ref),
            dataset_runtime=values["dataset_runtime"],
            team_store=values["team_store"],
            registration=values["registration"],
            blueprint=changed,
            request=values["request"],
            policy=values["policy"],
            requirement=values["requirement"],
            audit=values["audit"],
        )


def test_graph_bootstrap_rejects_aliased_authority_paths(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path)
    values["journal_path"] = values["dataset_runtime"].store.path

    with pytest.raises(ValueError, match="must be distinct"):
        FactoryDatasetGraphBootstrap.build(**values)
