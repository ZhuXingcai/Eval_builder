from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import (
    audit,
    capability_runtime,
    ref,
    team_fixture,
)

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
)
from eval_factory.harness import HarnessCapabilityRuntime
from eval_factory.packs import build_generic_agent_trace_execution_pack
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)
from eval_factory.packs.generic_agent_trace.trace_graph_refinement import (
    GenericAgentTraceGraphRefinement,
)
from eval_factory.packs.generic_agent_trace.workflow_bootstrap import (
    GenericAgentWorkflowBootstrap,
)
from eval_factory.team import TeamStore


def _inputs(tmp_path: Path) -> dict[str, object]:
    base = team_fixture()
    _registry, runtime, _providers = capability_runtime(base)
    requirement = EvaluationRequirementSpecV2.create(
        requirement_spec_id="requirement.workflow-bootstrap",
        run_id="factory-run.workflow-bootstrap",
        source_ref=ref(
            "evaluation-requirement-source",
            "workflow-bootstrap",
            version="v2",
        ),
        goals=("Build governed evaluation data.",),
        constraints=(),
        assumptions=(),
        open_questions=(),
        requirement_version=1,
        audit=audit(),
    )
    return {
        "registration": base.registration,
        "capability_runtime": runtime,
        "materials": CompositeCapabilityMaterialResolver(),
        "private_store": FactoryPrivateObjectStore(tmp_path / "private"),
        "request_store_path": tmp_path / "requests.sqlite3",
        "team_store": TeamStore(tmp_path / "team.sqlite3"),
        "requirement": requirement,
        "source_refs": (
            ref("trace-source", "source-a", version="v2"),
            ref("trace-source", "source-b", version="v2"),
        ),
        "independent_session_refs": {
            role: ref("harness-session", f"workflow-bootstrap-{role}")
            for role in (
                "control",
                "coordinator",
                "quality",
                "requirement",
                "task",
                "trace",
            )
        },
        "team_id": "team-workflow-bootstrap",
        "team_incarnation_id": ("team-workflow-bootstrap-incarnation-1"),
        "max_model_requests": 100,
        "max_model_tokens": 1_000_000,
        "max_cost_micro_usd": 100_000_000,
        "audit": audit(),
    }


def test_workflow_bootstrap_materializes_current_multi_source_team(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)

    first = GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]
    replay = GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]

    assert len(first.materialization.task_instances) == 23
    assert first.snapshot == replay.snapshot
    assert first.checkpoint == replay.checkpoint
    assert first.blueprint == replay.blueprint
    assert (
        first.workflow.validate_team(
            first.snapshot.team.team_id,
        ).team
        == first.snapshot.team
    )
    team_store = cast(TeamStore, values["team_store"])
    envelopes = team_store.current_artifact_envelopes(
        first.snapshot.team.team_id,
    )
    assert tuple(value.semantic_role for value in envelopes) == ("evaluation-requirement",)
    assert first.request_store.path != team_store.path


def test_workflow_bootstrap_rejects_store_and_resolver_drift(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    values["request_store_path"] = cast(
        TeamStore,
        values["team_store"],
    ).path
    with pytest.raises(ValueError, match="must be distinct"):
        GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]

    values = _inputs(tmp_path / "resolver")
    runtime = cast(
        HarnessCapabilityRuntime,
        values["capability_runtime"],
    )
    provider = cast(Any, runtime.registry.providers[0])
    provider.materials = object()
    with pytest.raises(ValueError, match="another material resolver"):
        GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]


def test_execution_workflow_bootstrap_requires_and_installs_refinement(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    registration = build_generic_agent_trace_execution_pack(audit=audit())
    values["registration"] = registration
    values["capability_runtime"] = cast(
        Any,
        type(
            "_Runtime",
            (),
            {
                "registry": type(
                    "_Registry",
                    (),
                    {
                        "registration": registration,
                        "providers": (),
                    },
                )(),
            },
        )(),
    )

    with pytest.raises(
        ValueError,
        match="requires Trace candidate graph refinement",
    ):
        GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]

    candidate_store = cast(Any, object())
    values["trace_candidate_store"] = candidate_store
    result = GenericAgentWorkflowBootstrap.build(**values)  # type: ignore[arg-type]

    refinement = result.workflow.graph_refinement
    assert isinstance(refinement, GenericAgentTraceGraphRefinement)
    assert refinement.candidate_store is candidate_store
