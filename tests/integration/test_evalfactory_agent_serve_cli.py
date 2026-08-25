from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.eval_factory.integration.test_harness_agent_loop import (
    _ref,
)
from tests.eval_factory.unit.test_harness_requirement_fixture import (
    fixture_requirement_config,
)
from tests.integration.test_evalfactory_agent_run_cli import (
    _attachment_config,
    _config,
    _core_config,
    _fixture,
    _policy,
    _r4_config,
    _release_config,
    _request,
    _specialist_config,
)
from typer.testing import CliRunner

from eval_factory.cli import _agent_serve_application, app
from eval_factory.console_api.fixture_host import (
    AgentShellFixtureHostConfigV1,
)
from eval_factory.packs.generic_agent_trace.shell_composition import (
    GenericAgentShellCompositionConfigV1,
)


def _host_config() -> AgentShellFixtureHostConfigV1:
    specialist = _specialist_config()
    policy = _policy(
        include_attachment=True,
        include_specialists=True,
    )
    request = _request(
        include_attachment=True,
        include_specialists=True,
    )
    return AgentShellFixtureHostConfigV1(
        requirement_agent=fixture_requirement_config(),
        dataset_runtime=_config(
            include_core=True,
            include_r4=True,
            include_attachment=True,
            specialist_config=specialist,
        ),
        planning_fixture=_fixture(),
        core_runtime=_core_config(),
        r4_runtime=_r4_config(),
        attachment_runtime=_attachment_config(),
        specialist_runtime=specialist,
        release_runtime=_release_config(
            approval_policy=specialist.job_store.approval_policy,
        ),
        shell_composition=GenericAgentShellCompositionConfigV1(
            allowed_task_kinds=policy.allowed_task_kinds,
            pipeline_policy_refs=request.pipeline_policy_refs,
            gateway_registry_refs=tuple(
                sorted(
                    set(request.gateway_registry_refs),
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.object_version,
                        value.object_sha256,
                    ),
                )
            ),
            output_target_ref=request.output_target_ref,
            max_transitions=policy.max_transitions,
            max_plan_revisions=policy.max_plan_revisions,
            max_agent_attempts=policy.max_agent_attempts,
        ),
        trace_schema_ref=_ref(
            "json-schema",
            "raw-traj-v1",
            version="v1",
        ),
    )


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "factory_store": tmp_path / "factory.sqlite3",
        "private_store": tmp_path / "private",
        "gateway_store": tmp_path / "gateway.sqlite3",
        "harness_store": tmp_path / "harness.sqlite3",
        "source_store": tmp_path / "source",
        "team_store": tmp_path / "team.sqlite3",
        "capability_request_store": (tmp_path / "capability-requests.sqlite3"),
        "graph_journal": tmp_path / "graph-journal.sqlite3",
        "graph_checkpoint": tmp_path / "graph-checkpoint.sqlite3",
        "core_workspace": tmp_path / "core-workspace",
        "attachment_workspace": tmp_path / "attachment-workspace",
        "job_store": tmp_path / "job-store.sqlite3",
        "specialist_workspace": tmp_path / "specialist-workspace",
        "candidate_output": tmp_path / "candidate-output",
    }


def test_agent_serve_builds_complete_fixture_host(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "shell-config.json"
    config_path.write_text(
        _host_config().model_dump_json(indent=2),
        encoding="utf-8",
    )
    application = _agent_serve_application(
        **_paths(tmp_path),
        registry=None,
        shell_config=config_path,
        user="user://agent-shell",
        plan_review_only=False,
    )
    client = TestClient(application)

    assert application.version == "1.3.0"
    created = client.post(
        "/api/harness/sessions",
        headers={
            "X-Eval-Factory-Principal": "user://agent-shell",
        },
        json={
            "session_id": "session-serve-cli",
            "incarnation_id": "session-serve-cli-incarnation",
            "idempotency_key": "create-session-serve-cli",
        },
    )
    assert created.status_code == 200
    assert created.json()["execution_phase"] == "WAITING_REQUIREMENT"
    paths = {route.path for route in application.routes}
    assert "/api/harness/sessions/{session_id}/team" in paths
    assert "/api/plan-reviews/contract" in paths


def test_agent_serve_requires_explicit_mode_and_all_authorities(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="every explicit authority"):
        _agent_serve_application(
            factory_store=tmp_path / "factory.sqlite3",
            registry=None,
            shell_config=None,
            private_store=None,
            gateway_store=None,
            harness_store=None,
            source_store=None,
            team_store=None,
            capability_request_store=None,
            graph_journal=None,
            graph_checkpoint=None,
            core_workspace=None,
            attachment_workspace=None,
            job_store=None,
            specialist_workspace=None,
            candidate_output=None,
            user=None,
            plan_review_only=False,
        )

    result = CliRunner().invoke(
        app,
        (
            "agent",
            "serve",
            "--factory-store",
            str(tmp_path / "factory.sqlite3"),
        ),
    )
    assert result.exit_code == 2
    error = json.loads(result.stderr)
    assert error == {
        "schema_version": "eval-factory/agent-shell-cli-error/v1",
        "error_code": "INVALID_INPUT",
        "message": "input does not satisfy the required contract",
    }


def test_plan_review_only_mode_is_explicit(
    tmp_path: Path,
) -> None:
    registry = _config().model_dump(
        mode="python",
        include={"capabilities", "definitions"},
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            registry,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    application = _agent_serve_application(
        factory_store=tmp_path / "factory.sqlite3",
        registry=registry_path,
        shell_config=None,
        private_store=None,
        gateway_store=None,
        harness_store=None,
        source_store=None,
        team_store=None,
        capability_request_store=None,
        graph_journal=None,
        graph_checkpoint=None,
        core_workspace=None,
        attachment_workspace=None,
        job_store=None,
        specialist_workspace=None,
        candidate_output=None,
        user=None,
        plan_review_only=True,
    )

    assert application.version == "1.0.0"
    assert (
        TestClient(application)
        .get(
            "/api/plan-reviews/contract",
        )
        .status_code
        == 200
    )
