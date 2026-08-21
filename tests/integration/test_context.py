from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from env_mock_agent.context.bootstrap import collect_bootstrap_state, render_bootstrap
from env_mock_agent.context.checkpoint import (
    validate_context,
    validate_handoffs,
    validate_project_state,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HAS_LOCAL_CONTINUITY_STATE = (
    (REPOSITORY_ROOT / ".beads").is_dir()
    and (REPOSITORY_ROOT / "PROJECT_STATE.md").is_file()
)


@pytest.mark.skipif(
    not HAS_LOCAL_CONTINUITY_STATE,
    reason="local Beads and PROJECT_STATE continuity data is not published",
)
def test_fresh_session_bootstrap_contains_authoritative_context() -> None:
    root = REPOSITORY_ROOT
    state = collect_bootstrap_state(root)
    rendered = render_bootstrap(state)
    assert "Environment Mock Agent Bootstrap" in rendered
    active_beads = state["active_beads"]
    ready_beads = state["ready_beads"]
    closed_beads = state["closed_beads"]
    assert isinstance(active_beads, list)
    assert isinstance(ready_beads, list)
    assert isinstance(closed_beads, list)
    assert any(item["id"] == "env_mock_agent-3nj.6" for item in closed_beads)
    visible_work = [*active_beads, *ready_beads, *closed_beads]
    assert visible_work
    assert visible_work[0]["id"] in rendered
    assert "LangGraph control plane" in rendered
    assert ".specify/memory/constitution.md" in rendered
    assert state["spec"].endswith("specs/003-evaluation-agent-harness/spec.md")
    assert state["spec"] in rendered
    assert "uv run pytest" in rendered
    assert "docs/decisions/runtime-selection-after-m5.md" in state["project_state"]


@pytest.mark.skipif(
    not HAS_LOCAL_CONTINUITY_STATE,
    reason="local Beads and PROJECT_STATE continuity data is not published",
)
def test_current_handoff_records_are_valid() -> None:
    root = REPOSITORY_ROOT
    assert validate_context(root) == []


def test_handoff_validator_allows_multiple_active_beads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from env_mock_agent.context import checkpoint

    handoffs = tmp_path / "handoffs"
    handoffs.mkdir()
    for task_id in ("task-a", "task-b"):
        (handoffs / f"{task_id}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "status": "in_progress",
                    "task_id": task_id,
                    "branch": "main",
                    "head": "abc123",
                    "next_action": f"Continue {task_id}.",
                    "next_command": f"run-{task_id}",
                    "modified_files": [],
                    "validations": [],
                    "current_failures": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        checkpoint,
        "_load_bead",
        lambda root, task_id: {
            "id": task_id,
            "status": "in_progress",
        },
    )

    assert validate_handoffs(tmp_path) == []


@pytest.mark.skipif(
    not HAS_LOCAL_CONTINUITY_STATE,
    reason="local Beads and PROJECT_STATE continuity data is not published",
)
def test_fresh_process_bootstrap_recovers_without_agent_session_context() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = collect_bootstrap_state(root)
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("CLAUDE_", "PI_", "TRAE_"))
    }
    process = subprocess.run(
        [sys.executable, "-m", "env_mock_agent.cli", "context", "bootstrap", "--json"],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    state = json.loads(process.stdout)
    assert state["goal"].startswith("Build an evidence-grounded")
    assert "LangGraph control plane" in state["architecture"]
    assert state["active_beads"] == expected["active_beads"]
    assert state["ready_beads"] == expected["ready_beads"]
    assert state["blocked_beads"] == expected["blocked_beads"]
    assert any(item["id"] == "env_mock_agent-3nj.6" for item in state["closed_beads"])
    assert state["project_state"] == expected["project_state"]
    assert state["spec"] == expected["spec"]
    assert state["spec"].endswith("specs/003-evaluation-agent-harness/spec.md")
    assert state["project_state"].startswith("# Project State")
    assert "docs/decisions/runtime-selection-after-m5.md" in state["project_state"]
    if state["active_beads"]:
        assert state["active_beads"][0]["id"] in state["next_action"]
        assert "continue that task" in state["next_action"]
    elif state["ready_beads"]:
        selected = next(
            (item for item in state["ready_beads"] if item.get("issue_type") != "epic"),
            state["ready_beads"][0],
        )
        assert selected["id"] in state["next_action"]
        assert "bd update" in state["next_action"]
    else:
        assert state["next_action"].startswith("No open Beads work remains")
    assert "npm run build --prefix node/pi_bridge" in state["validation_commands"]


@pytest.mark.skipif(
    not HAS_LOCAL_CONTINUITY_STATE,
    reason="local Beads and PROJECT_STATE continuity data is not published",
)
def test_current_handoff_uses_rich_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    data = yaml.safe_load((root / "handoffs/env_mock_agent-3nj.yaml").read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert isinstance(data["modified_files"], list)
    assert data["validations"]
    assert data["current_failures"]
    assert data["next_command"]


def test_project_state_validator_detects_stale_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from env_mock_agent.context import checkpoint

    (tmp_path / "PROJECT_STATE.md").write_text(
        "- State fingerprint: `" + ("0" * 64) + "`\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint,
        "build_project_state",
        lambda root: {"state_fingerprint": "1" * 64},
    )
    assert "stale relative to Git or Beads" in validate_project_state(tmp_path)[0]


def test_state_fingerprint_detects_untracked_content_change(tmp_path: Path) -> None:
    from env_mock_agent.context import checkpoint

    source = tmp_path / "source.txt"
    source.write_text("first\n", encoding="utf-8")
    state = {
        "project_root": str(tmp_path),
        "branch": "main",
        "head": "abc123",
        "working_tree": "?? source.txt",
        "active_beads": [],
        "ready_beads": [],
        "blocked_beads": [],
        "closed_beads": [],
    }
    before = checkpoint._state_fingerprint(state)
    source.write_text("second\n", encoding="utf-8")

    assert checkpoint._state_fingerprint(state) != before
