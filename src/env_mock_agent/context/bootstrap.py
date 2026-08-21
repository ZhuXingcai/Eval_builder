from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from env_mock_agent.config import find_project_root

PROJECT_GOAL = (
    "Build an evidence-grounded Eval Dataset Factory that turns traces into safe, auditable "
    "evaluation items while retaining multi-runtime attachment reconstruction and zero final-answer "
    "leakage."
)
PROJECT_ARCHITECTURE = (
    "LangGraph control plane; deterministic Trace Intelligence, Provider, Validator, policy, and "
    "release planes; stage-specialist semantic agents; Claude Agent SDK, Claude Code CLI, and Pi "
    "runtime plane; pytest, Inspect AI, and user-directed evaluation checkpoints."
)


def _run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return f"[unavailable: {' '.join(command)}: {result.stderr.strip()}]"
    return result.stdout.strip()


def _parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": raw}


def _next_action(active: Any, ready: Any) -> str:
    if isinstance(active, list) and active:
        task_id = str(active[0].get("id", ""))
        return f"Run `bd show {task_id} --json`, continue that task, then checkpoint its evidence."
    if isinstance(ready, list) and ready:
        selected = next(
            (item for item in ready if isinstance(item, dict) and item.get("issue_type") != "epic"),
            ready[0],
        )
        task_id = str(selected.get("id", ""))
        return f"Run `bd update {task_id} --claim`, then read its governing spec and ADRs."
    return (
        "No open Beads work remains. Review the latest decision document before creating the next milestone."
    )


def working_tree_overview(value: str) -> str:
    if value == "clean":
        return value
    count = len(value.splitlines())
    return f"{count} changed paths; run git status --short --untracked-files=all for details"


def _closed_bead_summaries(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return [
        {
            key: item.get(key)
            for key in ("id", "title", "status", "updated_at", "close_reason")
            if item.get(key) is not None
        }
        for item in value
        if isinstance(item, dict)
    ]


def _bead_spec(project_root: Path, beads: Any) -> Path | None:
    if not isinstance(beads, list):
        return None
    specs_root = (project_root / "specs").resolve()
    for bead in beads:
        if not isinstance(bead, dict):
            continue
        spec_id = bead.get("spec_id")
        if not isinstance(spec_id, str):
            continue
        candidate = (project_root / spec_id).resolve()
        try:
            candidate.relative_to(specs_root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _approved_specs(project_root: Path) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for path in (project_root / "specs").glob("[0-9][0-9][0-9]-*/spec.md"):
        match = re.match(r"^(\d+)-", path.parent.name)
        if match is None:
            continue
        content = path.read_text(encoding="utf-8")
        if re.search(r"^\*\*Status\*\*:\s*Approved\s*$", content, flags=re.MULTILINE):
            candidates.append((int(match.group(1)), path.resolve()))
    return [path for _, path in sorted(candidates, reverse=True)]


def _resolve_active_spec(project_root: Path, active_beads: Any) -> Path:
    from_active = _bead_spec(project_root, active_beads)
    if from_active is not None:
        return from_active
    approved = _approved_specs(project_root)
    if approved:
        return approved[0]
    legacy = (project_root / "specs/001-agent-foundation/spec.md").resolve()
    return legacy


def collect_bootstrap_state(root: Path | None = None) -> dict[str, Any]:
    project_root = (root or find_project_root()).resolve()
    git_branch = _run(["git", "branch", "--show-current"], project_root)
    git_head = _run(["git", "rev-parse", "--short", "HEAD"], project_root)
    git_status = _run(["git", "status", "--short", "--untracked-files=all"], project_root)
    active_raw = _run(["bd", "list", "--status=in_progress", "--json"], project_root)
    ready_raw = _run(["bd", "ready", "--json"], project_root)
    blocked_raw = _run(["bd", "blocked", "--json"], project_root)
    closed_raw = _run(["bd", "list", "--status=closed", "--json"], project_root)
    active = _parse_json(active_raw)
    ready = _parse_json(ready_raw)
    active_spec = _resolve_active_spec(project_root, active)

    project_state = project_root / "PROJECT_STATE.md"
    latest_state = project_state.read_text(encoding="utf-8") if project_state.exists() else ""
    return {
        "project_root": str(project_root),
        "goal": PROJECT_GOAL,
        "architecture": PROJECT_ARCHITECTURE,
        "branch": git_branch,
        "head": git_head,
        "working_tree": git_status or "clean",
        "active_beads": active,
        "ready_beads": ready,
        "blocked_beads": _parse_json(blocked_raw),
        "closed_beads": _closed_bead_summaries(_parse_json(closed_raw)),
        "next_action": _next_action(active, ready),
        "constitution": str(project_root / ".specify/memory/constitution.md"),
        "spec": str(active_spec),
        "project_state": latest_state,
        "validation_commands": [
            "uv run ruff check .",
            "uv run ruff format --check .",
            "uv run mypy src scripts",
            "uv run pytest",
            "npm run check --prefix node/pi_bridge",
            "npm run build --prefix node/pi_bridge",
            "npm test --prefix node/pi_bridge",
        ],
    }


def render_bootstrap(state: dict[str, Any]) -> str:
    active = state["active_beads"]
    ready = state["ready_beads"]
    return "\n".join(
        [
            "# Environment Mock Agent Bootstrap",
            "",
            f"- Project: `{state['project_root']}`",
            f"- Goal: {state['goal']}",
            f"- Architecture: {state['architecture']}",
            f"- Git: `{state['branch']}` @ `{state['head']}`",
            f"- Working tree: `{working_tree_overview(str(state['working_tree']))}`",
            f"- Constitution: `{state['constitution']}`",
            f"- Active spec: `{state['spec']}`",
            "",
            "## Active Beads",
            "```json",
            json.dumps(active, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Ready Beads",
            "```json",
            json.dumps(ready, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Blockers",
            "```json",
            json.dumps(state["blocked_beads"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Closed Milestones",
            "```json",
            json.dumps(state["closed_beads"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Next Action",
            state["next_action"],
            "",
            "## Validation Commands",
            *[f"- `{command}`" for command in state["validation_commands"]],
            "",
            "## Persisted Project State",
            state["project_state"] or "_No PROJECT_STATE.md has been generated yet._",
        ]
    )
