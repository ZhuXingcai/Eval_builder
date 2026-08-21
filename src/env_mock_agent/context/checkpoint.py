from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from env_mock_agent.config import find_project_root
from env_mock_agent.context.bootstrap import (
    PROJECT_ARCHITECTURE,
    PROJECT_GOAL,
    _run,
    collect_bootstrap_state,
    working_tree_overview,
)


def _list_recent_decisions(root: Path) -> list[str]:
    adr_dir = root / "docs/adr"
    decision_dir = root / "docs/decisions"
    adrs = sorted(adr_dir.glob("*.md"))[-6:] if adr_dir.exists() else []
    decisions = sorted(decision_dir.glob("*.md"))[-4:] if decision_dir.exists() else []
    return [str(path.relative_to(root)) for path in [*adrs, *decisions]]


def _latest_commits(root: Path) -> str:
    return _run(["git", "log", "-5", "--oneline", "--decorate"], root)


def _bead_signature(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    return sorted(
        [
            {
                "id": str(item.get("id", "")),
                "status": str(item.get("status", "")),
                "updated_at": str(item.get("updated_at", "")),
            }
            for item in value
            if isinstance(item, dict)
        ],
        key=lambda item: item["id"],
    )


def _modified_file_lines(value: str) -> list[str]:
    generated_paths = ("PROJECT_STATE.md", "handoffs/", ".beads/interactions.jsonl")
    lines: list[str] = []
    for line in value.splitlines():
        path = line[3:].split(" -> ")[-1] if len(line) > 3 else line
        if path == generated_paths[0] or path.startswith(generated_paths[1]):
            continue
        if path == generated_paths[2]:
            continue
        lines.append(line)
    return sorted(lines)


def _working_tree_signature(value: str, root: Path) -> list[str]:
    signature: list[str] = []
    for line in _modified_file_lines(value):
        relative_path = line[3:].split(" -> ")[-1]
        path = root / relative_path
        if path.is_symlink():
            digest = f"symlink:{path.readlink()}"
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_dir():
            digest = "directory"
        else:
            digest = "missing"
        signature.append(f"{line}\t{digest}")
    return signature


def _state_fingerprint(state: dict[str, Any]) -> str:
    project_root = Path(str(state["project_root"]))
    payload = {
        "branch": state["branch"],
        "head": state["head"],
        "working_tree": _working_tree_signature(str(state["working_tree"]), project_root),
        "active_beads": _bead_signature(state["active_beads"]),
        "ready_beads": _bead_signature(state["ready_beads"]),
        "blocked_beads": _bead_signature(state["blocked_beads"]),
        "closed_beads": _bead_signature(state["closed_beads"]),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_project_state(
    root: Path | None = None,
    *,
    validation_results: list[str] | None = None,
    current_failures: list[str] | None = None,
    next_command: str | None = None,
) -> dict[str, Any]:
    project_root = (root or find_project_root()).resolve()
    bootstrap = collect_bootstrap_state(project_root)
    stats_raw = _run(["bd", "stats", "--json"], project_root)

    def parse_json(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": raw}

    state = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "project_root": str(project_root),
        "project": "Environment Mock Agent",
        "goal": PROJECT_GOAL,
        "architecture": PROJECT_ARCHITECTURE,
        "milestone": "Eval Dataset Factory R0-R8",
        "branch": bootstrap["branch"],
        "head": bootstrap["head"],
        "working_tree": bootstrap["working_tree"],
        "active_beads": bootstrap["active_beads"],
        "ready_beads": bootstrap["ready_beads"],
        "blocked_beads": bootstrap["blocked_beads"],
        "closed_beads": bootstrap["closed_beads"],
        "beads_stats": parse_json(stats_raw),
        "recent_commits": _latest_commits(project_root),
        "recent_decisions": _list_recent_decisions(project_root),
        "key_files": [
            "AGENTS.md",
            ".specify/memory/constitution.md",
            str(Path(str(bootstrap["spec"])).relative_to(project_root)),
            str(Path(str(bootstrap["spec"])).with_name("plan.md").relative_to(project_root)),
            str(Path(str(bootstrap["spec"])).with_name("tasks.md").relative_to(project_root)),
            "specs/001-agent-foundation/spec.md",
            "docs/decisions/runtime-selection-after-m5.md",
        ],
        "validation_commands": bootstrap["validation_commands"],
        "validation_results": validation_results or [],
        "current_failures": current_failures or [],
        "next_action": next_command or bootstrap["next_action"],
        "do_not_repeat": [
            "Do not use runtime transcripts as project memory.",
            "Do not give downstream agents unrestricted raw trace access.",
            "Do not schedule the superseded independent-human review workflow.",
            "Do not bypass permissions in local process mode.",
            "Do not silently degrade critical attachments.",
            "Do not modify source LH fusion packages during benchmark runs.",
        ],
    }
    state["state_fingerprint"] = _state_fingerprint(state)
    return state


def render_project_state(state: dict[str, Any]) -> str:
    def json_block(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2)

    return "\n".join(
        [
            "# Project State",
            "",
            "> Generated by `envmock context checkpoint`. Do not edit by hand.",
            "",
            f"- Generated: `{state['generated_at']}`",
            f"- Goal: {state['goal']}",
            f"- Architecture: {state['architecture']}",
            f"- Milestone: {state['milestone']}",
            f"- Git: `{state['branch']}` @ `{state['head']}`",
            f"- State fingerprint: `{state['state_fingerprint']}`",
            f"- Working tree: `{working_tree_overview(str(state['working_tree']))}`",
            "",
            "## Active Work",
            "```json",
            json_block(state["active_beads"]),
            "```",
            "",
            "## Ready Work",
            "```json",
            json_block(state["ready_beads"]),
            "```",
            "",
            "## Blockers",
            "```json",
            json_block(state["blocked_beads"]),
            "```",
            "",
            "## Closed Milestones",
            "```json",
            json_block(state["closed_beads"]),
            "```",
            "",
            "## Recent Decisions",
            *([f"- `{path}`" for path in state["recent_decisions"]] or ["- None."]),
            "",
            "## Key Files",
            *[f"- `{path}`" for path in state["key_files"]],
            "",
            "## Validation Commands",
            *[f"- `{command}`" for command in state["validation_commands"]],
            "",
            "## Latest Validation",
            *([f"- {result}" for result in state["validation_results"]] or ["- Not recorded."]),
            "",
            "## Current Failures",
            *([f"- {failure}" for failure in state["current_failures"]] or ["- None."]),
            "",
            "## Next Action",
            state["next_action"],
            "",
            "## Do Not Repeat",
            *[f"- {item}" for item in state["do_not_repeat"]],
            "",
            "## Recent Commits",
            "```text",
            state["recent_commits"],
            "```",
        ]
    )


def _load_bead(root: Path, task_id: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["bd", "show", task_id, "--json"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return None
    return value[0]


def _synchronize_handoff_statuses(root: Path) -> None:
    for path in sorted((root / "handoffs").glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(data, dict):
            continue
        task_id = str(data.get("task_id", ""))
        bead = _load_bead(root, task_id) if task_id else None
        if bead is None:
            continue
        status = str(bead.get("status", ""))
        if data.get("status") == status:
            continue
        data["status"] = status
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def write_checkpoint(
    root: Path | None = None,
    task_id: str | None = None,
    *,
    validation_results: list[str] | None = None,
    current_failures: list[str] | None = None,
    next_command: str | None = None,
) -> tuple[Path, Path | None]:
    project_root = (root or find_project_root()).resolve()
    _synchronize_handoff_statuses(project_root)
    state = build_project_state(
        project_root,
        validation_results=validation_results,
        current_failures=current_failures,
        next_command=next_command,
    )
    state_path = project_root / "PROJECT_STATE.md"
    state_path.write_text(render_project_state(state) + "\n", encoding="utf-8")

    handoff_path: Path | None = None
    if task_id:
        bead = _load_bead(project_root, task_id)
        if bead is None:
            raise ValueError(f"unknown Beads task: {task_id}")
        handoff_dir = project_root / "handoffs"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = handoff_dir / f"{task_id}.yaml"
        handoff = {
            "schema_version": 2,
            "status": str(bead.get("status", "")),
            "task_id": task_id,
            "branch": state["branch"],
            "head": state["head"],
            "generated_at": state["generated_at"],
            "goal": state["goal"],
            "modified_files": _modified_file_lines(str(state["working_tree"])),
            "validations": validation_results or [],
            "current_failures": current_failures or [],
            "next_command": next_command or state["next_action"],
            "next_action": next_command or state["next_action"],
            "working_tree": working_tree_overview(str(state["working_tree"])),
            "do_not_repeat": state["do_not_repeat"],
        }
        handoff_path.write_text(
            yaml.safe_dump(handoff, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return state_path, handoff_path


def validate_handoffs(root: Path | None = None) -> list[str]:
    project_root = (root or find_project_root()).resolve()
    errors: list[str] = []
    for path in sorted((project_root / "handoffs").glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: root must be a mapping")
            continue
        task_id = str(data.get("task_id", ""))
        required = ("branch", "head", "next_action")
        if not task_id or any(not data.get(field) for field in required):
            errors.append(f"{path}: task_id, branch, head, and next_action are required")
        if int(data.get("schema_version", 1)) >= 2:
            if not data.get("next_command"):
                errors.append(f"{path}: schema v2 requires next_command")
            for field in ("modified_files", "validations", "current_failures"):
                if not isinstance(data.get(field), list):
                    errors.append(f"{path}: schema v2 field {field} must be a list")
        if task_id:
            bead = _load_bead(project_root, task_id)
            if bead is None:
                errors.append(f"{path}: unknown Beads task {task_id}")
            elif data.get("status") != bead.get("status"):
                errors.append(
                    f"{path}: handoff status {data.get('status')} does not match "
                    f"Beads status {bead.get('status')}"
                )
    return errors


def validate_project_state(root: Path | None = None) -> list[str]:
    project_root = (root or find_project_root()).resolve()
    state_path = project_root / "PROJECT_STATE.md"
    if not state_path.exists():
        return [f"{state_path}: missing; run `envmock context checkpoint`"]
    text = state_path.read_text(encoding="utf-8")
    match = re.search(r"^- State fingerprint: `([a-f0-9]{64})`$", text, flags=re.MULTILINE)
    if match is None:
        return [f"{state_path}: missing state fingerprint; regenerate the checkpoint"]
    current = build_project_state(project_root)
    if match.group(1) != current["state_fingerprint"]:
        return [f"{state_path}: stale relative to Git or Beads; run `envmock context checkpoint`"]
    return []


def validate_context(root: Path | None = None) -> list[str]:
    return [*validate_project_state(root), *validate_handoffs(root)]
