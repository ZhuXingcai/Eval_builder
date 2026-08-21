from __future__ import annotations

import re
from pathlib import Path

BLOCKED_COMMAND_PATTERNS = (
    r"(^|\s)sudo(\s|$)",
    r"rm\s+-[^\n]*r[^\n]*\s+/(?:\s|$)",
    r"git\s+push(?:\s|$)",
    r"curl\b[^\n]*\|\s*(?:sh|bash)\b",
    r"wget\b[^\n]*\|\s*(?:sh|bash)\b",
)

PATH_TOOL_KEYS = {
    "Read": ("file_path", "path"),
    "Write": ("file_path", "path"),
    "Edit": ("file_path", "path"),
    "Glob": ("path",),
    "Grep": ("path",),
}


def is_path_inside(root: Path, candidate: str | Path) -> bool:
    resolved_root = root.expanduser().resolve()
    value = Path(candidate)
    resolved = (resolved_root / value).resolve() if not value.is_absolute() else value.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return False
    return True


def validate_tool_input(workspace: Path, tool_name: str, tool_input: dict[str, object]) -> str | None:
    if tool_name in PATH_TOOL_KEYS:
        for key in PATH_TOOL_KEYS[tool_name]:
            raw = tool_input.get(key)
            if isinstance(raw, str) and not is_path_inside(workspace, raw):
                return f"{tool_name} path escapes workspace: {raw}"
    if tool_name == "Bash":
        command = str(tool_input.get("command") or "")
        for pattern in BLOCKED_COMMAND_PATTERNS:
            if re.search(pattern, command):
                return f"Bash command blocked by local-process policy: {pattern}"
        absolute_paths = re.findall(
            r"(?<![\w.-])/(?:Users|home|etc|var|tmp|opt|private)/[^\s;&|<>`$()]+",
            command,
        )
        for candidate in absolute_paths:
            if not is_path_inside(workspace, candidate):
                return f"Bash command references path outside workspace: {candidate}"
        redirections = re.findall(r"(?:^|[\s\d])(?:>>?|<)\s*([^\s;&|]+)", command)
        for candidate in redirections:
            if candidate.startswith("/") and not is_path_inside(workspace, candidate):
                return f"Bash redirection escapes workspace: {candidate}"
    return None
