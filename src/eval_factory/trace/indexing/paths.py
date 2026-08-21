from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from eval_factory.contracts.core import RelativePath

_RELATIVE_PATH_ADAPTER: TypeAdapter[str] = TypeAdapter(RelativePath)
_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")


@dataclass(frozen=True)
class ProjectedPath:
    raw: str
    value: str


def project_logical_path(value: object) -> ProjectedPath | None:
    if not isinstance(value, str):
        return None
    raw = value
    candidate = value.strip().replace("\\", "/")
    if not candidate or _has_control(candidate) or "://" in candidate:
        return None
    if candidate.startswith("~/"):
        candidate = f"home/{candidate[2:]}"
    else:
        drive = _WINDOWS_DRIVE.match(candidate)
        if drive:
            drive_name, tail = drive.groups()
            candidate = f"{drive_name}/{tail}" if tail else drive_name
        elif candidate.startswith("/"):
            candidate = candidate.lstrip("/")
    if not candidate:
        return None
    parts = [part for part in candidate.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        return None
    projected = "/".join(parts)
    try:
        validated = _RELATIVE_PATH_ADAPTER.validate_python(projected, strict=True)
    except ValidationError:
        return None
    return ProjectedPath(raw=raw, value=validated)


def extract_raw_path(arguments: object) -> object | None:
    if not isinstance(arguments, dict):
        return None
    for key in ("file_path", "path", "pattern"):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return None


def _has_control(value: str) -> bool:
    return any(ord(char) < 0x20 for char in value)
