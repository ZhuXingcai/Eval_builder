from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Protocol

from env_mock_agent.schemas import (
    Criticality,
    DependencyKind,
    DependencySpec,
    EvidenceLevel,
    EvidenceRef,
    ForbiddenOutputSpec,
    ProductionSpec,
    ReconstructionStrategy,
    TaskSpec,
)


class InputAdapter(Protocol):
    def load(self, path: Path, task_id: str | None = None) -> list[TaskSpec]: ...


def enum_value[EnumT: Enum](enum_type: type[EnumT], value: object, default: EnumT) -> EnumT:
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        return default


def normalize_dependencies(raw_dependencies: object, source: str) -> list[DependencySpec]:
    if not isinstance(raw_dependencies, list):
        return []
    normalized: list[DependencySpec] = []
    for index, raw in enumerate(raw_dependencies, start=1):
        if not isinstance(raw, dict):
            continue
        evidence = str(raw.get("evidence") or "").strip()
        strategy = enum_value(
            ReconstructionStrategy,
            raw.get("mock_strategy"),
            ReconstructionStrategy.SYNTHESIZE,
        )
        normalized.append(
            DependencySpec(
                dependency_id=f"D-{index:03d}",
                path=str(raw.get("path") or f"unknown-{index}"),
                kind=enum_value(DependencyKind, raw.get("kind"), DependencyKind.UNKNOWN),
                must_exist_before_start=bool(raw.get("must_exist_before_start", True)),
                criticality=enum_value(Criticality, raw.get("importance"), Criticality.MEDIUM),
                evidence_level=EvidenceLevel.PARTIAL if evidence else EvidenceLevel.CLUE_ONLY,
                evidence_refs=(
                    [EvidenceRef(source=source, excerpt=evidence)]
                    if evidence
                    else [EvidenceRef(source=source)]
                ),
                expected_content=raw.get("expected_content") or "",
                reconstruction_strategy=strategy,
                asset_type=str(raw.get("asset_type") or "file_unspecified"),
                risk_notes=str(raw.get("risk_notes") or ""),
            )
        )
    return normalized


def parse_production(raw_list: object, raw_types: object = None) -> list[ProductionSpec]:
    if isinstance(raw_list, list):
        return [
            ProductionSpec(
                name=str(item.get("name") or item.get("path") or f"output-{index}")
                if isinstance(item, dict)
                else str(item),
                asset_type=str(item.get("asset_type") or "file_unspecified")
                if isinstance(item, dict)
                else "file_unspecified",
                path=str(item.get("path")) if isinstance(item, dict) and item.get("path") else None,
                description=str(item.get("description") or "") if isinstance(item, dict) else "",
            )
            for index, item in enumerate(raw_list, start=1)
        ]

    types = [str(item) for item in raw_types] if isinstance(raw_types, list) else []
    text = str(raw_list or "").strip()
    if not text:
        return []
    lines = [re.sub(r"^\s*\d+[.、]\s*", "", line).strip() for line in text.splitlines() if line.strip()]
    production: list[ProductionSpec] = []
    for index, line in enumerate(lines, start=1):
        path_match = re.search(r"`([^`]+\.[A-Za-z0-9]+)`", line)
        path = path_match.group(1) if path_match else None
        suffix = Path(path).suffix.lstrip(".").lower() if path else ""
        production.append(
            ProductionSpec(
                name=path or line[:80],
                asset_type=suffix or (types[index - 1] if index <= len(types) else "file_unspecified"),
                path=path,
                description=line,
            )
        )
    return production


def forbidden_from_production(production: list[ProductionSpec]) -> list[ForbiddenOutputSpec]:
    return [
        ForbiddenOutputSpec(
            forbidden_id=f"F-{index:03d}",
            description=f"Requested final output must not be pre-placed: {item.name}",
            path_patterns=[item.path] if item.path else [],
            semantic_patterns=[item.name, item.description],
            reason="Attachment packages contain task inputs, not completed deliverables.",
        )
        for index, item in enumerate(production, start=1)
    ]
