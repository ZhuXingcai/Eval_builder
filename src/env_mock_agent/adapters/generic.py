from __future__ import annotations

import json
from pathlib import Path

import yaml

from env_mock_agent.adapters.base import forbidden_from_production, normalize_dependencies, parse_production
from env_mock_agent.schemas import InputAdapterType, RubricSource, TaskSpec


class GenericAdapter:
    def load(self, path: Path, task_id: str | None = None) -> list[TaskSpec]:
        resolved = path.expanduser().resolve()
        text = resolved.read_text(encoding="utf-8")
        data = json.loads(text) if resolved.suffix.lower() == ".json" else yaml.safe_load(text)
        items = data if isinstance(data, list) else [data]
        tasks: list[TaskSpec] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"generic task #{index} must be a mapping")
            candidate_id = str(item.get("task_id") or item.get("instance_id") or index)
            if task_id and candidate_id != task_id:
                continue
            production = parse_production(
                item.get("production") or item.get("production_list"),
                item.get("production_type"),
            )
            rubrics_raw = item.get("rubrics")
            rubrics_count = len(rubrics_raw) if isinstance(rubrics_raw, list) else 0
            tasks.append(
                TaskSpec(
                    task_id=candidate_id,
                    request_id=str(item.get("request_id") or item.get("sid") or "") or None,
                    task_name=str(item.get("task_name") or f"task_{candidate_id}"),
                    prompt=str(item.get("prompt") or ""),
                    category=str(item.get("category") or "") or None,
                    business=str(item.get("business") or "") or None,
                    source_adapter=InputAdapterType.GENERIC,
                    source_path=str(resolved),
                    trace_sources=[str(value) for value in item.get("trace_sources", [])],
                    original_attachments=[str(value) for value in item.get("original_attachments", [])],
                    dependencies=normalize_dependencies(item.get("dependencies"), str(resolved)),
                    production=production,
                    forbidden_outputs=forbidden_from_production(production),
                    rubrics=RubricSource(raw=rubrics_raw, count=rubrics_count) if rubrics_raw else None,
                    profile=str(item.get("profile") or "generic"),
                    metadata={"source": item},
                )
            )
        return tasks
