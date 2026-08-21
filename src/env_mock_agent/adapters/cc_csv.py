from __future__ import annotations

import csv
import json
from pathlib import Path

from env_mock_agent.adapters.base import forbidden_from_production, normalize_dependencies, parse_production
from env_mock_agent.schemas import InputAdapterType, RubricSource, TaskSpec


def _parse_json(value: str | None, default: object) -> object:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


class CcCsvAdapter:
    def load(self, path: Path, task_id: str | None = None) -> list[TaskSpec]:
        resolved = path.expanduser().resolve()
        tasks: list[TaskSpec] = []
        with resolved.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=1):
                candidate_id = str(row.get("instance_id") or row.get("task_id") or row_index)
                if task_id and candidate_id != task_id:
                    continue
                dependency_output = _parse_json(row.get("dependency_output"), {})
                dependency_data = dependency_output if isinstance(dependency_output, dict) else {}
                raw_dependencies = _parse_json(row.get("dependencies"), None)
                if not isinstance(raw_dependencies, list):
                    raw_dependencies = dependency_data.get("dependencies", [])
                raw_types = _parse_json(
                    row.get("production_type"), dependency_data.get("production_type", [])
                )
                raw_production = row.get("production_list") or dependency_data.get("production_list", "")
                production = parse_production(raw_production, raw_types)
                rubrics = _parse_json(row.get("rubrics"), row.get("rubrics") or "")
                rubric_count = len(rubrics) if isinstance(rubrics, list) else 0
                tasks.append(
                    TaskSpec(
                        task_id=candidate_id,
                        request_id=str(row.get("sid") or row.get("request_id") or candidate_id),
                        task_name=str(
                            row.get("task_name")
                            or dependency_data.get("task_short_name")
                            or f"task_{candidate_id}"
                        ),
                        prompt=str(row.get("prompt") or ""),
                        category=str(row.get("category") or "") or None,
                        business=str(row.get("business") or "") or None,
                        source_adapter=InputAdapterType.CC,
                        source_path=str(resolved),
                        dependencies=normalize_dependencies(raw_dependencies, f"{resolved}:row:{row_index}"),
                        production=production,
                        forbidden_outputs=forbidden_from_production(production),
                        rubrics=RubricSource(raw=rubrics, count=rubric_count),
                        profile="cc",
                        metadata={
                            "row_index": row_index,
                            "env_dependency": row.get("env_dependency"),
                            "dependency_output": dependency_data,
                            "source_row": row,
                        },
                    )
                )
        return tasks
