from __future__ import annotations

import json
from pathlib import Path

import yaml

from env_mock_agent.adapters.base import forbidden_from_production, normalize_dependencies, parse_production
from env_mock_agent.schemas import InputAdapterType, RubricSource, TaskSpec


class LhAdapter:
    def load(self, path: Path, task_id: str | None = None) -> list[TaskSpec]:
        package = path.expanduser().resolve()
        query_path = package / "query.yaml"
        rubrics_path = package / ".eval/rubrics.json"
        workspace = package / "workspace"
        if not query_path.is_file() or not rubrics_path.is_file() or not workspace.is_dir():
            raise ValueError(f"invalid LH package boundary: {package}")

        query = yaml.safe_load(query_path.read_text(encoding="utf-8"))
        if not isinstance(query, dict):
            raise ValueError(f"query.yaml must contain a mapping: {query_path}")
        candidate_id = str(query.get("task_id") or package.name.split("_", 2)[0:2][-1])
        if task_id and candidate_id != task_id:
            return []

        rubrics = json.loads(rubrics_path.read_text(encoding="utf-8"))
        if isinstance(rubrics, list):
            rubric_count = len(rubrics)
        elif isinstance(rubrics, dict):
            rubric_items = rubrics.get("items") or rubrics.get("rubrics") or rubrics.get("criteria") or []
            rubric_count = len(rubric_items) if isinstance(rubric_items, list) else 0
        else:
            rubric_count = 0

        production = parse_production(query.get("production_list"), query.get("production_type"))
        trace_sources = [str(value) for value in query.get("trace_sources", [])]
        metadata = {
            key: value
            for key, value in query.items()
            if key
            not in {
                "prompt",
                "dependencies",
                "production_list",
                "production_type",
            }
        }
        return [
            TaskSpec(
                task_id=candidate_id,
                request_id=str(query.get("request_id") or candidate_id),
                task_name=str(query.get("task_name") or package.name),
                prompt=str(query.get("prompt") or ""),
                category=str(query.get("category") or "") or None,
                business=str(query.get("business") or "") or None,
                source_adapter=InputAdapterType.LH,
                source_path=str(package),
                trace_sources=trace_sources,
                original_attachments=[str(workspace)],
                dependencies=normalize_dependencies(query.get("dependencies"), str(query_path)),
                production=production,
                forbidden_outputs=forbidden_from_production(production),
                rubrics=RubricSource(path=str(rubrics_path), raw=rubrics, count=rubric_count),
                profile="lh",
                metadata={
                    "query": query,
                    "package_name": package.name,
                    "workspace_path": str(workspace),
                    **metadata,
                },
            )
        ]
