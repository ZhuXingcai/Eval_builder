from __future__ import annotations

import json
from pathlib import Path

import yaml

from env_mock_agent.profiles.base import PackageProfile
from env_mock_agent.schemas import TaskSpec
from env_mock_agent.store import RunStore


class LhProfile(PackageProfile):
    name = "lh"

    def export(self, store: RunStore, task: TaskSpec, destination: Path, *, overwrite: bool = False) -> Path:
        self.assert_exportable(store)
        output = self.prepare_destination(destination, overwrite)
        query = task.metadata.get("query")
        if not isinstance(query, dict):
            query = {
                "task_id": task.task_id,
                "request_id": task.request_id,
                "task_name": task.task_name,
                "prompt": task.prompt,
                "category": task.category,
                "business": task.business,
                "production_list": "\n".join(
                    f"{index}. {item.description or item.name}"
                    for index, item in enumerate(task.production, start=1)
                ),
                "dependencies": [
                    {
                        "path": item.path,
                        "kind": item.kind.value,
                        "must_exist_before_start": item.must_exist_before_start,
                        "importance": item.criticality.value,
                        "evidence": "; ".join(ref.excerpt or ref.source for ref in item.evidence_refs),
                        "expected_content": item.expected_content,
                        "mock_strategy": item.reconstruction_strategy.value,
                        "asset_type": item.asset_type,
                        "risk_notes": item.risk_notes,
                    }
                    for item in task.dependencies
                ],
            }
        (output / "query.yaml").write_text(
            yaml.safe_dump(query, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        eval_dir = output / ".eval"
        eval_dir.mkdir()
        rubrics = task.rubrics.raw if task.rubrics is not None else {"items": [], "total_score": 0}
        (eval_dir / "rubrics.json").write_text(
            json.dumps(rubrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.copy_workspace(store, output)
        return output
