from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    run_id: str
    runs_root: str
    profile: str
    export_path: str
    overwrite: bool
    task: dict[str, object]
    dependencies: list[dict[str, object]]
    forbidden_outputs: list[dict[str, object]]
    capabilities: list[dict[str, object]]
    evidence: list[dict[str, object]]
    world_ledger: dict[str, object]
    artifact_plans: list[dict[str, object]]
    artifact_results: dict[str, dict[str, object]]
    findings: list[dict[str, object]]
    reviews: list[dict[str, object]]
    problems: list[dict[str, object]]
    repair_attempts: dict[str, int]
    route_history: list[dict[str, object]]
    release_allowed: bool
    package_path: str
    final_status: str
