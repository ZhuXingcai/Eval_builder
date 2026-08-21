from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

from env_mock_agent.providers import TextProvider
from eval_factory.cli import app
from eval_factory.orchestration.job_store import JobStore

GOLD_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals/golden/eval_factory/orchestration"
    / "r6-parent-e2e-canary-v1.json"
)
UNIT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/unit"


def test_public_canary_command_runs_text_provider_once_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    fixture_module = import_module("test_canary_pipeline_driver")
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    manifest = fixture_module._manifest(fixture_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    calls = 0
    original_generate = TextProvider.generate

    def counted_generate(self, request):
        nonlocal calls
        calls += 1
        return original_generate(self, request)

    monkeypatch.setattr(
        TextProvider,
        "generate",
        counted_generate,
    )
    job_store_path = tmp_path / "job.sqlite3"
    arguments = [
        "pipeline",
        "canary-run",
        "--manifest",
        str(manifest_path),
        "--job-store",
        str(job_store_path),
        "--source-registry",
        str(tmp_path / "source.sqlite3"),
        "--trace-store",
        str(tmp_path / "trace-store"),
        "--stage-store",
        str(tmp_path / "stage-store"),
        "--json",
    ]

    first = CliRunner().invoke(app, arguments)
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    replay = CliRunner().invoke(app, arguments)
    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.output) == payload
    assert calls == 1

    store = JobStore(job_store_path)
    units = store.list_work_units(job_id=payload["job_id"])
    metrics = store.list_work_attempt_metrics(job_id=payload["job_id"])
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    observed = {
        "artifact_group_units": sum(unit.scope.value == "ARTIFACT_GROUP" for unit in units),
        "failed_items": len(payload["failed_item_ids"]),
        "item_quality_results": len(payload["quality_result_refs"]),
        "job_status": payload["job_status"],
        "package_manifests": len(payload["package_manifest_refs"]),
        "provider_invocations": calls,
        "semantic_review_units": sum(unit.semantic_review_round is not None for unit in units),
        "stage_runs": len(store.list_stage_runs(job_id=payload["job_id"])),
        "succeeded_items": len(payload["succeeded_item_ids"]),
        "work_attempt_metrics": len(metrics),
        "work_leases": len(store.list_work_leases(job_id=payload["job_id"])),
        "work_units": len(units),
    }

    assert payload["claim_scope"] == gold["claim_scope"]
    assert manifest.policy_version == gold["policy_version"]
    assert observed == gold["expected"]
    assert all(metric.resource_usage is not None for metric in metrics)
    serialized = first.output.casefold()
    for forbidden in (
        "safe input-state content",
        "visible_prompt",
        "provider_payload",
        "private_reference",
        "physical_path",
        "grader_rule",
        "hidden_condition",
    ):
        assert forbidden not in serialized
