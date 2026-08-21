from __future__ import annotations

import json
from pathlib import Path

import pytest

from env_mock_agent.evals import BenchmarkRunner, BenchmarkSuite


def make_lh_fixture(tmp_path: Path) -> Path:
    package = tmp_path / "LH_TEST_fixture"
    (package / ".eval").mkdir(parents=True)
    (package / "workspace").mkdir()
    (package / "query.yaml").write_text(
        """
task_id: LH_TEST
task_name: fixture
prompt: Use the provided input.
dependencies:
  - path: input.txt
    kind: file
    must_exist_before_start: true
    importance: critical
    expected_content: Neutral input.
    mock_strategy: synthesize
    asset_type: txt
""".lstrip(),
        encoding="utf-8",
    )
    (package / ".eval/rubrics.json").write_text('{"items": []}\n', encoding="utf-8")
    return package


def test_benchmark_emits_one_failure_record_per_blocked_trial(
    tmp_path: Path,
) -> None:
    suite = BenchmarkSuite.model_validate(
        {
            "suite_id": "blocked-suite",
            "cases": [{"task_id": "LH_TEST", "path": str(make_lh_fixture(tmp_path))}],
            "runtimes": [
                {
                    "name": "sdk",
                    "runtime": "claude_agent_sdk",
                    "model_profile": {
                        "provider": "anthropic",
                        "model": "test",
                    },
                    "required_env": ["ENVMOCK_TEST_MISSING_CREDENTIAL"],
                }
            ],
            "repetitions": 3,
            "include_ark_subset": False,
        }
    )
    output = BenchmarkRunner(suite, tmp_path / "results").run()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["trials"]) == 3
    assert all(item["status"] == "blocked_preflight" for item in manifest["trials"])
    assert all(
        item["preflight_evidence"]["missing_env"] == ["ENVMOCK_TEST_MISSING_CREDENTIAL"]
        for item in manifest["trials"]
    )
    assert "No harness-quality conclusion is valid" in (output / "report.md").read_text(encoding="utf-8")


def test_benchmark_reports_ark_subset_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    suite = BenchmarkSuite.model_validate(
        {
            "suite_id": "ark-suite",
            "cases": [{"task_id": "LH_TEST", "path": str(make_lh_fixture(tmp_path))}],
            "runtimes": [
                {
                    "name": "sdk",
                    "runtime": "claude_agent_sdk",
                    "model_profile": {
                        "provider": "anthropic",
                        "model": "test",
                    },
                    "required_env": ["ENVMOCK_TEST_MISSING_CREDENTIAL"],
                }
            ],
            "repetitions": 1,
            "include_ark_subset": True,
        }
    )

    output = BenchmarkRunner(suite, tmp_path / "results").run()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preflight"]["pi_ark_glm"]["ready"] is False
    assert len(manifest["trials"]) == 2
    assert "| pi_ark_glm | no | missing required credentials: ARK_API_KEY |" in (
        output / "report.md"
    ).read_text(encoding="utf-8")
