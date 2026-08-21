from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from eval_factory.cli import app

ROOT = Path(__file__).resolve().parents[2]
UNIT_TEST_ROOT = ROOT / "tests/eval_factory/unit"


def _fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    return import_module("test_production_release")


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    fixture = _fixture(monkeypatch)
    payload = fixture._payload()
    policy = fixture._policy(payload)
    policy_path = tmp_path / "policy.json"
    evidence_path = tmp_path / "evidence.json"
    policy_path.write_bytes(policy.canonical_json())
    evidence_path.write_bytes(payload)
    return policy_path, evidence_path


def _arguments(
    tmp_path: Path,
    policy_path: Path,
    evidence_path: Path,
) -> list[str]:
    return [
        "pipeline",
        "production-release",
        "--policy",
        str(policy_path),
        "--repository-evidence",
        str(evidence_path),
        "--report-store",
        str(tmp_path / "reports"),
        "--json",
    ]


def _inventory(root: Path) -> tuple[int, int]:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)


def test_production_release_cli_blocks_repository_and_replays_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, evidence = _inputs(tmp_path, monkeypatch)
    arguments = _arguments(tmp_path, policy, evidence)

    first = CliRunner().invoke(app, arguments)
    before = _inventory(tmp_path / "reports")
    second = CliRunner().invoke(app, arguments)

    assert first.exit_code == second.exit_code == 3
    first_value = json.loads(first.output)
    second_value = json.loads(second.output)
    assert first_value == second_value
    assert first_value["outcome"] == "PRODUCTION_RELEASE_BLOCKED"
    assert first_value["evidence_class"] == "REPOSITORY_PENDING_ONLY"
    assert first_value["reason_codes"] == ["ATTESTATION_PENDING"]
    assert first_value["attestation_authority"] is None
    assert first_value["release_manifest"] is None
    assert first_value["registry_entry"] is None
    assert first_value["satisfies_sc_015"] is False
    assert _inventory(tmp_path / "reports") == before


def test_production_release_cli_rejects_overlapping_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, evidence = _inputs(tmp_path, monkeypatch)
    arguments = _arguments(tmp_path, policy, evidence)
    arguments[arguments.index("--report-store") + 1] = str(tmp_path)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    value = json.loads(result.output)
    assert value["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_checked_repository_evidence_is_blocked_and_exactly_replayable(
    tmp_path: Path,
) -> None:
    policy = ROOT / "evals/manifests/r8-09-production-release-policy-v1.json"
    evidence = ROOT / "evals/manifests/r8-09-repository-pending-production-release-evidence-v1.json"
    gold = json.loads(
        (ROOT / "evals/golden/eval_factory/release/r8-09-production-release-v1.json").read_text(
            encoding="utf-8"
        )
    )
    arguments = _arguments(tmp_path, policy, evidence)

    first = CliRunner().invoke(app, arguments)
    inventory = _inventory(tmp_path / "reports")
    replay_one = CliRunner().invoke(app, arguments)
    replay_two = CliRunner().invoke(app, arguments)

    values = tuple(json.loads(value.output) for value in (first, replay_one, replay_two))
    expected = gold["repository_pending"]
    assert first.exit_code == replay_one.exit_code == replay_two.exit_code == 3
    assert values[0] == values[1] == values[2]
    assert values[0]["result_sha256"] == (expected["production_release_result_sha256"])
    assert values[0]["outcome"] == expected["expected_outcome"]
    assert values[0]["reason_codes"] == [expected["expected_reason"]]
    assert values[0]["published_decisions"] == []
    assert values[0]["published_items"] == []
    assert values[0]["registry_entry"] is None
    assert values[0]["satisfies_sc_015"] is False
    assert inventory == (
        expected["public_report_store_file_count"],
        expected["public_report_store_bytes"],
    )
    assert _inventory(tmp_path / "reports") == inventory
