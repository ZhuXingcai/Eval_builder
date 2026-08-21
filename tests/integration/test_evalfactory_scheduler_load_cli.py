from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.readiness import SchedulerLoadStoreIntegrityError

CONTRACT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/contract"


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "pipeline",
        "scheduler-load",
        "--policy",
        str(tmp_path / "policy.json"),
        "--job-store",
        str(tmp_path / "factory.sqlite3"),
        "--run-root",
        str(tmp_path / "run"),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "report"),
        "--json",
    ]


def _fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(CONTRACT_TEST_ROOT))
    return import_module("test_v2_scheduler_load_contracts")


def test_scheduler_load_cli_emits_complete_load_only_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _fixture(monkeypatch)._report()
    monkeypatch.setattr(cli_module, "_scheduler_load", lambda **_: report)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "COMPLETE"
    assert payload["claim_scope"] == "SCHEDULER_LOAD_ONLY"
    assert payload["unique_job_count"] == 1000
    assert payload["authorizes_attestation"] is False
    assert payload["authorizes_production_release"] is False
    serialized = result.output.casefold()
    for forbidden in (
        "job_id",
        "item_id",
        "work_unit_id",
        "lease_id",
        "stage_run_id",
        "source_uri",
        "raw_sha",
        "physical_path",
        "credential",
        "exception",
    ):
        assert forbidden not in serialized


def test_scheduler_load_cli_persists_incomplete_before_exit_three(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _fixture(monkeypatch)._report(replace_index=0)
    monkeypatch.setattr(cli_module, "_scheduler_load", lambda **_: report)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 3
    assert json.loads(result.output)["outcome"] == "INCOMPLETE"


def test_scheduler_load_cli_rejects_shared_roots(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    run_root = arguments[arguments.index("--run-root") + 1]
    arguments[arguments.index("--private-store") + 1] = run_root

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    ("run_root", "private_store"),
    (
        ("run", "run/private"),
        ("private/run", "private"),
    ),
)
def test_scheduler_load_cli_rejects_nested_roots(
    tmp_path: Path,
    run_root: str,
    private_store: str,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        cli_module._scheduler_load(
            policy_path=tmp_path / "policy.json",
            job_store=tmp_path / "factory.sqlite3",
            run_root=tmp_path / run_root,
            private_store=tmp_path / private_store,
            report_store=tmp_path / "report",
        )


def test_scheduler_load_cli_closes_store_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(**_) -> None:
        raise SchedulerLoadStoreIntegrityError("private scheduler path")

    monkeypatch.setattr(cli_module, "_scheduler_load", fail, raising=False)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INTEGRITY_ERROR"
    assert "private scheduler path" not in result.output
