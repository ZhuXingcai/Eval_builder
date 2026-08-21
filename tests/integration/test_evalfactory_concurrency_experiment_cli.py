from __future__ import annotations

import errno
import json
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentReportV2,
)
from eval_factory.readiness import (
    ConcurrencyExperimentStoreIntegrityError,
)

UNIT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/unit"


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "pipeline",
        "concurrency-experiment",
        "--policy",
        str(tmp_path / "policy.json"),
        "--canary-manifest",
        str(tmp_path / "canary.json"),
        "--raw-root",
        str(tmp_path / "raw"),
        "--template",
        str(tmp_path / "template.json"),
        "--run-root",
        str(tmp_path / "run"),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "report"),
        "--json",
    ]


def _report(monkeypatch) -> ConcurrencyExperimentReportV2:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    return import_module("concurrency_experiment_fixtures").report()


def test_concurrency_experiment_cli_emits_scoped_recommendation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _report(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_concurrency_experiment",
        lambda **_: report,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "NON_MODEL_RECOMMENDED"
    assert payload["claim_scope"] == "CANARY_WORKER_STORAGE_ONLY"
    assert payload["satisfies_sc_012"] is False
    assert payload["authorizes_attestation"] is False
    assert payload["authorizes_production_release"] is False
    serialized = result.output.casefold()
    for forbidden in (
        "instance_id",
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
        "final_output",
        "grader",
    ):
        assert forbidden not in serialized


def test_concurrency_experiment_cli_persists_pending_before_exit_three(
    tmp_path: Path,
    monkeypatch,
) -> None:
    accepted = _report(monkeypatch)
    pending = ConcurrencyExperimentReportV2.create(
        policy_ref=accepted.policy_ref,
        host_summary=accepted.host_summary,
        cohort_manifest_ref=accepted.cohort_manifest_ref,
        template_ref=accepted.template_ref,
        private_workload_ref=accepted.private_workload_ref,
        private_result_set_ref=accepted.private_result_set_ref,
        level_summaries=accepted.level_summaries,
        recovery_summary=None,
        recommendation=None,
        audit=accepted.audit,
    )
    monkeypatch.setattr(
        cli_module,
        "_concurrency_experiment",
        lambda **_: pending,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 3
    assert json.loads(result.output)["outcome"] == "RECOMMENDATION_PENDING"


@pytest.mark.parametrize(
    ("run_root", "private_store"),
    (
        ("run", "run/private"),
        ("private/run", "private"),
    ),
)
def test_concurrency_experiment_rejects_nested_roots(
    tmp_path: Path,
    run_root: str,
    private_store: str,
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        cli_module._concurrency_experiment(
            policy_path=tmp_path / "policy.json",
            canary_manifest=tmp_path / "canary.json",
            raw_root=tmp_path / "raw",
            template_path=tmp_path / "template.json",
            run_root=tmp_path / run_root,
            private_store=tmp_path / private_store,
            report_store=tmp_path / "report",
        )


def test_concurrency_experiment_cli_closes_store_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(**_) -> None:
        raise ConcurrencyExperimentStoreIntegrityError("private concurrency path")

    monkeypatch.setattr(
        cli_module,
        "_concurrency_experiment",
        fail,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INTEGRITY_ERROR"
    assert "private concurrency path" not in result.output
    assert str(tmp_path) not in result.output


def test_concurrency_experiment_cli_closes_file_descriptor_exhaustion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(**_) -> None:
        raise OSError(
            errno.EMFILE,
            "too many open files",
            tmp_path / "private-source.jsonl",
        )

    monkeypatch.setattr(
        cli_module,
        "_concurrency_experiment",
        fail,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload == {
        "error_code": "INTERNAL_ERROR",
        "message": "pipeline operation failed",
        "schema_version": "eval-factory/pipeline-cli-error/v1",
    }
    assert "too many open files" not in result.output
    assert str(tmp_path) not in result.output
