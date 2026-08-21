from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionReasonCodeV2,
    CanaryRegressionReportV2,
)
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.readiness import CanaryRegressionReportIntegrityError

UNIT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/unit"
CONTRACT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/contract"


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "pipeline",
        "canary-regression",
        "--cohort",
        str(tmp_path / "cohort.json"),
        "--raw-root",
        str(tmp_path / "raw"),
        "--template",
        str(tmp_path / "template.json"),
        "--policy",
        str(tmp_path / "policy.json"),
        "--run-root",
        str(tmp_path / "run"),
        "--report-store",
        str(tmp_path / "reports"),
        "--json",
    ]


def test_canary_regression_cli_emits_incomplete_report_before_nonzero_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    report = import_module("test_canary_regression_store")._report()
    monkeypatch.setattr(
        cli_module,
        "_canary_regression",
        lambda **_: report,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["outcome"] == "INCOMPLETE"
    assert payload["claim_scope"] == "CANARY_REGRESSION_ONLY"
    serialized = result.output.casefold()
    for forbidden in (
        "private injected detail",
        "raw_root",
        "physical_path",
        "provider_payload",
        "grader_rule",
        "hidden_condition",
    ):
        assert forbidden not in serialized


def test_canary_regression_cli_emits_failed_report_with_success_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(CONTRACT_TEST_ROOT))
    fixtures = import_module("test_v2_canary_regression_contracts")
    base = fixtures._report()
    original = base.case_results[15]
    failed_case = fixtures._recreate(
        original,
        observed_outcome=CanaryRegressionObservedOutcomeV2.FAILED,
        reason_code=CanaryRegressionReasonCodeV2.STAGE_BLOCKED_POLICY,
        item_status=ItemStatus.FAILED,
    )
    report = CanaryRegressionReportV2.create(
        cohort_manifest_ref=base.cohort_manifest_ref,
        policy_ref=base.policy_ref,
        template_ref=base.template_ref,
        case_results=tuple(
            failed_case if item.instance_id == failed_case.instance_id else item for item in base.case_results
        ),
        audit=base.audit,
    )
    monkeypatch.setattr(cli_module, "_canary_regression", lambda **_: report)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "FAILED"
    assert payload["mismatched_ids"] == []


def test_canary_regression_cli_closes_invalid_input(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_canary_regression_cli_rejects_shared_run_and_report_root(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    run_root = arguments[arguments.index("--run-root") + 1]
    arguments[arguments.index("--report-store") + 1] = run_root

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_canary_regression_cli_closes_report_store_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(**_) -> None:
        raise CanaryRegressionReportIntegrityError("private report path")

    monkeypatch.setattr(cli_module, "_canary_regression", fail)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INTEGRITY_ERROR"
    assert "private report path" not in result.output
