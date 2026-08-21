from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest
from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityReasonCodeV2,
    RealTraceStabilityReportV2,
)
from eval_factory.readiness import RealTraceStabilityStoreIntegrityError

CONTRACT_TEST_ROOT = Path(__file__).resolve().parents[1] / "eval_factory/contract"


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "pipeline",
        "real-trace-stability",
        "--source-manifest",
        str(tmp_path / "manifest.csv"),
        "--raw-root",
        str(tmp_path / "raw"),
        "--policy",
        str(tmp_path / "policy.json"),
        "--run-root",
        str(tmp_path / "run"),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "reports"),
        "--json",
    ]


def _fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(CONTRACT_TEST_ROOT))
    return import_module("test_v2_real_trace_stability_contracts")


def test_real_trace_stability_cli_emits_pending_report_with_success_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _fixture(monkeypatch)._report(91)
    monkeypatch.setattr(
        cli_module,
        "_real_trace_stability",
        lambda **_: report,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "STATISTICAL_GATE_PENDING"
    assert payload["claim_scope"] == "REAL_TRACE_STABILITY_ONLY"
    serialized = result.output.casefold()
    for forbidden in (
        "instance_id",
        "source_trace",
        "raw_sha",
        "physical_path",
        "sid",
        "final_output",
        "credential",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("unstable_index", "expected_outcome"),
    (
        (None, "PASSED"),
        (0, "STABILITY_THRESHOLD_NOT_MET"),
    ),
)
def test_real_trace_stability_cli_emits_sufficient_outcomes_with_success_exit(
    tmp_path: Path,
    monkeypatch,
    unstable_index: int | None,
    expected_outcome: str,
) -> None:
    report = _fixture(monkeypatch)._report(100, unstable_index=unstable_index)
    monkeypatch.setattr(
        cli_module,
        "_real_trace_stability",
        lambda **_: report,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 0
    assert json.loads(result.output)["outcome"] == expected_outcome


def test_real_trace_stability_cli_emits_incomplete_before_exit_three(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixtures = _fixture(monkeypatch)
    base = fixtures._report(91)
    original = base.case_summaries[0]
    incomplete_case = RealTraceStabilityCaseSummaryV2.create(
        private_case_result_ref=original.private_case_result_ref,
        outcome=RealTraceStabilityCaseOutcomeV2.INCOMPLETE,
        reason_code=RealTraceStabilityReasonCodeV2.CHILD_STATE_INCOMPLETE,
        parse_quality=None,
        assigned_fault_point=original.assigned_fault_point,
        fault_observed=False,
        resume_succeeded=False,
        replay_stable=False,
        attempt_count=0,
        unexpected_retry_count=0,
        resume_count=0,
        replay_count=0,
        stage_result_count=0,
        completion_witness_count=0,
        audit=base.audit,
    )
    report = RealTraceStabilityReportV2.create(
        policy_ref=base.policy_ref,
        corpus_summary=base.corpus_summary,
        private_result_set_ref=base.private_result_set_ref,
        case_summaries=(incomplete_case, *base.case_summaries[1:]),
        audit=base.audit,
    )
    monkeypatch.setattr(
        cli_module,
        "_real_trace_stability",
        lambda **_: report,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 3
    assert json.loads(result.output)["outcome"] == "INCOMPLETE"


def test_real_trace_stability_cli_rejects_shared_roots(
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


def test_real_trace_stability_cli_closes_store_corruption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail(**_) -> None:
        raise RealTraceStabilityStoreIntegrityError("private report path")

    monkeypatch.setattr(cli_module, "_real_trace_stability", fail)

    result = CliRunner().invoke(app, _arguments(tmp_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INTEGRITY_ERROR"
    assert "private report path" not in result.output
