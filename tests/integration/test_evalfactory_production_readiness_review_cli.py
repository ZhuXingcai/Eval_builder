from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Never

import pytest
from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewEvidenceClassV2,
)
from eval_factory.readiness import (
    ProductionReadinessReviewAuthorizationError,
    ProductionReadinessStoreIntegrityError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_TEST_ROOT = REPO_ROOT / "tests/eval_factory/contract"
UNIT_TEST_ROOT = REPO_ROOT / "tests/eval_factory/unit"
REPOSITORY_EVIDENCE = REPO_ROOT / "evals/manifests/r8-07-repository-pending-evidence-v1.json"


def _contract_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(CONTRACT_TEST_ROOT))
    return import_module("test_v2_production_readiness_review_contracts")


def _unit_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    return import_module("production_review_fixtures")


def _pending_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    policy = _contract_fixture(monkeypatch)._policy(
        ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY
    )
    path = tmp_path / "policy.json"
    path.write_bytes(policy.canonical_json())
    return path


def _pending_arguments(tmp_path: Path, policy_path: Path) -> list[str]:
    return [
        "pipeline",
        "readiness-review",
        "--policy",
        str(policy_path),
        "--repository-evidence",
        str(REPOSITORY_EVIDENCE),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "public"),
        "--json",
    ]


def _snapshot(path: Path) -> tuple[int, int]:
    files = tuple(value for value in path.rglob("*") if value.is_file())
    return len(files), sum(value.stat().st_size for value in files)


def test_readiness_review_cli_runs_repository_pending_and_exits_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _pending_policy(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        app,
        _pending_arguments(tmp_path, policy_path),
    )

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["outcome"] == "APPROVALS_PENDING"
    assert payload["evidence_class"] == "REPOSITORY_PENDING_ONLY"
    assert payload["approved_domain_count"] == 0
    assert payload["pending_domain_count"] == 4
    assert payload["rejected_domain_count"] == 0
    assert payload["private_request_closure_sha256"] is None
    assert payload["private_approval_closure_sha256"] is None
    assert payload["satisfies_sc_013"] is False
    assert payload["authorizes_attestation"] is False
    assert payload["authorizes_production_release"] is False
    assert all(value["approval_record"] is None for value in payload["domain_reviews"])
    serialized = result.output.casefold()
    for forbidden in (
        "authority_principal",
        "proof_value",
        "signature",
        "incident_contact",
        "private_ref",
        "physical_path",
        "credential",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "exception",
    ):
        assert forbidden not in serialized


def test_readiness_review_cli_pending_replay_is_physical_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _pending_policy(tmp_path, monkeypatch)
    arguments = _pending_arguments(tmp_path, policy_path)

    first = CliRunner().invoke(app, arguments)
    before = _snapshot(tmp_path)
    second = CliRunner().invoke(app, arguments)
    after = _snapshot(tmp_path)

    assert first.exit_code == second.exit_code == 3
    assert json.loads(first.output)["report_sha256"] == (json.loads(second.output)["report_sha256"])
    assert before == after


def test_readiness_review_cli_rejects_mixed_mode_even_on_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _pending_policy(tmp_path, monkeypatch)
    arguments = _pending_arguments(tmp_path, policy_path)
    assert CliRunner().invoke(app, arguments).exit_code == 3
    request_path = tmp_path / "request.json"
    request_path.write_text("{}")
    arguments.extend(["--review-request", str(request_path)])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_readiness_review_cli_full_mechanism_approved_uses_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _unit_fixture(monkeypatch)
    policy, request, registry = fixture.full_inputs()
    policy_path = tmp_path / "policy.json"
    request_path = tmp_path / "request.json"
    registry_path = tmp_path / "registry.json"
    policy_path.write_bytes(policy.canonical_json())
    request_path.write_bytes(request.canonical_json())
    registry_path.write_bytes(registry.canonical_json())
    arguments = [
        "pipeline",
        "readiness-review",
        "--policy",
        str(policy_path),
        "--review-request",
        str(request_path),
        "--approval-registry",
        str(registry_path),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "public"),
        "--json",
    ]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "APPROVED"
    assert payload["evidence_class"] == "MECHANISM_VALIDATION_ONLY"
    assert payload["satisfies_sc_013"] is False


def test_readiness_review_cli_maps_sensitive_errors_to_closed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = _pending_policy(tmp_path, monkeypatch)

    def denied(**_: object) -> Never:
        raise ProductionReadinessReviewAuthorizationError("principal://private-approver")

    monkeypatch.setattr(
        cli_module,
        "_production_readiness_review",
        denied,
        raising=False,
    )
    result = CliRunner().invoke(
        app,
        _pending_arguments(tmp_path, policy_path),
    )
    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "AUTHORIZATION_ERROR"
    assert "private-approver" not in result.output

    def corrupt(**_: object) -> Never:
        raise ProductionReadinessStoreIntegrityError("/private/proof/path")

    monkeypatch.setattr(
        cli_module,
        "_production_readiness_review",
        corrupt,
        raising=False,
    )
    result = CliRunner().invoke(
        app,
        _pending_arguments(tmp_path, policy_path),
    )
    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INTEGRITY_ERROR"
    assert "/private/proof/path" not in result.output
