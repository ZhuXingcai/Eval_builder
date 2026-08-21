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
from eval_factory.readiness import (
    ProductionAttestationAuthorizationError,
    ProductionAttestationPolicyError,
    ProductionAttestationStoreIntegrityError,
)

ROOT = Path(__file__).resolve().parents[2]
UNIT_TEST_ROOT = ROOT / "tests/eval_factory/unit"
POLICY = ROOT / "evals/manifests/r8-08-production-readiness-attestation-policy-v1.json"
EVIDENCE = ROOT / "evals/manifests/r8-08-repository-pending-attestation-evidence-v1.json"


def _unit_fixture(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    return import_module("production_attestation_fixtures")


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "pipeline",
        "readiness-attestation",
        "--policy",
        str(POLICY),
        "--repository-evidence",
        str(EVIDENCE),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "public"),
        "--json",
    ]


def _snapshot(path: Path) -> tuple[int, int]:
    files = tuple(item for item in path.rglob("*") if item.is_file())
    return len(files), sum(item.stat().st_size for item in files)


def test_attestation_cli_repository_pending_and_exact_replay(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    first = CliRunner().invoke(app, arguments)
    before = _snapshot(tmp_path)
    second = CliRunner().invoke(app, arguments)
    after = _snapshot(tmp_path)

    assert first.exit_code == second.exit_code == 3
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload["result_sha256"] == second_payload["result_sha256"]
    assert first_payload["outcome"] == "ATTESTATION_PENDING"
    assert first_payload["evidence_class"] == "REPOSITORY_PENDING_ONLY"
    assert first_payload["passed_gate_count"] == 0
    assert first_payload["failed_gate_count"] == 0
    assert first_payload["pending_gate_count"] == 5
    assert first_payload["projection"] is None
    assert first_payload["private_request_closure_sha256"] is None
    assert first_payload["private_issuer_closure_sha256"] is None
    assert first_payload["private_attestation_closure_sha256"] is None
    assert first_payload["active_attestation"] is False
    assert first_payload["satisfies_sc_015"] is False
    assert first_payload["authorizes_production_release"] is False
    assert before == after
    serialized = first.output.casefold()
    for forbidden in (
        "issuer_principal",
        "approved_by",
        "proof_value",
        "organization_contact",
        "private_ref",
        "physical_path",
        "credential",
        "raw_trace",
        "private_reference",
        "model_payload",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "exception",
    ):
        assert forbidden not in serialized


def test_attestation_cli_rejects_mixed_mode_before_replay(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    assert CliRunner().invoke(app, arguments).exit_code == 3
    request = tmp_path / "request.json"
    request.write_text("{}")
    arguments.extend(["--attestation-request", str(request)])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_attestation_cli_maps_sensitive_errors_to_closed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(**_: object) -> Never:
        raise ProductionAttestationAuthorizationError("principal://private-issuer")

    monkeypatch.setattr(
        cli_module,
        "_production_readiness_attestation",
        denied,
    )
    result = CliRunner().invoke(app, _arguments(tmp_path))
    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "AUTHORIZATION_ERROR"
    assert "private-issuer" not in result.output

    def corrupt(**_: object) -> Never:
        raise ProductionAttestationStoreIntegrityError("/private/frozen/path")

    monkeypatch.setattr(
        cli_module,
        "_production_readiness_attestation",
        corrupt,
    )
    result = CliRunner().invoke(app, _arguments(tmp_path))
    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INTEGRITY_ERROR"
    assert "/private/frozen/path" not in result.output

    def stale_policy(**_: object) -> Never:
        raise ProductionAttestationPolicyError("private policy detail")

    monkeypatch.setattr(
        cli_module,
        "_production_readiness_attestation",
        stale_policy,
    )
    result = CliRunner().invoke(app, _arguments(tmp_path))
    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "POLICY_ERROR"
    assert "private policy detail" not in result.output


@pytest.mark.parametrize(
    ("rejected", "exit_code", "outcome"),
    (
        (False, 0, "ISSUED"),
        (True, 3, "REJECTED"),
    ),
)
def test_attestation_cli_full_issuance_and_rejection_are_replay_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rejected: bool,
    exit_code: int,
    outcome: str,
) -> None:
    fixture = _unit_fixture(monkeypatch)
    policy, request, registry = fixture.production_inputs(
        tmp_path / "fixtures",
        rejected=rejected,
    )
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    policy_path = input_root / "policy.json"
    request_path = input_root / "request.json"
    registry_path = input_root / "registry.json"
    policy_path.write_bytes(policy.canonical_json())
    request_path.write_bytes(request.canonical_json())
    registry_path.write_bytes(registry.canonical_json())
    arguments = [
        "pipeline",
        "readiness-attestation",
        "--policy",
        str(policy_path),
        "--attestation-request",
        str(request_path),
        "--issuer-registry",
        str(registry_path),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "public"),
        "--json",
    ]

    first = CliRunner().invoke(app, arguments)
    before = _snapshot(tmp_path)
    second = CliRunner().invoke(app, arguments)
    after = _snapshot(tmp_path)

    assert first.exit_code == second.exit_code == exit_code
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    assert first_payload == second_payload
    assert first_payload["outcome"] == outcome
    assert first_payload["satisfies_sc_015"] is False
    assert first_payload["authorizes_production_release"] is False
    if rejected:
        assert first_payload["projection"] is None
        assert first_payload["private_request_closure_sha256"] is not None
        assert first_payload["private_issuer_closure_sha256"] is not None
        assert first_payload["private_attestation_closure_sha256"] is None
    else:
        assert first_payload["projection"]["state"] == "ACTIVE"
        assert first_payload["private_attestation_closure_sha256"] is not None
        assert all(first_payload[f"satisfies_sc_{number:03d}"] for number in range(10, 15))
    assert before == after
