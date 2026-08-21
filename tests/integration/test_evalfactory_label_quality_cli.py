from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from typer.testing import CliRunner

import eval_factory.cli as cli_module
from eval_factory.cli import app
from eval_factory.contracts.label_quality_v2 import LabelQualityEvidenceClassV2
from eval_factory.statistics import (
    LabelQualityEvaluationAuthorizationError,
    LabelQualityStoreIntegrityError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_TEST_ROOT = REPO_ROOT / "tests/eval_factory/contract"
UNIT_TEST_ROOT = REPO_ROOT / "tests/eval_factory/unit"
R8_01_GOLD = (
    REPO_ROOT / "evals/golden/eval_factory/statistics/r8-01-independent-label-test-set-freeze-v1.json"
)


def _fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(CONTRACT_TEST_ROOT))
    return import_module("test_v2_label_quality_contracts")


def _arguments(tmp_path: Path, policy_path: Path) -> list[str]:
    return [
        "pipeline",
        "label-quality-evaluation",
        "--policy",
        str(policy_path),
        "--r8-01-evidence",
        str(R8_01_GOLD),
        "--private-store",
        str(tmp_path / "private"),
        "--report-store",
        str(tmp_path / "report"),
        "--json",
    ]


def _write_policy(tmp_path: Path, monkeypatch) -> Path:
    policy = _fixture(monkeypatch)._policy(LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY)
    path = tmp_path / "policy.json"
    path.write_bytes(policy.canonical_json())
    return path


def _snapshot(path: Path) -> tuple[int, int]:
    files = tuple(value for value in path.rglob("*") if value.is_file())
    return len(files), sum(value.stat().st_size for value in files)


def test_label_quality_cli_runs_real_repository_pending_and_exits_three(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, _arguments(tmp_path, policy_path))

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["outcome"] == "STATISTICAL_GATE_PENDING"
    assert payload["evidence_class"] == "REPOSITORY_PENDING_ONLY"
    assert payload["prerequisite"]["available_semantic_count"] == 91
    assert payload["prerequisite"]["required_semantic_count"] == 100
    assert payload["prerequisite"]["dataset_manifest_ref"] is None
    assert payload["prerequisite"]["access_receipt_ref"] is None
    assert payload["observation_closure_sha256"] is None
    assert payload["result_closure_sha256"] is None
    assert payload["satisfies_nfr_008"] is False
    assert payload["satisfies_sc_010"] is False
    assert payload["authorizes_attestation"] is False
    assert payload["authorizes_production_release"] is False
    serialized = result.output.casefold()
    for forbidden in (
        "source_trace_id",
        "raw_sha256",
        "annotation_ref",
        "expected_decision",
        "private_material_ref",
        "physical_path",
        "credential",
        "final_output",
        "grader_rule",
        "hidden_condition",
        "exception",
    ):
        assert forbidden not in serialized


def test_label_quality_cli_exact_pending_replay_is_physical_noop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)
    arguments = _arguments(tmp_path, policy_path)

    first = CliRunner().invoke(app, arguments)
    before = _snapshot(tmp_path)
    second = CliRunner().invoke(app, arguments)
    after = _snapshot(tmp_path)

    assert first.exit_code == second.exit_code == 3
    assert json.loads(first.output)["report_sha256"] == json.loads(second.output)["report_sha256"]
    assert before == after


def test_label_quality_cli_pending_mode_rejects_frozen_only_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)
    arguments = _arguments(tmp_path, policy_path)
    arguments.extend(
        [
            "--evaluation-request",
            str(tmp_path / "request.json"),
        ]
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.output


def test_label_quality_cli_replay_cannot_bypass_pending_option_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)
    arguments = _arguments(tmp_path, policy_path)
    assert CliRunner().invoke(app, arguments).exit_code == 3
    arguments.extend(
        [
            "--evaluation-request",
            str(tmp_path / "request.json"),
        ]
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"


def test_label_quality_cli_rejects_changed_repository_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)
    changed = tmp_path / "changed-gold.json"
    payload = json.loads(R8_01_GOLD.read_text())
    payload["real_corpus"]["eligible_source_count"] = 92
    changed.write_text(json.dumps(payload))
    arguments = _arguments(tmp_path, policy_path)
    arguments[arguments.index("--r8-01-evidence") + 1] = str(changed)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"
    assert "92" not in result.output
    assert str(tmp_path) not in result.output


def test_label_quality_cli_rejects_symlinked_input_or_store_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)
    linked_policy = tmp_path / "linked-policy.json"
    linked_policy.symlink_to(policy_path)
    arguments = _arguments(tmp_path, linked_policy)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"

    real_private = tmp_path / "real-private"
    real_private.mkdir()
    linked_private = tmp_path / "linked-private"
    linked_private.symlink_to(real_private, target_is_directory=True)
    arguments = _arguments(tmp_path, policy_path)
    arguments[arguments.index("--private-store") + 1] = str(linked_private)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert json.loads(result.output)["error_code"] == "INVALID_INPUT"


def test_label_quality_cli_maps_store_corruption_to_closed_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)

    def fail(**_) -> None:
        raise LabelQualityStoreIntegrityError("private label path")

    monkeypatch.setattr(cli_module, "_label_quality_evaluation", fail)
    result = CliRunner().invoke(app, _arguments(tmp_path, policy_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "INTEGRITY_ERROR"
    assert "private label path" not in result.output
    assert str(tmp_path) not in result.output


def test_label_quality_cli_maps_denied_access_to_closed_authorization_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = _write_policy(tmp_path, monkeypatch)

    def fail(**_) -> None:
        raise LabelQualityEvaluationAuthorizationError("private principal")

    monkeypatch.setattr(cli_module, "_label_quality_evaluation", fail)
    result = CliRunner().invoke(app, _arguments(tmp_path, policy_path))

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error_code"] == "AUTHORIZATION_ERROR"
    assert "private principal" not in result.output


def test_label_quality_cli_passed_report_uses_exit_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fixtures = _fixture(monkeypatch)
    policy_path = _write_policy(tmp_path, monkeypatch)
    report = fixtures._report()
    monkeypatch.setattr(
        cli_module,
        "_label_quality_evaluation",
        lambda **_: report,
        raising=False,
    )

    result = CliRunner().invoke(app, _arguments(tmp_path, policy_path))

    assert result.exit_code == 0
    assert json.loads(result.output)["outcome"] == "PASSED"


def test_label_quality_cli_composes_real_r8_01_frozen_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    fixtures = import_module("label_quality_fixtures")
    persistence, specs, frozen = fixtures.frozen_authority(tmp_path / "source")
    manifest = frozen.dataset_manifest
    assert manifest is not None
    selected = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = fixtures.observation_set(frozen, specs, selected.members)
    request = fixtures.frozen_request(frozen, specs, observations)
    policy = fixtures.quality_policy(frozen, specs)
    policy_path = tmp_path / "frozen-policy.json"
    evidence_path = tmp_path / "frozen-result.json"
    request_path = tmp_path / "frozen-request.json"
    policy_path.write_bytes(policy.canonical_json())
    evidence_path.write_bytes(frozen.canonical_json())
    request_path.write_bytes(request.canonical_json())
    arguments = _arguments(tmp_path, policy_path)
    arguments[arguments.index("--r8-01-evidence") + 1] = str(evidence_path)
    arguments.extend(
        [
            "--evaluation-request",
            str(request_path),
            "--r8-01-database",
            str(persistence.path),
            "--r8-01-material-store",
            str(persistence.material_store.root),
        ]
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["outcome"] == "PASSED"
    assert payload["evidence_class"] == "MECHANISM_VALIDATION_ONLY"
    assert payload["total_reference_count"] == 300
    assert payload["satisfies_nfr_008"] is False
    assert payload["satisfies_sc_010"] is False


def test_label_quality_cli_accepts_typed_r8_01_pending_without_private_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(UNIT_TEST_ROOT))
    fixtures = import_module("label_quality_fixtures")
    specs = fixtures.label_specs()
    persistence = fixtures.service(tmp_path / "source")
    source = fixtures.freeze_source(specs)
    all_candidates = fixtures.candidates(specs)
    semantic_ref = all_candidates[-1].label_spec_ref
    selected = tuple(candidate for candidate in all_candidates if candidate.label_spec_ref != semantic_ref)
    selected += tuple(candidate for candidate in all_candidates if candidate.label_spec_ref == semantic_ref)[
        :91
    ]
    source["candidate_pool"] = fixtures._candidate_pool(selected)
    pending = persistence.freeze(
        idempotency_key="r8-02-cli-pending",
        **source,
    )
    policy = fixtures.quality_policy(pending, specs)
    policy_path = tmp_path / "pending-policy.json"
    evidence_path = tmp_path / "pending-result.json"
    policy_path.write_bytes(policy.canonical_json())
    evidence_path.write_bytes(pending.canonical_json())
    arguments = _arguments(tmp_path, policy_path)
    arguments[arguments.index("--r8-01-evidence") + 1] = str(evidence_path)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 3
    payload = json.loads(result.output)
    assert payload["outcome"] == "STATISTICAL_GATE_PENDING"
    assert payload["evidence_class"] == "MECHANISM_VALIDATION_ONLY"
    assert payload["prerequisite"]["freeze_result_ref"] == pending.to_ref().model_dump(mode="json")
    assert payload["prerequisite"]["shortage_count"] == 1
