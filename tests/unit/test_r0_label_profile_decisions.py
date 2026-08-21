from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval_factory.contracts.labeling import LabelSpec, PredicateOperator

ROOT = Path(__file__).resolve().parents[2]
LABEL_ROOT = ROOT / "specs/002-eval-dataset-factory/labels/v1"
PROFILE_PATH = ROOT / "specs/002-eval-dataset-factory/release-profiles/v1/decision.json"
DECISION_MANIFEST_PATH = (
    ROOT / "specs/002-eval-dataset-factory/decisions/r0-11-first-labels-release-profiles.manifest.json"
)
EXPECTED_CONTRACT_SHA = "d550d84262d3f727394c65e52ec0bf6e41fdaf532bfde0851c72e6d1495d59a9"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_representative_label_specs_validate() -> None:
    paths = sorted(LABEL_ROOT.glob("*.json"))
    labels = [LabelSpec.model_validate_json(path.read_text(encoding="utf-8")) for path in paths]

    assert {label.name for label in labels} == {
        "search_tool_usage",
        "powershell_error_signature",
        "contextual_recovery_after_tool_error",
    }
    assert all(
        any(
            version.component == "cross-stage-contracts" and version.sha256 == EXPECTED_CONTRACT_SHA
            for version in label.audit.governing_versions
        )
        for label in labels
    )


def test_structured_labels_do_not_invoke_semantic_residual() -> None:
    search = LabelSpec.model_validate_json(
        (LABEL_ROOT / "search-tool-usage.json").read_text(encoding="utf-8")
    )
    powershell = LabelSpec.model_validate_json(
        (LABEL_ROOT / "powershell-error-signature.json").read_text(encoding="utf-8")
    )

    assert search.semantic_residual is None
    assert search.positive_predicates[0].field_path == "tool_family"
    assert search.positive_predicates[0].expected_value == "search"
    assert search.positive_predicates[0].required_capability == "tool_events"
    assert powershell.semantic_residual is None
    assert powershell.positive_predicates[0].operator is PredicateOperator.ERROR_SIGNATURE
    assert powershell.positive_predicates[0].required_capability == "call_result_pairing"


def test_contextual_label_has_deterministic_prerequisite_and_abstention() -> None:
    label = LabelSpec.model_validate_json(
        (LABEL_ROOT / "contextual-recovery-after-tool-error.json").read_text(encoding="utf-8")
    )

    assert label.positive_predicates[0].operator is PredicateOperator.SEQUENCE
    assert label.semantic_residual is not None
    assert len(label.semantic_residual.abstain_conditions) >= 4
    assert "full_trace" not in label.semantic_residual.allowed_evidence_types
    assert label.review_threshold < label.decision_threshold


def test_release_profile_decision_keeps_generic_disabled() -> None:
    decision = _load_json(PROFILE_PATH)
    profiles = {item["profile"]: item for item in decision["profiles"]}

    assert decision["status"] == "APPROVED"
    assert decision["contract_manifest_sha256"] == EXPECTED_CONTRACT_SHA
    assert profiles["LH"]["decision"] == "REQUIRED"
    assert profiles["LH"]["enabled_channels"] == ["CANARY", "INTERNAL_REVIEW"]
    assert profiles["LH"]["production_enabled"] is False
    assert profiles["GENERIC"]["decision"] == "DEFERRED_APPROVAL_REQUIRED"
    assert profiles["GENERIC"]["enabled_channels"] == []
    assert profiles["GENERIC"]["production_enabled"] is False


def test_decisions_contain_no_canary_or_raw_trace_content() -> None:
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in [*sorted(LABEL_ROOT.glob("*.json")), PROFILE_PATH]
    ).casefold()
    for forbidden in ("raw_traj://lh_", '"prompt"', '"response"', '"sid"', '"account"'):
        assert forbidden not in serialized


def test_decision_manifest_binds_every_approved_file() -> None:
    manifest = _load_json(DECISION_MANIFEST_PATH)

    assert manifest["status"] == "APPROVED"
    assert manifest["contract_manifest_sha256"] == EXPECTED_CONTRACT_SHA
    assert manifest["canary_manifest_sha256"] == (
        "1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"
    )
    for item in manifest["files"]:
        assert item["sha256"] == _sha256(ROOT / item["path"])
