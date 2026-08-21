from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/validate_eval_factory_annotations_v2.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "validate_eval_factory_annotations_v2",
    MODULE_PATH,
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
validator = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = validator
MODULE_SPEC.loader.exec_module(validator)

ROOT = Path(__file__).resolve().parents[2]
CANARY_PATH = ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
CONTRACT_MANIFEST_PATH = ROOT / "evals/annotation/v2/contract-manifest.json"
CROSS_STAGE_MANIFEST_PATH = ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json"
CANARY = json.loads(CANARY_PATH.read_text(encoding="utf-8"))["items"][0]
CONTRACT_MANIFEST_SHA256 = hashlib.sha256(CONTRACT_MANIFEST_PATH.read_bytes()).hexdigest()
CROSS_STAGE_MANIFEST_SHA256 = hashlib.sha256(CROSS_STAGE_MANIFEST_PATH.read_bytes()).hexdigest()
CANARY_MANIFEST_SHA256 = hashlib.sha256(CANARY_PATH.read_bytes()).hexdigest()
HASH = "a" * 64
EVIDENCE_ID = "ev_direct01"


def _evidence() -> dict[str, Any]:
    return {
        "evidence_ref_id": EVIDENCE_ID,
        "source_ref": CANARY["source_ref"],
        "source_sha256": CANARY["raw_sha256"],
        "locator_type": "SOURCE_SPAN",
        "locator_ref": "span://LH_005/1",
        "supports": "POSITIVE",
        "capability": "tool_events",
        "capability_complete": True,
        "approximate": False,
    }


def _base(annotation_type: str) -> dict[str, Any]:
    suffix = annotation_type.casefold()
    return {
        "schema_version": "eval-factory-annotation/v2",
        "annotation_id": f"ann_{suffix}0002",
        "annotation_type": annotation_type,
        "canary_instance_id": CANARY["instance_id"],
        "source_ref": CANARY["source_ref"],
        "source_sha256": CANARY["raw_sha256"],
        "subject_ref": f"subject://LH_005/{suffix}",
        "subject_version": "v2",
        "subject_sha256": HASH,
        "guide_version": "eval-factory-annotation-guide/v2",
        "contract_manifest_sha256": CONTRACT_MANIFEST_SHA256,
        "governing_versions": {
            "spec": "002-eval-dataset-factory/approved-v2-2026-07-20",
            "trace_adr": "ADR-0009",
            "safety_adr": "ADR-0010",
            "user_approval_policy": "eval-factory-user-approval/v2",
            "active_contract_manifest_sha256": CROSS_STAGE_MANIFEST_SHA256,
            "canary_manifest_sha256": CANARY_MANIFEST_SHA256,
        },
        "author_id": "reference-builder",
        "author_kind": "DETERMINISTIC_SERVICE",
        "status": "REFERENCE_READY",
        "conclusion": "POSITIVE",
        "evidence": [_evidence()],
        "uncertainty_codes": [],
        "finding_refs": [],
        "rationale": "Bound to exact source evidence and active v2 contracts.",
        "checkpoint_preview_refs": [],
        "supersedes": [],
        "created_at": "2026-07-20T00:00:00Z",
        "payload": {},
    }


def _parser_record() -> dict[str, Any]:
    record = _base("PARSER")
    record["payload"] = {
        "parse_outcome": "STRICT",
        "capability_truth": {
            "conversation_events": "COMPLETE",
            "tool_events": "COMPLETE",
            "call_result_pairing": "COMPLETE",
            "file_timeline": "COMPLETE",
            "final_response": "COMPLETE",
            "attachment_observations": "COMPLETE",
        },
        "repair_expectations": [],
        "event_expectations": [
            {
                "event_type": "tool_call",
                "expected_count": 1,
                "evidence_ref_ids": [EVIDENCE_ID],
            }
        ],
        "call_result_expectations": [],
        "source_span_accuracy": "EXACT",
        "negative_predicate_capabilities": ["tool_events"],
        "blocking_reason": None,
    }
    return record


def _component(name: str) -> dict[str, str]:
    return {"ref": f"{name}://LH_005/v2", "sha256": HASH}


def _release_record() -> dict[str, Any]:
    record = _base("RELEASE")
    record["checkpoint_preview_refs"] = ["user-approval-request://LH_005/final-dataset-review/v2"]
    record["payload"] = {
        "component_hashes": {
            "query_spec": _component("query"),
            "environment_spec": _component("environment"),
            "rubric_set": _component("rubric"),
            "evaluator_spec": _component("evaluator"),
            "reference_policy": _component("reference-policy"),
            "tool_policy": _component("tool-policy"),
            "provenance_manifest": _component("provenance"),
            "quality_report": _component("quality"),
        },
        "package_manifest_sha256": HASH,
        "package_provenance_exact_set": True,
        "hard_gate_findings": [],
        "automated_quality_passed": True,
        "user_approval_policy_ref": _component("user-approval-policy"),
        "required_checkpoints": ["FINAL_DATASET_REVIEW"],
        "checkpoint_decisions": [
            {
                "checkpoint": "FINAL_DATASET_REVIEW",
                "request_ref": "user-approval-request://LH_005/final/v2",
                "decision_record_ref": "user-decision-record://LH_005/final/v2",
            }
        ],
        "channel": "CANARY",
        "registry_target": "registry://canary/lh",
        "registry_isolated": True,
        "production_attestation": None,
        "expected_action": "PUBLISH",
        "expected_state": "RELEASED",
        "release_allowed": True,
    }
    return record


def test_reference_ready_records_validate_without_adjudication() -> None:
    assert (
        validator.validate_records(
            [_parser_record(), _release_record()],
            require_reference_ready=True,
        )
        == []
    )


def test_historical_independent_review_fields_are_rejected() -> None:
    record = _parser_record()
    record["adjudication"] = {
        "adjudicator_id": "should-not-exist",
    }
    errors = validator.validate_records([record])
    assert any("Additional properties are not allowed" in error for error in errors)
    assert any("historical independent-review fields are forbidden" in error for error in errors)


def test_release_requires_configured_checkpoint_and_automated_gates() -> None:
    record = _release_record()
    record["payload"]["checkpoint_decisions"] = []
    record["payload"]["automated_quality_passed"] = False

    errors = validator.validate_records([record])
    assert any("passing automated quality gates" in error for error in errors)
    assert any("every configured user checkpoint" in error for error in errors)


def test_disabled_checkpoint_cannot_have_synthetic_decision() -> None:
    record = _release_record()
    record["payload"]["required_checkpoints"] = []
    errors = validator.validate_records([record])
    assert any("cannot create synthetic decisions" in error for error in errors)


def test_checkpoint_can_bind_only_one_current_decision() -> None:
    record = _release_record()
    record["payload"]["checkpoint_decisions"].append(
        copy.deepcopy(record["payload"]["checkpoint_decisions"][0])
    )
    record["payload"]["checkpoint_decisions"][1]["decision_record_ref"] = (
        "user-decision-record://LH_005/final/v3"
    )
    errors = validator.validate_records([record])
    assert any("at most one current user decision" in error for error in errors)


def test_open_non_waivable_finding_blocks_user_accepted_release() -> None:
    record = _release_record()
    record["payload"]["hard_gate_findings"] = [
        {
            "finding_ref": "finding://LH_005/leak",
            "severity": "P2",
            "status": "OPEN",
            "non_waivable": True,
            "evidence_ref_ids": [EVIDENCE_ID],
        }
    ]
    errors = validator.validate_records([record])
    assert any("open blocking findings" in error for error in errors)


def test_reference_ready_bundle_cannot_be_empty_or_draft() -> None:
    assert validator.validate_records([], require_reference_ready=True) == [
        "bundle: reference bundle cannot be empty"
    ]
    record = _parser_record()
    record["status"] = "DRAFT"
    errors = validator.validate_records([record], require_reference_ready=True)
    assert any("requires REFERENCE_READY or SUPERSEDED" in error for error in errors)


def test_annotation_v2_generation_is_reproducible_and_v1_is_unchanged() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/export_eval_factory_annotations_v2.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    manifest = json.loads(CONTRACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["architecture"] == "v1-domain-base-plus-v2-workflow-overlay"
    assert set(manifest["historical_only_fields"]) == {
        "adjudication",
        "adjudicator_id",
        "annotator_role",
        "required_review_roles",
        "review_records_current",
    }

    v1_manifest = ROOT / "evals/annotation/v1/contract-manifest.json"
    assert hashlib.sha256(v1_manifest.read_bytes()).hexdigest() == (
        "1e4514fca3b7c5ae1c8ed53e235d8c1da2d5c7b7de0eca5e187c77dc5326f7b4"
    )


def test_release_payload_is_closed() -> None:
    record = copy.deepcopy(_release_record())
    record["payload"]["independent_reviewer_quorum"] = True
    errors = validator.validate_records([record])
    assert any("Additional properties are not allowed" in error for error in errors)
