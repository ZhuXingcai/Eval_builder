from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/validate_eval_factory_annotations.py"
MODULE_SPEC = importlib.util.spec_from_file_location("validate_eval_factory_annotations", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
validator = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = validator
MODULE_SPEC.loader.exec_module(validator)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
CONTRACT_MANIFEST_PATH = REPO_ROOT / "evals/annotation/v1/contract-manifest.json"
CANARY = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["items"][0]
CONTRACT_MANIFEST_SHA256 = hashlib.sha256(CONTRACT_MANIFEST_PATH.read_bytes()).hexdigest()
HASH = "a" * 64
EVIDENCE_ID = "ev_direct01"


def _evidence(*, supports: str = "POSITIVE", complete: bool = True) -> dict[str, Any]:
    return {
        "evidence_ref_id": EVIDENCE_ID,
        "source_ref": CANARY["source_ref"],
        "source_sha256": CANARY["raw_sha256"],
        "locator_type": "SOURCE_SPAN",
        "locator_ref": "span://LH_005/1",
        "supports": supports,
        "capability": "tool_events",
        "capability_complete": complete,
        "approximate": False,
    }


def _base(annotation_type: str) -> dict[str, Any]:
    suffix = annotation_type.casefold()
    return {
        "schema_version": "eval-factory-annotation/v1",
        "annotation_id": f"ann_{suffix}0001",
        "annotation_type": annotation_type,
        "canary_instance_id": CANARY["instance_id"],
        "source_ref": CANARY["source_ref"],
        "source_sha256": CANARY["raw_sha256"],
        "subject_ref": f"subject://LH_005/{suffix}",
        "subject_version": "v1",
        "subject_sha256": HASH,
        "guide_version": "eval-factory-annotation-guide/v1",
        "contract_manifest_sha256": CONTRACT_MANIFEST_SHA256,
        "governing_versions": {
            "spec": "002-eval-dataset-factory/approved-2026-07-19",
            "trace_adr": "ADR-0009",
            "safety_adr": "ADR-0010",
            "human_review_policy": "eval-factory-human-review/v1",
            "canary_manifest_sha256": ("1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6"),
        },
        "annotator_id": f"annotator-{suffix}",
        "annotator_role": "AUDITOR",
        "status": "SUBMITTED",
        "conclusion": "POSITIVE",
        "evidence": [_evidence()],
        "uncertainty_codes": [],
        "finding_refs": [],
        "rationale": "Bound to exact evidence under the v1 annotation guide.",
        "adjudication": None,
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


def _safety_record() -> dict[str, Any]:
    record = _base("SAFETY")
    record["annotator_role"] = "PROVENANCE_REVIEWER"
    record["payload"] = {
        "subject_decisions": [
            {
                "subject_ref": "file-version://LH_005/input.txt/v0",
                "file_version_stage": "V0_PRE_MUTATION",
                "origin_class": "PREEXISTING_WORKSPACE_INPUT",
                "taint_labels": [],
                "content_risk_labels": [],
                "disposition": "ALLOW_INPUT_EVIDENCE",
                "non_waivable": False,
                "evidence_ref_ids": [EVIDENCE_ID],
            }
        ],
        "known_output_contamination": "ABSENT",
        "answer_leakage": "ABSENT",
        "inventory_provenance_equality": "NOT_EVALUATED",
        "expected_package_disposition": "ALLOW",
        "non_waivable_blockers": [],
    }
    return record


def _label_record() -> dict[str, Any]:
    record = _base("LABEL")
    record["annotator_role"] = "LABEL_REVIEWER"
    record["payload"] = {
        "label_spec_ref": "label-spec://search-tool-usage",
        "label_spec_version": "v1",
        "expected_decision": "MATCH",
        "positive_evidence_ref_ids": [EVIDENCE_ID],
        "negative_evidence_ref_ids": [],
        "semantic_evidence_ref_ids": [],
        "structured_capability_complete": True,
        "semantic_residual": "NOT_REQUIRED",
        "confidence": 1.0,
        "rule_version": "search-tool-rule/v1",
        "model_profile": None,
        "prompt_version": None,
        "review_status": "APPROVED",
    }
    return record


def _task_record() -> dict[str, Any]:
    record = _base("TASK")
    record["annotator_role"] = "TASK_REVIEWER"
    record["payload"] = {
        "task_episode_refs": ["segment://LH_005/1"],
        "visible_prompt": "Inspect the supplied input and create the requested analysis.",
        "task_intent": "Measure evidence-grounded analysis.",
        "evaluation_claim": "The contestant can use the supplied input to produce the analysis.",
        "required_capabilities": ["file_read", "analysis"],
        "allowed_tools": ["file_read"],
        "forbidden_outputs": ["original_agent_final_output"],
        "attachment_dependencies": [],
        "requirement_lineage": [
            {
                "requirement_id": "req-001",
                "statement": "The contestant needs the observed input file.",
                "evidence_priority": "DIRECT_OBSERVATION",
                "evidence_ref_ids": [EVIDENCE_ID],
                "conflict_status": "NONE",
            }
        ],
        "selection_firewall": {
            "final_answer_excluded": True,
            "completed_deliverable_excluded": True,
            "private_reference_excluded": True,
            "grader_rules_excluded": True,
            "hidden_selection_signals_excluded": True,
            "trajectory_specific_steps_excluded": True,
        },
        "task_status": "USABLE",
    }
    return record


def _attachment_record() -> dict[str, Any]:
    record = _base("ATTACHMENT")
    record["annotator_role"] = "ARTIFACT_REVIEWER"
    record["payload"] = {
        "task_mode": "TRACE_RICH",
        "artifacts": [
            {
                "artifact_ref": "artifact://LH_005/input.txt",
                "logical_path": "input.txt",
                "media_type": "text/plain",
                "criticality": "CRITICAL",
                "expected_mode": "TRACE_RICH",
                "path_evidence": "DIRECT",
                "type_evidence": "DIRECT",
                "structure_evidence": "COMPLETE",
                "untainted_content_coverage": "COMPLETE",
                "pre_mutation_evidence": "COMPLETE",
                "provenance_confidence": 1.0,
                "truncation": "NONE",
                "blocking_uncertainties": [],
                "expected_structure": ["UTF-8 text input"],
                "expected_outcome": "BUILD",
                "evidence_ref_ids": [EVIDENCE_ID],
            }
        ],
        "required_artifact_set_complete": True,
        "input_state_only": True,
        "silent_degradation_allowed": False,
        "expected_package_disposition": "BUILD",
    }
    return record


def _component(name: str) -> dict[str, str]:
    return {"ref": f"{name}://LH_005/v1", "sha256": HASH}


def _release_record() -> dict[str, Any]:
    record = _base("RELEASE")
    record["annotator_role"] = "RELEASE_AUTHORITY"
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
        "required_review_roles": ["ITEM_REVIEWER", "RELEASE_AUTHORITY"],
        "review_records_current": True,
        "channel": "CANARY",
        "registry_target": "registry://canary/lh",
        "registry_isolated": True,
        "production_attestation": None,
        "expected_action": "PUBLISH",
        "expected_state": "RELEASED",
        "release_allowed": True,
    }
    return record


def test_all_six_domain_records_validate() -> None:
    records = [
        _parser_record(),
        _safety_record(),
        _label_record(),
        _task_record(),
        _attachment_record(),
        _release_record(),
    ]

    assert validator.validate_records(records) == []


def test_closed_payload_schema_rejects_unknown_field() -> None:
    record = _task_record()
    record["payload"]["undeclared_truth"] = "must fail"

    errors = validator.validate_records([record])

    assert any("Additional properties are not allowed" in error for error in errors)


def test_record_must_bind_to_frozen_canary_hash() -> None:
    record = _parser_record()
    record["source_sha256"] = "b" * 64

    errors = validator.validate_records([record])

    assert any("source_sha256 does not match" in error for error in errors)


def test_contract_manifest_drift_fails_before_record_validation(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    contract["files"][0]["sha256"] = "b" * 64
    drifted = tmp_path / "contract-manifest.json"
    drifted.write_text(json.dumps(contract), encoding="utf-8")

    try:
        validator.load_contract_manifest(drifted)
    except ValueError as exc:
        assert "contract drift" in str(exc)
    else:
        raise AssertionError("contract drift must fail before annotation validation")


def test_negative_label_requires_complete_negative_evidence() -> None:
    record = _label_record()
    record["conclusion"] = "NEGATIVE"
    record["evidence"] = [_evidence(supports="NEGATIVE", complete=False)]
    record["payload"]["expected_decision"] = "NO_MATCH"
    record["payload"]["positive_evidence_ref_ids"] = []
    record["payload"]["negative_evidence_ref_ids"] = [EVIDENCE_ID]
    record["payload"]["structured_capability_complete"] = False

    errors = validator.validate_records([record])

    assert any("NO_MATCH requires complete structured capability" in error for error in errors)
    assert any("must be NEGATIVE and capability-complete" in error for error in errors)


def test_parser_negative_predicate_requires_complete_capability() -> None:
    record = _parser_record()
    record["payload"]["capability_truth"]["tool_events"] = "PARTIAL"

    errors = validator.validate_records([record])

    assert any("negative predicates require COMPLETE parser capabilities" in error for error in errors)


def test_non_waivable_answer_leak_cannot_be_allowed() -> None:
    record = _safety_record()
    decision = record["payload"]["subject_decisions"][0]
    decision["origin_class"] = "AGENT_GENERATED_FINAL"
    decision["taint_labels"] = ["FINAL_OUTPUT_DERIVED"]
    decision["content_risk_labels"] = ["ANSWER_BEARING"]
    decision["non_waivable"] = False

    errors = validator.validate_records([record])

    assert any("must be marked non_waivable" in error for error in errors)
    assert any("cannot allow non-waivable content" in error for error in errors)
    assert any("requires contamination=PRESENT" in error for error in errors)
    assert any("requires answer_leakage=PRESENT" in error for error in errors)
    assert any("package cannot ALLOW" in error for error in errors)


def test_unknown_derivation_must_fail_closed() -> None:
    record = _safety_record()
    decision = record["payload"]["subject_decisions"][0]
    decision["origin_class"] = "UNKNOWN"
    decision["taint_labels"] = ["UNKNOWN_DERIVATION"]

    errors = validator.validate_records([record])

    assert any("must fail closed for unknown or unscannable content" in error for error in errors)


def test_release_cannot_bypass_exact_set_or_open_p0() -> None:
    record = _release_record()
    record["payload"]["package_provenance_exact_set"] = False
    record["payload"]["hard_gate_findings"] = [
        {
            "finding_ref": "finding://LH_005/p0",
            "severity": "P0",
            "status": "OPEN",
            "non_waivable": True,
            "evidence_ref_ids": [EVIDENCE_ID],
        }
    ]

    errors = validator.validate_records([record])

    assert any("exact package/provenance set equality" in error for error in errors)
    assert any("open blocking findings" in error for error in errors)


def test_release_allowed_requires_consistent_action_and_state() -> None:
    record = _release_record()
    record["payload"]["expected_action"] = "REJECT"
    record["payload"]["expected_state"] = "REJECTED"

    errors = validator.validate_records([record])

    assert any("requires APPROVE or PUBLISH action" in error for error in errors)
    assert any("requires APPROVED or RELEASED state" in error for error in errors)


def test_adjudicated_gold_bundle_cannot_be_empty() -> None:
    assert validator.validate_records([], require_adjudicated=True) == ["bundle: gold bundle cannot be empty"]


def test_safety_gold_adjudication_requires_distinct_reviewers() -> None:
    first = _safety_record()
    first["annotation_id"] = "ann_safety_input01"
    first["annotator_id"] = "reviewer-one"
    second = copy.deepcopy(first)
    second["annotation_id"] = "ann_safety_input02"
    second["annotator_id"] = "reviewer-two"
    adjudicated = copy.deepcopy(first)
    adjudicated["annotation_id"] = "ann_safety_gold001"
    adjudicated["annotator_id"] = "gold-assembler"
    adjudicated["status"] = "ADJUDICATED"
    adjudicated["adjudication"] = {
        "adjudicator_id": "reviewer-three",
        "adjudicator_role": "PROVENANCE_REVIEWER",
        "input_annotation_ids": [first["annotation_id"], second["annotation_id"]],
        "decision": "ACCEPT",
        "disagreement_codes": [],
        "review_record_refs": ["review-record://LH_005/safety/1"],
        "reason": "Independent evidence review reached the same version-bound decision.",
        "decided_at": "2026-07-20T00:10:00Z",
    }

    assert (
        validator.validate_records(
            [first, second, adjudicated],
            require_adjudicated=True,
        )
        == []
    )

    adjudicated["adjudication"]["adjudicator_id"] = "reviewer-one"
    errors = validator.validate_records(
        [first, second, adjudicated],
        require_adjudicated=True,
    )
    assert any("adjudicator must differ from every input annotator" in error for error in errors)

    adjudicated["adjudication"]["adjudicator_id"] = "reviewer-three"
    adjudicated["adjudication"]["decision"] = "ABSTAIN"
    errors = validator.validate_records(
        [first, second, adjudicated],
        require_adjudicated=True,
    )
    assert any("requires ACCEPT or REVISE decision" in error for error in errors)
