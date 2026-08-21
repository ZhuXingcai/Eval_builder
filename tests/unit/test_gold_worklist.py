from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKLIST_PATH = ROOT / "evals/golden/eval_factory/evaluation_items/v1/gold-worklist.json"
CANARY_PATH = ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
ANNOTATION_CONTRACT_PATH = ROOT / "evals/annotation/v1/contract-manifest.json"
REVIEW_QUEUE_PATH = ROOT / "evals/golden/eval_factory/evaluation_items/v1/review-queue.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_gold_worklist_is_bound_and_explicitly_not_gold() -> None:
    worklist = _load(WORKLIST_PATH)

    assert worklist["status"] == "PREPARED_NOT_GOLD"
    assert worklist["cross_stage_contract"] == {
        "beads_id": "env_mock_agent-ujc.1.10",
        "status": "FROZEN",
        "manifest_path": "specs/002-eval-dataset-factory/contracts/v1/manifest.json",
        "manifest_sha256": ("d550d84262d3f727394c65e52ec0bf6e41fdaf532bfde0851c72e6d1495d59a9"),
        "required_before_gold_materialization": True,
    }
    assert worklist["cross_stage_contract"]["manifest_sha256"] == _sha256(
        ROOT / worklist["cross_stage_contract"]["manifest_path"]
    )
    assert worklist["canary_manifest"]["sha256"] == _sha256(CANARY_PATH)
    assert worklist["annotation_contract"]["sha256"] == _sha256(ANNOTATION_CONTRACT_PATH)
    assert worklist["review_policy"]["synthetic_reviewer_identities_forbidden"] is True


def test_gold_worklist_uses_five_unique_frozen_canaries() -> None:
    worklist = _load(WORKLIST_PATH)
    canary_manifest = _load(CANARY_PATH)
    canaries = {item["instance_id"]: item for item in canary_manifest["items"]}
    items = worklist["items"]
    ids = [item["instance_id"] for item in items]

    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert set(ids) == {"LH_005", "LH_011", "LH_015", "LH_058", "LH_067"}
    for item in items:
        canary = canaries[item["instance_id"]]
        assert item["source_ref"] == canary["source_ref"]
        assert item["source_sha256"] == canary["raw_sha256"]


def test_gold_worklist_covers_required_truth_surfaces() -> None:
    worklist = _load(WORKLIST_PATH)
    roles = {item["selection_role"] for item in worklist["items"]}
    focus = {value for item in worklist["items"] for value in item["required_focus"]}

    assert roles == {
        "STRICT_LARGE_PRE_POST_MUTATION",
        "MALFORMED_REQUEST_AND_RESPONSE",
        "SEARCH_AND_FETCH",
        "POWERSHELL_ERROR",
        "INVALID_UNICODE_AND_NO_READ_CANDIDATE",
    }
    assert {
        "search-tool structured label",
        "PowerShell call/result identity",
        "final-output quarantine",
        "no-read candidate adjudication",
        "abstain when capability is incomplete",
    } <= focus
    assert worklist["required_evaluation_components"] == [
        "QuerySpec",
        "EnvironmentSpec",
        "RubricSet",
        "EvaluatorSpec",
        "ReferencePolicy",
        "ToolPolicy",
        "ProvenanceManifest",
        "QualityReport",
        "ReleaseDecision",
    ]


def test_review_queue_has_complete_human_slots_without_raw_content() -> None:
    queue = _load(REVIEW_QUEUE_PATH)
    cases = queue["cases"]
    assert queue["status"] == "AWAITING_HUMAN_ASSIGNMENT"
    assert queue["case_count"] == 70 == len(cases)
    assert queue["worklist"]["sha256"] == _sha256(WORKLIST_PATH)
    assert queue["synthetic_reviewer_identities_forbidden"] is True
    assert all(case["synthetic_identity_allowed"] is False for case in cases)
    assert all(case["assignee"] is None and case["status"] == "PENDING" for case in cases)

    for instance_id in {"LH_005", "LH_011", "LH_015", "LH_058", "LH_067"}:
        item_cases = [case for case in cases if case["instance_id"] == instance_id]
        phases = Counter(case["phase"] for case in item_cases)
        assert phases == {"SUBMISSION": 8, "ADJUDICATION": 6}
        submissions = Counter(case["annotation_type"] for case in item_cases if case["phase"] == "SUBMISSION")
        assert submissions["SAFETY"] == 2
        assert submissions["RELEASE"] == 2
        release = next(
            case
            for case in item_cases
            if case["annotation_type"] == "RELEASE" and case["phase"] == "ADJUDICATION"
        )
        assert len(release["depends_on"]) == 7

    serialized = REVIEW_QUEUE_PATH.read_text(encoding="utf-8").casefold()
    for forbidden in ('"prompt"', '"response"', '"raw_text"', '"content"'):
        assert forbidden not in serialized
