from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from eval_factory.contracts.approval import (
    EnvironmentStrategy,
    LabelPlan,
    TaskRewritePlan,
    UserApprovalPolicy,
    UserApprovalRequest,
)
from eval_factory.contracts.attachment import AttachmentReconstructionResult
from eval_factory.contracts.core import ContractModel
from eval_factory.contracts.core_v2 import canonical_sha256_v2
from eval_factory.contracts.quality import QualityReport
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.safety import ProvenanceManifest
from eval_factory.contracts.task import (
    EnvironmentSpec,
    EvaluatorSpec,
    ProducerTaskView,
    QuerySpec,
    ReferencePolicy,
    RubricSet,
    TaskDraft,
    ToolPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "evals/golden/eval_factory/evaluation_items/v2"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
CANARY_PATH = ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
V1_WORKLIST = ROOT / "evals/golden/eval_factory/evaluation_items/v1/gold-worklist.json"
V1_QUEUE = ROOT / "evals/golden/eval_factory/evaluation_items/v1/review-queue.json"

MODEL_BY_KEY: dict[str, type[ContractModel]] = {
    "attachment_reconstruction_result": AttachmentReconstructionResult,
    "environment_spec": EnvironmentSpec,
    "environment_strategy": EnvironmentStrategy,
    "evaluation_item": EvaluationItemV2,
    "evaluator_spec": EvaluatorSpec,
    "label_plan": LabelPlan,
    "producer_task_view": ProducerTaskView,
    "provenance_manifest": ProvenanceManifest,
    "quality_report": QualityReport,
    "query_spec": QuerySpec,
    "reference_policy": ReferencePolicy,
    "release_decision": ReleaseDecisionV2,
    "rubric_set": RubricSet,
    "task_draft": TaskDraft,
    "task_rewrite_plan": TaskRewritePlan,
    "tool_policy": ToolPolicy,
    "user_approval_policy": UserApprovalPolicy,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_fixture_generator_is_reproducible() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/prepare_eval_factory_reference_fixtures_v2.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_manifest_binds_five_canaries_and_supersedes_v1_queue() -> None:
    manifest = _load(MANIFEST_PATH)
    canaries = _load(CANARY_PATH)
    source_hashes = {item["instance_id"]: item["raw_sha256"] for item in canaries["items"]}

    assert manifest["status"] == "REVIEW_READY_NOT_GOLD"
    assert manifest["fixture_count"] == 5
    assert {entry["instance_id"] for entry in manifest["fixtures"]} == {
        "LH_005",
        "LH_011",
        "LH_015",
        "LH_058",
        "LH_067",
    }
    assert manifest["legacy_v1_review_workflow"] == {
        "status": "SUPERSEDED_NOT_SCHEDULED",
        "superseded_by": "env_mock_agent-ujc.1.13",
        "worklist": {
            "path": "evals/golden/eval_factory/evaluation_items/v1/gold-worklist.json",
            "sha256": _sha256(V1_WORKLIST),
        },
        "review_queue": {
            "path": "evals/golden/eval_factory/evaluation_items/v1/review-queue.json",
            "sha256": _sha256(V1_QUEUE),
        },
    }
    for entry in manifest["fixtures"]:
        fixture = _load(FIXTURE_ROOT / entry["path"])
        assert fixture["source"]["source_sha256"] == source_hashes[entry["instance_id"]]
        assert entry["sha256"] == _sha256(FIXTURE_ROOT / entry["path"])


def test_every_fixture_is_contract_complete_hash_bound_and_not_gold() -> None:
    manifest = _load(MANIFEST_PATH)
    for entry in manifest["fixtures"]:
        fixture = _load(FIXTURE_ROOT / entry["path"])
        objects = fixture["objects"]

        assert fixture["status"] == "REVIEW_READY_NOT_GOLD"
        assert fixture["semantic_gold"] is False
        assert fixture["user_approved"] is False
        assert fixture["user_decision_records"] == []
        assert fixture["required_evaluation_components"] == [
            "QuerySpec",
            "EnvironmentSpec",
            "RubricSet",
            "EvaluatorSpec",
            "ReferencePolicy",
            "ToolPolicy",
            "ProvenanceManifest",
            "QualityReport",
            "ReleaseDecisionV2",
        ]

        for key, model_type in MODEL_BY_KEY.items():
            value = model_type.model_validate_json(json.dumps(objects[key]))
            assert canonical_sha256_v2(value) == fixture["object_sha256"][key]

        request_keys = sorted(key for key in objects if key.startswith("approval_request_"))
        assert len(request_keys) == 4
        for key in request_keys:
            request = UserApprovalRequest.model_validate_json(json.dumps(objects[key]))
            assert canonical_sha256_v2(request) == fixture["object_sha256"][key]

        quality = QualityReport.model_validate_json(json.dumps(objects["quality_report"]))
        release = ReleaseDecisionV2.model_validate_json(json.dumps(objects["release_decision"]))
        assert quality.approvable is False
        assert release.state is ReleaseStateV2.CANDIDATE
        assert release.automated_quality_passed is False
        assert release.checkpoint_decisions == ()


def test_active_fixtures_contain_no_independent_review_workflow() -> None:
    serialized = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FIXTURE_ROOT.glob("*.reference-fixture.json"))
    ).casefold()
    for forbidden in (
        "humanreviewrecord",
        "human_review_record",
        "reviewquorum",
        "review_quorum",
        '"adjudication"',
        '"reviewer_id"',
        '"assignee"',
        '"claim_lease"',
    ):
        assert forbidden not in serialized
