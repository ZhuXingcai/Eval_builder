from __future__ import annotations

import ast
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import AttachmentReconstructionRequest
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.labeling import LabelDecision, LabelDecisionValue, ReviewStatus
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    DatasetJobSpec,
    ExportTarget,
    ResourceBudget,
    StageName,
)
from eval_factory.contracts.quality import QualityReport
from eval_factory.contracts.release import (
    CompatibilityDeclaration,
    CompatibilityImpact,
    EvaluationItemComponents,
    ProductionReadinessAttestation,
    ReleaseAction,
    ReleaseChannel,
    ReleaseDecision,
    ReleaseState,
)
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)
from eval_factory.contracts.task import ProducerTaskView

ROOT = Path(__file__).resolve().parents[3]
HASH = "a" * 64


def _ref(name: str) -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://example/v1",
        object_version="v1",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        created_by="contract-test",
        governing_versions=(VersionBinding(component="spec", version="approved-v1", sha256=HASH),),
    )


def _components() -> EvaluationItemComponents:
    return EvaluationItemComponents(
        query_spec_ref=_ref("query-spec"),
        environment_spec_ref=_ref("environment-spec"),
        rubric_set_ref=_ref("rubric-set"),
        evaluator_spec_ref=_ref("evaluator-spec"),
        reference_policy_ref=_ref("reference-policy"),
        tool_policy_ref=_ref("tool-policy"),
        provenance_manifest_ref=_ref("provenance-manifest"),
        quality_report_ref=_ref("quality-report"),
    )


def test_contracts_are_closed_frozen_and_versioned() -> None:
    reference = _ref("query-spec")
    assert reference.model_config["extra"] == "forbid"
    assert reference.model_config["frozen"] is True
    with pytest.raises(ValidationError):
        ObjectRef.model_validate(
            {
                **reference.model_dump(mode="json"),
                "schema_version": "eval-factory/object-ref/v2",
            }
        )
    with pytest.raises(ValidationError):
        ObjectRef.model_validate(
            {
                **reference.model_dump(mode="json"),
                "unexpected": "not allowed",
            }
        )


def test_canonical_hash_is_stable_across_mapping_order() -> None:
    first = ObjectRef.model_validate(
        {
            "schema_version": "eval-factory/object-ref/v1",
            "object_type": "query-spec",
            "object_id": "query-spec://example/v1",
            "object_version": "v1",
            "object_sha256": HASH,
        }
    )
    second = ObjectRef.model_validate(
        {
            "object_sha256": HASH,
            "object_version": "v1",
            "object_id": "query-spec://example/v1",
            "object_type": "query-spec",
            "schema_version": "eval-factory/object-ref/v1",
        }
    )
    assert first.canonical_json() == second.canonical_json()
    assert first.canonical_sha256() == second.canonical_sha256()


def test_stale_object_reference_hash_fails_closed() -> None:
    value = _ref("query-spec")
    stale = ObjectRef(
        object_type="object-ref",
        object_id="object-ref://stale/v1",
        object_version="v1",
        object_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="stale or mismatched object reference"):
        stale.require_matching_hash(value)


def test_label_no_match_requires_complete_negative_evidence() -> None:
    with pytest.raises(ValidationError, match="NO_MATCH requires complete structured capability"):
        LabelDecision(
            label_decision_id="label-decision://example/v1",
            label_spec_ref=_ref("label-spec"),
            trace_envelope_ref=_ref("trace-envelope"),
            decision=LabelDecisionValue.NO_MATCH,
            structured_capability_complete=False,
            confidence=0.9,
            review_status=ReviewStatus.REVIEW_REQUIRED,
            audit=_audit(),
        )


def test_non_waivable_provenance_cannot_be_allowed() -> None:
    with pytest.raises(ValidationError, match="cannot be allowed"):
        ProvenanceDecision(
            provenance_decision_id="provenance-decision://example/v1",
            subject_ref=_ref("file-version"),
            origin_class=OriginClass.AGENT_GENERATED_FINAL,
            taint_labels=frozenset({TaintLabel.FINAL_OUTPUT_DERIVED}),
            content_risk_labels=frozenset({ContentRiskLabel.ANSWER_BEARING}),
            visibility=Visibility.CONTESTANT_VISIBLE,
            disposition=Disposition.ALLOW_INPUT_EVIDENCE,
            rule_ids=("final-output-rule/v1",),
            confidence=1.0,
            review_required=True,
            policy_version="safety/v1",
            subject_sha256=HASH,
            audit=_audit(),
        )


def test_quality_report_with_p0_cannot_be_approvable() -> None:
    with pytest.raises(ValidationError, match="cannot be approvable"):
        QualityReport(
            quality_report_id="quality-report://example/v1",
            item_subject_refs=(_ref("query-spec"),),
            deterministic_validation_refs=(_ref("validation"),),
            semantic_round_refs=(
                _ref("review-round-1"),
                _ref("review-round-2"),
                _ref("review-round-3"),
            ),
            finding_refs=(_ref("finding"),),
            open_p0_count=1,
            open_p1_count=0,
            unresolved_non_waivable_count=0,
            approvable=True,
            audit=_audit(),
        )


def test_release_channel_and_state_are_bound() -> None:
    base: dict[str, Any] = {
        "release_decision_id": "release-decision://example/v1",
        "chain_id": "release-chain://example",
        "previous_decision_ref": None,
        "item_id": "item://example",
        "item_version": "v1",
        "components": _components(),
        "release_subject_sha256": HASH,
        "package_manifest_ref": _ref("package-manifest"),
        "package_sha256": HASH,
        "batch_quality_report_ref": None,
        "human_review_record_refs": (_ref("human-review-record"),),
        "review_policy_versions": ("human-review/v1",),
        "registry": "registry://canary",
        "export_profile": "LH",
        "export_profile_version": "v1",
        "idempotency_key": "release-example-v1",
        "actor": "release-authority",
        "decided_at": datetime.now(UTC),
        "decision_sha256": HASH,
        "audit": _audit(),
    }
    with pytest.raises(ValidationError, match="requires readiness attestation"):
        ReleaseDecision(
            **base,
            channel=ReleaseChannel.PRODUCTION,
            production_attestation_ref=None,
            action=ReleaseAction.PUBLISH,
            state=ReleaseState.RELEASED,
        )
    with pytest.raises(ValidationError, match="PUBLISH action must create RELEASED"):
        ReleaseDecision(
            **base,
            channel=ReleaseChannel.CANARY,
            production_attestation_ref=None,
            action=ReleaseAction.PUBLISH,
            state=ReleaseState.APPROVED,
        )


def test_attestation_window_must_be_ordered() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="expiry must follow activation"):
        ProductionReadinessAttestation(
            attestation_id="attestation://example/v1",
            system_version="v1",
            contract_manifest_ref=_ref("contract-manifest"),
            policy_refs=(_ref("policy"),),
            schema_refs=(_ref("schema"),),
            statistical_evidence_ref=_ref("statistics"),
            safety_evidence_ref=_ref("safety"),
            privacy_evidence_ref=_ref("privacy"),
            stability_evidence_ref=_ref("stability"),
            operations_evidence_ref=_ref("operations"),
            approved_by=("approver-one", "approver-two"),
            valid_from=now,
            valid_until=now - timedelta(seconds=1),
            attestation_sha256=HASH,
        )


def test_producer_view_schema_excludes_evaluation_internals() -> None:
    fields = ProducerTaskView.model_fields
    forbidden = {
        "evaluation_claim",
        "rubric_set",
        "rubric_set_ref",
        "evaluator_spec",
        "evaluator_spec_ref",
        "reference_policy",
        "reference_policy_ref",
        "selection_signals",
        "requirement_lineage",
    }
    assert forbidden.isdisjoint(fields)


def test_env_mock_agent_never_imports_eval_factory() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src/env_mock_agent").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "eval_factory" or name.startswith("eval_factory.") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_eval_factory_cross_context_imports_only_facade() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "src/eval_factory").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.startswith("env_mock_agent") and not name.startswith("env_mock_agent.facade"):
                    offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []


def test_factory_contract_import_does_not_load_physical_runtime_or_vendor_sdk() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import eval_factory.contracts.attachment_v2; "
                "assert 'claude_agent_sdk' not in sys.modules; "
                "assert 'anthropic' not in sys.modules; "
                "assert not any(name.startswith('env_mock_agent.runtimes') "
                "for name in sys.modules)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_generated_contract_manifest_is_current() -> None:
    process = subprocess.run(
        [sys.executable, "scripts/export_eval_factory_contracts.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest_path = ROOT / "specs/002-eval-dataset-factory/contracts/v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["contracts"]) >= 70
    assert {item["owner"] for item in manifest["contracts"]} == {"factory", "facade"}
    for contract in manifest["contracts"]:
        schema_path = manifest_path.parent / contract["schema_path"]
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False


def test_built_wheel_contains_both_bounded_contexts(tmp_path: Path) -> None:
    process = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "env_mock_agent/facade/contracts.py" in names
    assert "env_mock_agent/py.typed" in names
    assert "eval_factory/contracts/core.py" in names
    assert "eval_factory/contracts/checkpoint_interaction_v2.py" in names
    assert "eval_factory/contracts/statistics_v2.py" in names
    assert "eval_factory/approval/interaction.py" in names
    assert "eval_factory/approval/interaction_store.py" in names
    assert "eval_factory/statistics/freeze.py" in names
    assert "eval_factory/statistics/material_store.py" in names
    assert "eval_factory/statistics/persistence.py" in names
    assert "eval_factory/py.typed" in names


def test_facade_rejects_factory_schema_version() -> None:
    with pytest.raises(ValidationError):
        AttachmentReconstructionRequest.model_validate(
            {
                "schema_version": "eval-factory/attachment-reconstruction-request/v1",
                "request_id": "request://example/v1",
                "task_id": "task://example/v1",
                "producer_task_view_ref": {
                    "object_type": "producer-task-view",
                    "object_id": "producer-task-view://example/v1",
                    "object_version": "v1",
                    "object_sha256": HASH,
                },
                "artifacts": [],
                "profile": "LH",
                "profile_version": "v1",
                "idempotency_key": "request-example-v1",
            }
        )


def test_job_stage_dag_must_begin_with_trace_index_and_remain_ordered() -> None:
    job: dict[str, Any] = {
        "job_id": "job://example/v1",
        "traces": (
            {
                "source_trace_id": "source-trace://example",
                "source_uri": "raw-traj://example",
                "raw_sha256": HASH,
                "adapter_name": "raw-traj-v1",
                "adapter_version": "v1",
                "processing_class": "RESTRICTED_TRACE_RAW",
            },
        ),
        "requested_stages": (StageName.LABEL, StageName.TRACE_INDEX),
        "privacy_profile": "trusted-monitored-local",
        "model_profiles": (),
        "budget": ResourceBudget(
            max_model_requests=0,
            max_model_tokens=0,
            max_processes=1,
            max_renderers=1,
            max_network_requests=0,
            max_storage_bytes=1024,
        ),
        "concurrency": ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=1,
            items=1,
        ),
        "export_target": ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://canary",
        ),
        "idempotency_key": "job-example-v1",
        "audit": _audit(),
    }
    with pytest.raises(ValidationError, match="must begin with trace_index"):
        DatasetJobSpec.model_validate(job)


def test_compatibility_declaration_fails_closed_without_evidence_or_migration() -> None:
    common: dict[str, Any] = {
        "declaration_id": "compatibility://query-spec/v1-v2",
        "changed_contract": "query-spec",
        "prior_version": "v1",
        "new_version": "v2",
        "affected_object_types": ("QuerySpec", "EvaluationItem"),
        "approved_review_record_refs": (_ref("human-review-record"),),
        "declaration_sha256": HASH,
    }
    with pytest.raises(ValidationError, match="NO_EFFECT requires compatibility evidence"):
        CompatibilityDeclaration(
            **common,
            impact=CompatibilityImpact.NO_EFFECT,
            migration_ref=None,
        )
    with pytest.raises(ValidationError, match="requires migration/invalidation procedure"):
        CompatibilityDeclaration(
            **common,
            impact=CompatibilityImpact.INVALIDATE,
            migration_ref=None,
        )
