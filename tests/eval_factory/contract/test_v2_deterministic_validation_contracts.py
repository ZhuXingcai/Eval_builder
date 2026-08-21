from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    DeterministicItemValidationOutcomeV2,
    DeterministicItemValidationResultV2,
    deterministic_item_validation_result_carried_sha256,
    deterministic_item_validation_result_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(*refs: ObjectRef, at: datetime | None = None) -> ContractAudit:
    return ContractAudit(
        created_at=at or datetime(2026, 7, 28, tzinfo=UTC),
        created_by="deterministic-validation-contract-test",
        governing_versions=(
            VersionBinding(
                component="deterministic-validation",
                version="r5-08",
            ),
        ),
        input_refs=refs,
    )


def _not_required(
    *,
    at: datetime | None = None,
) -> DeterministicItemValidationResultV2:
    reconstruction_ref = _ref("attachment-reconstruction-result", "none")
    producer_ref = _ref("producer-task-view", "current")
    leakage_ref = _ref("prompt-leakage-reference-set", "current")
    value = DeterministicItemValidationResultV2(
        deterministic_item_validation_result_id=("deterministic-item-validation-result://pending"),
        attachment_reconstruction_result_ref=reconstruction_ref,
        producer_task_view_ref=producer_ref,
        leakage_reference_set_ref=leakage_ref,
        artifact_validation_results=(),
        artifact_validation_result_refs=(),
        finding_refs=(),
        candidate_inventory=None,
        candidate_inventory_ref=None,
        outcome=DeterministicItemValidationOutcomeV2.NOT_REQUIRED,
        validated_artifact_ids=(),
        repair_required_artifact_ids=(),
        rejected_artifact_ids=(),
        validation_blocked_artifact_ids=(),
        upstream_incomplete_artifact_ids=(),
        accepted_artifact_refs=(),
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version="deterministic-validation/r5-08-v1",
        deterministic_item_validation_result_sha256=HASH,
        audit=_audit(
            reconstruction_ref,
            producer_ref,
            leakage_ref,
            at=at,
        ),
    )
    digest = deterministic_item_validation_result_carried_sha256(value)
    return value.model_copy(
        update={
            "deterministic_item_validation_result_id": (
                f"deterministic-item-validation-result://sha256/{digest}"
            ),
            "deterministic_item_validation_result_sha256": digest,
        }
    )


def test_not_required_result_is_strict_pre_semantic_handoff() -> None:
    result = _not_required()

    assert result.outcome is DeterministicItemValidationOutcomeV2.NOT_REQUIRED
    assert result.accepted_artifact_refs == ()
    assert result.environment_spec_ref is None
    assert result.provenance_manifest_ref is None
    assert result.quality_report_ref is None
    assert result.package_sha256 is None
    assert result.input_state_only is None
    assert deterministic_item_validation_result_ref(result).object_sha256 == (
        result.deterministic_item_validation_result_sha256
    )

    values = result.model_dump(mode="python")
    values["package_sha256"] = HASH
    with pytest.raises(ValidationError, match="package"):
        DeterministicItemValidationResultV2.model_validate(values)

    values = result.model_dump(mode="python")
    values["raw_finding_text"] = "forbidden diagnostic"
    with pytest.raises(ValidationError, match="extra"):
        DeterministicItemValidationResultV2.model_validate(values)


def test_item_validation_identity_ignores_audit_actor_and_time() -> None:
    first = _not_required(at=datetime(2026, 7, 27, tzinfo=UTC))
    second = _not_required(at=datetime(2026, 7, 29, tzinfo=UTC))
    second = second.model_copy(
        update={"audit": second.audit.model_copy(update={"created_by": "another-validator"})}
    )

    assert deterministic_item_validation_result_carried_sha256(first) == (
        deterministic_item_validation_result_carried_sha256(second)
    )
    assert first.canonical_sha256() != second.canonical_sha256()
