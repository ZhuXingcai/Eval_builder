from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.validation_v2 import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    AttachmentInventoryMemberTypeV2,
    AttachmentInventoryMemberV2,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingCategoryV2,
    AttachmentValidationFindingCodeV2,
    AttachmentValidationFindingV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    AttachmentValidationSeverityV2,
    AttachmentValidationStatusV2,
    attachment_inventory_member_carried_sha256,
    attachment_inventory_member_ref,
    attachment_validation_finding_carried_sha256,
    attachment_validation_finding_ref,
    attachment_validation_request_carried_sha256,
    attachment_validation_request_ref,
    attachment_validation_result_carried_sha256,
    attachment_validation_result_ref,
    validate_attachment_validation_request_identity,
    validate_attachment_validation_result_identity,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _request() -> AttachmentValidationRequestV2:
    request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id="artifact://input",
        artifact_result_ref=_ref("artifact-build-result", "input"),
        execution_request_ref=_ref("attachment-execution-request", "input"),
        execution_result_ref=_ref("attachment-execution-result", "input"),
        build_spec_ref=_ref("artifact-build-spec", "input"),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        output_ref=_ref("attachment-output", "input"),
        output_sha256=HASH,
        logical_path="inputs/source.txt",
        media_type="text/plain",
        declared_validator_ids=("text-validator",),
        configured_pii_rules=(),
        leakage_reference_set_ref=_ref(
            "prompt-leakage-reference-set",
            "current",
        ),
        leakage_fingerprints=(),
        complete_leakage_categories=(),
        forbidden_output_values=("original final answer",),
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        idempotency_key="attachment-validation-idempotency://input",
        validation_request_sha256=HASH,
    )
    digest = attachment_validation_request_carried_sha256(request)
    return request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )


def _member(output_ref: FacadeObjectRef) -> AttachmentInventoryMemberV2:
    member = AttachmentInventoryMemberV2(
        inventory_member_id="attachment-inventory-member://pending",
        artifact_id="artifact://input",
        output_ref=output_ref,
        normalized_path="inputs/source.txt",
        member_type=AttachmentInventoryMemberTypeV2.FILE,
        media_type="text/plain",
        size_bytes=12,
        content_sha256=HASH,
        container_ref=None,
        inventory_member_sha256=HASH,
    )
    digest = attachment_inventory_member_carried_sha256(member)
    return member.model_copy(
        update={
            "inventory_member_id": (f"attachment-inventory-member://sha256/{digest}"),
            "inventory_member_sha256": digest,
        }
    )


def _finding(output_ref: FacadeObjectRef) -> AttachmentValidationFindingV2:
    finding = AttachmentValidationFindingV2(
        finding_id="attachment-validation-finding://pending",
        artifact_id="artifact://input",
        subject_output_ref=output_ref,
        inventory_member_id=None,
        severity=AttachmentValidationSeverityV2.P0,
        category=AttachmentValidationFindingCategoryV2.SECURITY,
        code=AttachmentValidationFindingCodeV2.SECRET_DETECTED,
        rule_ids=("builtin-secret/assigned-secret/v1",),
        matched_fingerprint_ids=(),
        non_waivable=True,
        finding_sha256=HASH,
    )
    digest = attachment_validation_finding_carried_sha256(finding)
    return finding.model_copy(
        update={
            "finding_id": f"attachment-validation-finding://sha256/{digest}",
            "finding_sha256": digest,
        }
    )


def _result(
    request: AttachmentValidationRequestV2,
    *,
    status: AttachmentValidationStatusV2,
) -> AttachmentValidationResultV2:
    finding = _finding(request.output_ref)
    result = AttachmentValidationResultV2(
        validation_result_id="attachment-validation-result://pending",
        validation_request_ref=attachment_validation_request_ref(request),
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        output_sha256=request.output_sha256,
        status=status,
        executed_validator_ids=(
            "common",
            "configured-pii",
            "metadata",
            "package-inventory",
            "restricted-fingerprint",
            "secrets",
            "text",
        ),
        required_validator_ids=(
            "common",
            "configured-pii",
            "metadata",
            "package-inventory",
            "restricted-fingerprint",
            "secrets",
            "text",
        ),
        complete_leakage_categories=(),
        findings=(finding,) if status is AttachmentValidationStatusV2.FAILED else (),
        inventory_members=(_member(request.output_ref),),
        inventory_complete=True,
        scan_complete=True,
        failure_code=None,
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        validation_result_sha256=HASH,
    )
    digest = attachment_validation_result_carried_sha256(result)
    return result.model_copy(
        update={
            "validation_result_id": (f"attachment-validation-result://sha256/{digest}"),
            "validation_result_sha256": digest,
        }
    )


def test_validation_request_is_strict_hash_bound_and_content_minimized() -> None:
    request = _request()

    validate_attachment_validation_request_identity(request)
    assert attachment_validation_request_ref(request).object_sha256 == (request.validation_request_sha256)

    serialized = request.model_dump(mode="json")
    for denied in (
        "artifact_path",
        "extracted_text",
        "private_reference",
        "runtime_transcript",
    ):
        assert denied not in serialized

    with pytest.raises(ValidationError, match="extra"):
        AttachmentValidationRequestV2.model_validate(
            {
                **serialized,
                "artifact_path": "/private/staging/source.txt",
            }
        )


def test_validation_finding_policy_is_closed_and_non_waivable() -> None:
    finding = _finding(_ref("attachment-output", "input"))

    assert attachment_validation_finding_ref(finding).object_sha256 == (finding.finding_sha256)
    assert finding.non_waivable is True

    values = finding.model_dump(mode="python")
    values["non_waivable"] = False
    with pytest.raises(ValidationError, match="finding policy"):
        AttachmentValidationFindingV2.model_validate(values)


def test_inventory_member_enforces_type_hash_and_container_matrix() -> None:
    member = _member(_ref("attachment-output", "input"))

    assert attachment_inventory_member_ref(member).object_sha256 == (member.inventory_member_sha256)

    values = member.model_dump(mode="python")
    values.update(
        member_type=AttachmentInventoryMemberTypeV2.DIRECTORY,
        size_bytes=0,
    )
    with pytest.raises(ValidationError, match="directory"):
        AttachmentInventoryMemberV2.model_validate(values)


def test_validation_result_enforces_complete_status_matrix() -> None:
    request = _request()
    passed = _result(request, status=AttachmentValidationStatusV2.PASSED)

    validate_attachment_validation_result_identity(passed)
    assert attachment_validation_result_ref(passed).object_sha256 == (passed.validation_result_sha256)

    with pytest.raises(ValidationError, match="PASSED"):
        AttachmentValidationResultV2.model_validate(
            {
                **passed.model_dump(mode="python"),
                "findings": (_finding(request.output_ref),),
            }
        )

    blocked_values = passed.model_dump(mode="python")
    blocked_values.update(
        status=AttachmentValidationStatusV2.BLOCKED,
        inventory_members=(),
        inventory_complete=False,
        scan_complete=False,
        failure_code=AttachmentValidationFailureCodeV2.OUTPUT_HASH_MISMATCH,
    )
    blocked = AttachmentValidationResultV2.model_validate(blocked_values)
    assert blocked.status is AttachmentValidationStatusV2.BLOCKED

    serialized = json.dumps(blocked.model_dump(mode="json"), sort_keys=True)
    assert "/private/" not in serialized
    assert "exception" not in serialized
