from __future__ import annotations

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.cross_item_safety_v2 import (
    ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
    AttachmentCrossItemSafetyScanFailureCodeV2,
    AttachmentCrossItemSafetyScanLimitsV2,
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    attachment_cross_item_safety_scan_request_carried_sha256,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_carried_sha256,
    attachment_cross_item_safety_scan_result_ref,
    validate_attachment_cross_item_safety_scan_request_identity,
    validate_attachment_cross_item_safety_scan_result_identity,
)
from env_mock_agent.facade.validation_v2 import (
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    AttachmentValidationFingerprintCategoryV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str = "current",
    *,
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-02/{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _fingerprint(
    suffix: str = "answer",
    *,
    category: AttachmentValidationFingerprintCategoryV2 = (
        AttachmentValidationFingerprintCategoryV2.FINAL_ANSWER
    ),
    match_kind: AttachmentValidationFingerprintMatchKindV2 = (
        AttachmentValidationFingerprintMatchKindV2.TOKEN_WINDOW
    ),
) -> AttachmentValidationFingerprintV2:
    return AttachmentValidationFingerprintV2(
        fingerprint_id=f"prompt-leakage-fingerprint://r7-02/{suffix}",
        category=category,
        match_kind=match_kind,
        digest_sha256="b" * 64,
        token_count=8,
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    )


def _limits() -> AttachmentCrossItemSafetyScanLimitsV2:
    return AttachmentCrossItemSafetyScanLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        max_fingerprint_count=10_000,
        max_match_count=1_000,
    )


def _request() -> AttachmentCrossItemSafetyScanRequestV2:
    pending = AttachmentCrossItemSafetyScanRequestV2(
        scan_request_id="attachment-cross-item-safety-scan-request://pending",
        target_item_id="dataset-item://r7-02/target",
        item_quality_result_ref=_ref("item-quality-compilation-result"),
        environment_artifact_ref=_ref("environment-artifact"),
        candidate_artifact_version_ref=_ref("candidate-artifact-version"),
        output_ref=_ref("attachment-output"),
        artifact_validation_result_ref=_ref("artifact-deterministic-validation-result"),
        logical_path="inputs/source.txt",
        media_type="text/plain",
        content_sha256=HASH,
        size_bytes=128,
        foreign_reference_set_refs=(_ref("prompt-leakage-reference-set", "foreign"),),
        foreign_fingerprints=(_fingerprint(),),
        limits=_limits(),
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
        idempotency_key="attachment-cross-item-safety-idempotency://current",
        request_sha256=HASH,
    )
    digest = attachment_cross_item_safety_scan_request_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_request_id": (f"attachment-cross-item-safety-scan-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _result(
    request: AttachmentCrossItemSafetyScanRequestV2,
    *,
    status: AttachmentCrossItemSafetyScanStatusV2,
) -> AttachmentCrossItemSafetyScanResultV2:
    passed = status is AttachmentCrossItemSafetyScanStatusV2.PASSED
    pending = AttachmentCrossItemSafetyScanResultV2(
        scan_result_id="attachment-cross-item-safety-scan-result://pending",
        scan_request_ref=attachment_cross_item_safety_scan_request_ref(request),
        environment_artifact_ref=request.environment_artifact_ref,
        output_ref=request.output_ref,
        output_sha256=request.content_sha256,
        status=status,
        matched_fingerprint_ids=((request.foreign_fingerprints[0].fingerprint_id,) if passed else ()),
        scanned_member_count=1 if passed else 0,
        scan_complete=passed,
        failure_code=(None if passed else AttachmentCrossItemSafetyScanFailureCodeV2.CONTENT_UNSCANNABLE),
        extractor_version=("attachment-cross-item-safety-extractor/r7-02-v1" if passed else None),
        normalization_version=request.normalization_version,
        policy_version=request.policy_version,
        result_sha256=HASH,
    )
    digest = attachment_cross_item_safety_scan_result_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_result_id": (f"attachment-cross-item-safety-scan-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )


def test_limits_and_request_are_strict_sorted_and_hash_bound() -> None:
    request = _request()

    validate_attachment_cross_item_safety_scan_request_identity(request)
    assert attachment_cross_item_safety_scan_request_ref(request).object_sha256 == request.request_sha256
    assert request.output_ref.object_sha256 == request.content_sha256

    with pytest.raises(ValidationError, match="Extra inputs"):
        AttachmentCrossItemSafetyScanRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "physical_path": "/private/staging/source.txt",
            }
        )
    with pytest.raises(ValidationError, match="environment_artifact_ref"):
        AttachmentCrossItemSafetyScanRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "environment_artifact_ref": _ref("final-package-manifest"),
            }
        )


def test_request_rejects_duplicate_or_unsorted_foreign_inventories() -> None:
    request = _request()
    reference = request.foreign_reference_set_refs[0]
    fingerprint = request.foreign_fingerprints[0]

    with pytest.raises(ValidationError, match="foreign reference sets"):
        AttachmentCrossItemSafetyScanRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "foreign_reference_set_refs": (reference, reference),
            }
        )
    with pytest.raises(ValidationError, match="foreign fingerprints"):
        AttachmentCrossItemSafetyScanRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "foreign_fingerprints": (
                    fingerprint.model_copy(
                        update={"fingerprint_id": ("prompt-leakage-fingerprint://r7-02/z-last")}
                    ),
                    fingerprint,
                ),
            }
        )
    duplicate_id_values = tuple(
        sorted(
            (
                fingerprint,
                fingerprint.model_copy(update={"digest_sha256": "c" * 64}),
            ),
            key=lambda value: (
                value.category.value,
                value.match_kind.value,
                value.digest_sha256,
                value.token_count,
                value.fingerprint_id,
            ),
        )
    )
    with pytest.raises(ValidationError, match="fingerprint IDs"):
        AttachmentCrossItemSafetyScanRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "foreign_fingerprints": duplicate_id_values,
            }
        )


@pytest.mark.parametrize(
    "status",
    (
        AttachmentCrossItemSafetyScanStatusV2.PASSED,
        AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
    ),
)
def test_result_matrix_is_closed_and_content_free(
    status: AttachmentCrossItemSafetyScanStatusV2,
) -> None:
    result = _result(_request(), status=status)

    validate_attachment_cross_item_safety_scan_result_identity(result)
    assert attachment_cross_item_safety_scan_result_ref(result).object_sha256 == result.result_sha256
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "extracted_text",
        "matched_text",
        "physical_path",
        "member_path",
        "raw_bytes",
        "token_inventory",
        "exception",
    ):
        assert denied not in serialized


def test_passed_and_blocked_results_are_all_or_none() -> None:
    request = _request()
    passed = _result(
        request,
        status=AttachmentCrossItemSafetyScanStatusV2.PASSED,
    )
    blocked = _result(
        request,
        status=AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
    )

    with pytest.raises(ValidationError, match="PASSED"):
        AttachmentCrossItemSafetyScanResultV2.model_validate(
            {
                **passed.model_dump(mode="python"),
                "scan_complete": False,
            }
        )
    with pytest.raises(ValidationError, match="BLOCKED"):
        AttachmentCrossItemSafetyScanResultV2.model_validate(
            {
                **blocked.model_dump(mode="python"),
                "matched_fingerprint_ids": (request.foreign_fingerprints[0].fingerprint_id,),
            }
        )


def test_identities_reject_stale_request_and_result_values() -> None:
    request = _request()
    result = _result(
        request,
        status=AttachmentCrossItemSafetyScanStatusV2.PASSED,
    )

    with pytest.raises(ValueError, match="request identity"):
        validate_attachment_cross_item_safety_scan_request_identity(
            request.model_copy(update={"logical_path": "inputs/changed.txt"})
        )
    with pytest.raises(ValueError, match="result identity"):
        validate_attachment_cross_item_safety_scan_result_identity(
            result.model_copy(
                update={
                    "scanned_member_count": result.scanned_member_count + 1,
                }
            )
        )


def test_pure_cross_item_facade_import_is_cold_and_factory_independent() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import env_mock_agent.facade; "
                "import env_mock_agent.facade.cross_item_safety_v2; "
                "assert 'env_mock_agent.facade.cross_item_safety_adapter' "
                "not in sys.modules; "
                "assert not any(name == 'eval_factory' or "
                "name.startswith('eval_factory.') for name in sys.modules); "
                "assert 'anthropic' not in sys.modules; "
                "assert 'claude_agent_sdk' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
