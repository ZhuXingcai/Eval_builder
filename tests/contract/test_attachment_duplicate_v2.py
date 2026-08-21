from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.duplicate_v2 import (
    ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
    ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
    AttachmentDuplicateFingerprintFailureCodeV2,
    AttachmentDuplicateFingerprintLimitsV2,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
    attachment_duplicate_fingerprint_request_carried_sha256,
    attachment_duplicate_fingerprint_request_ref,
    attachment_duplicate_fingerprint_result_carried_sha256,
    attachment_duplicate_fingerprint_result_ref,
    validate_attachment_duplicate_fingerprint_request_identity,
    validate_attachment_duplicate_fingerprint_result_identity,
)

HASH = "a" * 64
FINGERPRINT = "5" * 64


def _ref(
    object_type: str,
    suffix: str = "current",
    *,
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-01/{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _limits() -> AttachmentDuplicateFingerprintLimitsV2:
    return AttachmentDuplicateFingerprintLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        min_token_count=4,
        shingle_size=3,
        fingerprint_bits=256,
    )


def _request() -> AttachmentDuplicateFingerprintRequestV2:
    pending = AttachmentDuplicateFingerprintRequestV2(
        fingerprint_request_id="attachment-duplicate-fingerprint-request://pending",
        item_quality_result_ref=_ref("item-quality-compilation-result"),
        environment_artifact_ref=_ref("environment-artifact"),
        candidate_artifact_version_ref=_ref("candidate-artifact-version"),
        output_ref=_ref("attachment-output"),
        artifact_validation_result_ref=_ref("artifact-deterministic-validation-result"),
        logical_path="inputs/source.txt",
        media_type="text/plain",
        content_sha256=HASH,
        size_bytes=128,
        limits=_limits(),
        normalization_version=ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
        idempotency_key="attachment-duplicate-fingerprint-idempotency://current",
        request_sha256=HASH,
    )
    digest = attachment_duplicate_fingerprint_request_carried_sha256(pending)
    return pending.model_copy(
        update={
            "fingerprint_request_id": (f"attachment-duplicate-fingerprint-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _result(
    request: AttachmentDuplicateFingerprintRequestV2,
    *,
    outcome: AttachmentDuplicateFingerprintOutcomeV2,
) -> AttachmentDuplicateFingerprintResultV2:
    if outcome is AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED:
        fingerprint = FINGERPRINT
        token_count = 32
        shingle_count = 30
        failure_code = None
        extractor_version = "attachment-duplicate-extractor/r7-01-v1"
    elif outcome is AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED:
        fingerprint = None
        token_count = 0
        shingle_count = 0
        failure_code = AttachmentDuplicateFingerprintFailureCodeV2.MEDIA_UNSUPPORTED
        extractor_version = None
    else:
        fingerprint = None
        token_count = 0
        shingle_count = 0
        failure_code = AttachmentDuplicateFingerprintFailureCodeV2.OUTPUT_HASH_MISMATCH
        extractor_version = None
    pending = AttachmentDuplicateFingerprintResultV2(
        fingerprint_result_id="attachment-duplicate-fingerprint-result://pending",
        fingerprint_request_ref=attachment_duplicate_fingerprint_request_ref(request),
        environment_artifact_ref=request.environment_artifact_ref,
        output_ref=request.output_ref,
        exact_content_sha256=request.content_sha256,
        outcome=outcome,
        similarity_fingerprint=fingerprint,
        token_count=token_count,
        shingle_count=shingle_count,
        failure_code=failure_code,
        extractor_version=extractor_version,
        normalization_version=request.normalization_version,
        policy_version=request.policy_version,
        result_sha256=HASH,
    )
    digest = attachment_duplicate_fingerprint_result_carried_sha256(pending)
    return pending.model_copy(
        update={
            "fingerprint_result_id": (f"attachment-duplicate-fingerprint-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )


def test_limits_and_request_are_strict_explicit_and_hash_bound() -> None:
    request = _request()

    validate_attachment_duplicate_fingerprint_request_identity(request)
    assert attachment_duplicate_fingerprint_request_ref(request).object_sha256 == request.request_sha256
    assert request.limits.fingerprint_bits == 256
    assert request.output_ref.object_sha256 == request.content_sha256

    with pytest.raises(ValidationError, match="Extra inputs"):
        AttachmentDuplicateFingerprintRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "physical_path": "/private/staging/source.txt",
            }
        )
    with pytest.raises(ValidationError, match="environment_artifact_ref"):
        AttachmentDuplicateFingerprintRequestV2.model_validate(
            {
                **request.model_dump(mode="python"),
                "environment_artifact_ref": _ref("final-package-manifest"),
            }
        )


@pytest.mark.parametrize(
    "outcome",
    (
        AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED,
        AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED,
        AttachmentDuplicateFingerprintOutcomeV2.BLOCKED,
    ),
)
def test_result_outcome_matrix_is_closed_and_content_free(
    outcome: AttachmentDuplicateFingerprintOutcomeV2,
) -> None:
    request = _request()
    result = _result(request, outcome=outcome)

    validate_attachment_duplicate_fingerprint_result_identity(result)
    assert attachment_duplicate_fingerprint_result_ref(result).object_sha256 == result.result_sha256
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "extracted_text",
        "physical_path",
        "raw_bytes",
        "token_inventory",
        "shingle_inventory",
        "exception",
    ):
        assert denied not in serialized


def test_supported_result_requires_fingerprint_counts_and_extractor() -> None:
    supported = _result(
        _request(),
        outcome=AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED,
    )

    for field, value in (
        ("similarity_fingerprint", None),
        ("token_count", 0),
        ("shingle_count", 0),
        ("extractor_version", None),
    ):
        with pytest.raises(ValidationError, match="SUPPORTED"):
            AttachmentDuplicateFingerprintResultV2.model_validate(
                {
                    **supported.model_dump(mode="python"),
                    field: value,
                }
            )


def test_unsupported_and_blocked_failure_codes_cannot_cross_outcomes() -> None:
    request = _request()
    unsupported = _result(
        request,
        outcome=AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED,
    )
    blocked = _result(
        request,
        outcome=AttachmentDuplicateFingerprintOutcomeV2.BLOCKED,
    )

    with pytest.raises(ValidationError, match="UNSUPPORTED"):
        AttachmentDuplicateFingerprintResultV2.model_validate(
            {
                **unsupported.model_dump(mode="python"),
                "failure_code": (AttachmentDuplicateFingerprintFailureCodeV2.OUTPUT_HASH_MISMATCH),
            }
        )
    with pytest.raises(ValidationError, match="BLOCKED"):
        AttachmentDuplicateFingerprintResultV2.model_validate(
            {
                **blocked.model_dump(mode="python"),
                "failure_code": (AttachmentDuplicateFingerprintFailureCodeV2.MEDIA_UNSUPPORTED),
            }
        )


def test_identities_reject_stale_request_and_result_values() -> None:
    request = _request()
    result = _result(
        request,
        outcome=AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED,
    )

    with pytest.raises(ValueError, match="request identity"):
        validate_attachment_duplicate_fingerprint_request_identity(
            request.model_copy(update={"logical_path": "inputs/changed.txt"})
        )
    with pytest.raises(ValueError, match="result identity"):
        validate_attachment_duplicate_fingerprint_result_identity(
            result.model_copy(update={"token_count": result.token_count + 1})
        )
