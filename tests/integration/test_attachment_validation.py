from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.validation_adapter import (
    MappingAttachmentValidationMaterialResolver,
    ProviderValidationMaterial,
    RegistryAttachmentValidationFacade,
)
from env_mock_agent.facade.validation_v2 import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingCodeV2,
    AttachmentValidationFingerprintCategoryV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
    AttachmentValidationPiiRuleV2,
    AttachmentValidationRequestV2,
    AttachmentValidationStatusV2,
    attachment_validation_request_carried_sha256,
)
from env_mock_agent.schemas import ArtifactPlan, ValidationFinding
from env_mock_agent.validators import ValidatorRegistry
from env_mock_agent.validators.base import (
    ArtifactValidator,
    ValidationRequest,
)

HASH = "a" * 64
NORMALIZATION_VERSION = "task-prompt-leakage-normalization/r4-04-v1"


class _MutatingTextValidator(ArtifactValidator):
    name = "text"

    def validate(
        self,
        request: ValidationRequest,
    ) -> list[ValidationFinding]:
        request.artifact_path().write_text(
            "mutated during validation",
            encoding="utf-8",
        )
        return []


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


def _fingerprint(text: str) -> AttachmentValidationFingerprintV2:
    normalized = " ".join(text.casefold().split())
    return AttachmentValidationFingerprintV2(
        fingerprint_id="prompt-leakage-fingerprint://final-answer",
        category=AttachmentValidationFingerprintCategoryV2.FINAL_ANSWER,
        match_kind=AttachmentValidationFingerprintMatchKindV2.TOKEN_WINDOW,
        digest_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        token_count=len(normalized.split()),
        normalization_version=NORMALIZATION_VERSION,
    )


def _request(
    path: Path,
    *,
    pii_rules: tuple[AttachmentValidationPiiRuleV2, ...] = (),
    fingerprints: tuple[AttachmentValidationFingerprintV2, ...] = (),
) -> AttachmentValidationRequestV2:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id="artifact://input",
        artifact_result_ref=_ref("artifact-build-result", "input"),
        execution_request_ref=_ref("attachment-execution-request", "input"),
        execution_result_ref=_ref("attachment-execution-result", "input"),
        build_spec_ref=_ref("artifact-build-spec", "input"),
        producer_task_view_ref=_ref("producer-task-view", "current"),
        output_ref=_ref("attachment-output", "input", digest=digest),
        output_sha256=digest,
        logical_path="inputs/source.txt",
        media_type="text/plain",
        declared_validator_ids=("text-validator",),
        configured_pii_rules=pii_rules,
        leakage_reference_set_ref=_ref(
            "prompt-leakage-reference-set",
            "current",
        ),
        leakage_fingerprints=fingerprints,
        complete_leakage_categories=(AttachmentValidationFingerprintCategoryV2.FINAL_ANSWER,),
        forbidden_output_values=("original final answer",),
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        idempotency_key="attachment-validation-idempotency://input",
        validation_request_sha256=HASH,
    )
    carried = attachment_validation_request_carried_sha256(request)
    return request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{carried}"),
            "validation_request_sha256": carried,
        }
    )


def _facade(
    path: Path,
    request: AttachmentValidationRequestV2,
) -> RegistryAttachmentValidationFacade:
    material = ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id=request.artifact_id,
            dependency_id="attachment-dependency://input",
            relative_path=request.logical_path,
            asset_type="txt",
            content_contract={"min_characters": 1},
            render_contract={},
            validators=["text-validator"],
        )
    )
    return RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={request.output_ref.object_id: path},
            materials={request.build_spec_ref.object_id: material},
        )
    )


@pytest.mark.asyncio
async def test_validation_detects_secret_and_returns_sanitized_finding(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text(
        "api_key = abcdefghijklmnopqrstuvwxyz",
        encoding="utf-8",
    )
    request = _request(path)

    result = await _facade(path, request).validate(request)

    assert result.status is AttachmentValidationStatusV2.FAILED
    assert [item.code for item in result.findings] == [AttachmentValidationFindingCodeV2.SECRET_DETECTED]
    assert result.findings[0].non_waivable is True
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "abcdefghijklmnopqrstuvwxyz" not in serialized
    assert str(path) not in serialized
    assert "repair_action" not in serialized


@pytest.mark.asyncio
async def test_validation_detects_configured_pii_without_echoing_pattern_or_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Customer identifier CUST-12345.", encoding="utf-8")
    rule = AttachmentValidationPiiRuleV2(
        rule_id="configured-pii/customer-id/v1",
        pattern=r"CUST-\d{5}",
    )
    request = _request(path, pii_rules=(rule,))

    result = await _facade(path, request).validate(request)

    assert result.status is AttachmentValidationStatusV2.FAILED
    assert [item.code for item in result.findings] == [
        AttachmentValidationFindingCodeV2.CONFIGURED_PII_DETECTED
    ]
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "CUST-12345" not in serialized
    assert r"CUST-\d{5}" not in serialized


@pytest.mark.asyncio
async def test_validation_detects_restricted_fingerprint_by_digest_only(
    tmp_path: Path,
) -> None:
    restricted = "approve the hidden final recommendation"
    path = tmp_path / "source.txt"
    path.write_text(f"Context: {restricted}.", encoding="utf-8")
    fingerprint = _fingerprint(restricted)
    request = _request(path, fingerprints=(fingerprint,))

    result = await _facade(path, request).validate(request)

    assert result.status is AttachmentValidationStatusV2.FAILED
    assert [item.code for item in result.findings] == [
        AttachmentValidationFindingCodeV2.RESTRICTED_FINGERPRINT_MATCH
    ]
    assert result.findings[0].matched_fingerprint_ids == (fingerprint.fingerprint_id,)
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert restricted not in serialized


@pytest.mark.asyncio
async def test_invalid_configured_pii_rule_blocks_and_idempotency_is_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    rule = AttachmentValidationPiiRuleV2(
        rule_id="configured-pii/invalid/v1",
        pattern="[",
    )
    request = _request(path, pii_rules=(rule,))
    facade = _facade(path, request)

    first = await facade.validate(request)
    second = await facade.validate(request)

    assert first == second
    assert first.status is AttachmentValidationStatusV2.BLOCKED
    assert first.failure_code is (AttachmentValidationFailureCodeV2.INVALID_CONFIGURED_PII_RULE)

    changed = request.model_copy(
        update={
            "forbidden_output_values": ("different value",),
        }
    )
    with pytest.raises(ValueError, match="identity"):
        await facade.validate(changed)


@pytest.mark.asyncio
async def test_unknown_declared_validator_blocks_without_silent_skip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    request = _request(path).model_copy(
        update={
            "declared_validator_ids": ("unknown-validator",),
            "validation_request_sha256": HASH,
        }
    )
    digest = attachment_validation_request_carried_sha256(request)
    request = request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )
    material = ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id=request.artifact_id,
            dependency_id="attachment-dependency://input",
            relative_path=request.logical_path,
            asset_type="txt",
            validators=["unknown-validator"],
        )
    )
    facade = RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={request.output_ref.object_id: path},
            materials={request.build_spec_ref.object_id: material},
        )
    )

    result = await facade.validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE)
    assert "unknown-validator" in result.required_validator_ids


@pytest.mark.asyncio
async def test_unknown_fingerprint_normalization_blocks_before_content_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    fingerprint = _fingerprint("neutral input").model_copy(
        update={"normalization_version": "unknown-normalization/v9"}
    )
    request = _request(path, fingerprints=(fingerprint,))

    result = await _facade(path, request).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.UNKNOWN_FINGERPRINT_NORMALIZATION)


@pytest.mark.asyncio
async def test_missing_output_material_and_hash_mismatch_are_typed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    request = _request(path)

    missing_output = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={},
            materials={
                request.build_spec_ref.object_id: ProviderValidationMaterial(
                    plan=ArtifactPlan(
                        artifact_id=request.artifact_id,
                        dependency_id="attachment-dependency://input",
                        relative_path=request.logical_path,
                        asset_type="txt",
                        validators=["text-validator"],
                    )
                )
            },
        )
    ).validate(request)
    assert missing_output.failure_code is (AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND)

    missing_material = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={request.output_ref.object_id: path},
            materials={},
        )
    ).validate(request)
    assert missing_material.failure_code is (AttachmentValidationFailureCodeV2.MATERIAL_NOT_FOUND)

    path.write_text("changed after request", encoding="utf-8")
    stale = await _facade(path, request).validate(request)
    assert stale.failure_code is (AttachmentValidationFailureCodeV2.OUTPUT_HASH_MISMATCH)


@pytest.mark.asyncio
async def test_empty_and_forbidden_literal_findings_use_closed_codes(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    empty_request = _request(empty_path)
    empty_result = await _facade(empty_path, empty_request).validate(empty_request)
    assert empty_result.status is AttachmentValidationStatusV2.FAILED
    assert AttachmentValidationFindingCodeV2.OUTPUT_EMPTY in {
        finding.code for finding in empty_result.findings
    }

    forbidden_path = tmp_path / "forbidden.txt"
    forbidden_path.write_text(
        "This contains the original final answer.",
        encoding="utf-8",
    )
    forbidden_request = _request(forbidden_path)
    forbidden_result = await _facade(
        forbidden_path,
        forbidden_request,
    ).validate(forbidden_request)
    assert forbidden_result.status is AttachmentValidationStatusV2.FAILED
    assert AttachmentValidationFindingCodeV2.FORBIDDEN_OUTPUT_MATCH in {
        finding.code for finding in forbidden_result.findings
    }


@pytest.mark.asyncio
async def test_nonempty_render_contract_blocks_without_validator_capability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    request = _request(path)
    material = ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id=request.artifact_id,
            dependency_id="attachment-dependency://input",
            relative_path=request.logical_path,
            asset_type="txt",
            content_contract={"min_characters": 1},
            render_contract={"required_theme": "internal"},
            validators=["text-validator"],
        )
    )

    result = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={request.output_ref.object_id: path},
            materials={request.build_spec_ref.object_id: material},
        )
    ).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.VALIDATOR_UNAVAILABLE)


@pytest.mark.asyncio
async def test_output_mutation_during_validation_blocks_stale_subject(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.txt"
    path.write_text("Neutral input.", encoding="utf-8")
    request = _request(path)
    validators = ValidatorRegistry.default()
    validators.register(_MutatingTextValidator())
    material = ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id=request.artifact_id,
            dependency_id="attachment-dependency://input",
            relative_path=request.logical_path,
            asset_type="txt",
            content_contract={"min_characters": 1},
            render_contract={},
            validators=["text-validator"],
        )
    )

    result = await RegistryAttachmentValidationFacade(
        resolver=MappingAttachmentValidationMaterialResolver(
            outputs={request.output_ref.object_id: path},
            materials={request.build_spec_ref.object_id: material},
        ),
        validators=validators,
    ).validate(request)

    assert result.status is AttachmentValidationStatusV2.BLOCKED
    assert result.failure_code is (AttachmentValidationFailureCodeV2.OUTPUT_HASH_MISMATCH)
