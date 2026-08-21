from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.semantic_review_adapter import (
    MappingAttachmentSemanticReviewMaterialResolver,
    ProviderAttachmentRepairContext,
    ProviderAttachmentRepairDecision,
    ProviderAttachmentRepairMaterial,
    ProviderSemanticFindingDecision,
    ProviderSemanticReviewContext,
    ProviderSemanticReviewDecision,
    ProviderSemanticReviewMaterial,
    RegistryAttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.semantic_review_v2 import (
    ATTACHMENT_REPAIR_POLICY_VERSION,
    ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
    AttachmentRepairFailureCodeV2,
    AttachmentRepairOutcomeV2,
    AttachmentRepairRequestV2,
    AttachmentRepairSourceV2,
    AttachmentSemanticReviewerRoleV2,
    AttachmentSemanticReviewFailureCodeV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewRoundV2,
    attachment_repair_request_carried_sha256,
    attachment_semantic_review_request_carried_sha256,
)
from env_mock_agent.facade.validation_adapter import (
    ProviderValidationMaterial,
    RegistryAttachmentValidationFacade,
    StagingAttachmentValidationMaterialResolver,
)
from env_mock_agent.facade.validation_v2 import (
    AttachmentValidationRequestV2,
    AttachmentValidationStatusV2,
    attachment_validation_request_carried_sha256,
)
from env_mock_agent.schemas import ArtifactPlan

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v2",
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _review_request(
    round_: AttachmentSemanticReviewRoundV2,
) -> AttachmentSemanticReviewRequestV2:
    role = {
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY: (
            AttachmentSemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER
        ),
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY: (
            AttachmentSemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER
        ),
        AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: (
            AttachmentSemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER
        ),
    }[round_]
    request = AttachmentSemanticReviewRequestV2(
        semantic_review_request_id="attachment-semantic-review-request://pending",
        round=round_,
        reviewer_role=role,
        stage_run_ref=_ref("stage-run", round_.value.lower(), version="identity/v1"),
        candidate_revision_ref=_ref("attachment-candidate-revision", "current"),
        deterministic_validation_result_ref=_ref(
            "revision-deterministic-validation",
            "current",
        ),
        context_view_ref=_ref("semantic-review-context-view", round_.value.lower()),
        current_artifact_version_refs=(_ref("candidate-artifact-version", "input"),),
        current_output_refs=(_ref("attachment-output", "input"),),
        prior_round_result_ref=(
            None
            if round_ is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY
            else _ref("semantic-review-round-result", "prior")
        ),
        prior_finding_refs=(),
        prior_resolution_refs=(),
        model_profile_ref=_ref("model-profile", round_.value.lower(), version="v1"),
        prompt_version=f"semantic-review/{round_.value.lower()}/v1",
        policy_version=ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
        idempotency_key=f"semantic-review-idempotency://{round_.value.lower()}",
        semantic_review_request_sha256=HASH,
    )
    digest = attachment_semantic_review_request_carried_sha256(request)
    return request.model_copy(
        update={
            "semantic_review_request_id": (f"attachment-semantic-review-request://sha256/{digest}"),
            "semantic_review_request_sha256": digest,
        }
    )


def _repair_request(source: bytes) -> AttachmentRepairRequestV2:
    source_hash = hashlib.sha256(source).hexdigest()
    request = AttachmentRepairRequestV2(
        repair_request_id="attachment-repair-request://pending",
        repair_plan_ref=_ref("targeted-repair-plan", "coverage"),
        source_round=AttachmentRepairSourceV2.COVERAGE_SOLVABILITY,
        candidate_revision_ref=_ref("attachment-candidate-revision", "current"),
        artifact_id="artifact://input",
        artifact_version_ref=_ref("candidate-artifact-version", "input"),
        build_spec_ref=_ref("artifact-build-spec", "input"),
        execution_result_ref=_ref("attachment-execution-result", "input"),
        output_ref=_ref("attachment-output", "input", digest=source_hash),
        output_sha256=source_hash,
        targeted_finding_refs=(_ref("semantic-review-finding", "coverage"),),
        targeted_finding_codes=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING,),
        attempt=1,
        prior_repair_result_ref=None,
        policy_version=ATTACHMENT_REPAIR_POLICY_VERSION,
        idempotency_key="attachment-repair-idempotency://coverage/input",
        repair_request_sha256=HASH,
    )
    digest = attachment_repair_request_carried_sha256(request)
    return request.model_copy(
        update={
            "repair_request_id": f"attachment-repair-request://sha256/{digest}",
            "repair_request_sha256": digest,
        }
    )


class _RecordingReviewBackend:
    def __init__(
        self,
        decisions: dict[
            AttachmentSemanticReviewRoundV2,
            ProviderSemanticReviewDecision,
        ],
    ) -> None:
        self.decisions = decisions
        self.context_ids: list[str] = []
        self.materials: list[ProviderSemanticReviewMaterial] = []

    async def review(
        self,
        context: ProviderSemanticReviewContext,
    ) -> ProviderSemanticReviewDecision:
        self.context_ids.append(context.context_id)
        self.materials.append(context.material)
        return self.decisions[context.request.round]


class _RepairBackend:
    def __init__(self, output_bytes: bytes) -> None:
        self.output_bytes = output_bytes
        self.contexts: list[ProviderAttachmentRepairContext] = []

    async def repair(
        self,
        context: ProviderAttachmentRepairContext,
    ) -> ProviderAttachmentRepairDecision:
        self.contexts.append(context)
        return ProviderAttachmentRepairDecision.succeeded(
            output_bytes=self.output_bytes,
            worker_version="fixture-repair/v1",
        )


class _RecordingOutputRegistrar:
    def __init__(self) -> None:
        self.output_refs: list[FacadeObjectRef] = []

    def register_repaired_output(
        self,
        *,
        request: AttachmentRepairRequestV2,
        material: ProviderAttachmentRepairMaterial,
        output_ref: FacadeObjectRef,
        output_bytes: bytes,
    ) -> None:
        assert material.source_output_ref == request.output_ref
        assert hashlib.sha256(output_bytes).hexdigest() == output_ref.object_sha256
        self.output_refs.append(output_ref)


def _facade(
    requests: tuple[AttachmentSemanticReviewRequestV2, ...],
    *,
    review_backend: _RecordingReviewBackend,
    repair_request: AttachmentRepairRequestV2 | None = None,
    repair_backend: _RepairBackend | None = None,
    source_bytes: bytes = b"original input",
) -> RegistryAttachmentSemanticReviewFacade:
    review_materials = {
        request.context_view_ref.object_id: ProviderSemanticReviewMaterial(
            context_view_ref=request.context_view_ref,
            included_ref_inventory=tuple(
                sorted(
                    (
                        request.candidate_revision_ref,
                        request.deterministic_validation_result_ref,
                        request.context_view_ref,
                        *request.current_artifact_version_refs,
                        *request.current_output_refs,
                        *request.prior_finding_refs,
                        *request.prior_resolution_refs,
                        *request.prior_repair_plan_refs,
                        *request.prior_repair_result_refs,
                    ),
                    key=lambda ref: (
                        ref.object_type,
                        ref.object_id,
                        ref.object_version,
                        ref.object_sha256,
                    ),
                )
            ),
            denied_data_families=(
                "BUILD_TRANSCRIPT",
                "HIDDEN_REASONING",
                "RAW_PRIVATE_REFERENCE",
                "RAW_TRACE",
            ),
            allowed_evidence_ref_ids=frozenset(
                {
                    *(
                        evidence_id
                        for decision in review_backend.decisions.values()
                        for finding in decision.findings
                        for evidence_id in finding.evidence_ref_ids
                    ),
                    *(
                        evidence_id
                        for decision in review_backend.decisions.values()
                        for resolution in decision.resolutions
                        for evidence_id in resolution.evidence_ref_ids
                    ),
                }
            ),
            safe_payload={"fixture": request.round.value},
        )
        for request in requests
    }
    repair_materials = (
        {
            repair_request.repair_plan_ref.object_id: (
                ProviderAttachmentRepairMaterial(
                    source_output_ref=repair_request.output_ref,
                    source_output_bytes=source_bytes,
                    logical_path="inputs/source.txt",
                    media_type="text/plain",
                )
            )
        }
        if repair_request is not None
        else {}
    )
    return RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials=review_materials,
            repair_materials=repair_materials,
        ),
        review_backends={role: review_backend for role in AttachmentSemanticReviewerRoleV2},
        repair_backend=repair_backend,
        repaired_output_registrar=_RecordingOutputRegistrar(),
    )


@pytest.mark.asyncio
async def test_each_round_receives_a_fresh_purpose_specific_context() -> None:
    first = _review_request(AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    second = _review_request(AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY)
    backend = _RecordingReviewBackend(
        {
            first.round: ProviderSemanticReviewDecision.accepted(),
            second.round: ProviderSemanticReviewDecision.accepted(),
        }
    )
    facade = _facade((first, second), review_backend=backend)

    first_result = await facade.review(first)
    second_result = await facade.review(second)

    assert first_result.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED
    assert second_result.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED
    assert len(set(backend.context_ids)) == 2
    assert backend.materials[0].safe_payload != backend.materials[1].safe_payload
    for result in (first_result, second_result):
        attestation = result.clean_context_attestation
        assert attestation.fresh_context is True
        assert attestation.runtime_resume_used is False
        assert attestation.build_transcript_included is False
        assert attestation.hidden_reasoning_included is False
        assert attestation.raw_private_reference_included is False


@pytest.mark.asyncio
async def test_backend_finding_is_policy_mapped_and_content_minimized() -> None:
    request = _review_request(AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    decision = ProviderSemanticReviewDecision.requires_repair(
        findings=(
            ProviderSemanticFindingDecision(
                code=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING),
                subject_refs=(
                    request.candidate_revision_ref,
                    *request.current_artifact_version_refs,
                    *request.current_output_refs,
                ),
                artifact_ids=("artifact://input",),
                evidence_ref_ids=("semantic-evidence://coverage/input",),
            ),
        )
    )
    backend = _RecordingReviewBackend({request.round: decision})
    facade = _facade((request,), review_backend=backend)

    result = await facade.review(request)

    assert result.outcome is AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR
    assert result.findings[0].artifact_repair_allowed is True
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "safe_payload" not in serialized
    assert "artifact_content" not in serialized
    assert "hidden_reasoning" in serialized
    assert "runtime_transcript" not in serialized


@pytest.mark.asyncio
async def test_backend_cannot_fabricate_unregistered_evidence_id() -> None:
    request = _review_request(AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    backend = _RecordingReviewBackend(
        {
            request.round: ProviderSemanticReviewDecision.requires_repair(
                findings=(
                    ProviderSemanticFindingDecision(
                        code=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING),
                        subject_refs=(
                            request.candidate_revision_ref,
                            *request.current_artifact_version_refs,
                            *request.current_output_refs,
                        ),
                        artifact_ids=("artifact://input",),
                        evidence_ref_ids=("semantic-evidence://unregistered/value",),
                    ),
                )
            )
        }
    )
    material = ProviderSemanticReviewMaterial(
        context_view_ref=request.context_view_ref,
        included_ref_inventory=tuple(
            sorted(
                (
                    request.candidate_revision_ref,
                    request.deterministic_validation_result_ref,
                    request.context_view_ref,
                    *request.current_artifact_version_refs,
                    *request.current_output_refs,
                ),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
        denied_data_families=(
            "BUILD_TRANSCRIPT",
            "HIDDEN_REASONING",
            "RAW_PRIVATE_REFERENCE",
            "RAW_TRACE",
        ),
        allowed_evidence_ref_ids=frozenset(),
        safe_payload={},
    )
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={
                request.context_view_ref.object_id: material,
            },
            repair_materials={},
        ),
        review_backends={request.reviewer_role: backend},
        repair_backend=None,
    )

    result = await facade.review(request)

    assert result.outcome is AttachmentSemanticReviewOutcomeV2.BLOCKED
    assert result.failure_code is (AttachmentSemanticReviewFailureCodeV2.BACKEND_OUTPUT_INVALID)
    assert result.findings == ()


@pytest.mark.asyncio
async def test_repair_returns_new_opaque_output_and_rejects_unchanged_bytes() -> None:
    source = b"original input"
    request = _repair_request(source)
    review_request = _review_request(AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    review_backend = _RecordingReviewBackend(
        {review_request.round: ProviderSemanticReviewDecision.accepted()}
    )
    changed_backend = _RepairBackend(b"repaired input with required context")
    facade = _facade(
        (review_request,),
        review_backend=review_backend,
        repair_request=request,
        repair_backend=changed_backend,
        source_bytes=source,
    )

    result = await facade.repair(request)

    assert result.outcome is AttachmentRepairOutcomeV2.SUCCEEDED
    assert result.output_ref is not None
    assert result.output_sha256 == hashlib.sha256(changed_backend.output_bytes).hexdigest()
    assert result.output_sha256 != request.output_sha256
    assert changed_backend.contexts[0].material.source_output_bytes == source

    unchanged_backend = _RepairBackend(source)
    unchanged_facade = _facade(
        (review_request,),
        review_backend=review_backend,
        repair_request=request,
        repair_backend=unchanged_backend,
        source_bytes=source,
    )
    unchanged = await unchanged_facade.repair(request)
    assert unchanged.outcome is AttachmentRepairOutcomeV2.BLOCKED_POLICY
    assert unchanged.output_ref is None
    assert unchanged.output_sha256 is None


@pytest.mark.asyncio
async def test_repair_fails_closed_when_validation_registration_is_unavailable() -> None:
    source = b"original input"
    request = _repair_request(source)
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={},
            repair_materials={
                request.repair_plan_ref.object_id: (
                    ProviderAttachmentRepairMaterial(
                        source_output_ref=request.output_ref,
                        source_output_bytes=source,
                        logical_path="inputs/source.txt",
                        media_type="text/plain",
                    )
                )
            },
        ),
        review_backends={},
        repair_backend=_RepairBackend(b"changed input"),
    )

    result = await facade.repair(request)

    assert result.outcome is AttachmentRepairOutcomeV2.BLOCKED_CAPABILITY
    assert result.failure_code is (AttachmentRepairFailureCodeV2.VALIDATION_REGISTRATION_UNAVAILABLE)
    assert result.output_ref is None


@pytest.mark.asyncio
async def test_repaired_output_is_registered_for_complete_validation(
    tmp_path: Path,
) -> None:
    source = b"original input"
    repaired = b"repaired input with complete safe context"
    repair_request = _repair_request(source)
    validation_material = ProviderValidationMaterial(
        plan=ArtifactPlan(
            artifact_id=repair_request.artifact_id,
            dependency_id="attachment-dependency://input",
            relative_path="inputs/source.txt",
            asset_type="txt",
            content_contract={"min_characters": 1},
            render_contract={},
            validators=["text-validator"],
        )
    )
    validation_resolver = StagingAttachmentValidationMaterialResolver(
        staging_root=tmp_path / "validation-staging",
        execution_requests={},
        materials={
            repair_request.build_spec_ref.object_id: validation_material,
        },
    )
    semantic_facade = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={},
            repair_materials={
                repair_request.repair_plan_ref.object_id: (
                    ProviderAttachmentRepairMaterial(
                        source_output_ref=repair_request.output_ref,
                        source_output_bytes=source,
                        logical_path="inputs/source.txt",
                        media_type="text/plain",
                    )
                )
            },
        ),
        review_backends={},
        repair_backend=_RepairBackend(repaired),
        repaired_output_registrar=validation_resolver,
    )

    repair_result = await semantic_facade.repair(repair_request)
    assert repair_result.output_ref is not None
    assert repair_result.output_sha256 is not None
    validation_request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id=repair_request.artifact_id,
        artifact_result_ref=repair_request.artifact_version_ref,
        execution_request_ref=_ref("attachment-execution-request", "input"),
        execution_result_ref=repair_request.execution_result_ref,
        build_spec_ref=repair_request.build_spec_ref,
        producer_task_view_ref=_ref("producer-task-view", "current"),
        output_ref=repair_result.output_ref,
        output_sha256=repair_result.output_sha256,
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
        idempotency_key="attachment-validation-idempotency://repaired",
        validation_request_sha256=HASH,
    )
    digest = attachment_validation_request_carried_sha256(validation_request)
    validation_request = validation_request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )

    validation_result = await RegistryAttachmentValidationFacade(resolver=validation_resolver).validate(
        validation_request
    )

    assert validation_result.status is AttachmentValidationStatusV2.PASSED
    assert validation_result.output_ref == repair_result.output_ref
    assert validation_result.inventory_complete is True
    assert validation_result.scan_complete is True


@pytest.mark.asyncio
async def test_missing_context_and_backend_fail_closed_and_replay_exactly() -> None:
    request = _review_request(AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY)
    backend = _RecordingReviewBackend({request.round: ProviderSemanticReviewDecision.accepted()})
    missing = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={},
            repair_materials={},
        ),
        review_backends={request.reviewer_role: backend},
        repair_backend=None,
    )

    first = await missing.review(request)
    replay = await missing.review(request)

    assert replay == first
    assert first.outcome is AttachmentSemanticReviewOutcomeV2.BLOCKED
    assert first.failure_code is (AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISSING)
    assert backend.context_ids == []

    incomplete_material = ProviderSemanticReviewMaterial(
        context_view_ref=request.context_view_ref,
        included_ref_inventory=(),
        denied_data_families=(
            "BUILD_TRANSCRIPT",
            "HIDDEN_REASONING",
            "RAW_PRIVATE_REFERENCE",
            "RAW_TRACE",
        ),
        allowed_evidence_ref_ids=frozenset(),
        safe_payload={},
    )
    mismatched = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={
                request.context_view_ref.object_id: incomplete_material,
            },
            repair_materials={},
        ),
        review_backends={request.reviewer_role: backend},
        repair_backend=None,
    )
    mismatch = await mismatched.review(request)
    assert mismatch.outcome is AttachmentSemanticReviewOutcomeV2.BLOCKED
    assert mismatch.failure_code is (AttachmentSemanticReviewFailureCodeV2.CONTEXT_MISMATCH)
    assert backend.context_ids == []

    material = ProviderSemanticReviewMaterial(
        context_view_ref=request.context_view_ref,
        included_ref_inventory=tuple(
            sorted(
                (
                    request.candidate_revision_ref,
                    request.deterministic_validation_result_ref,
                    request.context_view_ref,
                    *request.current_artifact_version_refs,
                    *request.current_output_refs,
                ),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
        denied_data_families=(
            "BUILD_TRANSCRIPT",
            "HIDDEN_REASONING",
            "RAW_PRIVATE_REFERENCE",
            "RAW_TRACE",
        ),
        allowed_evidence_ref_ids=frozenset(),
        safe_payload={},
    )
    unavailable = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={
                request.context_view_ref.object_id: material,
            },
            repair_materials={},
        ),
        review_backends={},
        repair_backend=None,
    )
    blocked = await unavailable.review(request)
    assert blocked.outcome is AttachmentSemanticReviewOutcomeV2.BLOCKED
    assert blocked.failure_code is (AttachmentSemanticReviewFailureCodeV2.MODEL_UNAVAILABLE)


@pytest.mark.asyncio
async def test_repair_material_hash_mismatch_is_policy_blocked() -> None:
    source = b"original input"
    request = _repair_request(source)
    backend = _RepairBackend(b"changed")
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=MappingAttachmentSemanticReviewMaterialResolver(
            review_materials={},
            repair_materials={
                request.repair_plan_ref.object_id: (
                    ProviderAttachmentRepairMaterial(
                        source_output_ref=request.output_ref,
                        source_output_bytes=b"different source bytes",
                        logical_path="inputs/source.txt",
                        media_type="text/plain",
                    )
                )
            },
        ),
        review_backends={},
        repair_backend=backend,
    )

    result = await facade.repair(request)

    assert result.outcome is AttachmentRepairOutcomeV2.BLOCKED_POLICY
    assert result.failure_code is (AttachmentRepairFailureCodeV2.REPAIR_MATERIAL_MISMATCH)
    assert backend.contexts == []
