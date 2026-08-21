from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_deterministic_validation import (
    _facade_result,
    _FakeValidationFacade,
    _NoCallFacade,
    _not_required_result,
    _producer_view,
    _reconstruction_result,
    _reference_set,
)
from test_semantic_review import (
    _AcceptingBackend,
    _audit,
    _DynamicResolver,
    _ItemScopedRepairBackend,
    _NoRepairBackend,
    _NoRevalidation,
    _policy,
    _sources,
    _store,
)

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.semantic_review_adapter import (
    RegistryAttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.validation_v2 import (
    AttachmentInventoryMemberTypeV2,
    AttachmentInventoryMemberV2,
    AttachmentValidationRequestV2,
    AttachmentValidationStatusV2,
    attachment_inventory_member_carried_sha256,
    validate_attachment_validation_request_identity,
)
from eval_factory.attachment_planning.quality import (
    ItemQualityCompiler,
    ItemQualityPolicyError,
)
from eval_factory.attachment_planning.review import (
    CandidateRevisionCompiler,
    IsolatedSemanticReviewOrchestrator,
)
from eval_factory.attachment_planning.validation import (
    DeterministicValidationCompiler,
)
from eval_factory.contracts.core import ObjectRef, VersionBinding
from eval_factory.contracts.quality_v2 import (
    EnvironmentArtifactV2,
    EnvironmentSpecV2,
    FinalPackageManifestV2,
    ItemQualityCompilationResultV2,
    ItemQualityFailureCodeV2,
    ItemQualityOutcomeV2,
    PackageMemberProvenanceV2,
    QualityReportV2,
    environment_artifact_ref,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    provenance_decision_stable_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    RevisionDeterministicValidationOutcomeV2,
    RevisionDeterministicValidationV2,
    attachment_candidate_revision_ref,
    candidate_artifact_version_ref,
)
from eval_factory.contracts.safety import OriginClass
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    producer_task_view_ref,
    r4_task_contract_set_carried_sha256,
)

HASH = "a" * 64
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/item_quality"
    / "r5-10-version-bound-item-quality-v1.json"
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _inventory_member(
    request: AttachmentValidationRequestV2,
    *,
    normalized_path: str,
    member_type: AttachmentInventoryMemberTypeV2,
    media_type: str,
    content_sha256: str,
    container_ref: FacadeObjectRef | None,
) -> AttachmentInventoryMemberV2:
    value = AttachmentInventoryMemberV2(
        inventory_member_id="attachment-inventory-member://pending",
        artifact_id=request.artifact_id,
        output_ref=request.output_ref,
        normalized_path=normalized_path,
        member_type=member_type,
        media_type=media_type,
        size_bytes=32,
        content_sha256=content_sha256,
        container_ref=container_ref,
        inventory_member_sha256="0" * 64,
    )
    digest = attachment_inventory_member_carried_sha256(value)
    return value.model_copy(
        update={
            "inventory_member_id": f"attachment-inventory-member://sha256/{digest}",
            "inventory_member_sha256": digest,
        }
    )


class _NestedContainerValidationFacade:
    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ):
        validate_attachment_validation_request_identity(request)
        nested_digest = hashlib.sha256(b"nested archive").hexdigest()
        child_digest = hashlib.sha256(b"child input").hexdigest()
        root = _inventory_member(
            request,
            normalized_path=request.logical_path,
            member_type=AttachmentInventoryMemberTypeV2.FILE,
            media_type=request.media_type,
            content_sha256=request.output_sha256,
            container_ref=None,
        )
        nested = _inventory_member(
            request,
            normalized_path="nested.zip",
            member_type=AttachmentInventoryMemberTypeV2.NESTED_MEMBER,
            media_type="application/zip",
            content_sha256=nested_digest,
            container_ref=request.output_ref,
        )
        child_container_id = hashlib.sha256(f"{request.output_ref.object_id}|nested.zip".encode()).hexdigest()
        child_container_ref = FacadeObjectRef(
            object_type="attachment-output",
            object_id=f"attachment-output://container/{child_container_id}",
            object_version="v2",
            object_sha256=nested_digest,
        )
        child = _inventory_member(
            request,
            normalized_path="child.txt",
            member_type=AttachmentInventoryMemberTypeV2.NESTED_MEMBER,
            media_type="text/plain",
            content_sha256=child_digest,
            container_ref=child_container_ref,
        )
        inventory = tuple(
            sorted(
                (root, nested, child),
                key=lambda item: (
                    item.container_ref.object_id if item.container_ref else "",
                    item.normalized_path,
                    item.member_type.value,
                ),
            )
        )
        required = (
            "common",
            "configured-pii",
            "metadata",
            "package-inventory",
            "restricted-fingerprint",
            "secrets",
            "text",
        )
        return _facade_result(
            request,
            status=AttachmentValidationStatusV2.PASSED,
            required=required,
            findings=(),
            inventory=inventory,
            failure_code=None,
        )


def _task_contract_set(
    producer_ref: ObjectRef,
    *,
    task_draft_ref_value: ObjectRef | None = None,
) -> R4TaskContractSetV2:
    value = R4TaskContractSetV2(
        contract_set_id="r4-task-contract-set://pending",
        task_draft_ref=(task_draft_ref_value or _ref("task-draft", "current")),
        task_prompt_safety_gate_ref=_ref("task-prompt-safety-gate", "current"),
        rubric_set_ref=_ref("rubric-set", "current"),
        evaluator_spec_ref=_ref("evaluator-spec", "current"),
        reference_policy_ref=_ref("reference-policy", "current"),
        tool_policy_ref=_ref("tool-policy", "current"),
        contestant_tool_policy_ref=_ref("contestant-tool-policy", "current"),
        producer_storage_authorization_ref=_ref(
            "producer-storage-authorization",
            "current",
        ),
        producer_task_view_ref=producer_ref,
        contract_set_sha256="0" * 64,
        audit=_audit(),
    )
    digest = r4_task_contract_set_carried_sha256(value)
    return value.model_copy(
        update={
            "contract_set_id": f"r4-task-contract-set://sha256/{digest}",
            "contract_set_sha256": digest,
        }
    )


def test_item_quality_gold_is_content_free_and_closed() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["policy_version"] == "item-quality/r5-10-v1"
    assert len(payload["scenarios"]) == 9
    assert payload["downstream_boundary"] == {
        "evaluation_item_ref": None,
        "batch_quality_report_ref": None,
        "release_decision_ref": None,
    }
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "artifact_content",
        "raw_trace",
        "private_reference",
        "grader_rule",
        "final_answer",
        "runtime_transcript",
        "physical_path",
        "user_decision",
        "registry",
    ):
        assert forbidden not in serialized


async def _current_sources(
    tmp_path: Path,
    *,
    blocked: bool,
    artifacts: tuple[tuple[str, str], ...] = (),
    item_scoped_repair: bool = False,
    mismatched_candidate_artifact: bool = False,
    validation_outcomes: dict[str, str] | None = None,
    nested_container: bool = False,
):
    producer_view = _producer_view(artifacts)
    reconstruction = (
        await _reconstruction_result(
            tmp_path,
            view=producer_view,
            artifacts=artifacts,
        )
        if artifacts
        else _not_required_result(producer_view)
    )
    validation_facade = (
        _NestedContainerValidationFacade()
        if nested_container
        else _FakeValidationFacade(outcomes=validation_outcomes)
        if artifacts
        else _NoCallFacade()
    )
    source_deterministic = await DeterministicValidationCompiler().compile(
        reconstruction_result=reconstruction,
        producer_task_view=producer_view,
        leakage_reference_set=_reference_set(),
        configured_pii_rules=(),
        facade=validation_facade,
        audit=_audit(),
    )
    candidate_revision, deterministic_validation = CandidateRevisionCompiler().compile_initial(
        reconstruction_result=reconstruction,
        deterministic_validation=source_deterministic,
        audit=_audit(),
    )
    if mismatched_candidate_artifact:
        original = candidate_revision.artifact_versions[0]
        replacement = CandidateArtifactVersionV2.create_base(
            artifact_id="artifact://other",
            attachment_dependency_id="attachment-dependency://other",
            artifact_build_result_ref=(original.base_artifact_build_result_ref),
            build_spec_ref=original.build_spec_ref,
            execution_request_ref=original.execution_request_ref,
            execution_result_ref=original.execution_result_ref,
            output_ref=original.output_ref,
            logical_path=original.logical_path,
            media_type=original.media_type,
            declared_validator_ids=original.declared_validator_ids,
            derivation_root_refs=original.derivation_root_refs,
        )
        candidate_revision = AttachmentCandidateRevisionV2.create_initial(
            base_reconstruction_result_ref=(candidate_revision.base_reconstruction_result_ref),
            artifact_versions=(replacement,),
            audit=_audit(
                candidate_revision.base_reconstruction_result_ref,
                candidate_artifact_version_ref(replacement),
            ),
        )
        deterministic_validation = RevisionDeterministicValidationV2.create(
            candidate_revision_ref=attachment_candidate_revision_ref(candidate_revision),
            source_deterministic_validation_result_ref=(
                deterministic_validation.source_deterministic_validation_result_ref
            ),
            artifact_validation_result_refs=(deterministic_validation.artifact_validation_result_refs),
            finding_refs=deterministic_validation.finding_refs,
            output_refs=candidate_revision.candidate_output_refs,
            outcome=RevisionDeterministicValidationOutcomeV2.PASSED,
            audit=_audit(
                attachment_candidate_revision_ref(candidate_revision),
                deterministic_validation.source_deterministic_validation_result_ref,
            ),
        )
    store, job_id, item_id = _store(tmp_path)
    policy = _policy()
    backend = _ItemScopedRepairBackend() if item_scoped_repair else _AcceptingBackend()
    facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends=({} if blocked else {role: backend for role in policy.roles_in_order}),
        repair_backend=None if blocked else _NoRepairBackend(),
    )
    workflow = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        review_policy=policy,
        context_sources=_sources(),
        review_facade=facade,
        revision_validator=_NoRevalidation(),
        audit=_audit(),
    )
    return (
        _task_contract_set(producer_task_view_ref(producer_view)),
        policy,
        workflow,
        candidate_revision,
        deterministic_validation,
        source_deterministic,
        store,
        job_id,
        item_id,
    )


@pytest.mark.asyncio
async def test_item_quality_compiler_finalizes_current_empty_package(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=False)
    compiler = ItemQualityCompiler()

    result = compiler.compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.outcome is ItemQualityOutcomeV2.PASSED
    assert result.quality_report.approvable is True
    assert result.final_package_manifest is not None
    assert result.final_package_manifest.entries == ()
    assert result.final_package_manifest.package_sha256 == hashlib.sha256(b"[]").hexdigest()
    assert result.provenance_manifest is not None
    assert result.provenance_manifest.entries == ()
    assert result.environment_spec is not None
    assert result.environment_spec.artifacts == ()
    assert result.input_state_only is True
    compiler.validate_current(
        result,
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
    )


@pytest.mark.asyncio
async def test_item_quality_compiler_finalizes_current_file_package(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
    )

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.approvable is True
    assert result.final_package_manifest is not None
    assert len(result.final_package_manifest.entries) == 1
    assert result.provenance_manifest is not None
    assert len(result.provenance_manifest.entries) == 1
    binding = result.provenance_manifest.entries[0]
    assert binding.provenance_decision.disposition.value == ("ALLOW_INPUT_EVIDENCE")
    assert binding.provenance_decision.visibility.value == ("CONTESTANT_VISIBLE")
    assert result.environment_spec is not None
    assert len(result.environment_spec.artifacts) == 1
    environment_artifact = result.environment_spec.artifacts[0]
    assert environment_artifact.artifact_id == "artifact://input"
    assert environment_artifact.logical_path == "inputs/source.txt"
    assert environment_artifact.package_inventory_entry_refs == (result.final_package_manifest.entry_refs)


@pytest.mark.asyncio
async def test_item_quality_compiler_finalizes_nested_container_package(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/archive.zip"),),
        nested_container=True,
    )

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.approvable is True
    assert result.final_package_manifest is not None
    assert len(result.final_package_manifest.entries) == 3
    assert result.provenance_manifest is not None
    assert len(result.provenance_manifest.entries) == 3
    assert result.environment_spec is not None
    assert result.environment_spec.artifacts[0].package_inventory_entry_refs == (
        result.final_package_manifest.entry_refs
    )


@pytest.mark.asyncio
async def test_package_member_provenance_rejects_non_harness_origin(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
    )
    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    assert result.provenance_manifest is not None
    binding = result.provenance_manifest.entries[0]
    decision = binding.provenance_decision.model_copy(
        update={"origin_class": OriginClass.USER_SUPPLIED_INPUT}
    )
    values = binding.model_dump(mode="python")
    values["provenance_decision"] = decision
    values["provenance_decision_ref"] = provenance_decision_stable_ref(decision)

    with pytest.raises(
        ValidationError,
        match="not safe and exact",
    ):
        PackageMemberProvenanceV2.model_validate(values)


@pytest.mark.asyncio
async def test_final_package_rejects_stale_nested_inventory_entry(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
    )
    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    assert result.final_package_manifest is not None
    values = result.final_package_manifest.model_dump(mode="python")
    values["entries"][0]["inventory_member"]["normalized_path"] = "inputs/tampered.txt"

    with pytest.raises(ValidationError, match="identity is stale"):
        FinalPackageManifestV2.model_validate(values)


@pytest.mark.asyncio
async def test_final_package_rejects_entry_owner_set_mismatch(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
    )
    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    assert result.final_package_manifest is not None
    values = result.final_package_manifest.model_dump(mode="python")
    values["output_refs"] = ()

    with pytest.raises(
        ValidationError,
        match="exactly cover output and validation refs",
    ):
        FinalPackageManifestV2.model_validate(values)


@pytest.mark.asyncio
async def test_item_quality_result_rejects_cross_package_values(
    tmp_path: Path,
) -> None:
    first_sources = await _current_sources(
        tmp_path / "first",
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
    )
    second_sources = await _current_sources(
        tmp_path / "second",
        blocked=False,
    )
    compiler = ItemQualityCompiler()
    first = compiler.compile(
        task_contract_set=first_sources[0],
        review_policy=first_sources[1],
        semantic_workflow=first_sources[2],
        candidate_revision=first_sources[3],
        deterministic_validation=first_sources[4],
        source_deterministic_validation=first_sources[5],
        job_store=first_sources[6],
        job_id=first_sources[7],
        item_id=first_sources[8],
        audit=_audit(),
    )
    second = compiler.compile(
        task_contract_set=second_sources[0],
        review_policy=second_sources[1],
        semantic_workflow=second_sources[2],
        candidate_revision=second_sources[3],
        deterministic_validation=second_sources[4],
        source_deterministic_validation=second_sources[5],
        job_store=second_sources[6],
        job_id=second_sources[7],
        item_id=second_sources[8],
        audit=_audit(),
    )
    assert first.final_package_manifest is not None
    assert second.provenance_manifest is not None
    values = first.model_dump(mode="python")
    values["provenance_manifest"] = second.provenance_manifest.model_dump(mode="python")
    values["provenance_manifest_ref"] = second.provenance_manifest_ref

    with pytest.raises(
        ValidationError,
        match="nested package refs",
    ):
        ItemQualityCompilationResultV2.model_validate(values)


@pytest.mark.asyncio
async def test_item_quality_result_rejects_cross_artifact_environment_ownership(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(
            ("artifact://first", "inputs/first.txt"),
            ("artifact://second", "inputs/second.txt"),
        ),
    )
    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    assert result.final_package_manifest is not None
    assert result.provenance_manifest is not None
    assert result.environment_spec is not None
    first, second = result.environment_spec.artifacts
    tampered_artifacts = (
        EnvironmentArtifactV2.create(
            artifact_id=first.artifact_id,
            attachment_dependency_id=first.attachment_dependency_id,
            candidate_artifact_version_ref=first.candidate_artifact_version_ref,
            output_ref=first.output_ref,
            logical_path=first.logical_path,
            media_type=first.media_type,
            size_bytes=first.size_bytes,
            artifact_validation_result_ref=first.artifact_validation_result_ref,
            package_inventory_entry_refs=second.package_inventory_entry_refs,
        ),
        EnvironmentArtifactV2.create(
            artifact_id=second.artifact_id,
            attachment_dependency_id=second.attachment_dependency_id,
            candidate_artifact_version_ref=second.candidate_artifact_version_ref,
            output_ref=second.output_ref,
            logical_path=second.logical_path,
            media_type=second.media_type,
            size_bytes=second.size_bytes,
            artifact_validation_result_ref=second.artifact_validation_result_ref,
            package_inventory_entry_refs=first.package_inventory_entry_refs,
        ),
    )
    environment = result.environment_spec
    tampered_environment = EnvironmentSpecV2.create(
        candidate_revision_ref=environment.candidate_revision_ref,
        final_package_manifest_ref=environment.final_package_manifest_ref,
        provenance_manifest_ref=environment.provenance_manifest_ref,
        candidate_artifact_version_refs=environment.candidate_artifact_version_refs,
        artifacts=tampered_artifacts,
        package_sha256=environment.package_sha256,
        audit=_audit(
            environment.candidate_revision_ref,
            environment.final_package_manifest_ref,
            environment.provenance_manifest_ref,
            *environment.candidate_artifact_version_refs,
            *(environment_artifact_ref(item) for item in tampered_artifacts),
        ),
    )
    tampered_environment_ref = environment_spec_v2_ref(tampered_environment)
    report = result.quality_report
    report_audit_refs = tuple(
        tampered_environment_ref if ref == report.environment_spec_ref else ref
        for ref in report.audit.input_refs
    )
    tampered_report = QualityReportV2.create(
        r4_task_contract_set_ref=report.r4_task_contract_set_ref,
        review_policy_ref=report.review_policy_ref,
        semantic_workflow_result_ref=report.semantic_workflow_result_ref,
        candidate_revision_ref=report.candidate_revision_ref,
        deterministic_validation_ref=report.deterministic_validation_ref,
        source_deterministic_validation_ref=report.source_deterministic_validation_ref,
        candidate_inventory_ref=report.candidate_inventory_ref,
        artifact_version_refs=report.artifact_version_refs,
        output_refs=report.output_refs,
        artifact_validation_result_refs=report.artifact_validation_result_refs,
        semantic_round_result_refs=report.semantic_round_result_refs,
        stage_result_refs=report.stage_result_refs,
        repair_plan_refs=report.repair_plan_refs,
        repair_result_refs=report.repair_result_refs,
        current_finding_refs=report.current_finding_refs,
        stale_finding_refs=report.stale_finding_refs,
        resolution_refs=report.resolution_refs,
        accepted_artifact_refs=report.accepted_artifact_refs,
        final_package_manifest_ref=report.final_package_manifest_ref,
        provenance_manifest_ref=report.provenance_manifest_ref,
        environment_spec_ref=tampered_environment_ref,
        package_sha256=report.package_sha256,
        input_state_only=report.input_state_only,
        outcome=report.outcome,
        failure_code=report.failure_code,
        open_p0_count=report.open_p0_count,
        open_p1_count=report.open_p1_count,
        unresolved_non_waivable_count=report.unresolved_non_waivable_count,
        approvable=report.approvable,
        audit=_audit(*report_audit_refs),
    )

    with pytest.raises(
        ValidationError,
        match="cross-artifact",
    ):
        ItemQualityCompilationResultV2.create(
            quality_report=tampered_report,
            final_package_manifest=result.final_package_manifest,
            provenance_manifest=result.provenance_manifest,
            environment_spec=tampered_environment,
            audit=_audit(
                quality_report_v2_ref(tampered_report),
                final_package_manifest_ref(result.final_package_manifest),
                provenance_manifest_v2_ref(result.provenance_manifest),
                tampered_environment_ref,
                *tampered_report.accepted_artifact_refs,
            ),
        )


@pytest.mark.asyncio
async def test_item_quality_compiler_preserves_blocked_without_package_truth(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=True)

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.outcome is ItemQualityOutcomeV2.BLOCKED
    assert result.quality_report.approvable is False
    assert result.final_package_manifest is None
    assert result.provenance_manifest is None
    assert result.environment_spec is None
    assert result.accepted_artifact_refs == ()
    assert result.package_sha256 is None
    assert result.input_state_only is None


@pytest.mark.asyncio
async def test_item_quality_compiler_derives_current_p1_count(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        item_scoped_repair=True,
    )

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.outcome is (ItemQualityOutcomeV2.REQUIRES_REPAIR)
    assert result.quality_report.open_p0_count == 0
    assert result.quality_report.open_p1_count == 1
    assert result.quality_report.unresolved_non_waivable_count == 0
    assert result.quality_report.current_finding_refs == (sources[2].current_finding_refs)


@pytest.mark.asyncio
async def test_item_quality_compiler_derives_non_waivable_p0_count(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
        validation_outcomes={"artifact://input": "secret"},
    )

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.outcome is ItemQualityOutcomeV2.REJECTED
    assert result.quality_report.open_p0_count == 1
    assert result.quality_report.open_p1_count == 0
    assert result.quality_report.unresolved_non_waivable_count == 1
    assert result.quality_report.approvable is False
    assert result.final_package_manifest is None


@pytest.mark.asyncio
async def test_item_quality_compiler_blocks_cross_artifact_inventory(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(
        tmp_path,
        blocked=False,
        artifacts=(("artifact://input", "inputs/source.txt"),),
        mismatched_candidate_artifact=True,
    )

    result = ItemQualityCompiler().compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )

    assert result.quality_report.outcome is ItemQualityOutcomeV2.BLOCKED
    assert result.quality_report.failure_code is (ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
    assert result.quality_report.approvable is False
    assert result.final_package_manifest is None


@pytest.mark.asyncio
async def test_item_quality_compiler_rejects_stale_source_identities(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=False)
    compiler = ItemQualityCompiler()
    cases = (
        ({"task_contract_set": sources[0].model_copy(update={"contract_set_sha256": "b" * 64})},),
        ({"review_policy": sources[1].model_copy(update={"policy_sha256": "b" * 64})},),
        ({"candidate_revision": sources[3].model_copy(update={"candidate_revision_sha256": "b" * 64})},),
        ({"semantic_workflow": sources[2].model_copy(update={"workflow_result_sha256": "b" * 64})},),
    )
    base = {
        "task_contract_set": sources[0],
        "review_policy": sources[1],
        "semantic_workflow": sources[2],
        "candidate_revision": sources[3],
        "deterministic_validation": sources[4],
        "source_deterministic_validation": sources[5],
        "job_store": sources[6],
        "job_id": sources[7],
        "item_id": sources[8],
        "audit": _audit(),
    }

    for (override,) in cases:
        with pytest.raises(ItemQualityPolicyError, match="stale"):
            compiler.compile(**(base | override))


@pytest.mark.asyncio
async def test_item_quality_validate_current_rejects_result_hash_drift(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=False)
    compiler = ItemQualityCompiler()
    result = compiler.compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    stale = result.model_copy(update={"result_sha256": "b" * 64})

    with pytest.raises(
        ItemQualityPolicyError,
        match="current item quality validation",
    ):
        compiler.validate_current(
            stale,
            task_contract_set=sources[0],
            review_policy=sources[1],
            semantic_workflow=sources[2],
            candidate_revision=sources[3],
            deterministic_validation=sources[4],
            source_deterministic_validation=sources[5],
            job_store=sources[6],
            job_id=sources[7],
            item_id=sources[8],
        )


@pytest.mark.asyncio
async def test_item_quality_validate_current_rejects_nested_audit_version_drift(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=False)
    compiler = ItemQualityCompiler()
    result = compiler.compile(
        task_contract_set=sources[0],
        review_policy=sources[1],
        semantic_workflow=sources[2],
        candidate_revision=sources[3],
        deterministic_validation=sources[4],
        source_deterministic_validation=sources[5],
        job_store=sources[6],
        job_id=sources[7],
        item_id=sources[8],
        audit=_audit(),
    )
    stale_report = result.quality_report.model_copy(
        update={
            "audit": result.quality_report.audit.model_copy(
                update={
                    "governing_versions": (
                        VersionBinding(
                            component="item-quality",
                            version="item-quality/tampered",
                        ),
                    )
                }
            )
        }
    )
    stale = result.model_copy(update={"quality_report": stale_report})

    with pytest.raises(ItemQualityPolicyError, match="stale"):
        compiler.validate_current(
            stale,
            task_contract_set=sources[0],
            review_policy=sources[1],
            semantic_workflow=sources[2],
            candidate_revision=sources[3],
            deterministic_validation=sources[4],
            source_deterministic_validation=sources[5],
            job_store=sources[6],
            job_id=sources[7],
            item_id=sources[8],
        )


@pytest.mark.asyncio
async def test_item_quality_compiler_rejects_stale_task_binding(
    tmp_path: Path,
) -> None:
    sources = await _current_sources(tmp_path, blocked=False)
    stale_task_set = _task_contract_set(_ref("producer-task-view", "stale"))

    with pytest.raises(ItemQualityPolicyError, match="producer task view"):
        ItemQualityCompiler().compile(
            task_contract_set=stale_task_set,
            review_policy=sources[1],
            semantic_workflow=sources[2],
            candidate_revision=sources[3],
            deterministic_validation=sources[4],
            source_deterministic_validation=sources[5],
            job_store=sources[6],
            job_id=sources[7],
            item_id=sources[8],
            audit=_audit(),
        )
