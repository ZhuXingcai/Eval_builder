from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest
from test_artifact_execution import _audit as _execution_audit
from test_artifact_results import _result_audit
from test_artifact_routing import (
    _audit as _routing_audit,
)
from test_artifact_routing import (
    _ref as _routing_ref,
)
from test_artifact_routing import _routing_policy
from test_attachment_evidence_modes import (
    _case_inputs,
)
from test_attachment_evidence_modes import (
    _compile as _compile_mode,
)
from test_deterministic_validation import (
    _audit as _validation_audit,
)
from test_deterministic_validation import _NoCallFacade, _reference_set
from test_item_quality import _task_contract_set
from test_semantic_review import (
    _AcceptingBackend,
    _DynamicResolver,
    _NoRepairBackend,
    _NoRevalidation,
    _policy,
    _sources,
    _store,
)
from test_semantic_review import (
    _audit as _review_audit,
)

from env_mock_agent.facade import (
    FacadeObjectRef,
    RegistryAttachmentRoutingFacade,
)
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    ProviderExecutionMaterial,
    RegistryAttachmentExecutionFacade,
    world_ledger_object_ref,
)
from env_mock_agent.facade.semantic_review_adapter import (
    RegistryAttachmentSemanticReviewFacade,
)
from env_mock_agent.facade.validation_adapter import (
    ProviderValidationMaterial,
    RegistryAttachmentValidationFacade,
    StagingAttachmentValidationMaterialResolver,
)
from env_mock_agent.providers import ProviderRegistry, TextProvider
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.schemas import ArtifactPlan, WorldLedger
from eval_factory.attachment_planning import (
    ArtifactBuildContractDefinition,
    ArtifactBuildSpecCompiler,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompiler,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPreparationBuilder,
    ArtifactGroupExecutor,
    ArtifactResultCompiler,
    ArtifactRoutingRequestBuilder,
    ArtifactRoutingRunner,
    DeterministicValidationCompiler,
    WorldLedgerSnapshotRunner,
)
from eval_factory.attachment_planning.quality import ItemQualityCompiler
from eval_factory.attachment_planning.review import (
    CandidateRevisionCompiler,
    IsolatedSemanticReviewOrchestrator,
)
from eval_factory.contracts import (
    AttachmentReconstructionOutcomeV2,
    DeterministicItemValidationOutcomeV2,
    artifact_build_spec_v2_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.quality_v2 import (
    ItemQualityOutcomeV2,
    final_package_content_sha256,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewRoundV2,
    SemanticReviewWorkflowOutcomeV2,
)
from eval_factory.contracts.task_v2 import producer_task_view_ref

ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/attachment_modes" / "r5-three-mode-package-canaries-v1.json"
NOW = _execution_audit().created_at
PRIVATE_PROVIDER_CONTENT = "Safe input state generated for the package canary."


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _definition(
    dependency_id: str,
    *,
    provider_payload: bool,
) -> ArtifactBuildContractDefinition:
    suffix = dependency_id.rsplit("/", 1)[-1]
    return ArtifactBuildContractDefinition(
        definition_id=f"artifact-build-contract-definition://canary/{suffix}",
        attachment_dependency_id=dependency_id,
        asset_type="txt",
        content_contract_ref=_routing_ref(
            "artifact-content-contract",
            f"canary/{suffix}",
            version="v2",
        ),
        render_contract_ref=_routing_ref(
            "artifact-render-contract",
            f"canary/{suffix}",
            version="v2",
        ),
        provider_payload_ref=(
            _routing_ref(
                "attachment-provider-payload",
                f"canary/{suffix}",
                version="v2",
            )
            if provider_payload
            else None
        ),
        source_evidence_set_ref=None,
        required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
        required_runtime_tools=("write",),
        runtime_role="attachment-writer",
        runtime_resume_required=True,
        validator_ids=("secret-validator", "text-validator"),
    )


async def _routing_case(
    setup: str,
    *,
    include_definition: bool,
    provider_payload: bool,
) -> SimpleNamespace:
    inputs = _case_inputs((setup,))
    mode_result = _compile_mode(inputs)
    matrix = mode_result.artifact_evidence_matrix
    assert matrix is not None
    target = inputs[6][0]
    ArtifactEvidenceModeCompiler().validate_current(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        evidence_bundle=inputs[3],
        evidence_view_result=inputs[4],
        timelines=inputs[5],
        targets=inputs[6],
        artifact_evidence_matrix=matrix,
    )

    definitions = (
        (
            _definition(
                target.attachment_dependency_id,
                provider_payload=provider_payload,
            ),
        )
        if include_definition
        else ()
    )
    policy = _routing_policy()
    request = ArtifactRoutingRequestBuilder().build(
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        producer_view_result=inputs[4],
        producer_evidence_bundle=inputs[3],
        artifact_evidence_matrix=matrix,
        artifact_targets=inputs[6],
        routing_policy=policy,
        contract_definitions=definitions,
        audit=_routing_audit(),
    )
    providers = ProviderRegistry()
    providers.register(TextProvider())
    execution = await ArtifactRoutingRunner().run(
        request,
        facade=RegistryAttachmentRoutingFacade(
            providers,
            RuntimeRegistry(),
            clock=lambda: NOW,
        ),
        audit=_routing_audit(),
    )
    compiler = ArtifactBuildSpecCompiler()
    result = compiler.compile(
        request=request,
        execution_result=execution,
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        producer_view_result=inputs[4],
        producer_evidence_bundle=inputs[3],
        artifact_evidence_matrix=matrix,
        artifact_targets=inputs[6],
        routing_policy=policy,
        audit=_routing_audit(),
    )
    compiler.validate_current(
        request=request,
        execution_result=execution,
        compilation_result=result,
        attachment_planning_context=inputs[0],
        producer_task_view=inputs[1],
        storage_authorization=inputs[2],
        producer_view_result=inputs[4],
        producer_evidence_bundle=inputs[3],
        artifact_evidence_matrix=matrix,
        artifact_targets=inputs[6],
        routing_policy=policy,
    )
    return SimpleNamespace(
        mode_result=mode_result,
        matrix=matrix,
        target=target,
        context=inputs[0],
        producer_view=inputs[1],
        authorization=inputs[2],
        bundle=inputs[3],
        view_result=inputs[4],
        policy=policy,
        request=request,
        routing_execution=execution,
        routing_result=result,
    )


def _provider_plan(route: SimpleNamespace) -> ArtifactPlan:
    assert route.routing_result.routing_plan is not None
    entry = route.routing_result.routing_plan.entries[0]
    assert entry.build_spec is not None
    frozen = entry.build_spec.build_spec
    return ArtifactPlan(
        artifact_id=route.target.artifact_id,
        dependency_id=route.target.attachment_dependency_id,
        relative_path=frozen.relative_path,
        asset_type="txt",
        provider="text",
        content_contract={"content": PRIVATE_PROVIDER_CONTENT},
        render_contract={},
        validators=list(frozen.validator_ids),
        source_evidence_ids=[item.evidence_ref_id for item in frozen.authorized_evidence_refs],
    )


async def _successful_chain(
    mode: str,
    tmp_path: Path,
) -> SimpleNamespace:
    route = await _routing_case(
        mode,
        include_definition=True,
        provider_payload=True,
    )
    assert route.routing_result.routing_plan is not None
    entry = route.routing_result.routing_plan.entries[0]
    assert entry.build_spec is not None
    contract = route.request.build_contracts[0]
    assert contract.provider_payload_ref is not None
    plan = _provider_plan(route)
    build_spec_ref = artifact_build_spec_v2_ref(entry.build_spec)

    ledger = WorldLedger()
    ledger_ref = world_ledger_object_ref(
        f"world-ledger://r5-canary/{mode.casefold()}",
        ledger,
    )
    provider_material = ProviderExecutionMaterial(
        producer_task_view_ref=_facade_ref(entry.build_spec.build_spec.producer_task_view_ref),
        content_contract_ref=_facade_ref(entry.build_spec.build_spec.content_contract_ref),
        render_contract_ref=_facade_ref(entry.build_spec.build_spec.render_contract_ref),
        provider_payload_ref=_facade_ref(contract.provider_payload_ref),
        authorized_evidence_ref_ids=tuple(
            item.evidence_ref_id for item in entry.build_spec.build_spec.authorized_evidence_refs
        ),
        plan=plan,
    )
    providers = ProviderRegistry()
    providers.register(TextProvider())
    staging_root = tmp_path / "private-staging"
    execution_facade = RegistryAttachmentExecutionFacade(
        providers,
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
            provider_materials={
                build_spec_ref.object_id: provider_material,
            },
        ),
        staging_root=staging_root,
        clock=lambda: NOW,
    )
    preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=route.request,
        routing_result=route.routing_result,
        definitions=(
            ArtifactConsistencyDefinition(
                artifact_id=route.target.artifact_id,
                dependency_artifact_ids=(),
                locked_fact_ids=(),
            ),
        ),
        world_ledger_ref=ledger_ref,
        audit=_execution_audit(),
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        preparation,
        facade=execution_facade,
    )
    execution_result = ArtifactExecutionPlanCompiler().compile(
        routing_request=route.request,
        routing_result=route.routing_result,
        preparation=preparation,
        world_ledger_snapshot=snapshot,
        audit=_execution_audit(),
    )
    ArtifactExecutionPlanCompiler().validate_current(execution_result)
    assert execution_result.execution_plan is not None
    batch = await ArtifactGroupExecutor().run(
        execution_result.execution_plan,
        facade=execution_facade,
        audit=_execution_audit(),
    )
    ArtifactGroupExecutor().validate_current(
        execution_result.execution_plan,
        batch,
    )

    result_compiler = ArtifactResultCompiler()
    reconstruction = result_compiler.compile(
        routing_request=route.request,
        routing_result=route.routing_result,
        execution_result=execution_result,
        execution_batch=batch,
        audit=_result_audit(),
    )
    result_compiler.validate_current(
        reconstruction,
        routing_request=route.request,
        routing_result=route.routing_result,
        execution_result=execution_result,
        execution_batch=batch,
    )

    execution_request = batch.receipts[0].facade_request
    assert execution_request is not None
    validation_resolver = StagingAttachmentValidationMaterialResolver(
        staging_root=staging_root,
        execution_requests={
            execution_request.execution_request_id: execution_request,
        },
        materials={
            build_spec_ref.object_id: ProviderValidationMaterial(plan=plan),
        },
    )
    leakage_reference_set = _reference_set()
    validation_compiler = DeterministicValidationCompiler()
    source_validation = await validation_compiler.compile(
        reconstruction_result=reconstruction,
        producer_task_view=route.producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
        facade=RegistryAttachmentValidationFacade(
            resolver=validation_resolver,
        ),
        audit=_validation_audit(),
    )
    validation_compiler.validate_current(
        source_validation,
        reconstruction_result=reconstruction,
        producer_task_view=route.producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
    )
    candidate_revision, deterministic_validation = CandidateRevisionCompiler().compile_initial(
        reconstruction_result=reconstruction,
        deterministic_validation=source_validation,
        audit=_review_audit(),
    )

    store, job_id, item_id = _store(tmp_path / "semantic-review")
    review_policy = _policy()
    backend = _AcceptingBackend()
    review_facade = RegistryAttachmentSemanticReviewFacade(
        resolver=_DynamicResolver(),
        review_backends={role: backend for role in review_policy.roles_in_order},
        repair_backend=_NoRepairBackend(),
    )
    workflow = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        review_policy=review_policy,
        context_sources=_sources(),
        review_facade=review_facade,
        revision_validator=_NoRevalidation(),
        audit=_review_audit(),
    )
    quality_compiler = ItemQualityCompiler()
    task_contract_set = _task_contract_set(producer_task_view_ref(route.producer_view))
    quality = quality_compiler.compile(
        task_contract_set=task_contract_set,
        review_policy=review_policy,
        semantic_workflow=workflow,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        source_deterministic_validation=source_validation,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        audit=_review_audit(),
    )
    quality_compiler.validate_current(
        quality,
        task_contract_set=task_contract_set,
        review_policy=review_policy,
        semantic_workflow=workflow,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        source_deterministic_validation=source_validation,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
    )
    return SimpleNamespace(
        **vars(route),
        execution_result=execution_result,
        batch=batch,
        reconstruction=reconstruction,
        source_validation=source_validation,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        workflow=workflow,
        quality=quality,
        backend=backend,
        store=store,
        job_id=job_id,
        item_id=item_id,
        staging_root=staging_root,
    )


async def _blocked_chain(
    setup: str,
    *,
    include_definition: bool,
    provider_payload: bool,
    tmp_path: Path,
) -> SimpleNamespace:
    route = await _routing_case(
        setup,
        include_definition=include_definition,
        provider_payload=provider_payload,
    )
    ledger = WorldLedger()
    ledger_ref = world_ledger_object_ref(
        f"world-ledger://r5-canary/{setup.casefold()}",
        ledger,
    )
    execution_facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
        ),
        staging_root=tmp_path / "private-staging",
        clock=lambda: NOW,
    )
    preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=route.request,
        routing_result=route.routing_result,
        definitions=(),
        world_ledger_ref=ledger_ref,
        audit=_execution_audit(),
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        preparation,
        facade=execution_facade,
    )
    execution_result = ArtifactExecutionPlanCompiler().compile(
        routing_request=route.request,
        routing_result=route.routing_result,
        preparation=preparation,
        world_ledger_snapshot=snapshot,
        audit=_execution_audit(),
    )
    ArtifactExecutionPlanCompiler().validate_current(execution_result)
    result_compiler = ArtifactResultCompiler()
    reconstruction = result_compiler.compile(
        routing_request=route.request,
        routing_result=route.routing_result,
        execution_result=execution_result,
        execution_batch=None,
        audit=_result_audit(),
    )
    result_compiler.validate_current(
        reconstruction,
        routing_request=route.request,
        routing_result=route.routing_result,
        execution_result=execution_result,
        execution_batch=None,
    )

    leakage_reference_set = _reference_set()
    validation_compiler = DeterministicValidationCompiler()
    source_validation = await validation_compiler.compile(
        reconstruction_result=reconstruction,
        producer_task_view=route.producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
        facade=_NoCallFacade(),
        audit=_validation_audit(),
    )
    validation_compiler.validate_current(
        source_validation,
        reconstruction_result=reconstruction,
        producer_task_view=route.producer_view,
        leakage_reference_set=leakage_reference_set,
        configured_pii_rules=(),
    )
    candidate_revision, deterministic_validation = CandidateRevisionCompiler().compile_initial(
        reconstruction_result=reconstruction,
        deterministic_validation=source_validation,
        audit=_review_audit(),
    )
    store, job_id, item_id = _store(tmp_path / "semantic-review")
    review_policy = _policy()
    workflow = await IsolatedSemanticReviewOrchestrator().run(
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        review_policy=review_policy,
        context_sources=_sources(),
        review_facade=RegistryAttachmentSemanticReviewFacade(
            resolver=_DynamicResolver(),
            review_backends={},
            repair_backend=None,
        ),
        revision_validator=_NoRevalidation(),
        audit=_review_audit(),
    )
    task_contract_set = _task_contract_set(producer_task_view_ref(route.producer_view))
    quality_compiler = ItemQualityCompiler()
    quality = quality_compiler.compile(
        task_contract_set=task_contract_set,
        review_policy=review_policy,
        semantic_workflow=workflow,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        source_deterministic_validation=source_validation,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
        audit=_review_audit(),
    )
    quality_compiler.validate_current(
        quality,
        task_contract_set=task_contract_set,
        review_policy=review_policy,
        semantic_workflow=workflow,
        candidate_revision=candidate_revision,
        deterministic_validation=deterministic_validation,
        source_deterministic_validation=source_validation,
        job_store=store,
        job_id=job_id,
        item_id=item_id,
    )
    return SimpleNamespace(
        **vars(route),
        execution_result=execution_result,
        batch=None,
        reconstruction=reconstruction,
        source_validation=source_validation,
        workflow=workflow,
        quality=quality,
        store=store,
        job_id=job_id,
        item_id=item_id,
    )


def _gold() -> dict[str, object]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def _assert_content_free(value: object) -> None:
    forbidden = (
        "artifact_content",
        "provider_bytes",
        "raw_trace_payload",
        "private-reference://",
        "grader_rule",
        "final_answer",
        "hidden_condition",
        "secret_value",
        "runtime_transcript",
        "physical_path",
        "reviewer_reasoning",
        PRIVATE_PROVIDER_CONTENT.casefold(),
        "/users/bytedance/",
        "file://",
    )
    absent_only_key_tokens = (
        "release_decision",
        "release_state",
        "user_decision",
        "registry",
    )
    forbidden_path_keys = frozenset(
        {
            "absolute_path",
            "content_path",
            "local_path",
            "output_location",
            "path",
            "physical_path",
            "staging_path",
            "staging_root",
            "workspace",
        }
    )
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized_key = str(key).casefold()
            assert normalized_key not in forbidden_path_keys
            if any(token in normalized_key for token in absent_only_key_tokens):
                assert nested in (None, (), [], {})
                continue
            _assert_content_free(key)
            _assert_content_free(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_content_free(nested)
        return
    if isinstance(value, str):
        lowered = value.casefold()
        assert not any(item in lowered for item in forbidden)
        assert not PurePosixPath(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()


def test_parent_canary_gold_is_content_free_and_closed() -> None:
    payload = _gold()

    assert payload["schema_version"] == ("eval-factory/r5-three-mode-package-canaries-gold/v1")
    assert payload["claim_scope"] == ("DIRECT_CROSS_CHILD_EXECUTION_AND_FINAL_PACKAGE")
    assert {item["mode"] for item in payload["success_cases"]} == {
        "TRACE_RICH",
        "SKELETON_GUIDED",
        "PROMPT_ONLY",
    }
    assert {item["routing_outcome"] for item in payload["blocked_cases"]} == {
        "BLOCKED_POLICY",
        "BLOCKED_CAPABILITY",
    }
    _assert_content_free(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"staging_path": "relative-but-private"},
        {"opaque": "/private/tmp/provider-output.txt"},
        {"opaque": r"C:\private\provider-output.txt"},
    ],
)
def test_content_free_audit_rejects_physical_paths(
    payload: dict[str, str],
) -> None:
    with pytest.raises(AssertionError):
        _assert_content_free(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "TRACE_RICH",
        "SKELETON_GUIDED",
        "PROMPT_ONLY",
    ],
)
async def test_each_mode_reaches_real_provider_and_approvable_package(
    mode: str,
    tmp_path: Path,
) -> None:
    expected = next(item for item in _gold()["success_cases"] if item["mode"] == mode)
    chain = await _successful_chain(mode, tmp_path)
    assert chain.routing_result.routing_plan is not None
    route_entry = chain.routing_result.routing_plan.entries[0]
    assert route_entry.build_spec is not None
    execution_request = chain.batch.receipts[0].facade_request
    assert execution_request is not None

    assert chain.matrix.rows[0].row.selected_mode.value == mode
    assert chain.request.build_contracts[0].mode.value == mode
    assert route_entry.build_spec.build_spec.mode.value == mode
    assert execution_request.mode.value == mode
    assert chain.routing_result.outcome.value == expected["routing_outcome"]
    assert route_entry.outcome.value == expected["route_entry_outcome"]
    assert chain.batch.receipts[0].facade_result is not None
    assert chain.batch.receipts[0].facade_result.status.value == expected["execution_status"]
    assert execution_request.selected_route_id == "text"
    assert chain.reconstruction.outcome is (AttachmentReconstructionOutcomeV2.PENDING_VALIDATION)
    assert chain.source_validation.outcome.value == expected["validation_outcome"]
    assert chain.source_validation.outcome is (DeterministicItemValidationOutcomeV2.PASSED)
    assert chain.workflow.outcome is SemanticReviewWorkflowOutcomeV2.PASSED
    assert len(chain.workflow.round_results) == expected["semantic_round_count"]
    assert [item.round for item in chain.workflow.round_results] == list(SemanticReviewRoundV2)
    assert all(item.accepted for item in chain.workflow.round_results)
    assert len(set(chain.backend.context_ids)) == expected["semantic_round_count"]

    stage_runs = chain.store.list_stage_runs(
        job_id=chain.job_id,
        item_id=chain.item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )
    assert len(stage_runs) == expected["semantic_round_count"]
    assert all(item.status is StageRunStatus.SUCCEEDED for item in stage_runs)
    assert all(chain.store.get_stage_result_for_run(item.stage_run_id) is not None for item in stage_runs)

    quality = chain.quality
    assert quality.quality_report.outcome.value == expected["quality_outcome"]
    assert quality.quality_report.outcome is ItemQualityOutcomeV2.PASSED
    assert quality.quality_report.approvable is expected["approvable"]
    assert quality.input_state_only is expected["input_state_only"]
    assert quality.final_package_manifest is not None
    assert quality.provenance_manifest is not None
    assert quality.environment_spec is not None
    manifest = quality.final_package_manifest
    provenance = quality.provenance_manifest
    environment = quality.environment_spec
    assert len(manifest.entries) == expected["package_member_count"]
    assert len(provenance.entries) == expected["provenance_binding_count"]
    assert len(environment.artifacts) == expected["environment_artifact_count"]
    assert manifest.package_sha256 == final_package_content_sha256(manifest.entries)
    assert manifest.entry_refs == provenance.package_inventory_entry_refs
    assert environment.artifacts[0].package_inventory_entry_refs == (manifest.entry_refs)
    assert manifest.output_refs == chain.candidate_revision.candidate_output_refs
    assert manifest.accepted_artifact_refs == (chain.candidate_revision.artifact_version_refs)
    assert {
        "build_spec": route_entry.build_spec.artifact_build_spec_v2_sha256,
        "execution_request": execution_request.execution_request_sha256,
        "reconstruction": (chain.reconstruction.attachment_reconstruction_result_v2_sha256),
        "source_validation": (chain.source_validation.deterministic_item_validation_result_sha256),
        "candidate_revision": chain.candidate_revision.candidate_revision_sha256,
        "revision_validation": chain.deterministic_validation.validation_sha256,
        "stage_results": [item.object_sha256 for item in chain.workflow.stage_result_refs],
        "workflow": chain.workflow.workflow_result_sha256,
        "package": manifest.package_sha256,
        "package_entry": manifest.entry_refs[0].object_sha256,
        "provenance_entry": provenance.entry_refs[0].object_sha256,
        "environment_artifact": environment.artifacts[0].artifact_sha256,
        "quality_report": quality.quality_report.quality_report_sha256,
    } == expected["identity_sha256"]
    _assert_content_free(
        {
            "mode_result": chain.mode_result.model_dump(mode="json"),
            "routing_result": chain.routing_result.model_dump(mode="json"),
            "execution_batch": chain.batch.model_dump(mode="json"),
            "reconstruction": chain.reconstruction.model_dump(mode="json"),
            "validation": chain.source_validation.model_dump(mode="json"),
            "workflow": chain.workflow.model_dump(mode="json"),
            "quality": quality.model_dump(mode="json"),
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "case_id",
        "include_definition",
        "provider_payload",
    ),
    [
        (
            "critical-first-write-policy-block",
            False,
            False,
        ),
        (
            "provider-and-runtime-capability-block",
            True,
            False,
        ),
    ],
)
async def test_blocked_modes_never_gain_execution_or_package_truth(
    case_id: str,
    include_definition: bool,
    provider_payload: bool,
    tmp_path: Path,
) -> None:
    expected = next(item for item in _gold()["blocked_cases"] if item["case_id"] == case_id)
    chain = await _blocked_chain(
        expected["setup"],
        include_definition=include_definition,
        provider_payload=provider_payload,
        tmp_path=tmp_path,
    )
    assert chain.routing_result.routing_plan is not None
    entry = chain.routing_result.routing_plan.entries[0]

    assert chain.matrix.rows[0].row.selected_mode.value == expected["mode"]
    assert chain.routing_result.outcome.value == expected["routing_outcome"]
    assert entry.outcome.value == expected["route_entry_outcome"]
    assert (
        sum(item.build_spec is not None for item in chain.routing_result.routing_plan.entries)
        == expected["build_spec_count"]
    )
    assert (chain.execution_result.execution_plan is not None) is expected["execution_plan"]
    assert (chain.batch is not None) is expected["execution_batch"]
    assert chain.reconstruction.outcome is (AttachmentReconstructionOutcomeV2.BLOCKED)
    assert chain.reconstruction.artifact_execution_plan_ref is None
    assert chain.reconstruction.artifact_execution_batch_ref is None
    assert chain.reconstruction.candidate_output_refs == ()
    assert chain.source_validation.outcome is (DeterministicItemValidationOutcomeV2.UPSTREAM_INCOMPLETE)
    assert chain.workflow.outcome is (SemanticReviewWorkflowOutcomeV2.UPSTREAM_INCOMPLETE)
    assert chain.workflow.stage_result_refs == ()
    assert chain.quality.quality_report.outcome.value == expected["quality_outcome"]
    assert chain.quality.quality_report.approvable is expected["approvable"]
    assert (chain.quality.final_package_manifest is not None) is expected["final_package"]
    assert chain.quality.provenance_manifest is None
    assert chain.quality.environment_spec is None
    assert chain.quality.package_sha256 is None
    assert chain.quality.input_state_only is None
