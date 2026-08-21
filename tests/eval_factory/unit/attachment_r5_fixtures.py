from __future__ import annotations

from pathlib import Path

from test_approved_source_evidence import (
    _FakeFacade as _SourceFacade,
)
from test_approved_source_evidence import (
    _policy as _source_policy,
)
from test_approved_source_evidence import (
    _search_intent,
    _verified_checks,
)
from test_artifact_execution import (
    NOW,
    _ref,
)
from test_artifact_execution import (
    _audit as _execution_audit,
)
from test_artifact_routing import (
    _routing_policy,
)
from test_prompt_only_dependencies import (
    _audit,
    _case,
    _compile,
    _request,
)

from env_mock_agent.facade import (
    RegistryAttachmentRoutingFacade,
)
from env_mock_agent.facade.execution_adapter import (
    MappingAttachmentExecutionMaterialResolver,
    RegistryAttachmentExecutionFacade,
    world_ledger_object_ref,
)
from env_mock_agent.providers import ProviderRegistry, TextProvider
from env_mock_agent.runtimes import RuntimeRegistry
from env_mock_agent.schemas import WorldLedger
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
)
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceCompiler,
    ApprovedSourceRetrievalRequestBuilder,
    ApprovedSourceRetrievalRunner,
    ArtifactBuildContractDefinition,
    ArtifactBuildSpecCompiler,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompiler,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPreparationBuilder,
    ArtifactRoutingRequestBuilder,
    ArtifactRoutingRunner,
    WorldLedgerSnapshotRunner,
)
from eval_factory.contracts import (
    artifact_evidence_target_ref,
    source_evidence_set_ref,
)


async def build_attachment_r5_preparation(
    tmp_path: Path,
    *,
    descriptions: tuple[str, ...] = (
        "Use inputs/a.txt.",
        "Use inputs/b.txt.",
    ),
    include_source_evidence: bool = False,
) -> AttachmentR5PreparationAuthority:
    case = _case(descriptions=descriptions)
    dependency_request = _request(case)
    dependency_result = _compile(
        case,
        request=dependency_request,
    )
    assert dependency_result.discovery is not None
    discovery = dependency_result.discovery

    evidence_mode_result = ArtifactEvidenceModeCompiler().compile(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        evidence_bundle=case[4],
        evidence_view_result=case[3],
        timelines=(),
        targets=discovery.targets,
        audit=_audit(),
    )
    assert evidence_mode_result.artifact_evidence_matrix is not None

    retrieval_policy = _source_policy()
    intent_definitions = (_search_intent(case),) if include_source_evidence else ()
    retrieval_request = ApprovedSourceRetrievalRequestBuilder().build(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        dependency_discovery=discovery,
        retrieval_policy=retrieval_policy,
        intent_definitions=intent_definitions,
        audit=_audit(),
    )
    retrieval_execution_result = None
    verified_checks = ()
    if include_source_evidence:
        retrieval_execution_result = await ApprovedSourceRetrievalRunner().run(
            retrieval_request,
            facade=_SourceFacade(),
            audit=_audit(),
        )
        verified_checks = tuple(
            _verified_checks(execution) for execution in retrieval_execution_result.executions
        )
    source_evidence_result = ApprovedSourceEvidenceCompiler().compile(
        request=retrieval_request,
        execution_result=retrieval_execution_result,
        verified_checks=verified_checks,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        dependency_discovery=discovery,
        retrieval_policy=retrieval_policy,
        audit=_audit(),
    )

    evidence_set = source_evidence_result.source_evidence_set
    evidence_set_ref = source_evidence_set_ref(evidence_set) if evidence_set is not None else None
    covered_target_refs = set(evidence_set.covered_target_refs) if evidence_set is not None else set()
    contract_definitions = tuple(
        ArtifactBuildContractDefinition(
            definition_id=(f"artifact-build-contract-definition://{target.artifact_id.rsplit('://', 1)[-1]}"),
            attachment_dependency_id=target.attachment_dependency_id,
            asset_type="txt",
            content_contract_ref=_ref(
                "artifact-content-contract",
                target.artifact_id.rsplit("://", 1)[-1],
            ),
            render_contract_ref=_ref(
                "artifact-render-contract",
                target.artifact_id.rsplit("://", 1)[-1],
            ),
            provider_payload_ref=_ref(
                "attachment-provider-payload",
                target.artifact_id.rsplit("://", 1)[-1],
            ),
            source_evidence_set_ref=(
                evidence_set_ref if artifact_evidence_target_ref(target) in covered_target_refs else None
            ),
            required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
            required_runtime_tools=("write",),
            runtime_role="attachment-writer",
            runtime_resume_required=True,
            validator_ids=("secret-validator", "text-validator"),
        )
        for target in discovery.targets
    )
    routing_policy = _routing_policy()
    routing_request = ArtifactRoutingRequestBuilder().build(
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        artifact_evidence_matrix=(evidence_mode_result.artifact_evidence_matrix),
        artifact_targets=discovery.targets,
        routing_policy=routing_policy,
        contract_definitions=contract_definitions,
        audit=_audit(),
    )
    provider_registry = ProviderRegistry()
    provider_registry.register(TextProvider())
    routing_execution_result = await ArtifactRoutingRunner().run(
        routing_request,
        facade=RegistryAttachmentRoutingFacade(
            provider_registry,
            RuntimeRegistry(),
            clock=lambda: NOW,
        ),
        audit=_audit(),
    )
    routing_result = ArtifactBuildSpecCompiler().compile(
        request=routing_request,
        execution_result=routing_execution_result,
        attachment_planning_context=case[5],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        producer_evidence_bundle=case[4],
        artifact_evidence_matrix=(evidence_mode_result.artifact_evidence_matrix),
        artifact_targets=discovery.targets,
        routing_policy=routing_policy,
        audit=_audit(),
    )

    consistency_definitions = tuple(
        ArtifactConsistencyDefinition(
            artifact_id=target.artifact_id,
            dependency_artifact_ids=(),
            locked_fact_ids=(),
        )
        for target in discovery.targets
    )
    ledger = WorldLedger()
    ledger_ref = world_ledger_object_ref(
        "world-ledger://attachment-specialist",
        ledger,
    )
    execution_preparation = ArtifactExecutionPreparationBuilder().build(
        routing_request=routing_request,
        routing_result=routing_result,
        definitions=consistency_definitions,
        world_ledger_ref=ledger_ref,
        audit=_execution_audit(),
    )
    snapshot_facade = RegistryAttachmentExecutionFacade(
        ProviderRegistry(),
        RuntimeRegistry(),
        resolver=MappingAttachmentExecutionMaterialResolver(
            world_ledgers={ledger_ref.object_id: ledger},
        ),
        staging_root=tmp_path / "snapshot",
        clock=lambda: NOW,
    )
    snapshot = await WorldLedgerSnapshotRunner().run(
        execution_preparation,
        facade=snapshot_facade,
    )
    execution_result = ArtifactExecutionPlanCompiler().compile(
        routing_request=routing_request,
        routing_result=routing_result,
        preparation=execution_preparation,
        world_ledger_snapshot=snapshot,
        audit=_execution_audit(),
    )

    return AttachmentR5PreparationAuthority(
        task_draft=case[0],
        producer_task_view=case[1],
        storage_authorization=case[2],
        producer_view_result=case[3],
        evidence_bundle=case[4],
        attachment_planning_context=case[5],
        dependency_planning_context=case[6],
        dependency_request=dependency_request,
        dependency_proposal=None,
        existing_targets=(),
        dependency_result=dependency_result,
        timelines=(),
        evidence_mode_result=evidence_mode_result,
        retrieval_request=retrieval_request,
        retrieval_execution_result=retrieval_execution_result,
        verified_source_checks=verified_checks,
        retrieval_policy=retrieval_policy,
        source_evidence_result=source_evidence_result,
        routing_request=routing_request,
        routing_execution_result=routing_execution_result,
        routing_result=routing_result,
        routing_policy=routing_policy,
        consistency_definitions=consistency_definitions,
        world_ledger_ref=ledger_ref,
        execution_preparation=execution_preparation,
        execution_result=execution_result,
    )


__all__ = ["build_attachment_r5_preparation"]
