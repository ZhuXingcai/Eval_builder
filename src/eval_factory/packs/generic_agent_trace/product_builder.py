from __future__ import annotations

from pathlib import Path

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
    FactoryDatasetRuntime,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    FactoryDatasetFixtureComponents,
)
from eval_factory.agent_system.graph_bootstrap import (
    FactoryDatasetGraphBootstrap,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalNotFoundError,
    FactoryGraphJournalStore,
)
from eval_factory.agent_system.item_run_materializer import (
    FactoryItemRunMaterializer,
)
from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
)
from eval_factory.harness import (
    CapabilityRuntimeRegistry,
    HarnessCapabilityRuntime,
    HarnessGraphExecutionBindingV1,
    HarnessSessionProjectionV1,
    HarnessSessionStatusV1,
)
from eval_factory.harness.session_store import HarnessSessionStore
from eval_factory.packs.generic_agent_trace.candidate_aggregate import (
    GenericAgentCandidateAggregate,
)
from eval_factory.packs.generic_agent_trace.delivery_preparation import (
    GenericAgentDeliveryPreparation,
)
from eval_factory.packs.generic_agent_trace.execution_context import (
    GenericAgentBatchExecutionContext,
    GenericAgentFactoryExecutionContext,
)
from eval_factory.packs.generic_agent_trace.item_materials import (
    GenericAgentFactoryItemMaterialSource,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_execution_pack,
)
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)
from eval_factory.packs.generic_agent_trace.owner_runtime import (
    GenericAgentOwnerRuntime,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentPlanReviewPreparation,
)
from eval_factory.packs.generic_agent_trace.product_runtime import (
    GenericAgentGraphProduct,
    GenericAgentGraphProductError,
    GenericAgentGraphProductPaths,
    GenericAgentGraphRuntimeConfigV1,
)
from eval_factory.packs.generic_agent_trace.provider_registry import (
    build_first_party_capability_providers,
)
from eval_factory.packs.generic_agent_trace.request_coordinator import (
    GenericAgentFactoryRequestCoordinator,
)
from eval_factory.packs.generic_agent_trace.source_admission import (
    GenericAgentTraceSourceAdmissionService,
)
from eval_factory.packs.generic_agent_trace.workflow_bootstrap import (
    GenericAgentWorkflowBootstrap,
)
from eval_factory.team import TeamStore
from eval_factory.trace.source_registry import TraceSourceRegistry


class GenericAgentFirstPartyGraphBuilder:
    @staticmethod
    def build(
        *,
        components: FactoryDatasetFixtureComponents,
        session_store: HarnessSessionStore,
        config: GenericAgentGraphRuntimeConfigV1,
        paths: GenericAgentGraphProductPaths,
        core_input: FactoryDatasetCoreInput,
        request: FactoryDatasetRunRequestV2,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        candidate_output_root: Path,
        audit: ContractAudit,
    ) -> GenericAgentGraphProduct:
        runtime = components.runtime
        private_store = components.private_store
        registration = build_generic_agent_trace_execution_pack(
            audit=audit,
        )
        resolved_paths = paths.resolved()
        GenericAgentFirstPartyGraphBuilder._validate_paths(
            runtime=runtime,
            private_store=private_store,
            session_store=session_store,
            paths=resolved_paths,
        )
        main_session, member_sessions = GenericAgentFirstPartyGraphBuilder._sessions(
            session_store=session_store,
            config=config,
            composition_ref=registration.composition.to_ref(),
        )
        owner = GenericAgentOwnerRuntime.require(components)
        source_registry = TraceSourceRegistry(
            owner.core_runner.workspace / "trace-source-registry.sqlite3",
        )
        admission = GenericAgentTraceSourceAdmissionService().admit(
            core_input=core_input,
            registry=source_registry,
        )
        materials = CompositeCapabilityMaterialResolver()
        candidate_store = FactoryTraceCandidateMaterialStore(
            runtime.store,
        )
        item_materializer = FactoryItemRunMaterializer(
            runtime.store,
        )
        execution_context = GenericAgentFactoryExecutionContext(
            request=request,
            requirement=requirement,
            policy=policy,
            audit=audit,
            source_refs=admission.source_refs,
            trace_sources=admission.sources,
            store=runtime.store,
            candidate_store=candidate_store,
            item_materializer=item_materializer,
            job_store_bridge=owner.job_store_bridge,
            item_context_factory=owner.item_context_factory,
            batch_context_factory=owner.batch_context_factory,
            release_context_factory=owner.release_context_factory,
        )
        task_materials = owner.task_materials
        attachment_materials = FactoryAttachmentExecutionMaterialStore(
            private_store,
        )
        item_materials = GenericAgentFactoryItemMaterialSource(
            dataset_run_id=request.dataset_run_id,
            store=runtime.store,
            plan_reviews=runtime.plan_reviews,
            task_authoring_materials=task_materials,
            quality_materials=owner.quality_runtime.materials,
            criteria_materials=(owner.criteria_runtime.authority_materials),
            attachment_materials=attachment_materials,
            grading_materials=(owner.grading_runtime.authority_materials),
            context_source=execution_context,
        )
        batch_execution = GenericAgentBatchExecutionContext(
            execution=execution_context,
            item_materials=item_materials,
            witness=owner.witness_bridge,
        )
        delivery_preparation = GenericAgentDeliveryPreparation(
            dataset_run_id=request.dataset_run_id,
            store=runtime.store,
            batch_materials=owner.batch_runtime.materials,
            item_materials=item_materials,
            policy=policy,
            release_context_source=execution_context,
        )
        plan_review_preparation = GenericAgentPlanReviewPreparation(
            dataset_run_id=request.dataset_run_id,
            store=runtime.store,
            policy=policy,
            task_authoring_materials=task_materials,
            attachment_runtime=owner.attachment_runtime,
            criteria_runtime=owner.criteria_runtime,
            grading_runtime=owner.grading_runtime,
            item_materials=item_materials,
            delivery_preparation=delivery_preparation,
        )
        providers = build_first_party_capability_providers(
            registration=registration,
            runtime=runtime,
            owner=owner,
            materials=materials,
            source_registry=source_registry,
            candidate_store=candidate_store,
            attachment_materials=attachment_materials,
            item_materials=item_materials,
            execution_context=execution_context,
            batch_execution=batch_execution,
            plan_review_preparation=plan_review_preparation,
            dataset_run_id=request.dataset_run_id,
        )
        capability_runtime = HarnessCapabilityRuntime(
            CapabilityRuntimeRegistry(
                registration=registration,
                providers=providers,
            ),
        )
        team_store = TeamStore(resolved_paths.team_store)
        candidate_aggregate = GenericAgentCandidateAggregate(
            request=request,
            requirement=requirement,
            factory_store=runtime.store,
            candidate_store=candidate_store,
            source_refs=admission.source_refs,
        )
        workflow = GenericAgentWorkflowBootstrap.build(
            registration=registration,
            capability_runtime=capability_runtime,
            materials=materials,
            private_store=private_store,
            request_store_path=resolved_paths.request_store,
            team_store=team_store,
            requirement=requirement,
            source_refs=admission.source_refs,
            independent_session_refs={
                role: projection.session.session_ref for role, projection in member_sessions.items()
            },
            team_id=config.team_id,
            team_incarnation_id=config.team_incarnation_id,
            max_model_requests=policy.max_model_requests,
            max_model_tokens=policy.max_model_tokens,
            max_cost_micro_usd=policy.max_cost_micro_usd,
            audit=audit,
            request_preparation_factory=lambda materialization, preparation: (
                GenericAgentFactoryRequestCoordinator(
                    request=request,
                    requirement=requirement,
                    policy=policy,
                    factory_store=runtime.store,
                    candidate_store=candidate_store,
                    item_materializer=item_materializer,
                    materialization=materialization,
                    preparation=preparation,
                    trace_sources=admission.sources,
                    attachment_preparation_materials=(owner.attachment_runtime.preparation_materials),
                    item_materials=item_materials,
                    candidate_aggregate=candidate_aggregate,
                    delivery_preparation=delivery_preparation,
                    execution_context=execution_context,
                )
            ),
            trace_candidate_store=candidate_store,
        )
        journal_probe = FactoryGraphJournalStore(
            resolved_paths.graph_journal,
        )
        try:
            binding = journal_probe.get_binding(
                config.graph_binding_id,
            )
        except FactoryGraphJournalNotFoundError:
            requirement_policy_ref = main_session.session.current_requirement_policy_ref
            assert requirement_policy_ref is not None
            binding = FactoryDatasetGraphBootstrap.create_binding(
                binding_id=config.graph_binding_id,
                thread_id=config.thread_id,
                session_ref=main_session.session.session_ref,
                requirement_policy_ref=requirement_policy_ref,
                dataset_runtime=runtime,
                team_store=team_store,
                registration=registration,
                blueprint=workflow.blueprint,
                request=request,
                policy=policy,
                requirement=requirement,
                audit=audit,
            )
        GenericAgentFirstPartyGraphBuilder._validate_binding(
            binding=binding,
            config=config,
            session_ref=main_session.session.session_ref,
            blueprint_ref=workflow.blueprint.to_ref(),
        )
        graph = FactoryDatasetGraphBootstrap.build(
            dataset_runtime=runtime,
            workflow=workflow.workflow,
            output_reader=CandidateDatasetOutputAssembler(
                store=runtime.store,
                root=candidate_output_root,
            ),
            session_store=session_store,
            team_store=team_store,
            registration=registration,
            blueprint=workflow.blueprint,
            binding=binding,
            request=request,
            policy=policy,
            requirement=requirement,
            journal_path=resolved_paths.graph_journal,
            checkpoint_path=resolved_paths.graph_checkpoint,
            audit=audit,
        )
        return GenericAgentGraphProduct(
            graph=graph,
            workflow=workflow,
            dataset_runtime=runtime,
            dataset_store=runtime.store,
            request=request,
            audit=audit,
        )

    @staticmethod
    def _sessions(
        *,
        session_store: HarnessSessionStore,
        config: GenericAgentGraphRuntimeConfigV1,
        composition_ref: ObjectRef,
    ) -> tuple[
        HarnessSessionProjectionV1,
        dict[str, HarnessSessionProjectionV1],
    ]:
        main = session_store.get_projection(
            config.main_session_id,
        )
        members = {
            role: session_store.get_projection(session_id)
            for role, session_id in config.member_session_ids().items()
        }
        projections = (main, *members.values())
        if (
            main.session.status is not HarnessSessionStatusV1.ACTIVE
            or main.session.current_interpretation_ref is None
            or main.session.current_requirement_policy_ref is None
            or any(
                projection.session.status is not HarnessSessionStatusV1.ACTIVE
                or projection.session.composition_ref != composition_ref
                for projection in projections
            )
        ):
            raise GenericAgentGraphProductError(
                "Graph builder requires active Pack-bound READY sessions",
            )
        return main, members

    @staticmethod
    def _validate_paths(
        *,
        runtime: FactoryDatasetRuntime,
        private_store: FactoryPrivateObjectStore,
        session_store: HarnessSessionStore,
        paths: GenericAgentGraphProductPaths,
    ) -> None:
        values = (
            runtime.store.path,
            private_store.root,
            session_store.path,
            paths.team_store,
            paths.request_store,
            paths.graph_journal,
            paths.graph_checkpoint,
        )
        if len({value.expanduser().resolve() for value in values}) != len(
            values,
        ):
            raise GenericAgentGraphProductError(
                "Graph product authority paths must be distinct",
            )

    @staticmethod
    def _validate_binding(
        *,
        binding: HarnessGraphExecutionBindingV1,
        config: GenericAgentGraphRuntimeConfigV1,
        session_ref: ObjectRef,
        blueprint_ref: ObjectRef,
    ) -> None:
        if (
            binding.binding_id != config.graph_binding_id
            or binding.thread_id != config.thread_id
            or binding.session_ref != session_ref
            or binding.blueprint_ref != blueprint_ref
        ):
            raise GenericAgentGraphProductError(
                "existing Graph binding differs from product config",
            )


__all__ = [
    "GenericAgentFirstPartyGraphBuilder",
    "GenericAgentGraphProduct",
    "GenericAgentGraphProductError",
    "GenericAgentGraphProductPaths",
    "GenericAgentGraphRuntimeConfigV1",
]
