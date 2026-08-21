from __future__ import annotations

from typing import cast

from eval_factory.agent_system.attachment_execution_material import (
    FactoryAttachmentExecutionMaterialStore,
)
from eval_factory.agent_system.dataset_runtime import FactoryDatasetRuntime
from eval_factory.agent_system.task_authoring_commit import (
    FactoryTaskAuthoringCommitter,
)
from eval_factory.agent_system.trace_candidate_store import (
    FactoryTraceCandidateMaterialStore,
)
from eval_factory.harness import (
    CapabilityRuntimeProvider,
    StaticPackRegistrationV1,
)
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.packs.generic_agent_trace.attachment_projection import (
    GenericAgentAttachmentProjection,
)
from eval_factory.packs.generic_agent_trace.capabilities import (
    AttachmentQualityWorkflowCapabilityProvider,
    AttachmentReconstructionCapabilityProvider,
    BatchQualityCapabilityProvider,
    CriteriaRubricCapabilityProvider,
    DeliveryCapabilityProvider,
    GradingDesignCapabilityProvider,
    PlanReviewCapabilityProvider,
    RequirementPlanningCapabilityProvider,
    TaskAuthoringCapabilityProvider,
    TraceIngestionCapabilityProvider,
    generic_agent_trace_providers,
)
from eval_factory.packs.generic_agent_trace.execution_context import (
    GenericAgentBatchExecutionContext,
    GenericAgentFactoryExecutionContext,
)
from eval_factory.packs.generic_agent_trace.item_materials import (
    GenericAgentFactoryItemMaterialSource,
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
from eval_factory.packs.generic_agent_trace.quality_execution import (
    GenericAgentAttachmentQualityExecution,
)
from eval_factory.packs.generic_agent_trace.specialist_projections import (
    GenericAgentCriteriaProjection,
    GenericAgentGradingProjection,
)
from eval_factory.packs.generic_agent_trace.task_authoring_projection import (
    GenericAgentTaskAuthoringProjection,
)
from eval_factory.trace.source_registry import TraceSourceRegistry


def build_first_party_capability_providers(
    *,
    registration: StaticPackRegistrationV1,
    runtime: FactoryDatasetRuntime,
    owner: GenericAgentOwnerRuntime,
    materials: CompositeCapabilityMaterialResolver,
    source_registry: TraceSourceRegistry,
    candidate_store: FactoryTraceCandidateMaterialStore,
    attachment_materials: FactoryAttachmentExecutionMaterialStore,
    item_materials: GenericAgentFactoryItemMaterialSource,
    execution_context: GenericAgentFactoryExecutionContext,
    batch_execution: GenericAgentBatchExecutionContext,
    plan_review_preparation: GenericAgentPlanReviewPreparation,
    dataset_run_id: str,
) -> tuple[CapabilityRuntimeProvider, ...]:
    attachment_runner = owner.attachment_runtime.runner
    assert attachment_runner is not None
    providers = generic_agent_trace_providers(
        RequirementPlanningCapabilityProvider(
            registration,
            compiler=runtime.compiler,
            materials=materials,
        ),
        TraceIngestionCapabilityProvider(
            registration,
            service=TraceIndexStageService(
                source_registry=source_registry,
                trace_store=owner.core_runner.trace_store,
            ),
            materials=materials,
            candidate_preparer=owner.core_runner.candidate_preparer,
            candidate_materials=candidate_store,
        ),
        TaskAuthoringCapabilityProvider(
            registration,
            bridge=owner.task_bridge,
            materials=materials,
            result_projection=GenericAgentTaskAuthoringProjection(
                dataset_run_id=dataset_run_id,
                store=runtime.store,
                committer=FactoryTaskAuthoringCommitter(
                    store=runtime.store,
                    materials=owner.task_materials,
                ),
            ),
        ),
        AttachmentReconstructionCapabilityProvider(
            registration,
            runner=attachment_runner,
            materials=materials,
            facade=owner.attachment_runtime.execution_facade,
            result_projection=GenericAgentAttachmentProjection(
                dataset_run_id=dataset_run_id,
                store=runtime.store,
                materials=attachment_materials,
            ),
        ),
        CriteriaRubricCapabilityProvider(
            registration,
            runner=owner.criteria_runtime.runner,
            materials=materials,
            result_projection=GenericAgentCriteriaProjection(
                dataset_run_id=dataset_run_id,
                store=runtime.store,
                runtime=owner.criteria_runtime,
                materials=item_materials,
            ),
        ),
        GradingDesignCapabilityProvider(
            registration,
            runner=owner.grading_runtime.runner,
            materials=materials,
            result_projection=GenericAgentGradingProjection(
                dataset_run_id=dataset_run_id,
                store=runtime.store,
                runtime=owner.grading_runtime,
                materials=item_materials,
            ),
        ),
        AttachmentQualityWorkflowCapabilityProvider(
            registration,
            execution=GenericAgentAttachmentQualityExecution(
                dataset_run_id=dataset_run_id,
                store=runtime.store,
                task_authoring_materials=owner.task_materials,
                attachment_materials=attachment_materials,
                runtime=owner.quality_runtime,
                contexts=item_materials,
            ),
        ),
        BatchQualityCapabilityProvider(
            registration,
            runtime=owner.batch_runtime,
            materials=materials,
            context_source=batch_execution,
        ),
        PlanReviewCapabilityProvider(
            registration,
            service=runtime.plan_reviews,
            review_preparation=plan_review_preparation,
        ),
        DeliveryCapabilityProvider(
            registration,
            materials=materials,
            runtime_source=execution_context,
        ),
    )
    return cast(
        tuple[CapabilityRuntimeProvider, ...],
        providers,
    )


__all__ = ["build_first_party_capability_providers"]
