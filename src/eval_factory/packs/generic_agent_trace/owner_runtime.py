from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.attachment_item_runtime import (
    FactoryAttachmentItemRuntime,
)
from eval_factory.agent_system.attachment_quality_runtime import (
    FactoryAttachmentQualityRuntime,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityRuntime,
)
from eval_factory.agent_system.core_vertical import CoreVerticalRunner
from eval_factory.agent_system.criteria_item_runtime import (
    FactoryCriteriaItemRuntime,
)
from eval_factory.agent_system.dataset_runtime import (
    FactoryBatchQualityContextFactory,
    FactoryDatasetReleaseContextFactory,
    FactoryItemSpecialistContextFactory,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    FactoryDatasetFixtureComponents,
)
from eval_factory.agent_system.grading_item_runtime import (
    FactoryGradingItemRuntime,
)
from eval_factory.agent_system.job_store_bridge import FactoryJobStoreBridge
from eval_factory.agent_system.job_store_witness_bridge import (
    FactoryJobStoreWitnessBridge,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridge,
)
from eval_factory.agent_system.task_authoring_material import (
    FactoryTaskAuthoringMaterialStore,
)


class GenericAgentOwnerRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenericAgentOwnerRuntime:
    core_runner: CoreVerticalRunner
    task_bridge: FactoryTaskAuthoringBridge
    task_materials: FactoryTaskAuthoringMaterialStore
    attachment_runtime: FactoryAttachmentItemRuntime
    quality_runtime: FactoryAttachmentQualityRuntime
    criteria_runtime: FactoryCriteriaItemRuntime
    grading_runtime: FactoryGradingItemRuntime
    batch_runtime: FactoryBatchQualityRuntime
    job_store_bridge: FactoryJobStoreBridge
    item_context_factory: FactoryItemSpecialistContextFactory
    batch_context_factory: FactoryBatchQualityContextFactory
    release_context_factory: FactoryDatasetReleaseContextFactory
    witness_bridge: FactoryJobStoreWitnessBridge

    @classmethod
    def require(
        cls,
        components: FactoryDatasetFixtureComponents,
    ) -> GenericAgentOwnerRuntime:
        runtime = components.runtime
        required = (
            runtime.core_runner,
            runtime.task_authoring_bridge,
            runtime.task_authoring_materials,
            runtime.attachment_runtime,
            runtime.item_specialist_runtime,
            runtime.job_store_bridge,
            runtime.item_specialist_context_factory,
            runtime.batch_quality_runtime,
            runtime.batch_quality_context_factory,
            runtime.release_context_factory,
            runtime.job_store_witness_bridge,
            components.specialist,
            components.release,
        )
        if any(value is None for value in required):
            raise GenericAgentOwnerRuntimeError(
                "Pack 1.2.0 requires the complete first-party owner runtime",
            )
        assert runtime.core_runner is not None
        assert runtime.task_authoring_bridge is not None
        assert runtime.task_authoring_materials is not None
        assert runtime.attachment_runtime is not None
        assert runtime.item_specialist_runtime is not None
        assert runtime.job_store_bridge is not None
        assert runtime.item_specialist_context_factory is not None
        assert runtime.batch_quality_runtime is not None
        assert runtime.batch_quality_context_factory is not None
        assert runtime.release_context_factory is not None
        assert runtime.job_store_witness_bridge is not None
        grading = runtime.item_specialist_runtime.grading_runtime
        if runtime.attachment_runtime.runner is None or grading is None:
            raise GenericAgentOwnerRuntimeError(
                "Pack 1.2.0 owner runners are incomplete",
            )
        return cls(
            core_runner=runtime.core_runner,
            task_bridge=runtime.task_authoring_bridge,
            task_materials=runtime.task_authoring_materials,
            attachment_runtime=runtime.attachment_runtime,
            quality_runtime=(runtime.item_specialist_runtime.quality_runtime),
            criteria_runtime=(runtime.item_specialist_runtime.criteria_runtime),
            grading_runtime=grading,
            batch_runtime=runtime.batch_quality_runtime,
            job_store_bridge=runtime.job_store_bridge,
            item_context_factory=(runtime.item_specialist_context_factory),
            batch_context_factory=(runtime.batch_quality_context_factory),
            release_context_factory=(runtime.release_context_factory),
            witness_bridge=runtime.job_store_witness_bridge,
        )


__all__ = [
    "GenericAgentOwnerRuntime",
    "GenericAgentOwnerRuntimeError",
]
