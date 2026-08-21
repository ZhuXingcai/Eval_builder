from __future__ import annotations

from collections.abc import Callable

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.harness import CapabilityRuntimeRegistry
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentFactoryWorkflowError,
)
from eval_factory.packs.generic_agent_trace.request_store import (
    GenericAgentCapabilityRequestBindingV1,
    GenericAgentCapabilityRequestStore,
)
from eval_factory.packs.generic_agent_trace.task_graph_materializer import (
    GenericAgentTaskGraphMaterialization,
)
from eval_factory.team import TeamTaskV1


class GenericAgentTaskRequestPreparer:
    """Commits a typed request for one exact materialized Team task."""

    def __init__(
        self,
        *,
        materialization: GenericAgentTaskGraphMaterialization,
        registry: CapabilityRuntimeRegistry,
        store: GenericAgentCapabilityRequestStore,
        current_task_resolver: Callable[[str], TeamTaskV1] | None = None,
    ) -> None:
        self.materialization = materialization
        self.registry = registry
        self.store = store
        self.current_task_resolver = current_task_resolver
        if registry.registration.composition.to_ref() != materialization.authority.team.composition_ref:
            raise ValueError(
                "request preparer composition differs from Team authority",
            )

    def commit(
        self,
        *,
        task_id: str,
        request: ContractModelV2,
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        instance = self.materialization.instance(task_id)
        source_task = next(
            value for value in self.materialization.authority.graph.tasks if value.task_id == task_id
        )
        task = self.current_task_resolver(task_id) if self.current_task_resolver is not None else source_task
        if (
            task.task_id != source_task.task_id
            or task.capability_definition_ref != source_task.capability_definition_ref
        ):
            raise GenericAgentFactoryWorkflowError(
                "current Team task widens materialized capability authority",
            )
        definition = next(
            value
            for value in self.registry.registration.capability_definitions
            if value.to_ref() == task.capability_definition_ref
        )
        if definition.capability_id != instance.capability_id:
            raise GenericAgentFactoryWorkflowError(
                "materialized task capability metadata drifted",
            )
        provider_binding = next(
            value
            for value in self.registry.registration.provider_bindings
            if value.capability_definition_ref == definition.to_ref()
        )
        provider = next(
            value
            for value in self.registry.providers
            if value.provider_binding_ref == provider_binding.to_ref()
        )
        if type(request) is not provider.request_model:
            raise GenericAgentFactoryWorkflowError(
                "prepared request model differs from task provider",
            )
        ordered_sources = sorted_refs(source_authority_refs)
        if ordered_sources != source_authority_refs:
            raise ValueError(
                "prepared request source refs must be sorted",
            )
        return self.store.commit(
            task_ref=task.to_ref(),
            capability_definition_ref=definition.to_ref(),
            request=request,
            source_authority_refs=ordered_sources,
            audit=audit,
            idempotency_key=idempotency_key,
        )


__all__ = ["GenericAgentTaskRequestPreparer"]
