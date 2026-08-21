from __future__ import annotations

from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.agent_system.task_authoring_commit import (
    FactoryTaskAuthoringCommitter,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.harness.contracts import sorted_refs
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    TaskAuthoringCapabilityRequestV1,
)


class GenericAgentTaskAuthoringProjectionError(RuntimeError):
    pass


class GenericAgentTaskAuthoringProjection:
    """Projects a Team TaskAuthoring result into Factory item authority."""

    def __init__(
        self,
        *,
        dataset_run_id: str,
        store: FactoryControlStore,
        committer: FactoryTaskAuthoringCommitter,
    ) -> None:
        self.dataset_run_id = dataset_run_id
        self.store = store
        self.committer = committer

    def commit(
        self,
        *,
        request: TaskAuthoringCapabilityRequestV1,
        result: FactoryTaskAuthoringBridgeResult,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> tuple[ObjectRef, ...]:
        matches = tuple(
            binding
            for binding in self.store.list_item_bindings(
                self.dataset_run_id,
            )
            if binding.trace_candidate_decision_ref == request.decision_ref
        )
        if len(matches) != 1:
            raise GenericAgentTaskAuthoringProjectionError(
                "task authoring item binding is not unique",
            )
        binding = matches[0]
        if (
            binding.extracted_prompt_ref != request.extracted_prompt_ref
            or binding.inferred_intent_ref != request.intent_ref
            or binding.rewrite_candidate_ref != request.rewrite_ref
        ):
            raise GenericAgentTaskAuthoringProjectionError(
                "task authoring request differs from item binding",
            )
        committed = self.committer.commit(
            dataset_run_id=self.dataset_run_id,
            binding=binding,
            result=result,
            audit=audit,
            idempotency_key=idempotency_key,
        )
        return sorted_refs(
            (
                committed.stage_head.to_ref(),
                committed.material_ref,
            )
        )


__all__ = [
    "GenericAgentTaskAuthoringProjection",
    "GenericAgentTaskAuthoringProjectionError",
]
