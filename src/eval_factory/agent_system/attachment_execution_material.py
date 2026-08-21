from __future__ import annotations

from typing import Literal

from eval_factory.agent_system.attachment_subgraph import (
    AttachmentR5Execution,
    AttachmentSupervisedExecution,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    AttachmentGroupResultV2,
    AttachmentSubgraphResultV2,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionBatchV2,
    AttachmentReconstructionResultV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2


class FactoryAttachmentExecutionMaterialError(RuntimeError):
    pass


class _FactoryAttachmentExecutionMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-attachment-execution-material/v1"] = (
        "eval-factory/private-attachment-execution-material/v1"
    )
    execution_batch: ArtifactExecutionBatchV2
    reconstruction_result: AttachmentReconstructionResultV2
    group_results: tuple[AttachmentGroupResultV2, ...]
    envelopes: tuple[AgentResultEnvelopeV2, ...]
    subgraph_result: AttachmentSubgraphResultV2

    @classmethod
    def from_execution(
        cls,
        value: AttachmentSupervisedExecution,
    ) -> _FactoryAttachmentExecutionMaterialV1:
        return cls(
            execution_batch=value.r5_execution.execution_batch,
            reconstruction_result=(value.r5_execution.reconstruction_result),
            group_results=value.group_results,
            envelopes=value.envelopes,
            subgraph_result=value.subgraph_result,
        )

    def to_execution(self) -> AttachmentSupervisedExecution:
        return AttachmentSupervisedExecution(
            r5_execution=AttachmentR5Execution(
                execution_batch=self.execution_batch,
                reconstruction_result=self.reconstruction_result,
            ),
            group_results=self.group_results,
            envelopes=self.envelopes,
            subgraph_result=self.subgraph_result,
        )


class FactoryAttachmentExecutionMaterialStore:
    def __init__(self, private_store: FactoryPrivateObjectStore) -> None:
        self.private_store = private_store

    def put(
        self,
        execution: AttachmentSupervisedExecution,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="attachment-execution-material",
            value=(
                _FactoryAttachmentExecutionMaterialV1.from_execution(
                    execution,
                )
            ),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> AttachmentSupervisedExecution:
        if reference.object_type != "attachment-execution-material" or reference.object_version != "v2":
            raise FactoryAttachmentExecutionMaterialError(
                "attachment execution material reference type is invalid",
            )
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryAttachmentExecutionMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryAttachmentExecutionMaterialError(
                "attachment execution material is unavailable",
            ) from exc
        return value.to_execution()


__all__ = [
    "FactoryAttachmentExecutionMaterialError",
    "FactoryAttachmentExecutionMaterialStore",
]
