from __future__ import annotations

from typing import Literal

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelSpecV2,
    SelectionContextV2,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    TaskDraftV2,
)
from eval_factory.contracts.trace import TraceEnvelope
from eval_factory.provenance.views import EvidenceViewResult
from eval_factory.task_authoring.draft_models import (
    TaskDraftAuthoringResult,
)
from eval_factory.task_authoring.evaluation_models import (
    EvaluationContractCompileResult,
)
from eval_factory.task_authoring.models import (
    TaskEpisodeGroupingResult,
)
from eval_factory.task_authoring.producer_models import (
    ProducerTaskViewProjectionResult,
)
from eval_factory.task_authoring.prompt_safety_models import (
    TaskPromptSafetyResult,
)
from eval_factory.task_authoring.rubric_models import (
    RubricAuthoringResult,
)
from eval_factory.task_authoring.tool_models import (
    ToolPolicyCompilationResult,
)


class FactoryTaskAuthoringMaterialError(RuntimeError):
    pass


class _FactoryTaskAuthoringMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-task-authoring-material/v1"] = (
        "eval-factory/private-task-authoring-material/v1"
    )
    source_trace_ref: ObjectRef
    trace_envelope: TraceEnvelope
    label_spec: LabelSpecV2
    label_decision: LabelDecisionV2
    selection_context: SelectionContextV2
    task_episode_result: TaskEpisodeGroupingResult
    task_draft_result: TaskDraftAuthoringResult
    prompt_safety_result: TaskPromptSafetyResult
    task_draft: TaskDraftV2
    rubric_result: RubricAuthoringResult
    evaluation_contract_result: EvaluationContractCompileResult
    tool_policy_result: ToolPolicyCompilationResult
    producer_evidence_view: EvidenceViewResult
    producer_evidence_bundle: EvidenceBundle
    leakage_reference_set: PromptLeakageReferenceSetV2
    producer_task_view_result: ProducerTaskViewProjectionResult
    task_contract_set: R4TaskContractSetV2

    @classmethod
    def from_result(
        cls,
        result: FactoryTaskAuthoringBridgeResult,
    ) -> _FactoryTaskAuthoringMaterialV1:
        return cls(
            source_trace_ref=result.source_trace_ref,
            trace_envelope=result.trace_envelope,
            label_spec=result.label_spec,
            label_decision=result.label_decision,
            selection_context=result.selection_context,
            task_episode_result=result.task_episode_result,
            task_draft_result=result.task_draft_result,
            prompt_safety_result=result.prompt_safety_result,
            task_draft=result.task_draft,
            rubric_result=result.rubric_result,
            evaluation_contract_result=(result.evaluation_contract_result),
            tool_policy_result=result.tool_policy_result,
            producer_evidence_view=(result.producer_evidence_view),
            producer_evidence_bundle=(result.producer_evidence_bundle),
            leakage_reference_set=result.leakage_reference_set,
            producer_task_view_result=(result.producer_task_view_result),
            task_contract_set=result.task_contract_set,
        )

    def to_result(self) -> FactoryTaskAuthoringBridgeResult:
        return FactoryTaskAuthoringBridgeResult(
            source_trace_ref=self.source_trace_ref,
            trace_envelope=self.trace_envelope,
            label_spec=self.label_spec,
            label_decision=self.label_decision,
            selection_context=self.selection_context,
            task_episode_result=self.task_episode_result,
            task_draft_result=self.task_draft_result,
            prompt_safety_result=self.prompt_safety_result,
            task_draft=self.task_draft,
            rubric_result=self.rubric_result,
            evaluation_contract_result=(self.evaluation_contract_result),
            tool_policy_result=self.tool_policy_result,
            producer_evidence_view=self.producer_evidence_view,
            producer_evidence_bundle=self.producer_evidence_bundle,
            leakage_reference_set=self.leakage_reference_set,
            producer_task_view_result=(self.producer_task_view_result),
            task_contract_set=self.task_contract_set,
        )


class FactoryTaskAuthoringMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        result: FactoryTaskAuthoringBridgeResult,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="task-authoring-material",
            value=_FactoryTaskAuthoringMaterialV1.from_result(result),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryTaskAuthoringBridgeResult:
        if reference.object_type != "task-authoring-material" or reference.object_version != "v2":
            raise FactoryTaskAuthoringMaterialError("task authoring material reference type is invalid")
        try:
            material = self.private_store.get_model(
                reference,
                _FactoryTaskAuthoringMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryTaskAuthoringMaterialError("task authoring material is unavailable") from exc
        return material.to_result()


__all__ = [
    "FactoryTaskAuthoringMaterialError",
    "FactoryTaskAuthoringMaterialStore",
]
