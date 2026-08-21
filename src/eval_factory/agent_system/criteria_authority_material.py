from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.criteria_agent import (
    CriteriaRubricExecution,
)
from eval_factory.agent_system.criteria_subgraph import (
    CriteriaRubricSupervisedExecution,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    CriteriaRubricResultV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelRouteDecisionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
)
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    EvaluatorSpecV2,
    R4TaskContractSetV2,
    ReferencePolicyV2,
    RubricSetV2,
    ToolPolicyV2,
)


class FactoryCriteriaAuthorityMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryCriteriaAuthorityResult:
    supervised: CriteriaRubricSupervisedExecution
    execution: CriteriaRubricExecution
    reviewed_task_contract_set: R4TaskContractSetV2 | None
    reviewed_item_quality: ItemQualityCompilationResultV2 | None


class _FactoryCriteriaAuthorityMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-criteria-authority-material/v1"] = (
        "eval-factory/private-criteria-authority-material/v1"
    )

    result: CriteriaRubricResultV2
    envelope: AgentResultEnvelopeV2
    route: ModelRouteDecisionV2
    invocation_result_ref: ObjectRef
    rubric_set: RubricSetV2 | None
    evaluator_spec: EvaluatorSpecV2 | None
    reference_policy: ReferencePolicyV2 | None
    tool_policy: ToolPolicyV2 | None
    contestant_tool_policy: ContestantToolPolicyV2 | None
    reviewed_task_contract_set: R4TaskContractSetV2 | None
    reviewed_item_quality: ItemQualityCompilationResultV2 | None

    @classmethod
    def from_result(
        cls,
        value: FactoryCriteriaAuthorityResult,
    ) -> _FactoryCriteriaAuthorityMaterialV1:
        execution = value.execution
        return cls(
            result=execution.result,
            envelope=value.supervised.envelope,
            route=execution.route,
            invocation_result_ref=(execution.invocation_result_ref),
            rubric_set=execution.rubric_set,
            evaluator_spec=execution.evaluator_spec,
            reference_policy=execution.reference_policy,
            tool_policy=execution.tool_policy,
            contestant_tool_policy=(execution.contestant_tool_policy),
            reviewed_task_contract_set=(value.reviewed_task_contract_set),
            reviewed_item_quality=value.reviewed_item_quality,
        )

    def to_result(self) -> FactoryCriteriaAuthorityResult:
        execution = CriteriaRubricExecution(
            result=self.result,
            route=self.route,
            invocation_result_ref=self.invocation_result_ref,
            rubric_set=self.rubric_set,
            evaluator_spec=self.evaluator_spec,
            reference_policy=self.reference_policy,
            tool_policy=self.tool_policy,
            contestant_tool_policy=self.contestant_tool_policy,
        )
        return FactoryCriteriaAuthorityResult(
            supervised=CriteriaRubricSupervisedExecution(
                result=self.result,
                envelope=self.envelope,
            ),
            execution=execution,
            reviewed_task_contract_set=(self.reviewed_task_contract_set),
            reviewed_item_quality=self.reviewed_item_quality,
        )


class FactoryCriteriaAuthorityMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        value: FactoryCriteriaAuthorityResult,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="criteria-authority-material",
            value=_FactoryCriteriaAuthorityMaterialV1.from_result(value),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryCriteriaAuthorityResult:
        if reference.object_type != "criteria-authority-material" or reference.object_version != "v2":
            raise FactoryCriteriaAuthorityMaterialError(
                "criteria authority material reference type is invalid"
            )
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryCriteriaAuthorityMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryCriteriaAuthorityMaterialError("criteria authority material is unavailable") from exc
        return value.to_result()


__all__ = [
    "FactoryCriteriaAuthorityMaterialError",
    "FactoryCriteriaAuthorityMaterialStore",
    "FactoryCriteriaAuthorityResult",
]
