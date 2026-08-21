from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.grading_agent import (
    GradingDesignExecution,
)
from eval_factory.agent_system.grading_subgraph import (
    GradingDesignSupervisedExecution,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentResultEnvelopeV2,
    GradingDesignResultV2,
    JudgeDesignSpecV2,
    JudgeDesignValidationV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelRouteDecisionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.task_v2 import (
    EvaluatorReferenceGrantV2,
)


class FactoryGradingAuthorityMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryGradingAuthorityResult:
    supervised: GradingDesignSupervisedExecution
    execution: GradingDesignExecution


class _FactoryGradingAuthorityMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-grading-authority-material/v1"] = (
        "eval-factory/private-grading-authority-material/v1"
    )

    result: GradingDesignResultV2
    envelope: AgentResultEnvelopeV2
    route: ModelRouteDecisionV2 | None
    invocation_result_ref: ObjectRef | None
    design_spec: JudgeDesignSpecV2 | None
    validation: JudgeDesignValidationV2
    reference_grants: tuple[EvaluatorReferenceGrantV2, ...]

    @classmethod
    def from_result(
        cls,
        value: FactoryGradingAuthorityResult,
    ) -> _FactoryGradingAuthorityMaterialV1:
        execution = value.execution
        return cls(
            result=execution.result,
            envelope=value.supervised.envelope,
            route=execution.route,
            invocation_result_ref=(execution.invocation_result_ref),
            design_spec=execution.design_spec,
            validation=execution.validation,
            reference_grants=execution.reference_grants,
        )

    def to_result(self) -> FactoryGradingAuthorityResult:
        execution = GradingDesignExecution(
            result=self.result,
            route=self.route,
            invocation_result_ref=self.invocation_result_ref,
            design_spec=self.design_spec,
            validation=self.validation,
            reference_grants=self.reference_grants,
        )
        return FactoryGradingAuthorityResult(
            supervised=GradingDesignSupervisedExecution(
                result=self.result,
                envelope=self.envelope,
            ),
            execution=execution,
        )


class FactoryGradingAuthorityMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        value: FactoryGradingAuthorityResult,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="grading-authority-material",
            value=_FactoryGradingAuthorityMaterialV1.from_result(value),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryGradingAuthorityResult:
        if reference.object_type != "grading-authority-material" or reference.object_version != "v2":
            raise FactoryGradingAuthorityMaterialError("grading authority material reference type is invalid")
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryGradingAuthorityMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryGradingAuthorityMaterialError("grading authority material is unavailable") from exc
        return value.to_result()


__all__ = [
    "FactoryGradingAuthorityMaterialError",
    "FactoryGradingAuthorityMaterialStore",
    "FactoryGradingAuthorityResult",
]
