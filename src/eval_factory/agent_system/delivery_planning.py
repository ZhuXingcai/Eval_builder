from __future__ import annotations

from eval_factory.contracts.agent_system_v2 import (
    FactoryRunPolicyV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
)


class DatasetDeliveryPlanCompilerError(RuntimeError):
    pass


class DatasetDeliveryPlanCompiler:
    def compile(
        self,
        *,
        plan: DatasetDeliveryPlanV2,
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CompiledDatasetDeliveryPlanV2:
        canonical_plan = DatasetDeliveryPlanV2.model_validate_json(plan.canonical_json())
        canonical_policy = FactoryRunPolicyV2.model_validate_json(policy.canonical_json())
        if canonical_plan.plan_version > (canonical_policy.max_plan_revisions + 1):
            raise DatasetDeliveryPlanCompilerError("delivery plan exceeds revision policy")
        if len(canonical_plan.candidate_item_refs) > canonical_plan.max_files:
            raise DatasetDeliveryPlanCompilerError("delivery plan file limit is below candidate count")
        return CompiledDatasetDeliveryPlanV2.create(
            audit=audit,
            compiled_plan_id=(f"compiled-dataset-delivery-plan://{canonical_plan.object_sha256}"),
            source_plan_ref=canonical_plan.to_ref(),
            policy_ref=canonical_policy.to_ref(),
            aggregate_result_ref=(canonical_plan.aggregate_result_ref),
            output_target_ref=canonical_plan.output_target_ref,
            candidate_count=len(canonical_plan.candidate_item_refs),
        )


__all__ = [
    "DatasetDeliveryPlanCompiler",
    "DatasetDeliveryPlanCompilerError",
]
