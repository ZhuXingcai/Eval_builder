from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
    CandidateDatasetOutputAssembler,
    CandidateOutputWrite,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionResult,
)
from eval_factory.agent_system.delivery_planning import (
    DatasetDeliveryPlanCompiler,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewError,
    PlanReviewService,
)
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.dataset_runtime_v2 import (
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
    FactoryDatasetAggregateResultV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.release_projection_v2 import (
    evaluation_item_v2_ref,
)


class FactoryDeliveryRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryDeliveryRuntimeConfig:
    output_target_ref: ObjectRef
    max_files: int
    max_total_bytes: int


@dataclass(frozen=True, slots=True)
class FactoryDeliveryWaitingView:
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    review_ref: ObjectRef


class FactoryDeliveryRuntime:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        plan_reviews: PlanReviewService,
        assembler: CandidateDatasetOutputAssembler,
        config: FactoryDeliveryRuntimeConfig,
        requested_by: str,
    ) -> None:
        self.store = store
        self.plan_reviews = plan_reviews
        self.assembler = assembler
        self.config = config
        self.requested_by = requested_by
        self.compiler = DatasetDeliveryPlanCompiler()

    def advance(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        candidates: tuple[
            FactoryCandidateProjectionResult,
            ...,
        ],
        exports: tuple[AuthorizedCandidateExport, ...],
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDeliveryWaitingView | CandidateOutputWrite:
        review = self.prepare_review(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            candidates=candidates,
            policy=policy,
            audit=audit,
        )
        try:
            self.plan_reviews.require_resumed_plan(
                run_id=dataset_run_id,
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                plan_ref=review.plan_ref,
            )
        except PlanReviewError:
            return review
        return self.execute_reviewed(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            candidates=candidates,
            exports=exports,
            policy=policy,
            audit=audit,
        )

    def prepare_review(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        candidates: tuple[
            FactoryCandidateProjectionResult,
            ...,
        ],
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDeliveryWaitingView:
        current = self.store.get_run(dataset_run_id)
        if (
            self.store.get_dataset_aggregate(dataset_run_id) != aggregate
            or current.policy_ref != policy.to_ref()
        ):
            raise FactoryDeliveryRuntimeError("delivery parent authority is stale")
        try:
            material = self.store.get_domain_plan(
                dataset_run_id,
                PlanKindV2.FINAL_DELIVERY,
            )
        except FactoryControlNotFoundError:
            plan = DatasetDeliveryPlanV2.create(
                plan_id=(f"dataset-delivery-plan://{aggregate.object_sha256}"),
                run_ref=current.to_ref(),
                plan_version=1,
                predecessor_plan_ref=None,
                aggregate_result_ref=aggregate.to_ref(),
                candidate_item_refs=tuple(
                    evaluation_item_v2_ref(value.projection.evaluation_item) for value in candidates
                ),
                candidate_projection_refs=tuple(value.projection.to_ref() for value in candidates),
                candidate_stage_head_refs=tuple(
                    self.store.get_item_stage_head(
                        value.source.item.item_id,
                        FactoryItemStageV2.RELEASE_CANDIDATE,
                    ).to_ref()
                    for value in candidates
                ),
                rejected_binding_refs=(aggregate.rejected_binding_refs),
                blocked_binding_refs=(aggregate.blocked_binding_refs),
                output_target_ref=self.config.output_target_ref,
                max_files=self.config.max_files,
                max_total_bytes=self.config.max_total_bytes,
                audit=audit,
            )
            compiled = self.compiler.compile(
                plan=plan,
                policy=policy,
                audit=audit,
            )
            self.store.commit_domain_plan(
                run_id=dataset_run_id,
                expected_run_version=current.run_version,
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                plan=plan,
                compiled_plan=compiled,
                audit=audit,
                idempotency_key=(f"commit-factory-delivery-plan-{plan.object_sha256}"),
            )
        else:
            plan = DatasetDeliveryPlanV2.model_validate_json(material.plan_record_json)
            compiled = CompiledDatasetDeliveryPlanV2.model_validate_json(material.compiled_plan_record_json)
        try:
            review = self.plan_reviews.require_resumed_plan(
                run_id=dataset_run_id,
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                plan_ref=plan.to_ref(),
            )
        except PlanReviewError:
            review = self.plan_reviews.open_plan(
                run_id=dataset_run_id,
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                requested_by=self.requested_by,
                idempotency_key=(f"open-factory-delivery-review-{plan.object_sha256}"),
                audit=audit,
            )
        return FactoryDeliveryWaitingView(
            plan_ref=plan.to_ref(),
            compiled_plan_ref=compiled.to_ref(),
            review_ref=review.request.to_ref(),
        )

    def execute_reviewed(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        candidates: tuple[
            FactoryCandidateProjectionResult,
            ...,
        ],
        exports: tuple[AuthorizedCandidateExport, ...],
        policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> CandidateOutputWrite:
        current = self.store.get_run(dataset_run_id)
        if (
            self.store.get_dataset_aggregate(dataset_run_id) != aggregate
            or current.policy_ref != policy.to_ref()
        ):
            raise FactoryDeliveryRuntimeError(
                "delivery parent authority is stale",
            )
        material = self.store.get_domain_plan(
            dataset_run_id,
            PlanKindV2.FINAL_DELIVERY,
        )
        plan = DatasetDeliveryPlanV2.model_validate_json(
            material.plan_record_json,
        )
        compiled = CompiledDatasetDeliveryPlanV2.model_validate_json(
            material.compiled_plan_record_json,
        )
        self.plan_reviews.require_resumed_plan(
            run_id=dataset_run_id,
            plan_kind=PlanKindV2.FINAL_DELIVERY,
            plan_ref=plan.to_ref(),
        )
        parent_plan = self.store.get_plan(dataset_run_id)[1]
        return self.assembler.assemble(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            plan=plan,
            compiled_plan=compiled,
            parent_compiled_plan_ref=parent_plan.to_ref(),
            candidates=candidates,
            exports=exports,
            audit=audit,
            idempotency_key=(f"assemble-factory-candidate-output-{plan.object_sha256}"),
        )


__all__ = [
    "FactoryDeliveryRuntime",
    "FactoryDeliveryRuntimeConfig",
    "FactoryDeliveryRuntimeError",
    "FactoryDeliveryWaitingView",
]
