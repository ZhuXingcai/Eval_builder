from __future__ import annotations

from dataclasses import dataclass

from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
    CandidateOutputWrite,
)
from eval_factory.agent_system.candidate_projection_runtime import (
    FactoryCandidateProjectionInput,
    FactoryCandidateProjectionRuntime,
    FactoryCandidateProjectionView,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryRuntime,
    FactoryDeliveryWaitingView,
)
from eval_factory.agent_system.plan_review import PlanReviewError
from eval_factory.agent_system.release_source_builder import (
    FactoryReleaseSourceBuilder,
    FactoryReleaseSourceItem,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryRunPolicyV2,
    PlanKindV2,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalPolicy,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityReportV2,
)
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetAggregateResultV2,
)
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPolicyV2,
)


class FactoryDatasetReleaseRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryDatasetReleaseContext:
    source_builder: FactoryReleaseSourceBuilder
    runtime: FactoryDatasetReleaseRuntime
    approval_policy: UserApprovalPolicy
    release_policy: ReleaseProjectionPolicyV2
    exports: tuple[AuthorizedCandidateExport, ...]
    not_required_checkpoints: tuple[
        ApprovalCheckpoint,
        ...,
    ] = ()

    def advance(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        items: tuple[FactoryReleaseSourceItem, ...],
        batch_quality: BatchQualityReportV2,
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        review = self.prepare_review(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            items=items,
            batch_quality=batch_quality,
            factory_policy=factory_policy,
            audit=audit,
        )
        if isinstance(review.delivery, FactoryDeliveryWaitingView):
            try:
                self.runtime.delivery.plan_reviews.require_resumed_plan(
                    run_id=dataset_run_id,
                    plan_kind=PlanKindV2.FINAL_DELIVERY,
                    plan_ref=review.delivery.plan_ref,
                )
            except PlanReviewError:
                return review
        return self.execute_reviewed(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            items=items,
            batch_quality=batch_quality,
            factory_policy=factory_policy,
            audit=audit,
        )

    def prepare_review(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        items: tuple[FactoryReleaseSourceItem, ...],
        batch_quality: BatchQualityReportV2,
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        inputs = self.source_builder.build(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            items=items,
            batch_quality=batch_quality,
            approval_policy=self.approval_policy,
            not_required_checkpoints=(self.not_required_checkpoints),
        )
        return self.runtime.prepare_review(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            inputs=inputs,
            release_policy=self.release_policy,
            exports=self.exports,
            factory_policy=factory_policy,
            audit=audit,
        )

    def execute_reviewed(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        items: tuple[FactoryReleaseSourceItem, ...],
        batch_quality: BatchQualityReportV2,
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        inputs = self.source_builder.build(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            items=items,
            batch_quality=batch_quality,
            approval_policy=self.approval_policy,
            not_required_checkpoints=(self.not_required_checkpoints),
        )
        return self.runtime.execute_reviewed(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            inputs=inputs,
            release_policy=self.release_policy,
            exports=self.exports,
            factory_policy=factory_policy,
            audit=audit,
        )


@dataclass(frozen=True, slots=True)
class FactoryDatasetReleaseView:
    candidates: tuple[
        FactoryCandidateProjectionView,
        ...,
    ]
    delivery: FactoryDeliveryWaitingView | CandidateOutputWrite


class FactoryDatasetReleaseRuntime:
    def __init__(
        self,
        *,
        candidates: FactoryCandidateProjectionRuntime,
        delivery: FactoryDeliveryRuntime,
    ) -> None:
        self.candidates = candidates
        self.delivery = delivery

    def advance(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        inputs: tuple[FactoryCandidateProjectionInput, ...],
        release_policy: ReleaseProjectionPolicyV2,
        exports: tuple[AuthorizedCandidateExport, ...],
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        if not aggregate.candidate_binding_refs or aggregate.batch_quality_ref is None:
            raise FactoryDatasetReleaseRuntimeError("candidate delivery requires eligible items")
        candidate_views = self.candidates.advance(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            inputs=inputs,
            policy=release_policy,
            audit=audit,
        )
        candidate_results = tuple(
            self.candidates.materials.get(view.material_ref) for view in candidate_views
        )
        delivery = self.delivery.advance(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            candidates=candidate_results,
            exports=exports,
            policy=factory_policy,
            audit=audit,
        )
        return FactoryDatasetReleaseView(
            candidates=candidate_views,
            delivery=delivery,
        )

    def prepare_review(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        inputs: tuple[FactoryCandidateProjectionInput, ...],
        release_policy: ReleaseProjectionPolicyV2,
        exports: tuple[AuthorizedCandidateExport, ...],
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        if not aggregate.candidate_binding_refs or aggregate.batch_quality_ref is None:
            raise FactoryDatasetReleaseRuntimeError(
                "candidate delivery requires eligible items",
            )
        candidate_views = self.candidates.advance(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            inputs=inputs,
            policy=release_policy,
            audit=audit,
        )
        candidate_results = tuple(
            self.candidates.materials.get(view.material_ref) for view in candidate_views
        )
        delivery = self.delivery.prepare_review(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            candidates=candidate_results,
            policy=factory_policy,
            audit=audit,
        )
        del exports
        return FactoryDatasetReleaseView(
            candidates=candidate_views,
            delivery=delivery,
        )

    def execute_reviewed(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        inputs: tuple[FactoryCandidateProjectionInput, ...],
        release_policy: ReleaseProjectionPolicyV2,
        exports: tuple[AuthorizedCandidateExport, ...],
        factory_policy: FactoryRunPolicyV2,
        audit: ContractAudit,
    ) -> FactoryDatasetReleaseView:
        if not aggregate.candidate_binding_refs or aggregate.batch_quality_ref is None:
            raise FactoryDatasetReleaseRuntimeError(
                "candidate delivery requires eligible items",
            )
        candidate_views = self.candidates.advance(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            inputs=inputs,
            policy=release_policy,
            audit=audit,
        )
        candidate_results = tuple(
            self.candidates.materials.get(view.material_ref) for view in candidate_views
        )
        delivery = self.delivery.execute_reviewed(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            candidates=candidate_results,
            exports=exports,
            policy=factory_policy,
            audit=audit,
        )
        return FactoryDatasetReleaseView(
            candidates=candidate_views,
            delivery=delivery,
        )


__all__ = [
    "FactoryDatasetReleaseContext",
    "FactoryDatasetReleaseRuntime",
    "FactoryDatasetReleaseRuntimeError",
    "FactoryDatasetReleaseView",
]
