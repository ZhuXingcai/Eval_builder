from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_factory_graph_bootstrap import _factory_inputs
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.agent_system.store import FactoryControlNotFoundError
from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.dataset_runtime_v2 import FactoryItemStageV2
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    PlanReviewCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.plan_review_preparation import (
    GenericAgentPlanReviewPreparation,
    GenericAgentPlanReviewPreparationError,
)


class _Store:
    def __init__(self) -> None:
        self.item_run = SimpleNamespace(
            run_id="factory-run://item",
            to_ref=lambda: ref("factory-run", "item", version="v2"),
        )
        self.binding = SimpleNamespace(
            item_id="item-1",
            item_run_ref=self.item_run.to_ref(),
        )
        self.task_head = SimpleNamespace(
            result_ref=ref(
                "r4-task-contract-set",
                "item",
                version="v2",
            ),
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "task-authoring",
                version="v2",
            ),
        )
        self.quality_head = SimpleNamespace(
            result_ref=ref(
                "item-quality-compilation-result",
                "item",
                version="v2",
            ),
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "item-quality",
                version="v2",
            ),
        )
        self.criteria_head = SimpleNamespace(
            result_ref=ref(
                "criteria-rubric-result",
                "item",
                version="v2",
            ),
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "criteria",
                version="v2",
            ),
        )
        self.aggregate = SimpleNamespace(
            to_ref=lambda: ref(
                "factory-dataset-aggregate-result",
                "dataset",
                version="v2",
            ),
        )

    def get_run(self, run_id: str):
        assert run_id == self.item_run.run_id
        return self.item_run

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return self.item_run

    def list_item_bindings(self, dataset_run_id: str):
        assert dataset_run_id == "factory-run.graph-bootstrap"
        return (self.binding,)

    def get_item_stage_head(self, item_id: str, stage: object):
        assert item_id == self.binding.item_id
        return {
            FactoryItemStageV2.TASK_AUTHORING: self.task_head,
            FactoryItemStageV2.ITEM_QUALITY: self.quality_head,
            FactoryItemStageV2.CRITERIA_RUBRIC: self.criteria_head,
        }[stage]

    def get_item_stage_material_ref(self, head_ref: object):
        del head_ref
        return ref(
            "task-authoring-material",
            "item",
            version="v2",
        )

    def get_domain_plan(self, run_id: str, plan_kind: object):
        del run_id, plan_kind
        raise FactoryControlNotFoundError("not found")

    def get_dataset_aggregate(self, dataset_run_id: str):
        assert dataset_run_id == "factory-run.graph-bootstrap"
        return self.aggregate


class _AttachmentRuntime:
    def __init__(self) -> None:
        self.calls = 0

    async def prepare_review(self, **kwargs: object):
        del kwargs
        self.calls += 1
        return SimpleNamespace(
            plan_ref=ref(
                "attachment-generation-plan",
                "item",
                version="v2",
            ),
        )


class _DomainRuntime:
    def __init__(self, object_type: str) -> None:
        self.calls = 0
        self.object_type = object_type

    async def prepare_review(self, **kwargs: object):
        del kwargs
        self.calls += 1
        return SimpleNamespace(
            plan_ref=ref(
                self.object_type,
                "item",
                version="v2",
            ),
        )


class _ItemMaterials:
    def criteria(self, item_id: str):
        assert item_id == "item-1"
        return SimpleNamespace(
            task_draft=object(),
            task_contract_set=object(),
            quality=object(),
            quality_context=object(),
            binding_definitions=(),
            binding_definitions_ref=ref(
                "evaluator-binding-definitions",
                "item",
                version="v2",
            ),
            tool_catalog=object(),
        )

    def grading(self, item_id: str):
        assert item_id == "item-1"
        return SimpleNamespace(
            criteria=object(),
            model_authorizations=(),
            model_authorizations_ref=ref(
                "model-domain-authorizations",
                "item",
                version="v2",
            ),
            evaluated_at=audit().created_at,
        )


class _DeliveryPreparation:
    def __init__(self) -> None:
        self.calls = 0

    def prepare_review(self, *, audit: object):
        del audit
        self.calls += 1
        return SimpleNamespace(
            plan_ref=ref(
                "dataset-delivery-plan",
                "dataset",
                version="v2",
            ),
        )


@pytest.mark.asyncio
async def test_plan_review_preparation_opens_attachment_review() -> None:
    policy, _requirement, _request = _factory_inputs()
    store = _Store()
    attachment = _AttachmentRuntime()
    preparation = GenericAgentPlanReviewPreparation(
        dataset_run_id="factory-run.graph-bootstrap",
        store=cast(Any, store),
        policy=policy,
        task_authoring_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: object()),
        ),
        attachment_runtime=cast(Any, attachment),
    )

    result = await preparation.prepare(
        PlanReviewCapabilityRequestV1.create(
            run_id=store.item_run.run_id,
            plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
            plan_ref=store.task_head.result_ref,
            audit=audit(),
        ),
        audit=audit(),
    )

    assert attachment.calls == 1
    assert result.plan_kind is PlanKindV2.ATTACHMENT_GENERATION
    assert result.plan_ref.object_type == "attachment-generation-plan"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan_kind", "source_stage", "object_type"),
    (
        (
            PlanKindV2.CRITERIA_RUBRIC,
            FactoryItemStageV2.ITEM_QUALITY,
            "criteria-rubric-plan",
        ),
        (
            PlanKindV2.GRADING_DESIGN,
            FactoryItemStageV2.CRITERIA_RUBRIC,
            "grading-design-plan",
        ),
    ),
)
async def test_plan_review_preparation_opens_specialist_review(
    plan_kind: PlanKindV2,
    source_stage: FactoryItemStageV2,
    object_type: str,
) -> None:
    policy, _requirement, _request = _factory_inputs()
    store = _Store()
    criteria = _DomainRuntime("criteria-rubric-plan")
    grading = _DomainRuntime("grading-design-plan")
    preparation = GenericAgentPlanReviewPreparation(
        dataset_run_id="factory-run.graph-bootstrap",
        store=cast(Any, store),
        policy=policy,
        task_authoring_materials=cast(Any, object()),
        attachment_runtime=cast(Any, object()),
        criteria_runtime=cast(Any, criteria),
        grading_runtime=cast(Any, grading),
        item_materials=cast(Any, _ItemMaterials()),
    )

    result = await preparation.prepare(
        PlanReviewCapabilityRequestV1.create(
            run_id="factory-run://item",
            plan_kind=plan_kind,
            plan_ref=store.get_item_stage_head(
                store.binding.item_id,
                source_stage,
            ).result_ref,
            audit=audit(),
        ),
        audit=audit(),
    )

    assert result.plan_kind is plan_kind
    assert result.plan_ref.object_type == object_type
    assert criteria.calls == (1 if plan_kind is PlanKindV2.CRITERIA_RUBRIC else 0)
    assert grading.calls == (1 if plan_kind is PlanKindV2.GRADING_DESIGN else 0)


@pytest.mark.asyncio
async def test_plan_review_preparation_rejects_unowned_domain() -> None:
    policy, _requirement, _request = _factory_inputs()
    preparation = GenericAgentPlanReviewPreparation(
        dataset_run_id="factory-run.graph-bootstrap",
        store=cast(Any, _Store()),
        policy=policy,
        task_authoring_materials=cast(Any, object()),
        attachment_runtime=cast(Any, object()),
    )

    with pytest.raises(
        GenericAgentPlanReviewPreparationError,
        match="unavailable",
    ):
        await preparation.prepare(
            PlanReviewCapabilityRequestV1.create(
                run_id="factory-run.graph-bootstrap",
                plan_kind=PlanKindV2.FINAL_DELIVERY,
                plan_ref=ref(
                    "dataset-delivery-plan",
                    "dataset",
                    version="v2",
                ),
                audit=audit(),
            ),
            audit=audit(),
        )


@pytest.mark.asyncio
async def test_plan_review_preparation_opens_final_delivery_review() -> None:
    policy, _requirement, _request = _factory_inputs()
    store = _Store()
    delivery = _DeliveryPreparation()
    preparation = GenericAgentPlanReviewPreparation(
        dataset_run_id="factory-run.graph-bootstrap",
        store=cast(Any, store),
        policy=policy,
        task_authoring_materials=cast(Any, object()),
        attachment_runtime=cast(Any, object()),
        delivery_preparation=cast(Any, delivery),
    )

    result = await preparation.prepare(
        PlanReviewCapabilityRequestV1.create(
            run_id="factory-run.graph-bootstrap",
            plan_kind=PlanKindV2.FINAL_DELIVERY,
            plan_ref=store.aggregate.to_ref(),
            audit=audit(),
        ),
        audit=audit(),
    )

    assert delivery.calls == 1
    assert result.plan_kind is PlanKindV2.FINAL_DELIVERY
    assert result.plan_ref.object_type == "dataset-delivery-plan"
