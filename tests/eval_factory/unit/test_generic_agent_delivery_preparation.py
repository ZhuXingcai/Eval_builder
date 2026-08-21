from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_factory_graph_bootstrap import _factory_inputs
from test_support.team_runtime_fixtures import audit, ref

import eval_factory.packs.generic_agent_trace.delivery_preparation as module
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryWaitingView,
)
from eval_factory.packs.generic_agent_trace.delivery_preparation import (
    GenericAgentDeliveryPreparation,
    GenericAgentDeliveryPreparationError,
)


class _Store:
    def __init__(self) -> None:
        self.binding = SimpleNamespace(
            item_id="item-1",
            to_ref=lambda: ref(
                "factory-item-run-binding",
                "item-1",
                version="v2",
            ),
        )
        self.aggregate = SimpleNamespace(
            candidate_binding_refs=(self.binding.to_ref(),),
        )
        self.missing_binding = False

    def get_dataset_aggregate(self, dataset_run_id: str):
        assert dataset_run_id == "dataset-1"
        return self.aggregate

    def get_dataset_aggregate_material_ref(
        self,
        dataset_run_id: str,
    ):
        assert dataset_run_id == "dataset-1"
        return ref(
            "batch-quality-material",
            "dataset-1",
            version="v2",
        )

    def list_item_bindings(self, dataset_run_id: str):
        assert dataset_run_id == "dataset-1"
        return () if self.missing_binding else (self.binding,)

    def get_item_stage_head(self, item_id: str, stage: object):
        del stage
        assert item_id == "item-1"
        return SimpleNamespace(
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "release",
                version="v2",
            ),
        )

    def get_item_stage_material_ref(self, head_ref: object):
        del head_ref
        return ref(
            "release-projection-material",
            "item-1",
            version="v2",
        )


def _preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GenericAgentDeliveryPreparation, _Store, object]:
    _policy, _requirement, _request = _factory_inputs()
    store = _Store()
    batch = SimpleNamespace(report=object())
    monkeypatch.setattr(
        module,
        "FactoryBatchQualityResult",
        SimpleNamespace,
    )
    waiting = FactoryDeliveryWaitingView(
        plan_ref=ref(
            "dataset-delivery-plan",
            "dataset-1",
            version="v2",
        ),
        compiled_plan_ref=ref(
            "compiled-dataset-delivery-plan",
            "dataset-1",
            version="v2",
        ),
        review_ref=ref(
            "plan-review-request",
            "dataset-1",
            version="v2",
        ),
    )
    candidate = object()
    release_context = SimpleNamespace(
        exports=(),
        prepare_review=lambda **kwargs: SimpleNamespace(
            delivery=waiting,
        ),
        runtime=SimpleNamespace(
            candidates=SimpleNamespace(
                materials=SimpleNamespace(
                    get=lambda reference: candidate,
                ),
            ),
        ),
    )
    preparation = GenericAgentDeliveryPreparation(
        dataset_run_id="dataset-1",
        store=cast(Any, store),
        batch_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: batch),
        ),
        item_materials=cast(
            Any,
            SimpleNamespace(
                release_source=lambda item_id: object(),
            ),
        ),
        release_context=cast(Any, release_context),
        policy=_policy,
    )
    return preparation, store, candidate


def test_delivery_preparation_builds_review_and_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation, _store, candidate = _preparation(monkeypatch)

    review = preparation.prepare_review(audit=audit())
    material = preparation.material()

    assert review.plan_ref.object_type == "dataset-delivery-plan"
    assert material.candidates == (candidate,)
    assert material.exports == ()


def test_delivery_preparation_rejects_missing_candidate_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation, store, _candidate = _preparation(monkeypatch)
    store.missing_binding = True

    with pytest.raises(
        GenericAgentDeliveryPreparationError,
        match="binding",
    ):
        preparation.prepare_review(audit=audit())
    with pytest.raises(
        GenericAgentDeliveryPreparationError,
        match="binding",
    ):
        preparation.material()
