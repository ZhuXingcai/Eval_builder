from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

import eval_factory.packs.generic_agent_trace.item_materials as module
from eval_factory.packs.generic_agent_trace.item_materials import (
    GenericAgentCriteriaContext,
    GenericAgentFactoryItemMaterialError,
    GenericAgentFactoryItemMaterialSource,
    GenericAgentGradingContext,
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
        self.heads = {
            stage: SimpleNamespace(
                to_ref=lambda stage=stage: ref(
                    "factory-item-stage-head",
                    stage.value.lower(),
                    version="v2",
                ),
            )
            for stage in module.FactoryItemStageV2
        }

    def get_item_binding(self, dataset_run_id: str, item_id: str):
        assert dataset_run_id == "dataset-1"
        assert item_id == "item-1"
        return self.binding

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return SimpleNamespace(run_id="item-run-1")

    def get_item_stage_head(self, item_id: str, stage: object):
        assert item_id == "item-1"
        return self.heads[stage]

    def get_item_stage_material_ref(self, head_ref: object):
        return ref(
            "test-material",
            head_ref.object_id,
            version="v2",
        )

    def get_domain_plan(self, run_id: str, plan_kind: object):
        del plan_kind
        assert run_id == "item-run-1"
        return SimpleNamespace(
            plan_record_json="plan",
            compiled_plan_record_json="compiled",
        )


def _source(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[GenericAgentFactoryItemMaterialSource, object]:
    store = _Store()
    gate_ref = ref(
        "task-prompt-safety-gate",
        "item-1",
        version="v2",
    )
    gate = object()
    task_contract_set = SimpleNamespace(
        task_prompt_safety_gate_ref=gate_ref,
    )
    task_authoring = SimpleNamespace(
        task_draft=object(),
        task_contract_set=task_contract_set,
        prompt_safety_result=SimpleNamespace(
            task_prompt_safety_gate=gate,
        ),
        source_trace_ref=ref(
            "trace-source",
            "item-1",
            version="v1",
        ),
        trace_envelope=object(),
        label_decision=object(),
        selection_context=object(),
        leakage_reference_set=object(),
    )
    quality_result = SimpleNamespace(
        finalization=SimpleNamespace(
            item_quality=object(),
        ),
    )
    criteria_result = SimpleNamespace(
        to_ref=lambda: ref(
            "criteria-rubric-result",
            "item-1",
            version="v2",
        )
    )
    criteria_authority = SimpleNamespace(
        reviewed_task_contract_set=object(),
        reviewed_item_quality=object(),
        execution=SimpleNamespace(result=criteria_result),
    )
    grading_authority = SimpleNamespace(
        execution=SimpleNamespace(
            result=SimpleNamespace(
                to_ref=lambda: ref(
                    "grading-design-result",
                    "item-1",
                    version="v2",
                ),
            ),
        ),
    )
    plan = SimpleNamespace(
        to_ref=lambda: ref(
            "criteria-rubric-plan",
            "item-1",
            version="v2",
        ),
    )
    compiled = SimpleNamespace(
        to_ref=lambda: ref(
            "compiled-criteria-rubric-plan",
            "item-1",
            version="v2",
        ),
    )
    monkeypatch.setattr(
        module.CriteriaRubricPlanV2,
        "model_validate_json",
        lambda value: plan,
    )
    monkeypatch.setattr(
        module.CompiledCriteriaRubricPlanV2,
        "model_validate_json",
        lambda value: compiled,
    )
    monkeypatch.setattr(
        module,
        "task_prompt_safety_gate_ref",
        lambda value: gate_ref,
    )
    monkeypatch.setattr(
        module,
        "item_quality_compilation_result_ref",
        lambda value: ref(
            "item-quality-compilation-result",
            "item-1",
            version="v2",
        ),
    )
    criteria_context = GenericAgentCriteriaContext(
        quality_context=cast(Any, object()),
        binding_definitions=(),
        binding_definitions_ref=ref(
            "evaluator-binding-definitions",
            "item-1",
            version="v2",
        ),
        tool_catalog=cast(Any, object()),
    )
    grading_context = GenericAgentGradingContext(
        model_authorizations=(),
        model_authorizations_ref=ref(
            "model-domain-authorizations",
            "item-1",
            version="v2",
        ),
        evaluated_at=audit().created_at,
    )
    source = GenericAgentFactoryItemMaterialSource(
        dataset_run_id="dataset-1",
        store=cast(Any, store),
        plan_reviews=cast(
            Any,
            SimpleNamespace(
                require_resumed_plan=lambda **kwargs: SimpleNamespace(
                    request=SimpleNamespace(
                        to_ref=lambda: ref(
                            "plan-review-request",
                            "item-1",
                            version="v2",
                        ),
                    ),
                ),
            ),
        ),
        task_authoring_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: task_authoring),
        ),
        quality_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: quality_result),
        ),
        criteria_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: criteria_authority),
        ),
        attachment_materials=cast(
            Any,
            SimpleNamespace(
                get=lambda reference: SimpleNamespace(
                    r5_execution=SimpleNamespace(
                        reconstruction_result=object(),
                    ),
                ),
            ),
        ),
        grading_materials=cast(
            Any,
            SimpleNamespace(get=lambda reference: grading_authority),
        ),
        criteria_contexts={"item-1": criteria_context},
        grading_contexts={"item-1": grading_context},
    )
    return source, criteria_authority


def test_item_material_source_rebuilds_all_downstream_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, criteria_authority = _source(monkeypatch)

    criteria = source.criteria("item-1")
    grading = source.grading("item-1")
    batch = source.batch_source("item-1")
    release = source.release_source("item-1")

    assert source.get("item-1") is criteria.quality_context
    assert grading.criteria.authority is criteria_authority
    assert batch.item_id == "item-1"
    assert release.binding.item_id == "item-1"


def test_item_material_source_requires_registered_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _authority = _source(monkeypatch)

    with pytest.raises(
        GenericAgentFactoryItemMaterialError,
        match="Criteria context",
    ):
        source.criteria("missing")
    with pytest.raises(
        GenericAgentFactoryItemMaterialError,
        match="Grading context",
    ):
        source.grading("missing")
