from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

import eval_factory.packs.generic_agent_trace.specialist_projections as module
from eval_factory.packs.generic_agent_trace.specialist_projections import (
    GenericAgentCriteriaProjection,
    GenericAgentGradingProjection,
    GenericAgentSpecialistProjectionError,
)


class _Store:
    def __init__(self) -> None:
        self.binding = SimpleNamespace(item_id="item-1")
        self.plan_material = SimpleNamespace(
            plan_record_json="plan",
            compiled_plan_record_json="compiled",
        )
        self.unique = True

    def list_item_bindings(self, dataset_run_id: str):
        assert dataset_run_id == "dataset-1"
        return (self.binding,) if self.unique else ()

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return SimpleNamespace(run_id="item-run-1")

    def get_domain_plan(self, run_id: str, plan_kind: object):
        del plan_kind
        assert run_id == "item-run-1"
        return self.plan_material


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.plan_reviews = SimpleNamespace(
            require_resumed_plan=lambda **kwargs: SimpleNamespace(
                request=SimpleNamespace(
                    to_ref=lambda: ref(
                        "plan-review-request",
                        "specialist",
                        version="v2",
                    ),
                ),
            ),
        )

    def commit_execution(self, **kwargs: object):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stage_head_ref=ref(
                "factory-item-stage-head",
                "specialist",
                version="v2",
            ),
            material_ref=ref(
                "specialist-material",
                "specialist",
                version="v2",
            ),
        )


def _patch_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = SimpleNamespace(
        to_ref=lambda: ref(
            "criteria-rubric-plan",
            "specialist",
            version="v2",
        ),
    )
    compiled = SimpleNamespace(
        to_ref=lambda: ref(
            "compiled-criteria-rubric-plan",
            "specialist",
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
        module.GradingDesignPlanV2,
        "model_validate_json",
        lambda value: plan,
    )
    monkeypatch.setattr(
        module.CompiledGradingDesignPlanV2,
        "model_validate_json",
        lambda value: compiled,
    )


def test_criteria_projection_commits_current_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plans(monkeypatch)
    task_ref = ref("task-draft", "specialist", version="v2")
    quality_ref = ref(
        "attachment-quality-assessment",
        "specialist",
        version="v2",
    )
    solvability_ref = ref(
        "solvability-assessment",
        "specialist",
        version="v2",
    )
    catalog_ref = ref(
        "tool-capability-catalog",
        "specialist",
        version="v2",
    )
    monkeypatch.setattr(module, "task_draft_ref", lambda value: task_ref)
    monkeypatch.setattr(
        module,
        "generic_agent_tool_catalog_ref",
        lambda value: catalog_ref,
    )
    quality = SimpleNamespace(
        result=SimpleNamespace(
            finalization=SimpleNamespace(
                attachment_quality=SimpleNamespace(
                    to_ref=lambda: quality_ref,
                ),
            ),
            solvability=SimpleNamespace(
                assessment=SimpleNamespace(
                    to_ref=lambda: solvability_ref,
                ),
            ),
        ),
    )
    source = SimpleNamespace(
        task_draft=object(),
        task_contract_set=object(),
        quality=quality,
        quality_context=object(),
        tool_catalog=object(),
    )
    runtime = _Runtime()
    projection = GenericAgentCriteriaProjection(
        dataset_run_id="dataset-1",
        store=cast(Any, _Store()),
        runtime=cast(Any, runtime),
        materials=cast(
            Any,
            SimpleNamespace(criteria=lambda item_id: source),
        ),
    )
    request = SimpleNamespace(
        run_id="item-run-1",
        task_draft_ref=task_ref,
        attachment_quality_ref=quality_ref,
        solvability_ref=solvability_ref,
        tool_catalog_ref=catalog_ref,
    )

    refs = projection.commit(
        request=cast(Any, request),
        execution=cast(Any, object()),
        audit=audit(),
    )

    assert len(refs) == 2
    assert len(runtime.calls) == 1


def test_grading_projection_commits_current_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_plans(monkeypatch)
    refs = {
        name: ref(object_type, name, version="v2")
        for name, object_type in (
            ("result", "criteria-rubric-result"),
            ("route", "model-route-decision"),
            ("rubric", "rubric-set"),
            ("evaluator", "evaluator-spec"),
            ("reference", "reference-policy"),
            ("tool", "tool-policy"),
        )
    }
    for name in (
        "rubric_set_ref",
        "evaluator_spec_ref",
        "reference_policy_ref",
        "tool_policy_ref",
    ):
        key = (
            name.removesuffix("_set_ref")
            .removesuffix(
                "_spec_ref",
            )
            .removesuffix("_policy_ref")
        )
        monkeypatch.setattr(
            module,
            name,
            lambda value, key=key: refs[key],
        )
    criteria_execution = SimpleNamespace(
        result=SimpleNamespace(to_ref=lambda: refs["result"]),
        route=SimpleNamespace(to_ref=lambda: refs["route"]),
        rubric_set=object(),
        evaluator_spec=object(),
        reference_policy=object(),
        tool_policy=object(),
    )
    criteria = SimpleNamespace(
        authority=SimpleNamespace(execution=criteria_execution),
    )
    runtime = _Runtime()
    projection = GenericAgentGradingProjection(
        dataset_run_id="dataset-1",
        store=cast(Any, _Store()),
        runtime=cast(Any, runtime),
        materials=cast(
            Any,
            SimpleNamespace(
                grading=lambda item_id: SimpleNamespace(
                    criteria=criteria,
                ),
            ),
        ),
    )
    request = SimpleNamespace(
        run_id="item-run-1",
        criteria_result_ref=refs["result"],
        criteria_route_ref=refs["route"],
        rubric_set_ref=refs["rubric"],
        evaluator_spec_ref=refs["evaluator"],
        reference_policy_ref=refs["reference"],
        tool_policy_ref=refs["tool"],
    )

    result = projection.commit(
        request=cast(Any, request),
        execution=cast(Any, object()),
        audit=audit(),
    )

    assert len(result) == 2
    assert len(runtime.calls) == 1


def test_specialist_projection_rejects_missing_binding() -> None:
    store = _Store()
    store.unique = False
    with pytest.raises(
        GenericAgentSpecialistProjectionError,
        match="not unique",
    ):
        GenericAgentCriteriaProjection(
            dataset_run_id="dataset-1",
            store=cast(Any, store),
            runtime=cast(Any, object()),
            materials=cast(Any, object()),
        ).commit(
            request=cast(Any, SimpleNamespace(run_id="item-run-1")),
            execution=cast(Any, object()),
            audit=audit(),
        )
