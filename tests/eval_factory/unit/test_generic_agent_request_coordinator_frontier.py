from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_support.team_runtime_fixtures import audit, ref

import eval_factory.packs.generic_agent_trace.request_coordinator as module
from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentTeamTaskSpec,
)
from eval_factory.packs.generic_agent_trace.request_coordinator import (
    GenericAgentFactoryRequestCoordinator,
)
from eval_factory.team import TeamTaskStatusV1


class _Preparation:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        if name.startswith("prepare_"):

            def record(*args: object, **kwargs: object) -> None:
                del args, kwargs
                self.calls.append(name)

            return record
        raise AttributeError(name)


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
        self.item_run = SimpleNamespace(run_id="item-run-1")
        self.aggregate = SimpleNamespace(
            to_ref=lambda: ref(
                "factory-dataset-aggregate-result",
                "dataset-1",
                version="v2",
            ),
        )
        self.domain_exists = True

    def get_item_binding(self, *args: object):
        return self.binding

    def get_item_run(self, binding: object):
        assert binding is self.binding
        return self.item_run

    def get_domain_plan(self, run_id: str, plan_kind: object):
        if not self.domain_exists:
            raise module.FactoryControlNotFoundError("not found")
        return SimpleNamespace(
            plan_ref=ref(
                {
                    PlanKindV2.ATTACHMENT_GENERATION: ("attachment-generation-plan"),
                    PlanKindV2.CRITERIA_RUBRIC: ("criteria-rubric-plan"),
                    PlanKindV2.GRADING_DESIGN: ("grading-design-plan"),
                    PlanKindV2.FINAL_DELIVERY: ("dataset-delivery-plan"),
                }[plan_kind],
                "frontier",
                version="v2",
            ),
            plan_record_json="plan",
            compiled_plan_record_json="compiled",
        )

    def get_item_stage_head(self, item_id: str, stage: object):
        del stage
        assert item_id == "item-1"
        return SimpleNamespace(
            result_ref=ref(
                "stage-result",
                "frontier",
                version="v2",
            ),
            to_ref=lambda: ref(
                "factory-item-stage-head",
                "frontier",
                version="v2",
            ),
        )

    def get_item_stage_material_ref(self, head_ref: object):
        del head_ref
        return ref(
            "owner-material",
            "frontier",
            version="v2",
        )

    def get_domain_plan_material_ref(
        self,
        *,
        run_id: str,
        plan_kind: object,
    ):
        del run_id, plan_kind
        return ref(
            "attachment-r5-preparation",
            "frontier",
            version="v2",
        )

    def list_item_bindings(self, dataset_run_id: str):
        return (self.binding,)

    def get_dataset_aggregate(self, dataset_run_id: str):
        return self.aggregate


def _value(object_type: str):
    return SimpleNamespace(
        to_ref=lambda: ref(
            object_type,
            "frontier",
            version="v2",
        ),
    )


def _coordinator(
    capability_id: str,
    *,
    plan_kind: PlanKindV2 | None = None,
) -> tuple[
    GenericAgentFactoryRequestCoordinator,
    _Preparation,
    object,
    object,
]:
    coordinator = object.__new__(GenericAgentFactoryRequestCoordinator)
    preparation = _Preparation()
    store = _Store()
    source_ref = ref("trace-source", "frontier", version="v1")
    instance = SimpleNamespace(
        capability_id=capability_id,
        scope_ref=(
            None
            if capability_id
            in {
                "capability.batch-quality",
                "capability.delivery",
            }
            or plan_kind is PlanKindV2.FINAL_DELIVERY
            else source_ref
        ),
        plan_kind=plan_kind,
    )
    coordinator.request = SimpleNamespace()
    coordinator.dataset_run_id = "dataset-1"
    coordinator.requirement = SimpleNamespace()
    coordinator.policy = _value("factory-run-policy")
    coordinator.factory_store = cast(Any, store)
    coordinator.candidate_store = cast(Any, object())
    coordinator.item_materializer = cast(Any, object())
    coordinator.materialization = cast(
        Any,
        SimpleNamespace(instance=lambda task_id: instance),
    )
    coordinator.preparation = cast(Any, preparation)
    coordinator.trace_sources = {
        source_ref: TraceSourceRef(
            source_trace_id="trace-frontier",
            source_uri="file:///tmp/frontier.json",
            raw_sha256="a" * 64,
            adapter_name="raw_traj_v1",
            adapter_version="v1",
            processing_class="RESTRICTED_TRACE_RAW",
        ),
    }
    coordinator.attachment_preparation_materials = cast(
        Any,
        SimpleNamespace(get=lambda reference: object()),
    )
    quality = SimpleNamespace(
        result=SimpleNamespace(
            finalization=SimpleNamespace(
                attachment_quality=_value(
                    "attachment-quality-assessment",
                ),
            ),
            solvability=SimpleNamespace(
                assessment=_value("solvability-assessment"),
            ),
        ),
    )
    criteria_execution = SimpleNamespace(
        result=_value("criteria-rubric-result"),
        route=_value("model-route-decision"),
        rubric_set=object(),
        evaluator_spec=object(),
        reference_policy=object(),
        tool_policy=object(),
    )
    coordinator.item_materials = cast(
        Any,
        SimpleNamespace(
            criteria=lambda item_id: SimpleNamespace(
                task_draft=object(),
                quality=quality,
                binding_definitions_ref=ref(
                    "evaluator-binding-definitions",
                    "frontier",
                    version="v2",
                ),
                binding_definitions=(),
                tool_catalog=object(),
            ),
            grading=lambda item_id: SimpleNamespace(
                criteria=SimpleNamespace(
                    authority=SimpleNamespace(
                        execution=criteria_execution,
                    ),
                ),
                model_authorizations_ref=ref(
                    "model-domain-authorizations",
                    "frontier",
                    version="v2",
                ),
                model_authorizations=(),
                evaluated_at=audit().created_at,
            ),
            batch_source=lambda item_id: SimpleNamespace(
                item_id=item_id,
            ),
        ),
    )
    core_result = _value("core-vertical-result")
    core_result.object_sha256 = "b" * 64
    coordinator.candidate_aggregate = cast(
        Any,
        SimpleNamespace(build=lambda audit: core_result),
    )
    coordinator.batch_quality_context = SimpleNamespace(
        resolved_job_work_graph=object(),
        duplicate_policy=object(),
        cross_item_policy=object(),
        batch_policy=object(),
    )
    coordinator.delivery_preparation = cast(
        Any,
        SimpleNamespace(
            material=lambda: SimpleNamespace(
                aggregate=store.aggregate,
                candidates=(
                    SimpleNamespace(
                        projection=_value(
                            "release-projection-result",
                        ),
                    ),
                ),
                exports=(
                    SimpleNamespace(
                        item_id="item-1",
                        files=(),
                    ),
                ),
                policy=coordinator.policy,
            ),
        ),
    )
    work = SimpleNamespace(
        task=SimpleNamespace(
            task_id="team-frontier.task-test",
            object_sha256="c" * 64,
        ),
        status=TeamTaskStatusV1.READY,
        attempt=0,
    )
    snapshot = SimpleNamespace(
        team=SimpleNamespace(team_id="team-frontier"),
    )
    return coordinator, preparation, snapshot, work


@pytest.mark.parametrize(
    ("capability_id", "plan_kind", "expected"),
    (
        (
            "capability.attachment-reconstruction",
            None,
            "prepare_attachment_reconstruction",
        ),
        (
            "capability.quality-review",
            None,
            "prepare_attachment_quality_workflow",
        ),
        (
            "capability.criteria-rubric",
            None,
            "prepare_criteria_rubric",
        ),
        (
            "capability.grading-design",
            None,
            "prepare_grading_design",
        ),
        (
            "capability.batch-quality",
            None,
            "prepare_batch_quality",
        ),
        (
            "capability.plan-review",
            PlanKindV2.FINAL_DELIVERY,
            "prepare_plan_review",
        ),
        (
            "capability.delivery",
            None,
            "prepare_delivery",
        ),
    ),
)
def test_request_coordinator_prepares_remaining_frontier(
    monkeypatch: pytest.MonkeyPatch,
    capability_id: str,
    plan_kind: PlanKindV2 | None,
    expected: str,
) -> None:
    for model in (
        module.AttachmentGenerationPlanV2,
        module.CompiledAttachmentGenerationPlanV2,
        module.CriteriaRubricPlanV2,
        module.CompiledCriteriaRubricPlanV2,
        module.GradingDesignPlanV2,
        module.CompiledGradingDesignPlanV2,
    ):
        monkeypatch.setattr(
            model,
            "model_validate_json",
            lambda value, model=model: _value(
                {
                    module.AttachmentGenerationPlanV2: ("attachment-generation-plan"),
                    module.CompiledAttachmentGenerationPlanV2: ("compiled-attachment-generation-plan"),
                    module.CriteriaRubricPlanV2: "criteria-rubric-plan",
                    module.CompiledCriteriaRubricPlanV2: ("compiled-criteria-rubric-plan"),
                    module.GradingDesignPlanV2: "grading-design-plan",
                    module.CompiledGradingDesignPlanV2: ("compiled-grading-design-plan"),
                }[model]
            ),
        )
    for name, object_type in (
        ("task_draft_ref", "task-draft"),
        ("generic_agent_tool_catalog_ref", "tool-capability-catalog"),
        ("rubric_set_ref", "rubric-set"),
        ("evaluator_spec_ref", "evaluator-spec"),
        ("reference_policy_ref", "reference-policy"),
        ("tool_policy_ref", "tool-policy"),
        (
            "resolved_job_work_graph_v2_ref",
            "resolved-job-work-graph",
        ),
        ("duplicate_detection_policy_v2_ref", "duplicate-policy"),
        ("cross_item_safety_policy_v2_ref", "cross-item-policy"),
        ("batch_quality_policy_v2_ref", "batch-policy"),
        ("authorized_candidate_export_ref", "authorized-candidate-export"),
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda value, object_type=object_type: ref(
                object_type,
                "frontier",
                version="v2",
            ),
        )
    coordinator, preparation, snapshot, work = _coordinator(
        capability_id,
        plan_kind=plan_kind,
    )

    coordinator.prepare(
        snapshot=cast(Any, snapshot),
        work=cast(Any, work),
        spec=GenericAgentTeamTaskSpec(
            capability_id=capability_id,
            dependency_capability_ids=(),
            input_schema_refs=(),
            output_schema_refs=(),
        ),
        audit=audit(),
    )

    assert preparation.calls == [expected]
