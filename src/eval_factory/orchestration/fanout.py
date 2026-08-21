from __future__ import annotations

from typing import TYPE_CHECKING, Final

from eval_factory.contracts.attachment_v2 import (
    ArtifactExecutionBatchV2,
    ArtifactExecutionPlanV2,
    ArtifactExecutionReceiptOutcomeV2,
    artifact_execution_batch_ref,
    artifact_execution_group_ref,
    artifact_execution_plan_ref,
    artifact_execution_receipt_carried_sha256,
    artifact_execution_receipt_ref,
    artifact_execution_unit_ref,
)
from eval_factory.contracts.canary_execution_v2 import (
    SemanticReviewFanoutV2,
    SemanticReviewRoundWorkV2,
    semantic_review_fanout_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.model_control_v2 import (
    ModelAdmissionDecisionV2,
    ModelAdmissionOutcomeV2,
    WorkModelDemandV2,
    model_admission_decision_v2_ref,
    work_model_demand_v2_ref,
)
from eval_factory.contracts.orchestration import ItemStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    R6_WORK_GRAPH_POLICY_VERSION,
    ArtifactGroupFanoutV2,
    DatasetJobSpecV2,
    ResolvedDatasetJobPlanV2,
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkLeaseEventKindV2,
    WorkLeaseEventV2,
    WorkReadinessSnapshotV2,
    WorkReadinessV2,
    WorkRetryDecisionKindV2,
    WorkRetryDecisionV2,
    WorkUnitScopeV2,
    artifact_group_fanout_v2_ref,
    dataset_item_id_v2,
    dataset_job_spec_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    trace_source_v2_ref,
    work_lease_event_v2_ref,
    work_readiness_snapshot_v2_carried_sha256,
    work_retry_decision_v2_ref,
)
from eval_factory.contracts.resource_v2 import (
    ResourceAdmissionDecisionV2,
    ResourceAdmissionOutcomeV2,
    WorkResourceDemandV2,
    resource_admission_decision_v2_ref,
    work_resource_demand_v2_ref,
)
from eval_factory.contracts.review_v2 import SemanticReviewRoundV2
from eval_factory.orchestration.models import (
    ItemRecord,
    StageResultRecord,
    StageRunRecord,
    item_record_ref,
    stage_result_record_carried_sha256,
    stage_result_record_ref,
)
from eval_factory.orchestration.planning import DatasetJobPlanCompiler

if TYPE_CHECKING:
    from eval_factory.attachment_planning.execution_models import (
        ArtifactExecutionPlanningResult,
    )

_POLICY_COMPONENT: Final = "dataset-work-graph"
_ITEM_STAGES: Final = frozenset(
    {
        StageNameV2.TRACE_INDEX,
        StageNameV2.SAFETY,
        StageNameV2.LABEL,
        StageNameV2.TASK_AUTHORING,
        StageNameV2.ATTACHMENT,
        StageNameV2.ITEM_QUALITY,
    }
)
_TERMINAL_ITEM_STATUSES: Final = frozenset({ItemStatus.REJECTED, ItemStatus.FAILED, ItemStatus.REVOKED})


class WorkFanoutPolicyError(RuntimeError):
    pass


class StageWorkCompletion(ContractModelV2):
    work_unit_ref: ObjectRef
    stage_run: StageRunRecord
    stage_result: StageResultRecord


class DatasetJobWorkGraphCompiler:
    def compile(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        resolved_plan: ResolvedDatasetJobPlanV2,
        audit: ContractAudit,
    ) -> ResolvedJobWorkGraphV2:
        spec = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        DatasetJobPlanCompiler().validate_current(resolved_plan, job_spec=spec)
        source_refs = tuple(
            sorted(
                (trace_source_v2_ref(trace) for trace in spec.traces),
                key=_ref_key,
            )
        )
        if len(source_refs) != len(set(source_refs)):
            raise WorkFanoutPolicyError("job traces must be unique")
        item_ids = tuple(dataset_item_id_v2(spec.job_id, ref) for ref in source_refs)
        units: list[ResolvedWorkUnitV2] = []
        latest_item: dict[str, ResolvedWorkUnitV2] = {}
        last_job: ResolvedWorkUnitV2 | None = None
        previous_scope: WorkUnitScopeV2 | None = None
        previous_units: tuple[ResolvedWorkUnitV2, ...] = ()
        previous_stage: StageNameV2 | None = None

        for stage in resolved_plan.resolved_stages:
            if stage in _ITEM_STAGES:
                current: list[ResolvedWorkUnitV2] = []
                for item_id, source_ref in zip(item_ids, source_refs, strict=True):
                    dependencies: list[ObjectRef] = []
                    prior_item = latest_item.get(item_id)
                    if prior_item is not None:
                        dependencies.append(resolved_work_unit_v2_ref(prior_item))
                    if previous_scope is WorkUnitScopeV2.JOB and last_job is not None:
                        dependencies.append(resolved_work_unit_v2_ref(last_job))
                    join_mode = (
                        WorkDependencyJoinModeV2.ALL_TERMINAL
                        if stage is StageNameV2.ITEM_QUALITY and previous_stage is StageNameV2.ATTACHMENT
                        else WorkDependencyJoinModeV2.ALL_SUCCEEDED
                    )
                    unit = ResolvedWorkUnitV2.create(
                        scope=WorkUnitScopeV2.ITEM,
                        stage=stage,
                        job_id=spec.job_id,
                        item_id=item_id,
                        source_trace_ref=source_ref,
                        artifact_execution_group_ref=None,
                        depends_on_work_unit_refs=tuple(dependencies),
                        join_mode=join_mode,
                    )
                    current.append(unit)
                    latest_item[item_id] = unit
                previous_units = tuple(current)
                units.extend(current)
                previous_scope = WorkUnitScopeV2.ITEM
            else:
                if previous_units and previous_units[0].scope is WorkUnitScopeV2.ITEM:
                    job_dependencies = tuple(resolved_work_unit_v2_ref(unit) for unit in previous_units)
                    join_mode = WorkDependencyJoinModeV2.ALL_TERMINAL
                elif last_job is not None:
                    job_dependencies = (resolved_work_unit_v2_ref(last_job),)
                    join_mode = WorkDependencyJoinModeV2.ALL_SUCCEEDED
                else:
                    job_dependencies = ()
                    join_mode = WorkDependencyJoinModeV2.ALL_SUCCEEDED
                unit = ResolvedWorkUnitV2.create(
                    scope=WorkUnitScopeV2.JOB,
                    stage=stage,
                    job_id=spec.job_id,
                    item_id=None,
                    source_trace_ref=None,
                    artifact_execution_group_ref=None,
                    depends_on_work_unit_refs=job_dependencies,
                    join_mode=join_mode,
                )
                units.append(unit)
                previous_units = (unit,)
                last_job = unit
                previous_scope = WorkUnitScopeV2.JOB
            previous_stage = stage

        spec_ref = dataset_job_spec_v2_ref(spec)
        plan_ref = _resolved_plan_ref(resolved_plan)
        return ResolvedJobWorkGraphV2.create(
            job_id=spec.job_id,
            dataset_job_spec_ref=spec_ref,
            resolved_job_plan_ref=plan_ref,
            item_ids=item_ids,
            source_trace_refs=source_refs,
            work_units=tuple(units),
            audit=_safe_audit(audit, (spec_ref, plan_ref)),
        )

    def validate_current(
        self,
        graph: ResolvedJobWorkGraphV2,
        *,
        job_spec: DatasetJobSpecV2,
        resolved_plan: ResolvedDatasetJobPlanV2,
    ) -> None:
        canonical = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        expected = self.compile(
            job_spec=job_spec,
            resolved_plan=resolved_plan,
            audit=canonical.audit,
        )
        if canonical != expected:
            raise WorkFanoutPolicyError("work graph does not match current job plan and scope policy")


class ArtifactGroupFanoutCompiler:
    def compile(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        parent_attachment_work_unit: ResolvedWorkUnitV2,
        artifact_execution_planning_result: ArtifactExecutionPlanningResult,
        audit: ContractAudit,
    ) -> ArtifactGroupFanoutV2:
        graph = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        parent_ref = resolved_work_unit_v2_ref(parent_attachment_work_unit)
        if (
            parent_attachment_work_unit not in graph.work_units
            or parent_attachment_work_unit.scope is not WorkUnitScopeV2.ITEM
            or parent_attachment_work_unit.stage is not StageNameV2.ATTACHMENT
        ):
            raise WorkFanoutPolicyError("artifact fanout parent must be an ITEM ATTACHMENT work unit")
        artifact_execution_plan = _current_artifact_execution_plan(artifact_execution_planning_result)
        plan_ref = artifact_execution_plan_ref(artifact_execution_plan)
        group_units = tuple(
            ResolvedWorkUnitV2.create(
                scope=WorkUnitScopeV2.ARTIFACT_GROUP,
                stage=StageNameV2.ATTACHMENT,
                job_id=graph.job_id,
                item_id=parent_attachment_work_unit.item_id,
                source_trace_ref=None,
                artifact_execution_group_ref=artifact_execution_group_ref(group),
                depends_on_work_unit_refs=(),
                join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
            )
            for group in sorted(
                artifact_execution_plan.groups,
                key=lambda value: _ref_key(artifact_execution_group_ref(value)),
            )
        )
        graph_ref = resolved_job_work_graph_v2_ref(graph)
        group_refs = tuple(
            unit.artifact_execution_group_ref
            for unit in group_units
            if unit.artifact_execution_group_ref is not None
        )
        return ArtifactGroupFanoutV2.create(
            resolved_job_work_graph_ref=graph_ref,
            parent_attachment_work_unit_ref=parent_ref,
            artifact_execution_plan_ref=plan_ref,
            group_work_units=group_units,
            audit=_safe_audit(
                audit,
                (graph_ref, parent_ref, plan_ref, *group_refs),
            ),
        )

    def validate_current(
        self,
        fanout: ArtifactGroupFanoutV2,
        *,
        graph: ResolvedJobWorkGraphV2,
        parent_attachment_work_unit: ResolvedWorkUnitV2,
        artifact_execution_planning_result: ArtifactExecutionPlanningResult,
    ) -> None:
        canonical = ArtifactGroupFanoutV2.model_validate(fanout.model_dump(mode="python"))
        expected = self.compile(
            graph=graph,
            parent_attachment_work_unit=parent_attachment_work_unit,
            artifact_execution_planning_result=(artifact_execution_planning_result),
            audit=canonical.audit,
        )
        if canonical != expected:
            raise WorkFanoutPolicyError("artifact group fanout is stale")


class SemanticReviewFanoutCompiler:
    def compile(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        parent_item_quality_work_unit: ResolvedWorkUnitV2,
        audit: ContractAudit,
    ) -> SemanticReviewFanoutV2:
        graph = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        if (
            parent_item_quality_work_unit not in graph.work_units
            or parent_item_quality_work_unit.scope is not WorkUnitScopeV2.ITEM
            or parent_item_quality_work_unit.stage is not StageNameV2.ITEM_QUALITY
            or parent_item_quality_work_unit.semantic_review_round is not None
        ):
            raise WorkFanoutPolicyError(
                "semantic review fanout parent must be the base ITEM_QUALITY finalizer"
            )
        source_ref = parent_item_quality_work_unit.source_trace_ref
        if source_ref is None:
            raise WorkFanoutPolicyError("semantic review fanout parent has no trace")
        round_units: list[SemanticReviewRoundWorkV2] = []
        dependencies = parent_item_quality_work_unit.depends_on_work_unit_refs
        for round_ in SemanticReviewRoundV2:
            unit = ResolvedWorkUnitV2.create(
                scope=WorkUnitScopeV2.ITEM,
                stage=StageNameV2.ITEM_QUALITY,
                job_id=parent_item_quality_work_unit.job_id,
                item_id=parent_item_quality_work_unit.item_id,
                source_trace_ref=source_ref,
                artifact_execution_group_ref=None,
                depends_on_work_unit_refs=dependencies,
                join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
                semantic_review_round=round_,
            )
            round_units.append(
                SemanticReviewRoundWorkV2(
                    round=round_,
                    work_unit=unit,
                )
            )
            dependencies = (resolved_work_unit_v2_ref(unit),)
        return SemanticReviewFanoutV2.create(
            resolved_job_work_graph_ref=(resolved_job_work_graph_v2_ref(graph)),
            parent_item_quality_work_unit_ref=(resolved_work_unit_v2_ref(parent_item_quality_work_unit)),
            round_work=tuple(round_units),
            audit=audit,
        )

    def validate_current(
        self,
        fanout: SemanticReviewFanoutV2,
        *,
        graph: ResolvedJobWorkGraphV2,
        parent_item_quality_work_unit: ResolvedWorkUnitV2,
    ) -> None:
        canonical = SemanticReviewFanoutV2.model_validate(fanout.model_dump(mode="python"))
        expected = self.compile(
            graph=graph,
            parent_item_quality_work_unit=(parent_item_quality_work_unit),
            audit=canonical.audit,
        )
        if canonical != expected:
            raise WorkFanoutPolicyError("semantic review fanout is stale")


class WorkReadinessEvaluator:
    def evaluate(
        self,
        *,
        graph: ResolvedJobWorkGraphV2,
        work_unit: ResolvedWorkUnitV2,
        stage_completions: tuple[StageWorkCompletion, ...],
        item_records: tuple[ItemRecord, ...],
        audit: ContractAudit,
        work_retry_decisions: tuple[WorkRetryDecisionV2, ...] = (),
        work_lease_events: tuple[WorkLeaseEventV2, ...] = (),
        resource_admission_decisions: tuple[ResourceAdmissionDecisionV2, ...] = (),
        work_resource_demands: tuple[WorkResourceDemandV2, ...] = (),
        model_admission_decisions: tuple[ModelAdmissionDecisionV2, ...] = (),
        work_model_demands: tuple[WorkModelDemandV2, ...] = (),
        artifact_group_fanout: ArtifactGroupFanoutV2 | None = None,
        semantic_review_fanout: SemanticReviewFanoutV2 | None = None,
        artifact_execution_planning_result: (ArtifactExecutionPlanningResult | None) = None,
        artifact_execution_batch: ArtifactExecutionBatchV2 | None = None,
    ) -> WorkReadinessSnapshotV2:
        graph = ResolvedJobWorkGraphV2.model_validate(graph.model_dump(mode="python"))
        graph_units = {_ref_key(resolved_work_unit_v2_ref(unit)): unit for unit in graph.work_units}
        if artifact_group_fanout is not None:
            artifact_group_fanout = ArtifactGroupFanoutV2.model_validate(
                artifact_group_fanout.model_dump(mode="python")
            )
            _validate_artifact_fanout_binding(
                graph,
                artifact_group_fanout,
            )
            graph_units.update(
                {
                    _ref_key(resolved_work_unit_v2_ref(unit)): unit
                    for unit in artifact_group_fanout.group_work_units
                }
            )
        if semantic_review_fanout is not None:
            semantic_review_fanout = SemanticReviewFanoutV2.model_validate(
                semantic_review_fanout.model_dump(mode="python")
            )
            _validate_semantic_review_fanout_binding(
                graph,
                semantic_review_fanout,
                work_unit,
            )
            graph_units.update(
                {
                    _ref_key(resolved_work_unit_v2_ref(value.work_unit)): (value.work_unit)
                    for value in semantic_review_fanout.round_work
                }
            )
        retry_by_unit, event_by_unit = _control_evidence_by_unit(
            graph_units,
            work_retry_decisions,
            work_lease_events,
        )
        resource_decision_by_unit = _resource_evidence_by_unit(
            graph_units,
            resource_admission_decisions,
            work_resource_demands,
        )
        model_decision_by_unit = _model_evidence_by_unit(
            graph_units,
            model_admission_decisions,
            work_model_demands,
        )
        unit_ref = resolved_work_unit_v2_ref(work_unit)
        if _ref_key(unit_ref) not in graph_units:
            raise WorkFanoutPolicyError("work unit is not owned by the graph")

        items: dict[str, ItemRecord] = {}
        for item in item_records:
            if item.item_id in items or item.job_id != graph.job_id or item.item_id not in graph.item_ids:
                raise WorkFanoutPolicyError("Item readiness evidence is duplicate or cross-graph")
            items[item.item_id] = item
        if (
            work_unit.item_id is not None
            and work_unit.item_id in items
            and items[work_unit.item_id].status in _TERMINAL_ITEM_STATUSES
        ):
            return _snapshot(
                graph=graph,
                fanout=artifact_group_fanout,
                semantic_fanout=semantic_review_fanout,
                work_unit=work_unit,
                readiness=WorkReadinessV2.SKIPPED_ITEM,
                skipped_item_refs=(item_record_ref(items[work_unit.item_id]),),
                audit=audit,
            )

        if (
            artifact_group_fanout is not None
            and unit_ref == artifact_group_fanout.parent_attachment_work_unit_ref
        ):
            if stage_completions:
                raise WorkFanoutPolicyError("artifact group join cannot use StageRun completion evidence")
            return _artifact_join_snapshot(
                graph=graph,
                fanout=artifact_group_fanout,
                parent_work_unit=work_unit,
                artifact_execution_planning_result=(artifact_execution_planning_result),
                artifact_execution_batch=artifact_execution_batch,
                retry_by_unit=retry_by_unit,
                event_by_unit=event_by_unit,
                resource_decision_by_unit=resource_decision_by_unit,
                model_decision_by_unit=model_decision_by_unit,
                audit=audit,
            )
        if artifact_execution_planning_result is not None or artifact_execution_batch is not None:
            raise WorkFanoutPolicyError("R5 execution evidence requires the parent artifact fanout join")

        completion_by_unit: dict[tuple[str, str, str, str], StageWorkCompletion] = {}
        for completion in stage_completions:
            dependency = graph_units.get(_ref_key(completion.work_unit_ref))
            if dependency is None:
                raise WorkFanoutPolicyError("stage completion belongs to an unknown work unit")
            _validate_stage_completion(dependency, completion)
            key = _ref_key(completion.work_unit_ref)
            if key in completion_by_unit:
                raise WorkFanoutPolicyError("stage completions must be unique by work unit")
            completion_by_unit[key] = completion

        succeeded: list[ObjectRef] = []
        retryable: list[ObjectRef] = []
        terminal: list[ObjectRef] = []
        waiting: list[ObjectRef] = []
        skipped: list[ObjectRef] = []
        dependency_refs = work_unit.depends_on_work_unit_refs
        if (
            semantic_review_fanout is not None
            and unit_ref == semantic_review_fanout.parent_item_quality_work_unit_ref
        ):
            dependency_refs = (
                *dependency_refs,
                *(resolved_work_unit_v2_ref(value.work_unit) for value in semantic_review_fanout.round_work),
            )
        for dependency_ref in dependency_refs:
            dependency = graph_units.get(_ref_key(dependency_ref))
            if dependency is None:
                raise WorkFanoutPolicyError("work dependency is not owned by the graph")
            observed_completion = completion_by_unit.get(_ref_key(dependency_ref))
            if observed_completion is not None:
                result_ref = stage_result_record_ref(observed_completion.stage_result)
                if observed_completion.stage_result.status is StageRunStatus.SUCCEEDED:
                    succeeded.append(result_ref)
                elif (
                    observed_completion.stage_result.failure is not None
                    and observed_completion.stage_result.failure.retryable
                ):
                    decision = retry_by_unit.get(_ref_key(dependency_ref))
                    if decision is None:
                        retryable.append(result_ref)
                    else:
                        _validate_retry_decision_binding(
                            dependency,
                            observed_completion,
                            decision,
                        )
                        decision_ref = work_retry_decision_v2_ref(decision)
                        if decision.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
                            retryable.append(decision_ref)
                        else:
                            terminal.append(decision_ref)
                else:
                    terminal.append(result_ref)
                continue
            control_event = event_by_unit.get(_ref_key(dependency_ref))
            if (
                control_event is not None
                and control_event.event_kind
                in {
                    WorkLeaseEventKindV2.TERMINAL_FAILURE,
                    WorkLeaseEventKindV2.EXPIRED,
                    WorkLeaseEventKindV2.CANCELLED,
                }
                and control_event.failure is not None
                and not control_event.failure.retryable
            ):
                terminal.append(work_lease_event_v2_ref(control_event))
                continue
            resource_decision = resource_decision_by_unit.get(_ref_key(dependency_ref))
            if resource_decision is not None:
                terminal.append(resource_admission_decision_v2_ref(resource_decision))
                continue
            model_decision = model_decision_by_unit.get(_ref_key(dependency_ref))
            if model_decision is not None:
                terminal.append(model_admission_decision_v2_ref(model_decision))
                continue
            dependency_item = items.get(dependency.item_id or "")
            if dependency_item is not None and dependency_item.status in _TERMINAL_ITEM_STATUSES:
                skipped.append(item_record_ref(dependency_item))
            else:
                waiting.append(dependency_ref)

        if retryable or waiting:
            readiness = WorkReadinessV2.WAITING
        elif work_unit.join_mode is WorkDependencyJoinModeV2.ALL_SUCCEEDED and (terminal or skipped):
            readiness = WorkReadinessV2.BLOCKED_DEPENDENCY
        else:
            readiness = WorkReadinessV2.READY
        return _snapshot(
            graph=graph,
            fanout=artifact_group_fanout,
            semantic_fanout=semantic_review_fanout,
            work_unit=work_unit,
            readiness=readiness,
            succeeded_dependency_result_refs=tuple(succeeded),
            retryable_dependency_result_refs=tuple(retryable),
            terminal_non_success_result_refs=tuple(terminal),
            waiting_dependency_work_unit_refs=tuple(waiting),
            skipped_item_refs=tuple(skipped),
            audit=audit,
        )


def _validate_stage_completion(
    unit: ResolvedWorkUnitV2,
    completion: StageWorkCompletion,
) -> None:
    run = completion.stage_run
    result = completion.stage_result
    unit_ref = resolved_work_unit_v2_ref(unit)
    if (
        completion.work_unit_ref != unit_ref
        or run.job_id != unit.job_id
        or run.item_id != unit.item_id
        or run.stage is not unit.stage
        or unit_ref not in run.input_refs
        or result.stage_run_id != run.stage_run_id
        or result.stage_run_version != run.row_version
        or result.status is not run.status
        or stage_result_record_carried_sha256(result) != result.result_sha256
    ):
        raise WorkFanoutPolicyError("StageRun/StageResult does not bind the exact work unit")


def _control_evidence_by_unit(
    graph_units: dict[
        tuple[str, str, str, str],
        ResolvedWorkUnitV2,
    ],
    retry_decisions: tuple[WorkRetryDecisionV2, ...],
    lease_events: tuple[WorkLeaseEventV2, ...],
) -> tuple[
    dict[tuple[str, str, str, str], WorkRetryDecisionV2],
    dict[tuple[str, str, str, str], WorkLeaseEventV2],
]:
    retry_by_unit: dict[
        tuple[str, str, str, str],
        WorkRetryDecisionV2,
    ] = {}
    for observed in retry_decisions:
        decision = WorkRetryDecisionV2.model_validate(observed.model_dump(mode="python"))
        key = _ref_key(decision.work_unit_ref)
        if key not in graph_units or key in retry_by_unit:
            raise WorkFanoutPolicyError("retry decision is duplicate or belongs to unknown work")
        retry_by_unit[key] = decision

    event_by_unit: dict[
        tuple[str, str, str, str],
        WorkLeaseEventV2,
    ] = {}
    for observed_event in lease_events:
        event = WorkLeaseEventV2.model_validate(observed_event.model_dump(mode="python"))
        key = _ref_key(event.work_unit_ref)
        if key not in graph_units or key in event_by_unit:
            raise WorkFanoutPolicyError("lease event is duplicate or belongs to unknown work")
        event_by_unit[key] = event
    for key, decision in retry_by_unit.items():
        supplied_event = event_by_unit.get(key)
        if supplied_event is not None and decision.prior_lease_event_ref != work_lease_event_v2_ref(
            supplied_event
        ):
            raise WorkFanoutPolicyError("retry decision does not bind the supplied lease event")
    return retry_by_unit, event_by_unit


def _resource_evidence_by_unit(
    graph_units: dict[
        tuple[str, str, str, str],
        ResolvedWorkUnitV2,
    ],
    decisions: tuple[ResourceAdmissionDecisionV2, ...],
    demands: tuple[WorkResourceDemandV2, ...],
) -> dict[tuple[str, str, str, str], ResourceAdmissionDecisionV2]:
    by_unit: dict[tuple[str, str, str, str], ResourceAdmissionDecisionV2] = {}
    demand_by_ref: dict[
        tuple[str, str, str, str],
        WorkResourceDemandV2,
    ] = {}
    for observed_demand in demands:
        canonical_demand = WorkResourceDemandV2.model_validate(observed_demand.model_dump(mode="python"))
        key = _ref_key(work_resource_demand_v2_ref(canonical_demand))
        if key in demand_by_ref:
            raise WorkFanoutPolicyError("resource demands must be unique")
        demand_by_ref[key] = canonical_demand
    for observed in decisions:
        decision = ResourceAdmissionDecisionV2.model_validate(observed.model_dump(mode="python"))
        if decision.outcome not in {
            ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED,
            ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
        }:
            raise WorkFanoutPolicyError("resource admission readiness evidence must be terminal")
        current_demand = demand_by_ref.get(_ref_key(decision.work_resource_demand_ref))
        if current_demand is None:
            raise WorkFanoutPolicyError("resource admission decision is missing demand evidence")
        key = _ref_key(current_demand.work_unit_ref)
        if key not in graph_units:
            raise WorkFanoutPolicyError("resource admission decision belongs to unknown work")
        if key in by_unit:
            raise WorkFanoutPolicyError("resource admission decisions must be unique by work unit")
        by_unit[key] = decision
    return by_unit


def _model_evidence_by_unit(
    graph_units: dict[
        tuple[str, str, str, str],
        ResolvedWorkUnitV2,
    ],
    decisions: tuple[ModelAdmissionDecisionV2, ...],
    demands: tuple[WorkModelDemandV2, ...],
) -> dict[tuple[str, str, str, str], ModelAdmissionDecisionV2]:
    by_unit: dict[tuple[str, str, str, str], ModelAdmissionDecisionV2] = {}
    demand_by_ref: dict[
        tuple[str, str, str, str],
        WorkModelDemandV2,
    ] = {}
    for observed_demand in demands:
        demand = WorkModelDemandV2.model_validate(observed_demand.model_dump(mode="python"))
        key = _ref_key(work_model_demand_v2_ref(demand))
        if key in demand_by_ref:
            raise WorkFanoutPolicyError("model demands must be unique")
        demand_by_ref[key] = demand
    for observed_decision in decisions:
        decision = ModelAdmissionDecisionV2.model_validate(observed_decision.model_dump(mode="python"))
        if decision.outcome not in {
            ModelAdmissionOutcomeV2.BUDGET_EXHAUSTED,
            ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
        }:
            raise WorkFanoutPolicyError("model admission readiness evidence must be terminal")
        current_demand = demand_by_ref.get(_ref_key(decision.work_model_demand_ref))
        if current_demand is None:
            raise WorkFanoutPolicyError("model admission decision is missing demand evidence")
        key = _ref_key(current_demand.work_unit_ref)
        if key not in graph_units:
            raise WorkFanoutPolicyError("model admission decision belongs to unknown work")
        if key in by_unit:
            raise WorkFanoutPolicyError("model admission decisions must be unique by work unit")
        by_unit[key] = decision
    return by_unit


def _validate_retry_decision_binding(
    unit: ResolvedWorkUnitV2,
    completion: StageWorkCompletion,
    decision: WorkRetryDecisionV2,
) -> None:
    result_ref = stage_result_record_ref(completion.stage_result)
    if (
        decision.work_unit_ref != resolved_work_unit_v2_ref(unit)
        or result_ref not in decision.prior_result_refs
        or decision.completed_attempt != completion.stage_run.attempt
    ):
        raise WorkFanoutPolicyError("retry decision does not bind the retryable StageResult")


def _snapshot(
    *,
    graph: ResolvedJobWorkGraphV2,
    fanout: ArtifactGroupFanoutV2 | None,
    semantic_fanout: SemanticReviewFanoutV2 | None = None,
    work_unit: ResolvedWorkUnitV2,
    readiness: WorkReadinessV2,
    succeeded_dependency_result_refs: tuple[ObjectRef, ...] = (),
    retryable_dependency_result_refs: tuple[ObjectRef, ...] = (),
    terminal_non_success_result_refs: tuple[ObjectRef, ...] = (),
    waiting_dependency_work_unit_refs: tuple[ObjectRef, ...] = (),
    skipped_item_refs: tuple[ObjectRef, ...] = (),
    audit: ContractAudit,
) -> WorkReadinessSnapshotV2:
    graph_ref = resolved_job_work_graph_v2_ref(graph)
    fanout_ref = artifact_group_fanout_v2_ref(fanout) if fanout else None
    semantic_fanout_ref = semantic_review_fanout_v2_ref(semantic_fanout) if semantic_fanout else None
    unit_ref = resolved_work_unit_v2_ref(work_unit)
    refs = (
        graph_ref,
        *((fanout_ref,) if fanout_ref else ()),
        *((semantic_fanout_ref,) if semantic_fanout_ref else ()),
        unit_ref,
        *succeeded_dependency_result_refs,
        *retryable_dependency_result_refs,
        *terminal_non_success_result_refs,
        *waiting_dependency_work_unit_refs,
        *skipped_item_refs,
    )
    value = WorkReadinessSnapshotV2(
        work_readiness_snapshot_id="work-readiness-snapshot://pending",
        resolved_job_work_graph_ref=graph_ref,
        artifact_group_fanout_ref=fanout_ref,
        semantic_review_fanout_ref=semantic_fanout_ref,
        work_unit_ref=unit_ref,
        readiness=readiness,
        succeeded_dependency_result_refs=_sorted_refs(succeeded_dependency_result_refs),
        retryable_dependency_result_refs=_sorted_refs(retryable_dependency_result_refs),
        terminal_non_success_result_refs=_sorted_refs(terminal_non_success_result_refs),
        waiting_dependency_work_unit_refs=_sorted_refs(waiting_dependency_work_unit_refs),
        skipped_item_refs=_sorted_refs(skipped_item_refs),
        work_readiness_snapshot_sha256="0" * 64,
        audit=_safe_audit(audit, refs),
    )
    digest = work_readiness_snapshot_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "work_readiness_snapshot_id": (f"work-readiness-snapshot://sha256/{digest}"),
            "work_readiness_snapshot_sha256": digest,
        }
    )


def _artifact_join_snapshot(
    *,
    graph: ResolvedJobWorkGraphV2,
    fanout: ArtifactGroupFanoutV2,
    parent_work_unit: ResolvedWorkUnitV2,
    artifact_execution_planning_result: (ArtifactExecutionPlanningResult | None),
    artifact_execution_batch: ArtifactExecutionBatchV2 | None,
    retry_by_unit: dict[
        tuple[str, str, str, str],
        WorkRetryDecisionV2,
    ],
    event_by_unit: dict[
        tuple[str, str, str, str],
        WorkLeaseEventV2,
    ],
    resource_decision_by_unit: dict[
        tuple[str, str, str, str],
        ResourceAdmissionDecisionV2,
    ],
    model_decision_by_unit: dict[
        tuple[str, str, str, str],
        ModelAdmissionDecisionV2,
    ],
    audit: ContractAudit,
) -> WorkReadinessSnapshotV2:
    if artifact_execution_planning_result is None and artifact_execution_batch is None:
        control_retryable: list[ObjectRef] = []
        control_terminal: list[ObjectRef] = []
        control_waiting: list[ObjectRef] = []
        for unit in fanout.group_work_units:
            unit_ref = resolved_work_unit_v2_ref(unit)
            key = _ref_key(unit_ref)
            decision = retry_by_unit.get(key)
            if decision is not None:
                decision_ref = work_retry_decision_v2_ref(decision)
                if decision.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
                    control_retryable.append(decision_ref)
                else:
                    control_terminal.append(decision_ref)
                continue
            event = event_by_unit.get(key)
            if (
                event is not None
                and event.event_kind
                in {
                    WorkLeaseEventKindV2.TERMINAL_FAILURE,
                    WorkLeaseEventKindV2.EXPIRED,
                    WorkLeaseEventKindV2.CANCELLED,
                }
                and event.failure is not None
                and not event.failure.retryable
            ):
                control_terminal.append(work_lease_event_v2_ref(event))
                continue
            resource_decision = resource_decision_by_unit.get(key)
            if resource_decision is not None:
                control_terminal.append(resource_admission_decision_v2_ref(resource_decision))
                continue
            model_decision = model_decision_by_unit.get(key)
            if model_decision is not None:
                control_terminal.append(model_admission_decision_v2_ref(model_decision))
                continue
            control_waiting.append(unit_ref)
        readiness = WorkReadinessV2.WAITING if control_retryable or control_waiting else WorkReadinessV2.READY
        return _snapshot(
            graph=graph,
            fanout=fanout,
            work_unit=parent_work_unit,
            readiness=readiness,
            retryable_dependency_result_refs=tuple(control_retryable),
            terminal_non_success_result_refs=tuple(control_terminal),
            waiting_dependency_work_unit_refs=tuple(control_waiting),
            audit=audit,
        )
    if artifact_execution_planning_result is None or artifact_execution_batch is None:
        raise WorkFanoutPolicyError("artifact group join requires both the R5 plan and batch")
    artifact_execution_plan = _current_artifact_execution_plan(artifact_execution_planning_result)
    if artifact_execution_plan_ref(artifact_execution_plan) != (fanout.artifact_execution_plan_ref):
        raise WorkFanoutPolicyError("artifact group join plan does not match the fanout")
    from eval_factory.attachment_planning.execution import ArtifactGroupExecutor
    from eval_factory.attachment_planning.execution_models import (
        ArtifactExecutionPolicyError,
    )

    try:
        ArtifactGroupExecutor().validate_current(
            artifact_execution_plan,
            artifact_execution_batch,
        )
    except ArtifactExecutionPolicyError as exc:
        raise WorkFanoutPolicyError("artifact execution batch is not current") from exc
    _validate_artifact_execution_batch_bindings(
        artifact_execution_plan,
        artifact_execution_batch,
    )

    succeeded: list[ObjectRef] = []
    retryable: list[ObjectRef] = []
    terminal: list[ObjectRef] = []
    batch_ref = artifact_execution_batch_ref(artifact_execution_batch)
    group_unit_by_ref = {
        unit.artifact_execution_group_ref: unit
        for unit in fanout.group_work_units
        if unit.artifact_execution_group_ref is not None
    }
    for receipt in artifact_execution_batch.receipts:
        receipt_ref = artifact_execution_receipt_ref(receipt)
        if receipt.outcome is ArtifactExecutionReceiptOutcomeV2.SUCCEEDED:
            succeeded.append(receipt_ref)
        elif receipt.outcome is ArtifactExecutionReceiptOutcomeV2.RETRYABLE_FAILURE:
            group_unit = group_unit_by_ref.get(receipt.artifact_execution_group_ref)
            if group_unit is None:
                raise WorkFanoutPolicyError("retryable artifact receipt has no fanout group")
            decision = retry_by_unit.get(_ref_key(resolved_work_unit_v2_ref(group_unit)))
            if decision is None:
                retryable.append(receipt_ref)
                continue
            if (
                receipt_ref not in decision.prior_result_refs
                or batch_ref not in decision.prior_result_refs
                or decision.completed_attempt != receipt.attempt
            ):
                raise WorkFanoutPolicyError("artifact retry decision does not bind the R5 receipt")
            decision_ref = work_retry_decision_v2_ref(decision)
            if decision.decision is WorkRetryDecisionKindV2.RETRY_SCHEDULED:
                if decision_ref not in retryable:
                    retryable.append(decision_ref)
            elif decision_ref not in terminal:
                terminal.append(decision_ref)
        else:
            terminal.append(receipt_ref)

    if retryable:
        retryable.append(batch_ref)
        readiness = WorkReadinessV2.WAITING
    elif terminal:
        terminal.append(batch_ref)
        readiness = WorkReadinessV2.READY
    else:
        succeeded.append(batch_ref)
        readiness = WorkReadinessV2.READY
    return _snapshot(
        graph=graph,
        fanout=fanout,
        work_unit=parent_work_unit,
        readiness=readiness,
        succeeded_dependency_result_refs=tuple(succeeded),
        retryable_dependency_result_refs=tuple(retryable),
        terminal_non_success_result_refs=tuple(terminal),
        audit=audit,
    )


def _validate_artifact_fanout_binding(
    graph: ResolvedJobWorkGraphV2,
    fanout: ArtifactGroupFanoutV2,
) -> None:
    if fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph):
        raise WorkFanoutPolicyError("artifact group fanout belongs to another work graph")
    graph_by_ref = {resolved_work_unit_v2_ref(unit): unit for unit in graph.work_units}
    parent = graph_by_ref.get(fanout.parent_attachment_work_unit_ref)
    if (
        parent is None
        or parent.scope is not WorkUnitScopeV2.ITEM
        or parent.stage is not StageNameV2.ATTACHMENT
    ):
        raise WorkFanoutPolicyError("artifact group fanout parent is not a graph attachment unit")
    for unit in fanout.group_work_units:
        if unit.job_id != graph.job_id or unit.item_id != parent.item_id:
            raise WorkFanoutPolicyError("artifact group fanout work unit crosses its parent scope")


def _validate_semantic_review_fanout_binding(
    graph: ResolvedJobWorkGraphV2,
    fanout: SemanticReviewFanoutV2,
    target: ResolvedWorkUnitV2,
) -> None:
    if fanout.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph):
        raise WorkFanoutPolicyError("semantic review fanout belongs to another work graph")
    graph_by_ref = {resolved_work_unit_v2_ref(unit): unit for unit in graph.work_units}
    parent = graph_by_ref.get(fanout.parent_item_quality_work_unit_ref)
    if (
        parent is None
        or parent.scope is not WorkUnitScopeV2.ITEM
        or parent.stage is not StageNameV2.ITEM_QUALITY
        or parent.semantic_review_round is not None
    ):
        raise WorkFanoutPolicyError("semantic review fanout parent is not a base ITEM_QUALITY unit")
    round_units = tuple(value.work_unit for value in fanout.round_work)
    if target != parent and target not in round_units:
        raise WorkFanoutPolicyError("semantic review readiness target is not owned by the fanout")
    SemanticReviewFanoutCompiler().validate_current(
        fanout,
        graph=graph,
        parent_item_quality_work_unit=parent,
    )


def _validate_artifact_execution_batch_bindings(
    plan: ArtifactExecutionPlanV2,
    batch: ArtifactExecutionBatchV2,
) -> None:
    expected_by_artifact = {
        unit.artifact_id: (
            artifact_execution_group_ref(group),
            artifact_execution_unit_ref(unit),
        )
        for group in plan.groups
        for unit in group.units
    }
    receipt_by_artifact = {receipt.artifact_id: receipt for receipt in batch.receipts}
    if set(receipt_by_artifact) != set(expected_by_artifact):
        raise WorkFanoutPolicyError("artifact execution batch does not cover the exact R5 plan")
    receipt_refs = {artifact_execution_receipt_ref(receipt) for receipt in batch.receipts}
    for artifact_id, receipt in receipt_by_artifact.items():
        group_ref, unit_ref = expected_by_artifact[artifact_id]
        if (
            artifact_execution_receipt_carried_sha256(receipt) != receipt.artifact_execution_receipt_sha256
            or receipt.artifact_execution_plan_ref != batch.artifact_execution_plan_ref
            or receipt.artifact_execution_group_ref != group_ref
            or receipt.artifact_execution_unit_ref != unit_ref
            or not set(receipt.failed_dependency_receipt_refs) <= receipt_refs
        ):
            raise WorkFanoutPolicyError("artifact execution receipt is stale or cross-plan")


def _current_artifact_execution_plan(
    result: ArtifactExecutionPlanningResult,
) -> ArtifactExecutionPlanV2:
    from eval_factory.attachment_planning.execution import (
        ArtifactExecutionPlanCompiler,
    )
    from eval_factory.attachment_planning.execution_models import (
        ArtifactExecutionPlanningOutcome,
        ArtifactExecutionPlanningResult,
        ArtifactExecutionPolicyError,
    )

    try:
        canonical = ArtifactExecutionPlanningResult.model_validate(result.model_dump(mode="python"))
        ArtifactExecutionPlanCompiler().validate_current(canonical)
    except (ArtifactExecutionPolicyError, ValueError) as exc:
        raise WorkFanoutPolicyError("artifact execution planning result is not current") from exc
    if canonical.outcome is not ArtifactExecutionPlanningOutcome.PLANNED or canonical.execution_plan is None:
        raise WorkFanoutPolicyError("artifact fanout requires a planned R5 execution result")
    if canonical.audit.input_refs != (artifact_execution_plan_ref(canonical.execution_plan),):
        raise WorkFanoutPolicyError("artifact execution planning result audit is stale")
    if not any(
        binding.component == "artifact-execution" and binding.version == "r5-06"
        for binding in canonical.audit.governing_versions
    ):
        raise WorkFanoutPolicyError("artifact execution planning result version is stale")
    return canonical.execution_plan


def _resolved_plan_ref(plan: ResolvedDatasetJobPlanV2) -> ObjectRef:
    from eval_factory.contracts.orchestration_v2 import (
        resolved_dataset_job_plan_v2_ref,
    )

    return resolved_dataset_job_plan_v2_ref(plan)


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    versions = tuple(
        sorted(
            (
                *(binding for binding in audit.governing_versions if binding.component != _POLICY_COMPONENT),
                VersionBinding(
                    component=_POLICY_COMPONENT,
                    version=R6_WORK_GRAPH_POLICY_VERSION,
                ),
            ),
            key=lambda binding: (
                binding.component,
                binding.version,
                binding.sha256 or "",
            ),
        )
    )
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=versions,
        input_refs=_sorted_refs(refs),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(refs, key=_ref_key))
