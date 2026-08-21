from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import cast

from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.agent_system.workspace import (
    AgentWorkspaceManager,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentDefinitionV2,
    AgentResultEnvelopeV2,
    AgentTaskOutcomeV2,
    AgentTaskStatusV2,
    AgentTaskV2,
    AgentWorkspaceStateV2,
    AttachmentGenerationPlanV2,
    CompiledAttachmentGenerationPlanV2,
    CompiledCriteriaRubricPlanV2,
    CompiledDatasetBuildPlanV2,
    CompiledGradingDesignPlanV2,
    CriteriaRubricPlanV2,
    GradingDesignPlanV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2


class ExecutionSupervisorError(RuntimeError):
    pass


class StaleAgentLeaseError(ExecutionSupervisorError):
    pass


class AgentWorkEventKindV2(StrEnum):
    ACQUIRED = "ACQUIRED"
    HEARTBEAT = "HEARTBEAT"
    EXPIRED = "EXPIRED"
    RETRY_DECIDED = "RETRY_DECIDED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class AgentWorkStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    EXHAUSTED = "EXHAUSTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class AgentRetryDecisionKindV2(StrEnum):
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    EXHAUSTED = "EXHAUSTED"


class AgentCancellationStateV2(StrEnum):
    REQUESTED = "REQUESTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class AgentWorkerRef:
    worker_id: str

    def __post_init__(self) -> None:
        if len(self.worker_id) < 3 or len(self.worker_id) > 256:
            raise ValueError("Agent worker ID length is invalid")


@dataclass(frozen=True, slots=True)
class AgentLeaseV2:
    agent_task_ref: ObjectRef
    agent_definition_ref: ObjectRef
    lease_id: str
    worker: AgentWorkerRef
    attempt: int
    fencing_token: int
    acquired_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AgentWorkEventV2:
    event_id: str
    agent_task_ref: ObjectRef
    lease_id: str
    event_version: int
    event_kind: AgentWorkEventKindV2
    state: AgentWorkStateV2
    fencing_token: int
    occurred_at: datetime
    effective_expires_at: datetime | None
    reason_code: str | None
    workspace_receipt_ref: ObjectRef | None


@dataclass(frozen=True, slots=True)
class AgentRetryDecisionV2:
    retry_decision_id: str
    agent_task_ref: ObjectRef
    prior_lease_id: str
    attempt: int
    decision: AgentRetryDecisionKindV2
    next_attempt: int | None
    eligible_at: datetime | None
    reason_code: str
    workspace_receipt_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class AgentCancellationV2:
    cancellation_id: str
    agent_task_ref: ObjectRef
    lease_id: str
    state: AgentCancellationStateV2
    reason_code: str
    requested_by: str
    predecessor_cancellation_id: str | None
    workspace_receipt_ref: ObjectRef | None


class ExecutionSupervisor:
    def __init__(
        self,
        store: FactoryControlStore,
        registry: AgentRegistry,
        *,
        workspace_manager: AgentWorkspaceManager,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.workspace_manager = workspace_manager
        self._clock = clock or (lambda: datetime.now(UTC))

    def materialize(
        self,
        *,
        run_id: str,
        compiled_plan: CompiledDatasetBuildPlanV2,
        audit: ContractAudit,
    ) -> tuple[AgentTaskV2, ...]:
        plan, stored_compiled = self.store.get_plan(run_id)
        if stored_compiled != compiled_plan or compiled_plan.source_plan_ref != plan.to_ref():
            raise ExecutionSupervisorError("Supervisor received a stale compiled plan")
        current = self.store.get_run(run_id)
        by_key: dict[str, AgentTaskV2] = {}
        for plan_task in compiled_plan.tasks:
            definition = self.registry.resolve(plan_task.agent_role, plan_task.task_kind)
            capability_refs = tuple(
                sorted(
                    (
                        capability.to_ref()
                        for capability in self.registry.capabilities_for(definition)
                        if capability.capability_id in plan_task.required_capability_ids
                    ),
                    key=_ref_key,
                )
            )
            dependency_refs = tuple(
                sorted(
                    (by_key[key].to_ref() for key in plan_task.dependency_task_keys),
                    key=_ref_key,
                )
            )
            input_refs = dependency_refs if dependency_refs else (current.requirement_spec_ref,)
            task = AgentTaskV2.create(
                agent_task_id=f"agent-task://{_slug(run_id)}/{plan_task.task_key}",
                run_ref=current.to_ref(),
                compiled_plan_ref=compiled_plan.to_ref(),
                plan_task_key=plan_task.task_key,
                task_kind=plan_task.task_kind,
                agent_definition_ref=definition.to_ref(),
                input_refs=input_refs,
                expected_output_object_types=plan_task.output_object_types,
                required_capability_refs=capability_refs,
                status=(
                    AgentTaskStatusV2.READY
                    if not plan_task.dependency_task_keys
                    else AgentTaskStatusV2.PENDING
                ),
                attempt=0,
                max_attempts=plan_task.max_attempts,
                audit=audit,
            )
            by_key[plan_task.task_key] = task
        tasks = tuple(by_key[key] for key in compiled_plan.topological_task_keys)
        self._persist_registry_and_tasks(run_id, tasks)
        return self.list_tasks(
            run_id,
            compiled_plan_ref=compiled_plan.to_ref(),
        )

    def materialize_attachment(
        self,
        *,
        run_id: str,
        compiled_plan: CompiledAttachmentGenerationPlanV2,
        audit: ContractAudit,
    ) -> tuple[AgentTaskV2, ...]:
        material = self.store.get_domain_plan(
            run_id,
            PlanKindV2.ATTACHMENT_GENERATION,
        )
        plan = AttachmentGenerationPlanV2.model_validate_json(material.plan_record_json)
        stored_compiled = CompiledAttachmentGenerationPlanV2.model_validate_json(
            material.compiled_plan_record_json
        )
        if stored_compiled != compiled_plan or compiled_plan.source_plan_ref != plan.to_ref():
            raise ExecutionSupervisorError("Supervisor received a stale attachment plan")
        current = self.store.get_run(run_id)
        works = {work.work_key: work for work in compiled_plan.works}
        existing = {
            task.plan_task_key: task
            for task in self.list_tasks(
                run_id,
                compiled_plan_ref=compiled_plan.to_ref(),
            )
        }
        if existing and set(existing) != set(works):
            raise ExecutionSupervisorError("persisted attachment task inventory is incomplete")
        assignments = {assignment.work_key: assignment for assignment in compiled_plan.assignments}
        by_key: dict[str, AgentTaskV2] = {}
        for work_key in compiled_plan.topological_work_keys:
            work = works[work_key]
            assignment = assignments[work_key]
            definition = self.registry.resolve(
                work.agent_role,
                "attachment-mock",
            )
            capabilities = self.registry.capabilities_for(definition)
            expected_capability_refs = tuple(
                sorted(
                    (
                        capability.to_ref()
                        for capability in capabilities
                        if capability.capability_id in work.required_capability_ids
                    ),
                    key=_ref_key,
                )
            )
            if (
                definition.to_ref() != assignment.agent_definition_ref
                or expected_capability_refs != assignment.capability_refs
            ):
                raise ExecutionSupervisorError("attachment assignment differs from Agent registry")
            dependency_refs = tuple(by_key[key].to_ref() for key in work.dependency_work_keys)
            input_refs = tuple(
                sorted(
                    (
                        plan.producer_task_view_ref,
                        plan.evidence_bundle_ref,
                        plan.attachment_planning_context_ref,
                        work.artifact_group_ref,
                        *dependency_refs,
                    ),
                    key=_ref_key,
                )
            )
            task = AgentTaskV2.create(
                agent_task_id=(f"agent-task://{_slug(run_id)}/attachment/{work.work_key}"),
                run_ref=(existing[work_key].run_ref if work_key in existing else current.to_ref()),
                compiled_plan_ref=compiled_plan.to_ref(),
                plan_task_key=work.work_key,
                task_kind="attachment-mock",
                agent_definition_ref=definition.to_ref(),
                input_refs=input_refs,
                expected_output_object_types=(work.output_object_types),
                required_capability_refs=(assignment.capability_refs),
                status=(
                    AgentTaskStatusV2.READY if not work.dependency_work_keys else AgentTaskStatusV2.PENDING
                ),
                attempt=0,
                max_attempts=work.max_attempts,
                audit=audit,
            )
            by_key[work.work_key] = task
        tasks = tuple(by_key[key] for key in compiled_plan.topological_work_keys)
        self._persist_registry_and_tasks(run_id, tasks)
        return self.list_tasks(
            run_id,
            compiled_plan_ref=compiled_plan.to_ref(),
        )

    def materialize_criteria_rubric(
        self,
        *,
        run_id: str,
        compiled_plan: CompiledCriteriaRubricPlanV2,
        audit: ContractAudit,
    ) -> AgentTaskV2:
        material = self.store.get_domain_plan(
            run_id,
            PlanKindV2.CRITERIA_RUBRIC,
        )
        plan = CriteriaRubricPlanV2.model_validate_json(material.plan_record_json)
        stored_compiled = CompiledCriteriaRubricPlanV2.model_validate_json(material.compiled_plan_record_json)
        if stored_compiled != compiled_plan or compiled_plan.source_plan_ref != plan.to_ref():
            raise ExecutionSupervisorError("Supervisor received a stale criteria/rubric plan")
        definition = self.registry.resolve(
            plan.agent_role,
            "criteria-rubric",
        )
        capabilities = self.registry.capabilities_for(definition)
        expected_capability_refs = tuple(
            sorted(
                (
                    capability.to_ref()
                    for capability in capabilities
                    if capability.capability_id in plan.required_capability_ids
                ),
                key=_ref_key,
            )
        )
        if (
            definition.to_ref() != compiled_plan.agent_definition_ref
            or expected_capability_refs != compiled_plan.capability_refs
        ):
            raise ExecutionSupervisorError("criteria/rubric assignment differs from Agent registry")
        current = self.store.get_run(run_id)
        task = AgentTaskV2.create(
            agent_task_id=(f"agent-task://{_slug(run_id)}/criteria-rubric"),
            run_ref=current.to_ref(),
            compiled_plan_ref=compiled_plan.to_ref(),
            plan_task_key="criteria-rubric",
            task_kind="criteria-rubric",
            agent_definition_ref=definition.to_ref(),
            input_refs=tuple(
                sorted(
                    (
                        plan.task_draft_ref,
                        plan.attachment_quality_ref,
                        plan.solvability_ref,
                        plan.tool_catalog_ref,
                    ),
                    key=_ref_key,
                )
            ),
            expected_output_object_types=("criteria-rubric-result",),
            required_capability_refs=(compiled_plan.capability_refs),
            status=AgentTaskStatusV2.READY,
            attempt=0,
            max_attempts=plan.max_attempts,
            audit=audit,
        )
        self._persist_registry_and_tasks(run_id, (task,))
        tasks = self.list_tasks(
            run_id,
            compiled_plan_ref=compiled_plan.to_ref(),
        )
        if len(tasks) != 1 or tasks[0] != task:
            raise ExecutionSupervisorError("criteria/rubric task materialization drifted")
        return task

    def materialize_grading_design(
        self,
        *,
        run_id: str,
        compiled_plan: CompiledGradingDesignPlanV2,
        audit: ContractAudit,
    ) -> AgentTaskV2:
        material = self.store.get_domain_plan(
            run_id,
            PlanKindV2.GRADING_DESIGN,
        )
        plan = GradingDesignPlanV2.model_validate_json(material.plan_record_json)
        stored_compiled = CompiledGradingDesignPlanV2.model_validate_json(material.compiled_plan_record_json)
        if stored_compiled != compiled_plan or compiled_plan.source_plan_ref != plan.to_ref():
            raise ExecutionSupervisorError("Supervisor received a stale grading design plan")
        definition = self.registry.resolve(
            plan.agent_role,
            "grading-design",
        )
        expected_capability_refs = tuple(
            sorted(
                (
                    capability.to_ref()
                    for capability in self.registry.capabilities_for(definition)
                    if capability.capability_id in plan.required_capability_ids
                ),
                key=_ref_key,
            )
        )
        if (
            definition.to_ref() != compiled_plan.agent_definition_ref
            or expected_capability_refs != compiled_plan.capability_refs
        ):
            raise ExecutionSupervisorError("grading design assignment differs from Agent registry")
        current = self.store.get_run(run_id)
        task = AgentTaskV2.create(
            agent_task_id=(f"agent-task://{_slug(run_id)}/grading-design"),
            run_ref=current.to_ref(),
            compiled_plan_ref=compiled_plan.to_ref(),
            plan_task_key="grading-design",
            task_kind="grading-design",
            agent_definition_ref=definition.to_ref(),
            input_refs=tuple(
                sorted(
                    (
                        plan.criteria_rubric_result_ref,
                        plan.rubric_set_ref,
                        plan.evaluator_spec_ref,
                        plan.reference_policy_ref,
                        plan.tool_policy_ref,
                    ),
                    key=_ref_key,
                )
            ),
            expected_output_object_types=("grading-design-result",),
            required_capability_refs=(compiled_plan.capability_refs),
            status=AgentTaskStatusV2.READY,
            attempt=0,
            max_attempts=plan.max_attempts,
            audit=audit,
        )
        self._persist_registry_and_tasks(run_id, (task,))
        tasks = self.list_tasks(
            run_id,
            compiled_plan_ref=compiled_plan.to_ref(),
        )
        if len(tasks) != 1 or tasks[0] != task:
            raise ExecutionSupervisorError("grading design task materialization drifted")
        return task

    def list_tasks(
        self,
        run_id: str,
        *,
        compiled_plan_ref: ObjectRef | None = None,
    ) -> tuple[AgentTaskV2, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT record_json FROM agent_tasks WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            values = tuple(AgentTaskV2.model_validate_json(str(row["record_json"])) for row in rows)
            if compiled_plan_ref is not None:
                values = tuple(value for value in values if value.compiled_plan_ref == compiled_plan_ref)
            return tuple(sorted(values, key=lambda value: value.plan_task_key))
        finally:
            connection.close()

    def get_result(
        self,
        task_id: str,
    ) -> AgentResultEnvelopeV2:
        result = self.find_result(task_id)
        if result is None:
            raise ExecutionSupervisorError("Agent task result does not exist")
        return result

    def find_result(
        self,
        task_id: str,
    ) -> AgentResultEnvelopeV2 | None:
        connection = self._connect()
        try:
            task = self._load_task(connection, task_id)
            row = connection.execute(
                """
                SELECT record_json FROM agent_results
                WHERE agent_task_id = ?
                ORDER BY attempt DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return None
            result = AgentResultEnvelopeV2.model_validate_json(str(row["record_json"]))
            if result.task_ref != task.to_ref():
                raise ExecutionSupervisorError("Agent result belongs to another task")
            return result
        finally:
            connection.close()

    def get_lease(
        self,
        lease_id: str,
    ) -> AgentLeaseV2:
        connection = self._connect()
        try:
            lease = self._load_lease(connection, lease_id)
            task = self._load_task(
                connection,
                lease.agent_task_ref.object_id,
            )
            if lease.agent_task_ref != task.to_ref():
                raise ExecutionSupervisorError("Agent lease belongs to another task")
            return lease
        finally:
            connection.close()

    def dispatch_ready(
        self,
        run_id: str,
        *,
        compiled_plan_ref: ObjectRef | None = None,
    ) -> tuple[AgentTaskV2, ...]:
        all_tasks = self.list_tasks(run_id)
        tasks = tuple(
            task
            for task in all_tasks
            if compiled_plan_ref is None or task.compiled_plan_ref == compiled_plan_ref
        )
        tasks_by_ref = {task.to_ref(): task for task in all_tasks}
        connection = self._connect()
        try:
            ready: list[AgentTaskV2] = []
            for task in tasks:
                if self._has_result(connection, task.agent_task_id):
                    continue
                head = connection.execute(
                    "SELECT state FROM agent_work_current_heads WHERE agent_task_id = ?",
                    (task.agent_task_id,),
                ).fetchone()
                if head is not None:
                    state = AgentWorkStateV2(str(head["state"]))
                    if state in {
                        AgentWorkStateV2.ACTIVE,
                        AgentWorkStateV2.CANCEL_REQUESTED,
                        AgentWorkStateV2.CANCELLED,
                        AgentWorkStateV2.EXHAUSTED,
                    }:
                        continue
                    if state is AgentWorkStateV2.RETRY_SCHEDULED:
                        retry = self._load_retry_decision(
                            connection,
                            task.agent_task_id,
                        )
                        if retry.eligible_at is None or self._clock() < retry.eligible_at:
                            continue
                dependency_refs = tuple(
                    reference for reference in task.input_refs if reference.object_type == "agent-task"
                )
                try:
                    dependencies = tuple(tasks_by_ref[reference] for reference in dependency_refs)
                except KeyError as exc:
                    raise ExecutionSupervisorError("Agent task dependency is not persisted") from exc
                if all(
                    self._result_succeeded(
                        connection,
                        dependency.agent_task_id,
                    )
                    for dependency in dependencies
                ):
                    ready.append(task)
            return tuple(ready)
        finally:
            connection.close()

    def acquire(
        self,
        task_id: str,
        worker: AgentWorkerRef,
        *,
        lease_duration: timedelta,
        idempotency_key: str | None = None,
    ) -> AgentLeaseV2:
        if lease_duration <= timedelta(0):
            raise ValueError("Agent lease duration must be positive")
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            task = self._load_task(connection, task_id)
            request_sha256 = _request_sha256(
                task.to_ref(),
                worker.worker_id,
                lease_duration.total_seconds(),
            )
            scope = f"acquire-agent-lease:{task.agent_task_id}"
            if idempotency_key is not None:
                replay = self._idempotent_response(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_type="agent-lease",
                )
                if replay is not None:
                    lease = self._load_lease(connection, replay)
                    connection.rollback()
                    return lease
            if self._has_result(connection, task_id):
                raise StaleAgentLeaseError("Agent task already has a terminal result")
            head = connection.execute(
                "SELECT * FROM agent_work_current_heads WHERE agent_task_id = ?",
                (task_id,),
            ).fetchone()
            if head is not None:
                state = AgentWorkStateV2(str(head["state"]))
                if state in {
                    AgentWorkStateV2.ACTIVE,
                    AgentWorkStateV2.CANCEL_REQUESTED,
                }:
                    raise StaleAgentLeaseError("Agent task already has an active lease")
                if state is AgentWorkStateV2.RETRY_SCHEDULED:
                    retry = self._load_retry_decision(
                        connection,
                        task.agent_task_id,
                    )
                    if (
                        retry.decision is not AgentRetryDecisionKindV2.RETRY_SCHEDULED
                        or retry.eligible_at is None
                        or now < retry.eligible_at
                    ):
                        raise StaleAgentLeaseError("Agent task retry is not eligible")
                elif state in {
                    AgentWorkStateV2.CANCELLED,
                    AgentWorkStateV2.COMPLETED,
                    AgentWorkStateV2.EXHAUSTED,
                }:
                    raise StaleAgentLeaseError("Agent task is terminal")
            row = connection.execute(
                "SELECT COALESCE(MAX(attempt), 0), COALESCE(MAX(fencing_token), 0) "
                "FROM agent_work_leases WHERE agent_task_id = ?",
                (task_id,),
            ).fetchone()
            assert row is not None
            attempt = int(row[0]) + 1
            fence = int(row[1]) + 1
            if attempt > task.max_attempts:
                raise StaleAgentLeaseError("Agent task attempt budget is exhausted")
            definition = self._load_definition(connection, task.agent_definition_ref)
            seed = f"{task.object_id}|{attempt}|{fence}|{worker.worker_id}"
            lease = AgentLeaseV2(
                agent_task_ref=task.to_ref(),
                agent_definition_ref=definition.to_ref(),
                lease_id=f"agent-lease://sha256/{hashlib.sha256(seed.encode()).hexdigest()}",
                worker=worker,
                attempt=attempt,
                fencing_token=fence,
                acquired_at=now,
                expires_at=now + lease_duration,
            )
            workspace = self.workspace_manager.open(
                run_ref=task.run_ref,
                lease=lease,
                audit=task.audit,
            )
            record_json = _lease_json(lease)
            connection.execute(
                """
                INSERT INTO agent_attempts (
                    agent_task_id, attempt, lease_id, fencing_token,
                    state, record_json
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?)
                """,
                (task_id, attempt, lease.lease_id, fence, record_json),
            )
            connection.execute(
                """
                INSERT INTO agent_work_leases (
                    lease_id, agent_task_id, attempt, fencing_token, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (lease.lease_id, task_id, attempt, fence, record_json),
            )
            event = self._event(
                lease=lease,
                event_version=1,
                event_kind=AgentWorkEventKindV2.ACQUIRED,
                state=AgentWorkStateV2.ACTIVE,
                effective_expires_at=lease.expires_at,
                occurred_at=now,
                reason_code=None,
                workspace_receipt_ref=workspace.receipt.to_ref(),
            )
            self._insert_event(connection, event)
            connection.execute(
                """
                INSERT INTO agent_work_current_heads (
                    agent_task_id, lease_id, event_version, state, fencing_token
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(agent_task_id) DO UPDATE SET
                    lease_id = excluded.lease_id,
                    event_version = excluded.event_version,
                    state = excluded.state,
                    fencing_token = excluded.fencing_token
                """,
                (
                    task.agent_task_id,
                    lease.lease_id,
                    AgentWorkStateV2.ACTIVE.value,
                    fence,
                ),
            )
            if idempotency_key is not None:
                self._record_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_type="agent-lease",
                    response_id=lease.lease_id,
                )
            connection.commit()
            return lease
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent lease authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def heartbeat(
        self,
        *,
        lease: AgentLeaseV2,
        worker: AgentWorkerRef,
        expected_event_version: int,
        extension: timedelta,
        idempotency_key: str,
    ) -> AgentWorkEventV2:
        if extension <= timedelta(0):
            raise ValueError("Agent heartbeat extension must be positive")
        request_sha256 = _request_sha256(
            lease.lease_id,
            worker.worker_id,
            expected_event_version,
            extension.total_seconds(),
        )
        scope = f"heartbeat-agent-lease:{lease.lease_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-work-event",
            )
            if replay is not None:
                event = self._load_event(connection, replay)
                connection.rollback()
                return event
            stored = self._load_lease(connection, lease.lease_id)
            if stored != lease or stored.worker != worker:
                raise StaleAgentLeaseError("Agent heartbeat lease or worker is stale")
            head = self._require_head(
                connection,
                lease,
                state=AgentWorkStateV2.ACTIVE,
                expected_event_version=expected_event_version,
            )
            now = self._clock()
            current_expiry = self._effective_expiry(
                connection,
                lease,
            )
            if now >= current_expiry:
                raise StaleAgentLeaseError("Agent heartbeat uses an expired lease")
            effective_expiry = max(
                current_expiry,
                now + extension,
            )
            event = self._event(
                lease=lease,
                event_version=expected_event_version + 1,
                event_kind=AgentWorkEventKindV2.HEARTBEAT,
                state=AgentWorkStateV2.ACTIVE,
                effective_expires_at=effective_expiry,
                occurred_at=now,
                reason_code=None,
                workspace_receipt_ref=(
                    self._active_workspace_ref(
                        connection,
                        lease,
                    )
                ),
            )
            self._insert_event(connection, event)
            self._update_head(
                connection,
                lease=lease,
                prior_event_version=int(head["event_version"]),
                event=event,
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-work-event",
                response_id=event.event_id,
            )
            connection.commit()
            return event
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent heartbeat authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def expire(
        self,
        *,
        lease: AgentLeaseV2,
        retry_delay: timedelta,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> AgentRetryDecisionV2:
        if retry_delay < timedelta(0):
            raise ValueError("Agent retry delay cannot be negative")
        request_sha256 = _request_sha256(
            lease.lease_id,
            retry_delay.total_seconds(),
        )
        scope = f"expire-agent-lease:{lease.lease_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-retry-decision",
            )
            if replay is not None:
                decision = self._load_retry_decision_by_id(
                    connection,
                    replay,
                )
                connection.rollback()
                return decision
            stored = self._load_lease(connection, lease.lease_id)
            if stored != lease:
                raise StaleAgentLeaseError("Agent expiry lease is stale")
            task = self._load_task(
                connection,
                lease.agent_task_ref.object_id,
            )
            head = self._require_head(
                connection,
                lease,
                state=AgentWorkStateV2.ACTIVE,
                expected_event_version=None,
            )
            now = self._clock()
            if now < self._effective_expiry(connection, lease):
                raise ExecutionSupervisorError("Agent lease is not expired")
            quarantined = self.workspace_manager.quarantine_lease(
                run_ref=task.run_ref,
                lease=lease,
                reason_code="LEASE_EXPIRED",
                audit=audit,
            )
            expired = self._event(
                lease=lease,
                event_version=int(head["event_version"]) + 1,
                event_kind=AgentWorkEventKindV2.EXPIRED,
                state=AgentWorkStateV2.EXPIRED,
                effective_expires_at=None,
                occurred_at=now,
                reason_code="LEASE_EXPIRED",
                workspace_receipt_ref=quarantined.to_ref(),
            )
            self._insert_event(connection, expired)
            retryable = lease.attempt < task.max_attempts
            decision_kind = (
                AgentRetryDecisionKindV2.RETRY_SCHEDULED if retryable else AgentRetryDecisionKindV2.EXHAUSTED
            )
            decision = AgentRetryDecisionV2(
                retry_decision_id=(f"agent-retry-decision://{_slug(task.agent_task_id)}/{lease.attempt}"),
                agent_task_ref=task.to_ref(),
                prior_lease_id=lease.lease_id,
                attempt=lease.attempt,
                decision=decision_kind,
                next_attempt=(lease.attempt + 1 if retryable else None),
                eligible_at=(now + retry_delay if retryable else None),
                reason_code="LEASE_EXPIRED",
                workspace_receipt_ref=quarantined.to_ref(),
            )
            connection.execute(
                """
                INSERT INTO agent_retry_decisions (
                    retry_decision_id, agent_task_id,
                    attempt, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    decision.retry_decision_id,
                    task.agent_task_id,
                    lease.attempt,
                    _retry_json(decision),
                ),
            )
            final_state = AgentWorkStateV2.RETRY_SCHEDULED if retryable else AgentWorkStateV2.EXHAUSTED
            decided = self._event(
                lease=lease,
                event_version=expired.event_version + 1,
                event_kind=(AgentWorkEventKindV2.RETRY_DECIDED),
                state=final_state,
                effective_expires_at=None,
                occurred_at=now,
                reason_code=decision_kind.value,
                workspace_receipt_ref=quarantined.to_ref(),
            )
            self._insert_event(connection, decided)
            self._update_head(
                connection,
                lease=lease,
                prior_event_version=int(head["event_version"]),
                event=decided,
            )
            connection.execute(
                """
                UPDATE agent_attempts
                SET state = ?
                WHERE agent_task_id = ? AND attempt = ?
                """,
                (
                    final_state.value,
                    task.agent_task_id,
                    lease.attempt,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-retry-decision",
                response_id=decision.retry_decision_id,
            )
            connection.commit()
            return decision
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent expiry authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def request_cancel(
        self,
        *,
        task_id: str,
        requested_by: str,
        reason_code: str,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> AgentCancellationV2:
        if len(requested_by) < 3:
            raise ValueError("Agent cancellation principal is invalid")
        request_sha256 = _request_sha256(
            task_id,
            requested_by,
            reason_code,
        )
        scope = f"request-agent-cancel:{task_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-cancellation",
            )
            if replay is not None:
                cancellation = self._load_cancellation(
                    connection,
                    replay,
                )
                connection.rollback()
                return cancellation
            task = self._load_task(connection, task_id)
            head = connection.execute(
                """
                SELECT * FROM agent_work_current_heads
                WHERE agent_task_id = ?
                """,
                (task.agent_task_id,),
            ).fetchone()
            if head is None or head["state"] != AgentWorkStateV2.ACTIVE.value:
                raise StaleAgentLeaseError("Agent cancellation requires active work")
            lease = self._load_lease(
                connection,
                str(head["lease_id"]),
            )
            event = self._event(
                lease=lease,
                event_version=int(head["event_version"]) + 1,
                event_kind=(AgentWorkEventKindV2.CANCEL_REQUESTED),
                state=AgentWorkStateV2.CANCEL_REQUESTED,
                effective_expires_at=None,
                occurred_at=self._clock(),
                reason_code=reason_code,
                workspace_receipt_ref=(
                    self._active_workspace_ref(
                        connection,
                        lease,
                    )
                ),
            )
            self._insert_event(connection, event)
            self._update_head(
                connection,
                lease=lease,
                prior_event_version=int(head["event_version"]),
                event=event,
            )
            cancellation = AgentCancellationV2(
                cancellation_id=(f"agent-cancellation://{_slug(lease.lease_id)}/requested"),
                agent_task_ref=task.to_ref(),
                lease_id=lease.lease_id,
                state=AgentCancellationStateV2.REQUESTED,
                reason_code=reason_code,
                requested_by=requested_by,
                predecessor_cancellation_id=None,
                workspace_receipt_ref=None,
            )
            self._insert_cancellation(
                connection,
                cancellation,
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-cancellation",
                response_id=cancellation.cancellation_id,
            )
            connection.commit()
            return cancellation
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent cancellation request raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def acknowledge_cancel(
        self,
        *,
        lease: AgentLeaseV2,
        worker: AgentWorkerRef,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> AgentCancellationV2:
        request_sha256 = _request_sha256(
            lease.lease_id,
            worker.worker_id,
        )
        scope = f"acknowledge-agent-cancel:{lease.lease_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-cancellation",
            )
            if replay is not None:
                cancellation = self._load_cancellation(
                    connection,
                    replay,
                )
                connection.rollback()
                return cancellation
            stored = self._load_lease(connection, lease.lease_id)
            if stored != lease or stored.worker != worker:
                raise StaleAgentLeaseError("Agent cancellation acknowledgement is stale")
            head = self._require_head(
                connection,
                lease,
                state=AgentWorkStateV2.CANCEL_REQUESTED,
                expected_event_version=None,
            )
            requested = self._latest_cancellation(
                connection,
                lease.lease_id,
                AgentCancellationStateV2.REQUESTED,
            )
            task = self._load_task(
                connection,
                lease.agent_task_ref.object_id,
            )
            quarantined = self.workspace_manager.quarantine_lease(
                run_ref=task.run_ref,
                lease=lease,
                reason_code="CANCELLED",
                audit=audit,
            )
            event = self._event(
                lease=lease,
                event_version=int(head["event_version"]) + 1,
                event_kind=AgentWorkEventKindV2.CANCELLED,
                state=AgentWorkStateV2.CANCELLED,
                effective_expires_at=None,
                occurred_at=self._clock(),
                reason_code=requested.reason_code,
                workspace_receipt_ref=quarantined.to_ref(),
            )
            self._insert_event(connection, event)
            self._update_head(
                connection,
                lease=lease,
                prior_event_version=int(head["event_version"]),
                event=event,
            )
            cancelled = AgentCancellationV2(
                cancellation_id=(f"agent-cancellation://{_slug(lease.lease_id)}/cancelled"),
                agent_task_ref=task.to_ref(),
                lease_id=lease.lease_id,
                state=AgentCancellationStateV2.CANCELLED,
                reason_code=requested.reason_code,
                requested_by=requested.requested_by,
                predecessor_cancellation_id=(requested.cancellation_id),
                workspace_receipt_ref=quarantined.to_ref(),
            )
            self._insert_cancellation(connection, cancelled)
            connection.execute(
                """
                UPDATE agent_attempts SET state = ?
                WHERE agent_task_id = ? AND attempt = ?
                """,
                (
                    AgentWorkStateV2.CANCELLED.value,
                    task.agent_task_id,
                    lease.attempt,
                ),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="agent-cancellation",
                response_id=cancelled.cancellation_id,
            )
            connection.commit()
            return cancelled
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent cancellation acknowledgement raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete(
        self,
        *,
        lease: AgentLeaseV2,
        envelope: AgentResultEnvelopeV2,
    ) -> AgentResultEnvelopeV2:
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            task = self._load_task(connection, lease.agent_task_ref.object_id)
            prior = connection.execute(
                "SELECT record_json FROM agent_results WHERE agent_task_id = ? AND attempt = ?",
                (task.agent_task_id, lease.attempt),
            ).fetchone()
            if prior is not None:
                stored = AgentResultEnvelopeV2.model_validate_json(str(prior["record_json"]))
                if stored != envelope:
                    raise StaleAgentLeaseError("Agent attempt already has a different result")
                connection.rollback()
                return stored
            head = connection.execute(
                "SELECT * FROM agent_work_current_heads WHERE agent_task_id = ?",
                (task.agent_task_id,),
            ).fetchone()
            if (
                head is None
                or head["state"] != AgentWorkStateV2.ACTIVE.value
                or head["lease_id"] != lease.lease_id
                or head["fencing_token"] != lease.fencing_token
            ):
                raise StaleAgentLeaseError("Agent result uses a stale or expired lease")
            effective_expires_at = self._effective_expiry(
                connection,
                lease,
            )
            if now >= effective_expires_at:
                raise StaleAgentLeaseError("Agent result uses a stale or expired lease")
            if (
                envelope.task_ref != task.to_ref()
                or envelope.agent_definition_ref != lease.agent_definition_ref
                or envelope.lease_id != lease.lease_id
                or envelope.attempt != lease.attempt
                or envelope.fencing_token != lease.fencing_token
            ):
                raise StaleAgentLeaseError("Agent result envelope does not bind the lease")
            definition = self._load_definition(
                connection,
                task.agent_definition_ref,
            )
            workspace = self.workspace_manager.resolve(envelope.workspace_receipt_ref)
            if (
                workspace.state is not AgentWorkspaceStateV2.COMMITTED
                or workspace.run_ref != task.run_ref
                or workspace.task_ref != task.to_ref()
                or workspace.agent_definition_ref != definition.to_ref()
                or workspace.attempt != lease.attempt
                or workspace.fencing_token != lease.fencing_token
            ):
                raise ExecutionSupervisorError("Agent workspace receipt does not bind the lease")
            if envelope.outcome is AgentTaskOutcomeV2.SUCCEEDED and any(
                ref.object_type not in task.expected_output_object_types for ref in envelope.output_refs
            ):
                raise ExecutionSupervisorError("Agent output type violates task contract")
            if (
                definition.max_model_requests > 0
                and not envelope.gateway_receipt_refs
                and envelope.outcome is not AgentTaskOutcomeV2.BLOCKED
            ):
                raise ExecutionSupervisorError("Agent result lacks required Gateway receipts")
            if not envelope.validator_result_refs:
                raise ExecutionSupervisorError("Agent result lacks validator closure")
            connection.execute(
                """
                INSERT INTO agent_results (
                    result_id, agent_task_id, attempt,
                    object_id, object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.result_id,
                    task.agent_task_id,
                    lease.attempt,
                    envelope.object_id,
                    envelope.object_sha256,
                    envelope.canonical_json().decode(),
                ),
            )
            event_version = int(head["event_version"]) + 1
            event = self._event(
                lease=lease,
                event_version=event_version,
                event_kind=AgentWorkEventKindV2.COMPLETED,
                state=AgentWorkStateV2.COMPLETED,
                effective_expires_at=None,
                occurred_at=now,
                reason_code=None,
                workspace_receipt_ref=workspace.to_ref(),
            )
            self._insert_event(connection, event)
            connection.execute(
                """
                UPDATE agent_attempts SET state = ?
                WHERE agent_task_id = ? AND attempt = ?
                """,
                (
                    AgentWorkStateV2.COMPLETED.value,
                    task.agent_task_id,
                    lease.attempt,
                ),
            )
            changed = connection.execute(
                """
                UPDATE agent_work_current_heads
                SET event_version = ?, state = ?
                WHERE agent_task_id = ? AND lease_id = ?
                  AND fencing_token = ? AND event_version = ?
                """,
                (
                    event_version,
                    AgentWorkStateV2.COMPLETED.value,
                    task.agent_task_id,
                    lease.lease_id,
                    lease.fencing_token,
                    head["event_version"],
                ),
            ).rowcount
            if changed != 1:
                raise StaleAgentLeaseError("Agent result head changed during completion")
            self.store._insert_outbox(
                connection,
                aggregate_type="AGENT_TASK",
                aggregate_id=task.agent_task_id,
                aggregate_version=event_version,
                event_type="agent-result-completed",
                object_id=envelope.object_id,
                created_at=now.isoformat(),
            )
            connection.commit()
            return envelope
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise StaleAgentLeaseError("Agent result authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _persist_registry_and_tasks(
        self,
        run_id: str,
        tasks: tuple[AgentTaskV2, ...],
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for capability in self.registry.capabilities:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO agent_capabilities (
                        capability_id, object_id, object_sha256, record_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        capability.capability_id,
                        capability.object_id,
                        capability.object_sha256,
                        capability.canonical_json().decode(),
                    ),
                )
            for definition in self.registry.definitions:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO agent_definitions (
                        agent_definition_id, agent_role, agent_version,
                        object_id, object_sha256, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        definition.agent_definition_id,
                        definition.agent_role,
                        definition.agent_version,
                        definition.object_id,
                        definition.object_sha256,
                        definition.canonical_json().decode(),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO agent_registry_heads (
                        agent_role, agent_definition_id, object_id
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        definition.agent_role,
                        definition.agent_definition_id,
                        definition.object_id,
                    ),
                )
            for task in tasks:
                row = connection.execute(
                    "SELECT record_json FROM agent_tasks WHERE agent_task_id = ?",
                    (task.agent_task_id,),
                ).fetchone()
                if row is not None:
                    stored = AgentTaskV2.model_validate_json(str(row["record_json"]))
                    if stored != task:
                        raise ExecutionSupervisorError("Agent task authority conflicts")
                    continue
                connection.execute(
                    """
                    INSERT INTO agent_tasks (
                        agent_task_id, run_id, status, attempt,
                        object_id, object_sha256, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.agent_task_id,
                        run_id,
                        task.status.value,
                        task.attempt,
                        task.object_id,
                        task.object_sha256,
                        task.canonical_json().decode(),
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.store.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @staticmethod
    def _load_task(connection: sqlite3.Connection, task_id: str) -> AgentTaskV2:
        row = connection.execute(
            "SELECT record_json FROM agent_tasks WHERE agent_task_id = ? OR object_id = ?",
            (task_id, task_id),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent task does not exist")
        return AgentTaskV2.model_validate_json(str(row["record_json"]))

    @staticmethod
    def _load_definition(
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> AgentDefinitionV2:
        row = connection.execute(
            "SELECT record_json FROM agent_definitions WHERE object_id = ?",
            (reference.object_id,),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent definition does not exist")
        value = AgentDefinitionV2.model_validate_json(str(row["record_json"]))
        if value.to_ref() != reference:
            raise ExecutionSupervisorError("Agent definition ref drifted")
        return value

    @staticmethod
    def _has_result(connection: sqlite3.Connection, task_id: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM agent_results WHERE agent_task_id = ? LIMIT 1",
                (task_id,),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _result_succeeded(connection: sqlite3.Connection, task_id: str) -> bool:
        row = connection.execute(
            "SELECT record_json FROM agent_results WHERE agent_task_id = ? ORDER BY attempt DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        if row is None:
            return False
        result = AgentResultEnvelopeV2.model_validate_json(str(row["record_json"]))
        return result.outcome is AgentTaskOutcomeV2.SUCCEEDED

    @staticmethod
    def _event(
        *,
        lease: AgentLeaseV2,
        event_version: int,
        event_kind: AgentWorkEventKindV2,
        state: AgentWorkStateV2,
        effective_expires_at: datetime | None,
        occurred_at: datetime,
        reason_code: str | None,
        workspace_receipt_ref: ObjectRef | None,
    ) -> AgentWorkEventV2:
        return AgentWorkEventV2(
            event_id=(f"agent-work-event://{_slug(lease.lease_id)}/{event_version}"),
            agent_task_ref=lease.agent_task_ref,
            lease_id=lease.lease_id,
            event_version=event_version,
            event_kind=event_kind,
            state=state,
            fencing_token=lease.fencing_token,
            occurred_at=occurred_at,
            effective_expires_at=effective_expires_at,
            reason_code=reason_code,
            workspace_receipt_ref=workspace_receipt_ref,
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        event: AgentWorkEventV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO agent_work_events (
                event_id, lease_id, event_version, event_kind, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.lease_id,
                event.event_version,
                event.event_kind.value,
                _event_json(event),
            ),
        )

    @staticmethod
    def _load_lease(
        connection: sqlite3.Connection,
        lease_id: str,
    ) -> AgentLeaseV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_work_leases
            WHERE lease_id = ?
            """,
            (lease_id,),
        ).fetchone()
        if row is None:
            raise StaleAgentLeaseError("Agent lease does not exist")
        return _lease_from_json(str(row["record_json"]))

    @staticmethod
    def _load_event(
        connection: sqlite3.Connection,
        event_id: str,
    ) -> AgentWorkEventV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_work_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent work event does not exist")
        return _event_from_json(str(row["record_json"]))

    def _effective_expiry(
        self,
        connection: sqlite3.Connection,
        lease: AgentLeaseV2,
    ) -> datetime:
        rows = connection.execute(
            """
            SELECT record_json FROM agent_work_events
            WHERE lease_id = ?
            ORDER BY event_version
            """,
            (lease.lease_id,),
        ).fetchall()
        events = tuple(_event_from_json(str(row["record_json"])) for row in rows)
        if tuple(event.event_version for event in events) != tuple(range(1, len(events) + 1)):
            raise ExecutionSupervisorError("Agent work event history has a version gap")
        expiry = lease.expires_at
        for event in events:
            if (
                event.lease_id != lease.lease_id
                or event.agent_task_ref != lease.agent_task_ref
                or event.fencing_token != lease.fencing_token
            ):
                raise ExecutionSupervisorError("Agent work event history drifted")
            if event.effective_expires_at is not None:
                expiry = event.effective_expires_at
        return expiry

    def _active_workspace_ref(
        self,
        connection: sqlite3.Connection,
        lease: AgentLeaseV2,
    ) -> ObjectRef:
        rows = connection.execute(
            """
            SELECT record_json FROM agent_work_events
            WHERE lease_id = ?
            ORDER BY event_version DESC
            """,
            (lease.lease_id,),
        ).fetchall()
        for row in rows:
            event = _event_from_json(str(row["record_json"]))
            if event.workspace_receipt_ref is not None:
                receipt = self.workspace_manager.resolve(event.workspace_receipt_ref)
                if receipt.state is AgentWorkspaceStateV2.ACTIVE:
                    return receipt.to_ref()
        raise ExecutionSupervisorError("Agent lease lacks an active workspace")

    @staticmethod
    def _require_head(
        connection: sqlite3.Connection,
        lease: AgentLeaseV2,
        *,
        state: AgentWorkStateV2,
        expected_event_version: int | None,
    ) -> sqlite3.Row:
        task_id = ExecutionSupervisor._task_id_for_ref(
            connection,
            lease.agent_task_ref,
        )
        head = connection.execute(
            """
            SELECT * FROM agent_work_current_heads
            WHERE agent_task_id = ?
            """,
            (task_id,),
        ).fetchone()
        if (
            head is None
            or head["lease_id"] != lease.lease_id
            or int(head["fencing_token"]) != lease.fencing_token
            or head["state"] != state.value
            or (expected_event_version is not None and int(head["event_version"]) != expected_event_version)
        ):
            raise StaleAgentLeaseError("Agent work head is stale")
        return cast(sqlite3.Row, head)

    @staticmethod
    def _update_head(
        connection: sqlite3.Connection,
        *,
        lease: AgentLeaseV2,
        prior_event_version: int,
        event: AgentWorkEventV2,
    ) -> None:
        task_id = ExecutionSupervisor._task_id_for_ref(
            connection,
            lease.agent_task_ref,
        )
        changed = connection.execute(
            """
            UPDATE agent_work_current_heads
            SET event_version = ?, state = ?
            WHERE agent_task_id = ? AND lease_id = ?
              AND fencing_token = ? AND event_version = ?
            """,
            (
                event.event_version,
                event.state.value,
                task_id,
                lease.lease_id,
                lease.fencing_token,
                prior_event_version,
            ),
        ).rowcount
        if changed != 1:
            raise StaleAgentLeaseError("Agent work head changed during transition")

    @staticmethod
    def _task_id_for_ref(
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> str:
        row = connection.execute(
            """
            SELECT agent_task_id, object_sha256
            FROM agent_tasks WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if row is None or row["object_sha256"] != reference.object_sha256:
            raise ExecutionSupervisorError("Agent task ref is not persisted")
        return str(row["agent_task_id"])

    @staticmethod
    def _load_retry_decision(
        connection: sqlite3.Connection,
        task_id: str,
    ) -> AgentRetryDecisionV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_retry_decisions
            WHERE agent_task_id = ?
            ORDER BY attempt DESC LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent retry decision is missing")
        return _retry_from_json(str(row["record_json"]))

    @staticmethod
    def _load_retry_decision_by_id(
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> AgentRetryDecisionV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_retry_decisions
            WHERE retry_decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent retry decision does not exist")
        return _retry_from_json(str(row["record_json"]))

    @staticmethod
    def _insert_cancellation(
        connection: sqlite3.Connection,
        value: AgentCancellationV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO agent_cancellations (
                cancellation_id, agent_task_id,
                lease_id, state, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                value.cancellation_id,
                value.agent_task_ref.object_id,
                value.lease_id,
                value.state.value,
                _cancellation_json(value),
            ),
        )

    @staticmethod
    def _load_cancellation(
        connection: sqlite3.Connection,
        cancellation_id: str,
    ) -> AgentCancellationV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_cancellations
            WHERE cancellation_id = ?
            """,
            (cancellation_id,),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent cancellation does not exist")
        return _cancellation_from_json(str(row["record_json"]))

    @staticmethod
    def _latest_cancellation(
        connection: sqlite3.Connection,
        lease_id: str,
        state: AgentCancellationStateV2,
    ) -> AgentCancellationV2:
        row = connection.execute(
            """
            SELECT record_json FROM agent_cancellations
            WHERE lease_id = ? AND state = ?
            """,
            (lease_id, state.value),
        ).fetchone()
        if row is None:
            raise ExecutionSupervisorError("Agent cancellation authority is missing")
        return _cancellation_from_json(str(row["record_json"]))

    @staticmethod
    def _idempotent_response(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT request_sha256, response_type, response_id
            FROM factory_control_idempotency
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha256:
            raise ExecutionSupervisorError("Agent operation idempotency request changed")
        if row["response_type"] != response_type:
            raise ExecutionSupervisorError("Agent operation idempotency type drifted")
        return str(row["response_id"])

    @staticmethod
    def _record_idempotency(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO factory_control_idempotency (
                scope, idempotency_key, request_sha256,
                response_type, response_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scope,
                idempotency_key,
                request_sha256,
                response_type,
                response_id,
            ),
        )


def _lease_json(lease: AgentLeaseV2) -> str:
    value = asdict(lease)
    value["agent_task_ref"] = lease.agent_task_ref.model_dump(mode="json")
    value["agent_definition_ref"] = lease.agent_definition_ref.model_dump(mode="json")
    value["worker"] = asdict(lease.worker)
    value["acquired_at"] = lease.acquired_at.isoformat()
    value["expires_at"] = lease.expires_at.isoformat()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _lease_from_json(payload: str) -> AgentLeaseV2:
    value = json.loads(payload)
    lease = AgentLeaseV2(
        agent_task_ref=ObjectRef.model_validate(value["agent_task_ref"]),
        agent_definition_ref=ObjectRef.model_validate(value["agent_definition_ref"]),
        lease_id=str(value["lease_id"]),
        worker=AgentWorkerRef(worker_id=str(value["worker"]["worker_id"])),
        attempt=int(value["attempt"]),
        fencing_token=int(value["fencing_token"]),
        acquired_at=datetime.fromisoformat(value["acquired_at"]),
        expires_at=datetime.fromisoformat(value["expires_at"]),
    )
    if _lease_json(lease) != payload:
        raise ExecutionSupervisorError("Agent lease material is not canonical")
    return lease


def _event_json(event: AgentWorkEventV2) -> str:
    return json.dumps(
        {
            "agent_task_ref": event.agent_task_ref.model_dump(mode="json"),
            "effective_expires_at": (
                event.effective_expires_at.isoformat() if event.effective_expires_at is not None else None
            ),
            "event_id": event.event_id,
            "event_kind": event.event_kind.value,
            "event_version": event.event_version,
            "fencing_token": event.fencing_token,
            "lease_id": event.lease_id,
            "occurred_at": event.occurred_at.isoformat(),
            "reason_code": event.reason_code,
            "state": event.state.value,
            "workspace_receipt_ref": (
                event.workspace_receipt_ref.model_dump(mode="json")
                if event.workspace_receipt_ref is not None
                else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _event_from_json(payload: str) -> AgentWorkEventV2:
    value = json.loads(payload)
    workspace_value = value["workspace_receipt_ref"]
    event = AgentWorkEventV2(
        event_id=str(value["event_id"]),
        agent_task_ref=ObjectRef.model_validate(value["agent_task_ref"]),
        lease_id=str(value["lease_id"]),
        event_version=int(value["event_version"]),
        event_kind=AgentWorkEventKindV2(value["event_kind"]),
        state=AgentWorkStateV2(value["state"]),
        fencing_token=int(value["fencing_token"]),
        occurred_at=datetime.fromisoformat(value["occurred_at"]),
        effective_expires_at=(
            datetime.fromisoformat(value["effective_expires_at"])
            if value["effective_expires_at"] is not None
            else None
        ),
        reason_code=value["reason_code"],
        workspace_receipt_ref=(
            ObjectRef.model_validate(workspace_value) if workspace_value is not None else None
        ),
    )
    if _event_json(event) != payload:
        raise ExecutionSupervisorError("Agent event material is not canonical")
    return event


def _retry_json(value: AgentRetryDecisionV2) -> str:
    return json.dumps(
        {
            "agent_task_ref": value.agent_task_ref.model_dump(mode="json"),
            "attempt": value.attempt,
            "decision": value.decision.value,
            "eligible_at": (value.eligible_at.isoformat() if value.eligible_at is not None else None),
            "next_attempt": value.next_attempt,
            "prior_lease_id": value.prior_lease_id,
            "reason_code": value.reason_code,
            "retry_decision_id": value.retry_decision_id,
            "workspace_receipt_ref": (value.workspace_receipt_ref.model_dump(mode="json")),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _retry_from_json(payload: str) -> AgentRetryDecisionV2:
    value = json.loads(payload)
    decision = AgentRetryDecisionV2(
        retry_decision_id=str(value["retry_decision_id"]),
        agent_task_ref=ObjectRef.model_validate(value["agent_task_ref"]),
        prior_lease_id=str(value["prior_lease_id"]),
        attempt=int(value["attempt"]),
        decision=AgentRetryDecisionKindV2(value["decision"]),
        next_attempt=(int(value["next_attempt"]) if value["next_attempt"] is not None else None),
        eligible_at=(
            datetime.fromisoformat(value["eligible_at"]) if value["eligible_at"] is not None else None
        ),
        reason_code=str(value["reason_code"]),
        workspace_receipt_ref=ObjectRef.model_validate(value["workspace_receipt_ref"]),
    )
    if _retry_json(decision) != payload:
        raise ExecutionSupervisorError("Agent retry material is not canonical")
    return decision


def _cancellation_json(value: AgentCancellationV2) -> str:
    return json.dumps(
        {
            "agent_task_ref": value.agent_task_ref.model_dump(mode="json"),
            "cancellation_id": value.cancellation_id,
            "lease_id": value.lease_id,
            "predecessor_cancellation_id": (value.predecessor_cancellation_id),
            "reason_code": value.reason_code,
            "requested_by": value.requested_by,
            "state": value.state.value,
            "workspace_receipt_ref": (
                value.workspace_receipt_ref.model_dump(mode="json")
                if value.workspace_receipt_ref is not None
                else None
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cancellation_from_json(
    payload: str,
) -> AgentCancellationV2:
    value = json.loads(payload)
    workspace_value = value["workspace_receipt_ref"]
    cancellation = AgentCancellationV2(
        cancellation_id=str(value["cancellation_id"]),
        agent_task_ref=ObjectRef.model_validate(value["agent_task_ref"]),
        lease_id=str(value["lease_id"]),
        state=AgentCancellationStateV2(value["state"]),
        reason_code=str(value["reason_code"]),
        requested_by=str(value["requested_by"]),
        predecessor_cancellation_id=(
            str(value["predecessor_cancellation_id"])
            if value["predecessor_cancellation_id"] is not None
            else None
        ),
        workspace_receipt_ref=(
            ObjectRef.model_validate(workspace_value) if workspace_value is not None else None
        ),
    )
    if _cancellation_json(cancellation) != payload:
        raise ExecutionSupervisorError("Agent cancellation material is not canonical")
    return cancellation


def _request_sha256(*values: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    return value.rsplit("://", 1)[-1].replace("/", "-")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "AgentCancellationStateV2",
    "AgentCancellationV2",
    "AgentLeaseV2",
    "AgentRetryDecisionKindV2",
    "AgentRetryDecisionV2",
    "AgentWorkEventKindV2",
    "AgentWorkEventV2",
    "AgentWorkStateV2",
    "AgentWorkerRef",
    "ExecutionSupervisor",
    "ExecutionSupervisorError",
    "StaleAgentLeaseError",
]
