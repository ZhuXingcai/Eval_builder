from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import (
    CompiledDatasetBuildPlanV2,
    DatasetBuildPlanV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanKindV2,
    PlannerAssessmentV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetAggregateResultV2,
    FactoryDatasetPlanningAuthorityV2,
    FactoryDatasetRunRequestV2,
    FactoryItemRunBindingV2,
    FactoryItemStageHeadV2,
    FactoryItemStageV2,
)


class FactoryControlStoreError(RuntimeError):
    pass


class FactoryControlNotFoundError(FactoryControlStoreError):
    pass


class FactoryControlConflictError(FactoryControlStoreError):
    pass


class FactoryControlConcurrencyError(FactoryControlStoreError):
    pass


class FactoryControlIntegrityError(FactoryControlStoreError):
    pass


class FactoryControlInjectedCrash(FactoryControlStoreError):
    pass


class FactoryControlStoreFaultPoint(StrEnum):
    AFTER_POLICY = "after_policy"
    AFTER_REQUIREMENT = "after_requirement"
    AFTER_RUN = "after_run"
    AFTER_RUN_HEAD = "after_run_head"
    AFTER_PLAN = "after_plan"
    AFTER_COMPILED_PLAN = "after_compiled_plan"
    AFTER_PLAN_HEAD = "after_plan_head"
    AFTER_DOMAIN_PLAN = "after_domain_plan"
    AFTER_DOMAIN_PLAN_PROMOTION = "after_domain_plan_promotion"
    AFTER_DOMAIN_PLAN_HEAD = "after_domain_plan_head"
    AFTER_DOMAIN_RESULT = "after_domain_result"
    AFTER_DOMAIN_RESULT_HEAD = "after_domain_result_head"
    AFTER_DATASET_REQUEST = "after_dataset_request"
    AFTER_PLANNING_AUTHORITY = "after_planning_authority"
    AFTER_PLANNING_AUTHORITY_HEAD = "after_planning_authority_head"
    AFTER_ITEM_BINDING = "after_item_binding"
    AFTER_ITEM_BINDING_HEAD = "after_item_binding_head"
    AFTER_ITEM_STAGE = "after_item_stage"
    AFTER_ITEM_STAGE_HEAD = "after_item_stage_head"
    AFTER_DATASET_AGGREGATE = "after_dataset_aggregate"
    AFTER_DATASET_AGGREGATE_HEAD = "after_dataset_aggregate_head"
    AFTER_PLANNER_ASSESSMENT = "after_planner_assessment"
    AFTER_OUTBOX = "after_outbox"
    AFTER_IDEMPOTENCY = "after_idempotency"


class FactoryControlStoreFaultInjector(Protocol):
    def maybe_raise(self, point: FactoryControlStoreFaultPoint) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticFactoryControlStoreFaultInjector:
    crash_points: frozenset[FactoryControlStoreFaultPoint] = frozenset()

    def maybe_raise(self, point: FactoryControlStoreFaultPoint) -> None:
        if point in self.crash_points:
            raise FactoryControlInjectedCrash(f"injected factory control store crash at {point.value}")


@dataclass(frozen=True, slots=True)
class FactoryControlHeadRebuild:
    run_head_count: int
    plan_head_count: int
    planning_authority_head_count: int = 0
    domain_plan_head_count: int = 0
    domain_result_head_count: int = 0
    item_binding_head_count: int = 0
    item_stage_head_count: int = 0
    dataset_aggregate_head_count: int = 0


class DomainPlanObject(Protocol):
    object_id: str
    object_sha256: str
    run_ref: ObjectRef
    plan_version: int
    predecessor_plan_ref: ObjectRef | None

    def canonical_json(self) -> bytes: ...

    def to_ref(self) -> ObjectRef: ...


class CompiledDomainPlanObject(Protocol):
    object_id: str
    object_sha256: str
    source_plan_ref: ObjectRef
    policy_ref: ObjectRef

    def canonical_json(self) -> bytes: ...

    def to_ref(self) -> ObjectRef: ...


class DomainResultObject(Protocol):
    object_id: str
    object_sha256: str
    plan_ref: ObjectRef
    audit: ContractAudit

    def canonical_json(self) -> bytes: ...

    def to_ref(self) -> ObjectRef: ...


@dataclass(frozen=True, slots=True)
class DomainPlanMaterial:
    run_id: str
    plan_kind: PlanKindV2
    plan_version: int
    predecessor_plan_ref: ObjectRef | None
    plan_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    plan_record_json: bytes
    compiled_plan_record_json: bytes


@dataclass(frozen=True, slots=True)
class DomainResultMaterial:
    run_id: str
    plan_kind: PlanKindV2
    plan_ref: ObjectRef
    result_ref: ObjectRef
    result_record_json: bytes


class FactoryControlStore:
    """SQLite authority for global runs; it never opens a Dataset JobStore."""

    def __init__(
        self,
        path: Path,
        *,
        fault_injector: FactoryControlStoreFaultInjector | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fault_injector = fault_injector
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS factory_run_policies (
                    policy_id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_requirement_specs (
                    requirement_spec_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    requirement_version INTEGER NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_runs (
                    run_id TEXT NOT NULL,
                    run_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    policy_object_id TEXT NOT NULL,
                    requirement_object_id TEXT NOT NULL,
                    current_plan_object_id TEXT,
                    compiled_plan_object_id TEXT,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, run_version)
                );

                CREATE TABLE IF NOT EXISTS factory_run_current_heads (
                    run_id TEXT PRIMARY KEY,
                    run_version INTEGER NOT NULL,
                    run_object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS dataset_build_plans (
                    plan_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    predecessor_object_id TEXT,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(plan_id, plan_version)
                );

                CREATE TABLE IF NOT EXISTS compiled_dataset_build_plans (
                    compiled_plan_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    source_plan_object_id TEXT NOT NULL UNIQUE,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_current_heads (
                    run_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    plan_object_id TEXT NOT NULL UNIQUE,
                    compiled_plan_object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS domain_plan_materials (
                    plan_object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    plan_kind TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    predecessor_object_id TEXT,
                    source_run_object_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    record_json_sha256 TEXT NOT NULL,
                    compiled_object_id TEXT NOT NULL UNIQUE,
                    compiled_object_type TEXT NOT NULL,
                    compiled_object_version TEXT NOT NULL,
                    compiled_object_sha256 TEXT NOT NULL,
                    compiled_record_json TEXT NOT NULL,
                    compiled_record_json_sha256 TEXT NOT NULL,
                    compiled_source_plan_object_id TEXT NOT NULL,
                    compiled_policy_object_id TEXT NOT NULL,
                    UNIQUE(run_id, plan_kind, plan_version)
                );

                CREATE TABLE IF NOT EXISTS domain_plan_promotions (
                    run_id TEXT NOT NULL,
                    plan_kind TEXT NOT NULL,
                    promotion_version INTEGER NOT NULL,
                    plan_object_id TEXT NOT NULL,
                    compiled_object_id TEXT NOT NULL,
                    PRIMARY KEY(run_id, plan_kind, promotion_version),
                    UNIQUE(run_id, plan_kind, plan_object_id)
                );

                CREATE TABLE IF NOT EXISTS domain_plan_current_heads (
                    run_id TEXT NOT NULL,
                    plan_kind TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    plan_object_id TEXT NOT NULL UNIQUE,
                    compiled_object_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(run_id, plan_kind)
                );

                CREATE TABLE IF NOT EXISTS domain_plan_material_refs (
                    plan_object_id TEXT PRIMARY KEY,
                    material_object_type TEXT NOT NULL,
                    material_object_id TEXT NOT NULL,
                    material_object_version TEXT NOT NULL,
                    material_object_sha256 TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS domain_results (
                    result_object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    plan_kind TEXT NOT NULL,
                    source_plan_object_id TEXT NOT NULL,
                    object_type TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    record_json_sha256 TEXT NOT NULL,
                    UNIQUE(run_id, plan_kind, source_plan_object_id)
                );

                CREATE TABLE IF NOT EXISTS domain_result_current_heads (
                    run_id TEXT NOT NULL,
                    plan_kind TEXT NOT NULL,
                    plan_object_id TEXT NOT NULL,
                    result_object_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(run_id, plan_kind)
                );

                CREATE TABLE IF NOT EXISTS domain_result_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    result_object_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_run_requests (
                    request_object_id TEXT PRIMARY KEY,
                    dataset_run_id TEXT NOT NULL UNIQUE,
                    requirement_object_id TEXT NOT NULL,
                    policy_object_id TEXT NOT NULL,
                    manifest_object_id TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_planning_authorities (
                    authority_object_id TEXT PRIMARY KEY,
                    dataset_run_id TEXT NOT NULL,
                    dataset_run_object_id TEXT NOT NULL,
                    planning_version INTEGER NOT NULL,
                    predecessor_authority_object_id TEXT,
                    plan_object_id TEXT NOT NULL UNIQUE,
                    route_decision_object_id TEXT NOT NULL,
                    invocation_result_object_id TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(dataset_run_id, planning_version)
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_planning_current_heads (
                    dataset_run_id TEXT PRIMARY KEY,
                    planning_version INTEGER NOT NULL,
                    authority_object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS factory_item_run_bindings (
                    binding_object_id TEXT PRIMARY KEY,
                    dataset_run_id TEXT NOT NULL,
                    dataset_run_object_id TEXT NOT NULL,
                    item_run_id TEXT NOT NULL,
                    item_run_object_id TEXT NOT NULL UNIQUE,
                    item_id TEXT NOT NULL,
                    binding_version INTEGER NOT NULL,
                    predecessor_binding_object_id TEXT,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(dataset_run_id, item_id, binding_version)
                );

                CREATE TABLE IF NOT EXISTS factory_item_binding_current_heads (
                    dataset_run_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    binding_version INTEGER NOT NULL,
                    binding_object_id TEXT NOT NULL UNIQUE,
                    item_run_id TEXT NOT NULL,
                    PRIMARY KEY(dataset_run_id, item_id)
                );

                CREATE TABLE IF NOT EXISTS factory_item_stage_records (
                    stage_head_object_id TEXT PRIMARY KEY,
                    dataset_run_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    item_run_id TEXT NOT NULL,
                    binding_object_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    stage_version INTEGER NOT NULL,
                    predecessor_head_object_id TEXT,
                    outcome TEXT NOT NULL,
                    result_object_id TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(
                        item_id, binding_object_id, stage, stage_version
                    )
                );

                CREATE TABLE IF NOT EXISTS factory_item_stage_current_heads (
                    item_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    stage_version INTEGER NOT NULL,
                    stage_head_object_id TEXT NOT NULL UNIQUE,
                    binding_object_id TEXT NOT NULL,
                    PRIMARY KEY(item_id, stage)
                );

                CREATE TABLE IF NOT EXISTS factory_item_stage_material_refs (
                    stage_head_object_id TEXT PRIMARY KEY,
                    material_object_type TEXT NOT NULL,
                    material_object_id TEXT NOT NULL,
                    material_object_version TEXT NOT NULL,
                    material_object_sha256 TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_aggregate_results (
                    aggregate_object_id TEXT PRIMARY KEY,
                    dataset_run_id TEXT NOT NULL UNIQUE,
                    dataset_run_object_id TEXT NOT NULL,
                    core_vertical_result_object_id TEXT NOT NULL,
                    batch_quality_object_id TEXT,
                    outcome TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_aggregate_current_heads (
                    dataset_run_id TEXT PRIMARY KEY,
                    aggregate_object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS factory_dataset_aggregate_material_refs (
                    aggregate_object_id TEXT PRIMARY KEY,
                    material_object_type TEXT NOT NULL,
                    material_object_id TEXT NOT NULL,
                    material_object_version TEXT NOT NULL,
                    material_object_sha256 TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_capabilities (
                    capability_id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_definitions (
                    agent_definition_id TEXT PRIMARY KEY,
                    agent_role TEXT NOT NULL,
                    agent_version TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(agent_role, agent_version)
                );

                CREATE TABLE IF NOT EXISTS agent_registry_heads (
                    agent_role TEXT PRIMARY KEY,
                    agent_definition_id TEXT NOT NULL UNIQUE,
                    object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    agent_task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_attempts (
                    agent_task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    lease_id TEXT NOT NULL UNIQUE,
                    fencing_token INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY(agent_task_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS agent_results (
                    result_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(agent_task_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS agent_work_leases (
                    lease_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(agent_task_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS agent_work_events (
                    event_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL,
                    event_version INTEGER NOT NULL,
                    event_kind TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(lease_id, event_version)
                );

                CREATE TABLE IF NOT EXISTS agent_retry_decisions (
                    retry_decision_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(agent_task_id, attempt)
                );

                CREATE TABLE IF NOT EXISTS agent_cancellations (
                    cancellation_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(lease_id, state)
                );

                CREATE TABLE IF NOT EXISTS agent_work_current_heads (
                    agent_task_id TEXT PRIMARY KEY,
                    lease_id TEXT NOT NULL UNIQUE,
                    event_version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_review_requests (
                    review_request_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    plan_object_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_revisions (
                    revision_id TEXT PRIMARY KEY,
                    review_request_id TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_decisions (
                    decision_id TEXT PRIMARY KEY,
                    review_request_id TEXT NOT NULL UNIQUE,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_review_results (
                    result_id TEXT PRIMARY KEY,
                    review_request_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_review_current_heads (
                    review_request_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS model_route_decisions (
                    route_decision_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    route_version INTEGER NOT NULL,
                    predecessor_route_object_id TEXT,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(agent_task_id, route_version)
                );

                CREATE TABLE IF NOT EXISTS gateway_receipt_refs (
                    receipt_object_id TEXT PRIMARY KEY,
                    agent_task_id TEXT NOT NULL,
                    route_decision_object_id TEXT NOT NULL,
                    receipt_ref_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS planner_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_run_completions (
                    completion_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    outcome TEXT NOT NULL,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_delivery_manifests (
                    delivery_manifest_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_control_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS factory_control_outbox (
                    event_id TEXT PRIMARY KEY,
                    aggregate_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    aggregate_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    def create_run(
        self,
        *,
        policy: FactoryRunPolicyV2,
        requirement: EvaluationRequirementSpecV2,
        idempotency_key: str,
    ) -> FactoryRunV2:
        run = FactoryRunV2.create(
            run_id=requirement.run_id,
            run_version=0,
            status=FactoryRunStatusV2.CREATED,
            policy_ref=policy.to_ref(),
            requirement_spec_ref=requirement.to_ref(),
            current_plan_ref=None,
            compiled_plan_ref=None,
            active_task_refs=(),
            result_refs=(),
            pending_review_ref=None,
            planner_assessment_ref=None,
            completion_ref=None,
            delivery_manifest_ref=None,
            transition_count=0,
            model_requests_used=0,
            model_tokens_used=0,
            cost_micro_usd_used=0,
            audit=requirement.audit,
        )
        request_sha256 = _request_sha256(policy.to_ref(), requirement.to_ref())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope="create-run",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                replay = self._load_run_by_object_id(connection, replay_id)
                connection.rollback()
                return replay
            existing = connection.execute(
                "SELECT 1 FROM factory_runs WHERE run_id = ? LIMIT 1",
                (run.run_id,),
            ).fetchone()
            if existing is not None:
                raise FactoryControlConflictError("factory run already exists")
            self._insert_policy(connection, policy)
            self._fault(FactoryControlStoreFaultPoint.AFTER_POLICY)
            self._insert_requirement(connection, requirement)
            self._fault(FactoryControlStoreFaultPoint.AFTER_REQUIREMENT)
            self._insert_run(connection, run)
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN)
            connection.execute(
                """
                INSERT INTO factory_run_current_heads (
                    run_id, run_version, run_object_id
                ) VALUES (?, ?, ?)
                """,
                (run.run_id, run.run_version, run.object_id),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=run.run_version,
                event_type="factory-run-created",
                object_id=run.object_id,
                created_at=run.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope="create-run",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
                response_id=run.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run.run_id)

    def commit_plan(
        self,
        *,
        run_id: str,
        expected_run_version: int,
        plan: DatasetBuildPlanV2,
        compiled_plan: CompiledDatasetBuildPlanV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> FactoryRunV2:
        request_sha256 = _request_sha256(
            run_id,
            expected_run_version,
            plan.to_ref(),
            compiled_plan.to_ref(),
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope="commit-plan",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                replay = self._load_run_by_object_id(connection, replay_id)
                connection.rollback()
                return replay
            current = self._load_current_run(connection, run_id)
            if current.run_version != expected_run_version:
                raise FactoryControlConcurrencyError(
                    f"expected run version {expected_run_version}, observed {current.run_version}"
                )
            if plan.run_ref != current.to_ref():
                raise FactoryControlConflictError("plan does not bind the current run authority")
            if compiled_plan.source_plan_ref != plan.to_ref():
                raise FactoryControlConflictError("compiled plan does not bind the proposal")
            if compiled_plan.policy_ref != current.policy_ref:
                raise FactoryControlConflictError("compiled plan does not bind the run policy")
            self._validate_plan_successor(connection, run_id=run_id, plan=plan)
            self._insert_plan(connection, run_id=run_id, plan=plan)
            self._fault(FactoryControlStoreFaultPoint.AFTER_PLAN)
            self._insert_compiled_plan(
                connection,
                run_id=run_id,
                compiled_plan=compiled_plan,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_COMPILED_PLAN)
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=FactoryRunStatusV2.PLANNING,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=plan.to_ref(),
                compiled_plan_ref=compiled_plan.to_ref(),
                active_task_refs=current.active_task_refs,
                result_refs=current.result_refs,
                pending_review_ref=None,
                planner_assessment_ref=current.planner_assessment_ref,
                completion_ref=None,
                delivery_manifest_ref=None,
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self._insert_run(connection, successor)
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN)
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    run_id,
                    expected_run_version,
                ),
            ).rowcount
            if changed != 1:
                raise FactoryControlConcurrencyError("factory run head changed during plan commit")
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN_HEAD)
            connection.execute(
                """
                INSERT INTO plan_current_heads (
                    run_id, plan_id, plan_version, plan_object_id,
                    compiled_plan_object_id
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    plan_version = excluded.plan_version,
                    plan_object_id = excluded.plan_object_id,
                    compiled_plan_object_id = excluded.compiled_plan_object_id
                """,
                (
                    run_id,
                    plan.plan_id,
                    plan.plan_version,
                    plan.object_id,
                    compiled_plan.object_id,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_PLAN_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=successor.run_version,
                event_type="factory-plan-committed",
                object_id=compiled_plan.object_id,
                created_at=successor.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope="commit-plan",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
                response_id=successor.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def commit_planner_assessment(
        self,
        *,
        run_id: str,
        expected_run_version: int,
        assessment: PlannerAssessmentV2,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> FactoryRunV2:
        request_sha256 = _request_sha256(
            run_id,
            expected_run_version,
            assessment.to_ref(),
        )
        scope = "commit-planner-assessment"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                replay = self._load_run_by_object_id(
                    connection,
                    replay_id,
                )
                if replay.planner_assessment_ref is None:
                    raise FactoryControlIntegrityError(
                        "assessment replay run has no assessment",
                    )
                stored = self._load_planner_assessment_by_object_id(
                    connection,
                    replay.planner_assessment_ref.object_id,
                )
                if stored != assessment:
                    raise FactoryControlIntegrityError(
                        "assessment replay authority drifted",
                    )
                connection.rollback()
                return replay
            current = self._load_current_run(connection, run_id)
            if current.run_version != expected_run_version:
                raise FactoryControlConcurrencyError(
                    "planner assessment commit uses a stale run version",
                )
            if (
                assessment.run_ref != current.to_ref()
                or current.compiled_plan_ref is None
                or assessment.compiled_plan_ref != current.compiled_plan_ref
            ):
                raise FactoryControlConflictError(
                    "planner assessment does not bind current run authority",
                )
            connection.execute(
                """
                INSERT INTO planner_assessments (
                    assessment_id, run_id, object_id,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    assessment.assessment_id,
                    run_id,
                    assessment.object_id,
                    assessment.object_sha256,
                    assessment.canonical_json().decode(),
                ),
            )
            self._fault(
                FactoryControlStoreFaultPoint.AFTER_PLANNER_ASSESSMENT,
            )
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=current.status,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=current.current_plan_ref,
                compiled_plan_ref=current.compiled_plan_ref,
                active_task_refs=current.active_task_refs,
                result_refs=current.result_refs,
                pending_review_ref=current.pending_review_ref,
                planner_assessment_ref=assessment.to_ref(),
                completion_ref=current.completion_ref,
                delivery_manifest_ref=current.delivery_manifest_ref,
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self._insert_run(connection, successor)
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN)
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    run_id,
                    expected_run_version,
                ),
            ).rowcount
            if changed != 1:
                raise FactoryControlConcurrencyError(
                    "run head changed during assessment commit",
                )
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=successor.run_version,
                event_type="planner-assessment-committed",
                object_id=assessment.object_id,
                created_at=audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
                response_id=successor.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
            return successor
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError(
                "planner assessment authority already exists",
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_planner_assessment(
        self,
        reference: ObjectRef,
    ) -> PlannerAssessmentV2:
        if reference.object_type != "planner-assessment":
            raise FactoryControlIntegrityError(
                "planner assessment ref has the wrong type",
            )
        connection = self._connect()
        try:
            value = self._load_planner_assessment_by_object_id(
                connection,
                reference.object_id,
            )
            if value.to_ref() != reference:
                raise FactoryControlIntegrityError(
                    "planner assessment differs from its reference",
                )
            return value
        finally:
            connection.close()

    def list_planner_assessments(
        self,
        run_id: str,
    ) -> tuple[PlannerAssessmentV2, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT object_id FROM planner_assessments
                WHERE run_id = ?
                ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()
            return tuple(
                self._load_planner_assessment_by_object_id(
                    connection,
                    str(row["object_id"]),
                )
                for row in rows
            )
        finally:
            connection.close()

    def commit_domain_plan[
        PlanT: DomainPlanObject,
        CompiledT: CompiledDomainPlanObject,
    ](
        self,
        *,
        run_id: str,
        expected_run_version: int,
        plan_kind: PlanKindV2,
        plan: PlanT,
        compiled_plan: CompiledT,
        audit: ContractAudit,
        idempotency_key: str,
        material_ref: ObjectRef | None = None,
    ) -> FactoryRunV2:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            raise FactoryControlConflictError("global plans require commit_plan")
        request_sha256 = _request_sha256(
            run_id,
            expected_run_version,
            plan_kind.value,
            plan.to_ref(),
            compiled_plan.to_ref(),
            material_ref,
        )
        scope = f"commit-domain-plan:{plan_kind.value}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                replay = self._load_run_by_object_id(
                    connection,
                    replay_id,
                )
                replay_material_ref = self._load_domain_plan_private_material_ref(
                    connection,
                    plan.object_id,
                )
                if replay_material_ref != material_ref:
                    raise FactoryControlIntegrityError("domain plan material replay drifted")
                connection.rollback()
                return replay
            current = self._load_current_run(connection, run_id)
            if current.run_version != expected_run_version:
                raise FactoryControlConcurrencyError("domain plan commit uses a stale run version")
            if current.status is not FactoryRunStatusV2.PLANNING:
                raise FactoryControlConflictError("domain plan commit requires a planning run")
            if plan.run_ref != current.to_ref():
                raise FactoryControlConflictError("domain plan does not bind the current run")
            if compiled_plan.source_plan_ref != plan.to_ref():
                raise FactoryControlConflictError("compiled domain plan does not bind its source")
            if compiled_plan.policy_ref != current.policy_ref:
                raise FactoryControlConflictError("compiled domain plan does not bind run policy")
            self._validate_domain_plan_successor(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                plan=plan,
            )
            self._insert_domain_plan_material(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                plan=plan,
                compiled_plan=compiled_plan,
            )
            if material_ref is not None:
                connection.execute(
                    """
                    INSERT INTO domain_plan_material_refs (
                        plan_object_id,
                        material_object_type, material_object_id,
                        material_object_version,
                        material_object_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        plan.object_id,
                        material_ref.object_type,
                        material_ref.object_id,
                        material_ref.object_version,
                        material_ref.object_sha256,
                    ),
                )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN)
            self._promote_domain_plan(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                material=self._load_domain_plan_material_by_ref(
                    connection,
                    run_id=run_id,
                    plan_kind=plan_kind,
                    reference=plan.to_ref(),
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN_PROMOTION)
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=FactoryRunStatusV2.PLANNING,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=current.current_plan_ref,
                compiled_plan_ref=current.compiled_plan_ref,
                active_task_refs=current.active_task_refs,
                result_refs=current.result_refs,
                pending_review_ref=None,
                planner_assessment_ref=current.planner_assessment_ref,
                completion_ref=None,
                delivery_manifest_ref=None,
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self._insert_run(connection, successor)
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN)
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    run_id,
                    expected_run_version,
                ),
            ).rowcount
            if changed != 1:
                raise FactoryControlConcurrencyError("run head changed during domain plan commit")
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=successor.run_version,
                event_type="factory-domain-plan-committed",
                object_id=compiled_plan.object_id,
                created_at=successor.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
                response_id=successor.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> FactoryRunV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            value = self._load_current_run(connection, run_id)
            connection.rollback()
            return value
        finally:
            connection.close()

    def get_run_by_ref(self, reference: ObjectRef) -> FactoryRunV2:
        if reference.object_type != "factory-run":
            raise FactoryControlIntegrityError(
                "factory run ref has the wrong object type",
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            value = self._load_run_by_object_id(
                connection,
                reference.object_id,
            )
            if value.to_ref() != reference:
                raise FactoryControlIntegrityError(
                    "factory run differs from its reference",
                )
            connection.rollback()
            return value
        finally:
            connection.close()

    def get_run_history(self, run_id: str) -> tuple[FactoryRunV2, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT * FROM factory_runs
                WHERE run_id = ?
                ORDER BY run_version
                """,
                (run_id,),
            ).fetchall()
            values = tuple(self._parse_run_row(row) for row in rows)
            if not values:
                raise FactoryControlNotFoundError(f"factory run not found: {run_id}")
            self._validate_contiguous_run_history(values)
            connection.rollback()
            return values
        finally:
            connection.close()

    def get_policy(self, policy_id: str) -> FactoryRunPolicyV2:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM factory_run_policies
                WHERE policy_id = ? OR object_id = ?
                """,
                (policy_id, policy_id),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError(f"factory policy not found: {policy_id}")
            return self._parse_policy_row(row)
        finally:
            connection.close()

    def get_policy_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryRunPolicyV2:
        if reference.object_type != "factory-run-policy":
            raise FactoryControlIntegrityError(
                "factory policy ref has the wrong object type",
            )
        value = self.get_policy(reference.object_id)
        if value.to_ref() != reference:
            raise FactoryControlIntegrityError(
                "factory policy differs from its reference",
            )
        return value

    def get_requirement(self, requirement_spec_id: str) -> EvaluationRequirementSpecV2:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM evaluation_requirement_specs
                WHERE requirement_spec_id = ?
                """,
                (requirement_spec_id,),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError(f"factory requirement not found: {requirement_spec_id}")
            return self._parse_requirement_row(row)
        finally:
            connection.close()

    def get_requirement_by_ref(
        self,
        reference: ObjectRef,
    ) -> EvaluationRequirementSpecV2:
        if reference.object_type != "evaluation-requirement-spec":
            raise FactoryControlIntegrityError(
                "factory requirement ref has the wrong object type",
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM evaluation_requirement_specs
                WHERE object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError(
                    "factory requirement was not found",
                )
            value = self._parse_requirement_row(row)
            if value.to_ref() != reference:
                raise FactoryControlIntegrityError(
                    "factory requirement differs from its reference",
                )
            connection.rollback()
            return value
        finally:
            connection.close()

    def get_plan(
        self,
        run_id: str,
    ) -> tuple[DatasetBuildPlanV2, CompiledDatasetBuildPlanV2]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                "SELECT * FROM plan_current_heads WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if head is None:
                raise FactoryControlNotFoundError(f"current factory plan not found: {run_id}")
            plan_row = connection.execute(
                "SELECT * FROM dataset_build_plans WHERE object_id = ?",
                (head["plan_object_id"],),
            ).fetchone()
            compiled_row = connection.execute(
                """
                SELECT * FROM compiled_dataset_build_plans
                WHERE object_id = ?
                """,
                (head["compiled_plan_object_id"],),
            ).fetchone()
            if plan_row is None or compiled_row is None:
                raise FactoryControlIntegrityError("plan head points to missing immutable records")
            plan = self._parse_plan_row(plan_row)
            compiled = self._parse_compiled_plan_row(compiled_row)
            if (
                head["plan_id"] != plan.plan_id
                or head["plan_version"] != plan.plan_version
                or compiled.source_plan_ref != plan.to_ref()
                or plan_row["run_id"] != run_id
                or compiled_row["run_id"] != run_id
            ):
                raise FactoryControlIntegrityError("plan current head does not match immutable records")
            current_run = self._load_current_run(connection, run_id)
            if (
                current_run.current_plan_ref != plan.to_ref()
                or current_run.compiled_plan_ref != compiled.to_ref()
            ):
                raise FactoryControlIntegrityError("run and plan heads disagree")
            connection.rollback()
            return plan, compiled
        finally:
            connection.close()

    def get_domain_plan(
        self,
        run_id: str,
        plan_kind: PlanKindV2,
    ) -> DomainPlanMaterial:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT * FROM domain_plan_current_heads
                WHERE run_id = ? AND plan_kind = ?
                """,
                (run_id, plan_kind.value),
            ).fetchone()
            if head is None:
                raise FactoryControlNotFoundError("current domain plan was not found")
            material = self._load_domain_plan_material_by_ref(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                reference=_object_ref_from_head(
                    connection,
                    head,
                ),
            )
            if (
                head["plan_version"] != material.plan_version
                or head["compiled_object_id"] != material.compiled_plan_ref.object_id
            ):
                raise FactoryControlIntegrityError("domain plan head differs from immutable material")
            connection.rollback()
            return material
        finally:
            connection.close()

    def get_domain_plan_material_ref(
        self,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
    ) -> ObjectRef:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT plan_object_id
                FROM domain_plan_current_heads
                WHERE run_id = ? AND plan_kind = ?
                """,
                (run_id, plan_kind.value),
            ).fetchone()
            if head is None:
                raise FactoryControlNotFoundError("current domain plan was not found")
            material_ref = self._load_domain_plan_private_material_ref(
                connection,
                str(head["plan_object_id"]),
            )
            if material_ref is None:
                raise FactoryControlNotFoundError("domain plan material was not found")
            connection.rollback()
            return material_ref
        finally:
            connection.close()

    def commit_domain_result[
        ResultT: DomainResultObject,
    ](
        self,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        result: ResultT,
        idempotency_key: str,
    ) -> DomainResultMaterial:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            raise FactoryControlConflictError("global results cannot use domain result storage")
        request_sha256 = _request_sha256(
            run_id,
            plan_kind.value,
            result.plan_ref,
            result.to_ref(),
        )
        scope = f"commit-domain-result:{run_id}:{plan_kind.value}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = connection.execute(
                """
                SELECT request_sha256, result_object_id
                FROM domain_result_idempotency
                WHERE scope = ? AND idempotency_key = ?
                """,
                (scope, idempotency_key),
            ).fetchone()
            if replay is not None:
                if replay["request_sha256"] != request_sha256:
                    raise FactoryControlConflictError("domain result idempotency request changed")
                material = self._load_domain_result_material(
                    connection,
                    str(replay["result_object_id"]),
                )
                current_plan = connection.execute(
                    """
                    SELECT plan_object_id
                    FROM domain_plan_current_heads
                    WHERE run_id = ? AND plan_kind = ?
                    """,
                    (run_id, plan_kind.value),
                ).fetchone()
                if current_plan is None or material.plan_ref.object_id != str(current_plan["plan_object_id"]):
                    raise FactoryControlConflictError("domain result replay is stale for the current plan")
                connection.rollback()
                return material
            run = self._load_current_run(connection, run_id)
            if run.status in {
                FactoryRunStatusV2.COMPLETED,
                FactoryRunStatusV2.FAILED,
                FactoryRunStatusV2.CANCELLED,
            }:
                raise FactoryControlConflictError("terminal run cannot commit a domain result")
            plan_head = connection.execute(
                """
                SELECT plan_object_id
                FROM domain_plan_current_heads
                WHERE run_id = ? AND plan_kind = ?
                """,
                (run_id, plan_kind.value),
            ).fetchone()
            if plan_head is None or result.plan_ref.object_id != str(plan_head["plan_object_id"]):
                raise FactoryControlConflictError("domain result does not bind the current plan")
            plan_material = self._load_domain_plan_material_by_ref(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                reference=result.plan_ref,
            )
            if result.plan_ref != plan_material.plan_ref:
                raise FactoryControlConflictError("domain result plan authority is stale")
            payload = result.canonical_json()
            result_ref = result.to_ref()
            connection.execute(
                """
                INSERT INTO domain_results (
                    result_object_id, run_id, plan_kind,
                    source_plan_object_id, object_type, object_version,
                    object_sha256, record_json, record_json_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.object_id,
                    run_id,
                    plan_kind.value,
                    result.plan_ref.object_id,
                    result_ref.object_type,
                    result_ref.object_version,
                    result.object_sha256,
                    payload.decode(),
                    hashlib.sha256(payload).hexdigest(),
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT)
            connection.execute(
                """
                INSERT INTO domain_result_current_heads (
                    run_id, plan_kind, plan_object_id, result_object_id
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(run_id, plan_kind) DO UPDATE SET
                    plan_object_id = excluded.plan_object_id,
                    result_object_id = excluded.result_object_id
                """,
                (
                    run_id,
                    plan_kind.value,
                    result.plan_ref.object_id,
                    result.object_id,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DOMAIN_RESULT_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=run.run_version,
                event_type="factory-domain-result-committed",
                object_id=result.object_id,
                created_at=result.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            connection.execute(
                """
                INSERT INTO domain_result_idempotency (
                    scope, idempotency_key, request_sha256,
                    result_object_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    scope,
                    idempotency_key,
                    request_sha256,
                    result.object_id,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("domain result authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_domain_result(run_id, plan_kind)

    def get_domain_result(
        self,
        run_id: str,
        plan_kind: PlanKindV2,
    ) -> DomainResultMaterial:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT * FROM domain_result_current_heads
                WHERE run_id = ? AND plan_kind = ?
                """,
                (run_id, plan_kind.value),
            ).fetchone()
            if head is None:
                raise FactoryControlNotFoundError("current domain result was not found")
            material = self._load_domain_result_material(
                connection,
                str(head["result_object_id"]),
            )
            plan_head = connection.execute(
                """
                SELECT plan_object_id FROM domain_plan_current_heads
                WHERE run_id = ? AND plan_kind = ?
                """,
                (run_id, plan_kind.value),
            ).fetchone()
            if (
                material.run_id != run_id
                or material.plan_kind is not plan_kind
                or material.plan_ref.object_id != str(head["plan_object_id"])
                or plan_head is None
                or material.plan_ref.object_id != str(plan_head["plan_object_id"])
            ):
                raise FactoryControlIntegrityError("domain result head differs from current authority")
            connection.rollback()
            return material
        finally:
            connection.close()

    def commit_dataset_request(
        self,
        request: FactoryDatasetRunRequestV2,
        *,
        idempotency_key: str,
    ) -> FactoryDatasetRunRequestV2:
        request_sha256 = _request_sha256(request.to_ref())
        scope = f"commit-dataset-request:{request.dataset_run_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-dataset-run-request",
            )
            if replay_id is not None:
                replay = self._load_dataset_request_by_object_id(
                    connection,
                    replay_id,
                )
                connection.rollback()
                return replay
            run = self._load_current_run(
                connection,
                request.dataset_run_id,
            )
            if (
                run.requirement_spec_ref != request.requirement_spec_ref
                or run.policy_ref != request.factory_policy_ref
            ):
                raise FactoryControlConflictError("dataset request does not bind current run inputs")
            connection.execute(
                """
                INSERT INTO factory_dataset_run_requests (
                    request_object_id, dataset_run_id,
                    requirement_object_id, policy_object_id,
                    manifest_object_id, object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.object_id,
                    request.dataset_run_id,
                    request.requirement_spec_ref.object_id,
                    request.factory_policy_ref.object_id,
                    request.manifest_ref.object_id,
                    request.object_sha256,
                    _json(request),
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DATASET_REQUEST)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=request.dataset_run_id,
                aggregate_version=run.run_version,
                event_type="factory-dataset-request-committed",
                object_id=request.object_id,
                created_at=request.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-dataset-run-request",
                response_id=request.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("factory dataset request authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_dataset_request(request.dataset_run_id)

    def get_dataset_request(
        self,
        dataset_run_id: str,
    ) -> FactoryDatasetRunRequestV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM factory_dataset_run_requests
                WHERE dataset_run_id = ?
                """,
                (dataset_run_id,),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError("factory dataset request was not found")
            value = self._parse_dataset_request_row(
                connection,
                row,
            )
            connection.rollback()
            return value
        finally:
            connection.close()

    def get_dataset_request_by_ref(
        self,
        reference: ObjectRef,
    ) -> FactoryDatasetRunRequestV2:
        if reference.object_type != "factory-dataset-run-request":
            raise FactoryControlIntegrityError(
                "factory dataset request ref has the wrong object type",
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            value = self._load_dataset_request_by_object_id(
                connection,
                reference.object_id,
            )
            if value.to_ref() != reference:
                raise FactoryControlIntegrityError(
                    "factory dataset request differs from its reference",
                )
            connection.rollback()
            return value
        finally:
            connection.close()

    def commit_planning_authority(
        self,
        authority: FactoryDatasetPlanningAuthorityV2,
        *,
        idempotency_key: str,
    ) -> FactoryDatasetPlanningAuthorityV2:
        request_sha256 = _request_sha256(authority.to_ref())
        scope = f"commit-planning-authority:{authority.dataset_run_ref.object_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type=("factory-dataset-planning-authority"),
            )
            if replay_id is not None:
                replay = self._load_planning_authority_by_object_id(
                    connection,
                    replay_id,
                )
                connection.rollback()
                return replay
            run = self._load_run_by_object_id(
                connection,
                authority.dataset_run_ref.object_id,
            )
            if run.to_ref() != authority.dataset_run_ref:
                raise FactoryControlConflictError("planning authority does not bind current run")
            head = connection.execute(
                """
                SELECT * FROM
                    factory_dataset_planning_current_heads
                WHERE dataset_run_id = ?
                """,
                (run.run_id,),
            ).fetchone()
            if head is None:
                if authority.planning_version != 1 or authority.predecessor_authority_ref is not None:
                    raise FactoryControlConflictError("first planning authority must be version one")
            else:
                previous = self._load_planning_authority_by_object_id(
                    connection,
                    str(head["authority_object_id"]),
                )
                if (
                    authority.planning_version != previous.planning_version + 1
                    or authority.predecessor_authority_ref != previous.to_ref()
                ):
                    raise FactoryControlConflictError("planning authority predecessor is not current")
            connection.execute(
                """
                INSERT INTO factory_dataset_planning_authorities (
                    authority_object_id, dataset_run_id,
                    dataset_run_object_id, planning_version,
                    predecessor_authority_object_id,
                    plan_object_id, route_decision_object_id,
                    invocation_result_object_id, object_sha256,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    authority.object_id,
                    run.run_id,
                    authority.dataset_run_ref.object_id,
                    authority.planning_version,
                    (
                        authority.predecessor_authority_ref.object_id
                        if authority.predecessor_authority_ref
                        else None
                    ),
                    authority.plan_ref.object_id,
                    authority.route_decision_ref.object_id,
                    authority.invocation_result_ref.object_id,
                    authority.object_sha256,
                    _json(authority),
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_PLANNING_AUTHORITY)
            connection.execute(
                """
                INSERT INTO factory_dataset_planning_current_heads (
                    dataset_run_id, planning_version,
                    authority_object_id
                ) VALUES (?, ?, ?)
                ON CONFLICT(dataset_run_id) DO UPDATE SET
                    planning_version = excluded.planning_version,
                    authority_object_id =
                        excluded.authority_object_id
                """,
                (
                    run.run_id,
                    authority.planning_version,
                    authority.object_id,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_PLANNING_AUTHORITY_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=run.run_version,
                event_type="factory-planning-authority-committed",
                object_id=authority.object_id,
                created_at=authority.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type=("factory-dataset-planning-authority"),
                response_id=authority.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("factory planning authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_planning_authority(run.run_id)

    def get_planning_authority(
        self,
        dataset_run_id: str,
    ) -> FactoryDatasetPlanningAuthorityV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM
                    factory_dataset_planning_current_heads
                WHERE dataset_run_id = ?
                """,
                (dataset_run_id,),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError("factory planning authority was not found")
            value = self._load_planning_authority_by_object_id(
                connection,
                str(row["authority_object_id"]),
            )
            if row["dataset_run_id"] != dataset_run_id or row["planning_version"] != value.planning_version:
                raise FactoryControlIntegrityError("factory planning authority head drifted")
            connection.rollback()
            return value
        finally:
            connection.close()

    def commit_item_binding(
        self,
        binding: FactoryItemRunBindingV2,
        *,
        idempotency_key: str,
    ) -> FactoryItemRunBindingV2:
        dataset_run = binding.dataset_run_ref
        request_sha256 = _request_sha256(binding.to_ref())
        scope = f"commit-item-binding:{dataset_run.object_id}:{binding.item_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-item-run-binding",
            )
            if replay_id is not None:
                replay = self._load_item_binding_by_object_id(
                    connection,
                    replay_id,
                )
                connection.rollback()
                return replay
            dataset = self._load_run_by_object_id(
                connection,
                binding.dataset_run_ref.object_id,
            )
            item_run = self._load_run_by_object_id(
                connection,
                binding.item_run_ref.object_id,
            )
            if dataset.to_ref() != binding.dataset_run_ref or item_run.to_ref() != binding.item_run_ref:
                raise FactoryControlConflictError("item binding does not reference immutable run authority")
            current_row = connection.execute(
                """
                SELECT * FROM factory_item_binding_current_heads
                WHERE dataset_run_id = ? AND item_id = ?
                """,
                (dataset.run_id, binding.item_id),
            ).fetchone()
            if current_row is None:
                historical = connection.execute(
                    """
                    SELECT 1 FROM factory_item_run_bindings
                    WHERE dataset_run_id = ? AND item_id = ?
                    LIMIT 1
                    """,
                    (dataset.run_id, binding.item_id),
                ).fetchone()
                if historical is not None:
                    raise FactoryControlIntegrityError("factory item binding head is missing")
                if binding.binding_version != 1 or binding.predecessor_binding_ref is not None:
                    raise FactoryControlConflictError("first item binding must be version one")
            else:
                current = self._load_item_binding_by_object_id(
                    connection,
                    str(current_row["binding_object_id"]),
                )
                if (
                    binding.binding_version != current.binding_version + 1
                    or binding.predecessor_binding_ref != current.to_ref()
                ):
                    raise FactoryControlConflictError("item binding authority predecessor is not current")
            connection.execute(
                """
                INSERT INTO factory_item_run_bindings (
                    binding_object_id, dataset_run_id,
                    dataset_run_object_id, item_run_id,
                    item_run_object_id, item_id, binding_version,
                    predecessor_binding_object_id, object_sha256,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    binding.object_id,
                    dataset.run_id,
                    binding.dataset_run_ref.object_id,
                    item_run.run_id,
                    binding.item_run_ref.object_id,
                    binding.item_id,
                    binding.binding_version,
                    (binding.predecessor_binding_ref.object_id if binding.predecessor_binding_ref else None),
                    binding.object_sha256,
                    _json(binding),
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_ITEM_BINDING)
            connection.execute(
                """
                INSERT INTO factory_item_binding_current_heads (
                    dataset_run_id, item_id, binding_version,
                    binding_object_id, item_run_id
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_run_id, item_id) DO UPDATE SET
                    binding_version = excluded.binding_version,
                    binding_object_id = excluded.binding_object_id,
                    item_run_id = excluded.item_run_id
                """,
                (
                    dataset.run_id,
                    binding.item_id,
                    binding.binding_version,
                    binding.object_id,
                    item_run.run_id,
                ),
            )
            if current_row is not None:
                connection.execute(
                    """
                    DELETE FROM factory_item_stage_current_heads
                    WHERE item_id = ?
                    """,
                    (binding.item_id,),
                )
            self._fault(FactoryControlStoreFaultPoint.AFTER_ITEM_BINDING_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_DATASET_ITEM",
                aggregate_id=binding.item_id,
                aggregate_version=binding.binding_version,
                event_type="factory-item-binding-committed",
                object_id=binding.object_id,
                created_at=binding.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-item-run-binding",
                response_id=binding.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("factory item binding authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_item_binding(
            dataset.run_id,
            binding.item_id,
        )

    def get_item_binding(
        self,
        dataset_run_id: str,
        item_id: str,
    ) -> FactoryItemRunBindingV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM factory_item_binding_current_heads
                WHERE dataset_run_id = ? AND item_id = ?
                """,
                (dataset_run_id, item_id),
            ).fetchone()
            if row is None:
                historical = connection.execute(
                    """
                    SELECT 1 FROM factory_item_run_bindings
                    WHERE dataset_run_id = ? AND item_id = ?
                    LIMIT 1
                    """,
                    (dataset_run_id, item_id),
                ).fetchone()
                if historical is not None:
                    raise FactoryControlIntegrityError("factory item binding head is missing")
                raise FactoryControlNotFoundError("factory item binding was not found")
            binding = self._load_item_binding_by_object_id(
                connection,
                str(row["binding_object_id"]),
            )
            if (
                row["dataset_run_id"] != dataset_run_id
                or row["item_id"] != binding.item_id
                or row["binding_version"] != binding.binding_version
                or row["item_run_id"]
                != self._load_run_by_object_id(
                    connection,
                    binding.item_run_ref.object_id,
                ).run_id
            ):
                raise FactoryControlIntegrityError("factory item binding head drifted")
            connection.rollback()
            return binding
        finally:
            connection.close()

    def get_item_run(
        self,
        binding: FactoryItemRunBindingV2,
    ) -> FactoryRunV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            stored = self._load_item_binding_by_object_id(
                connection,
                binding.object_id,
            )
            if stored != binding:
                raise FactoryControlIntegrityError("factory item binding authority drifted")
            historical = self._load_run_by_object_id(
                connection,
                binding.item_run_ref.object_id,
            )
            if historical.to_ref() != binding.item_run_ref:
                raise FactoryControlIntegrityError("factory item run binding drifted")
            current = self._load_current_run(
                connection,
                historical.run_id,
            )
            connection.rollback()
            return current
        finally:
            connection.close()

    def begin_item_planning(
        self,
        binding: FactoryItemRunBindingV2,
        *,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> FactoryRunV2:
        request_sha256 = _request_sha256(
            binding.to_ref(),
            FactoryRunStatusV2.PLANNING.value,
        )
        scope = f"begin-item-planning:{binding.object_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
            )
            if replay_id is not None:
                replay = self._load_run_by_object_id(
                    connection,
                    replay_id,
                )
                connection.rollback()
                return replay
            stored = self._load_item_binding_by_object_id(
                connection,
                binding.object_id,
            )
            if stored != binding:
                raise FactoryControlIntegrityError("factory item binding authority drifted")
            dataset = self._load_run_by_object_id(
                connection,
                binding.dataset_run_ref.object_id,
            )
            current_binding = connection.execute(
                """
                SELECT binding_object_id, item_run_id
                FROM factory_item_binding_current_heads
                WHERE dataset_run_id = ? AND item_id = ?
                """,
                (dataset.run_id, binding.item_id),
            ).fetchone()
            historical = self._load_run_by_object_id(
                connection,
                binding.item_run_ref.object_id,
            )
            if (
                dataset.to_ref() != binding.dataset_run_ref
                or historical.to_ref() != binding.item_run_ref
                or current_binding is None
                or current_binding["binding_object_id"] != binding.object_id
                or current_binding["item_run_id"] != historical.run_id
            ):
                raise FactoryControlConflictError("item planning requires the current item binding")
            current = self._load_current_run(
                connection,
                historical.run_id,
            )
            if current.to_ref() != binding.item_run_ref or current.status is not FactoryRunStatusV2.CREATED:
                raise FactoryControlConflictError("item planning requires the created child run")
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=FactoryRunStatusV2.PLANNING,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=current.current_plan_ref,
                compiled_plan_ref=current.compiled_plan_ref,
                active_task_refs=current.active_task_refs,
                result_refs=current.result_refs,
                pending_review_ref=current.pending_review_ref,
                planner_assessment_ref=current.planner_assessment_ref,
                completion_ref=current.completion_ref,
                delivery_manifest_ref=current.delivery_manifest_ref,
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self._insert_run(connection, successor)
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN)
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                  AND run_object_id = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    current.run_id,
                    current.run_version,
                    current.object_id,
                ),
            ).rowcount
            if changed != 1:
                raise FactoryControlConcurrencyError("item child run head changed during planning")
            self._fault(FactoryControlStoreFaultPoint.AFTER_RUN_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=current.run_id,
                aggregate_version=successor.run_version,
                event_type="factory-item-planning-started",
                object_id=successor.object_id,
                created_at=successor.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-run",
                response_id=successor.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("item planning authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_item_run(binding)

    def list_item_bindings(
        self,
        dataset_run_id: str,
    ) -> tuple[FactoryItemRunBindingV2, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = connection.execute(
                """
                SELECT * FROM factory_item_binding_current_heads
                WHERE dataset_run_id = ?
                ORDER BY item_id
                """,
                (dataset_run_id,),
            ).fetchall()
            immutable_count = connection.execute(
                """
                SELECT COUNT(DISTINCT item_id)
                FROM factory_item_run_bindings
                WHERE dataset_run_id = ?
                """,
                (dataset_run_id,),
            ).fetchone()
            expected_count = int(immutable_count[0]) if immutable_count is not None else 0
            if expected_count != len(rows):
                raise FactoryControlIntegrityError("factory item binding heads are incomplete")
            values = tuple(
                self._load_item_binding_by_object_id(
                    connection,
                    str(row["binding_object_id"]),
                )
                for row in rows
            )
            connection.rollback()
            return values
        finally:
            connection.close()

    def commit_item_stage_head(
        self,
        head: FactoryItemStageHeadV2,
        *,
        idempotency_key: str,
        material_ref: ObjectRef | None = None,
    ) -> FactoryItemStageHeadV2:
        request_sha256 = _request_sha256(
            head.to_ref(),
            material_ref,
        )
        scope = f"commit-item-stage:{head.item_binding_ref.object_id}:{head.stage.value}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-item-stage-head",
            )
            if replay_id is not None:
                replay = self._load_item_stage_by_object_id(
                    connection,
                    replay_id,
                )
                replay_material_ref = self._load_item_stage_material_ref(
                    connection,
                    replay.object_id,
                )
                if replay_material_ref != material_ref:
                    raise FactoryControlIntegrityError("factory item stage material replay drifted")
                connection.rollback()
                return replay
            binding = self._load_item_binding_by_object_id(
                connection,
                head.item_binding_ref.object_id,
            )
            if binding.to_ref() != head.item_binding_ref or binding.item_run_ref != head.item_run_ref:
                raise FactoryControlConflictError("item stage does not bind current item authority")
            dataset = self._load_run_by_object_id(
                connection,
                binding.dataset_run_ref.object_id,
            )
            current_binding = connection.execute(
                """
                SELECT binding_object_id
                FROM factory_item_binding_current_heads
                WHERE dataset_run_id = ? AND item_id = ?
                """,
                (dataset.run_id, binding.item_id),
            ).fetchone()
            if current_binding is None or current_binding["binding_object_id"] != binding.object_id:
                raise FactoryControlConflictError("item stage binding is not current")
            current_row = connection.execute(
                """
                SELECT * FROM factory_item_stage_current_heads
                WHERE item_id = ? AND stage = ?
                """,
                (binding.item_id, head.stage.value),
            ).fetchone()
            if current_row is None:
                historical = connection.execute(
                    """
                    SELECT 1 FROM factory_item_stage_records
                    WHERE item_id = ? AND binding_object_id = ?
                      AND stage = ?
                    LIMIT 1
                    """,
                    (
                        binding.item_id,
                        binding.object_id,
                        head.stage.value,
                    ),
                ).fetchone()
                if historical is not None:
                    raise FactoryControlIntegrityError("factory item stage head is missing")
                if head.stage_version != 1 or head.predecessor_head_ref is not None:
                    raise FactoryControlConflictError("first item stage head must be version one")
            else:
                current = self._load_item_stage_by_object_id(
                    connection,
                    str(current_row["stage_head_object_id"]),
                )
                if (
                    current.item_binding_ref != binding.to_ref()
                    or head.stage_version != current.stage_version + 1
                    or head.predecessor_head_ref != current.to_ref()
                ):
                    raise FactoryControlConflictError("item stage predecessor is not current")
            connection.execute(
                """
                INSERT INTO factory_item_stage_records (
                    stage_head_object_id, dataset_run_id, item_id,
                    item_run_id, binding_object_id, stage,
                    stage_version, predecessor_head_object_id,
                    outcome, result_object_id, object_sha256,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    head.object_id,
                    dataset.run_id,
                    binding.item_id,
                    self._load_run_by_object_id(
                        connection,
                        binding.item_run_ref.object_id,
                    ).run_id,
                    binding.object_id,
                    head.stage.value,
                    head.stage_version,
                    (head.predecessor_head_ref.object_id if head.predecessor_head_ref else None),
                    head.outcome.value,
                    head.result_ref.object_id,
                    head.object_sha256,
                    _json(head),
                ),
            )
            if material_ref is not None:
                connection.execute(
                    """
                    INSERT INTO factory_item_stage_material_refs (
                        stage_head_object_id,
                        material_object_type, material_object_id,
                        material_object_version,
                        material_object_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        head.object_id,
                        material_ref.object_type,
                        material_ref.object_id,
                        material_ref.object_version,
                        material_ref.object_sha256,
                    ),
                )
            self._fault(FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE)
            connection.execute(
                """
                INSERT INTO factory_item_stage_current_heads (
                    item_id, stage, stage_version,
                    stage_head_object_id, binding_object_id
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(item_id, stage) DO UPDATE SET
                    stage_version = excluded.stage_version,
                    stage_head_object_id =
                        excluded.stage_head_object_id,
                    binding_object_id = excluded.binding_object_id
                """,
                (
                    binding.item_id,
                    head.stage.value,
                    head.stage_version,
                    head.object_id,
                    binding.object_id,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_ITEM_STAGE_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_DATASET_ITEM",
                aggregate_id=binding.item_id,
                aggregate_version=head.stage_version,
                event_type="factory-item-stage-promoted",
                object_id=head.object_id,
                created_at=head.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-item-stage-head",
                response_id=head.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("factory item stage authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_item_stage_head(
            binding.item_id,
            head.stage,
        )

    def get_item_stage_material_ref(
        self,
        stage_head_ref: ObjectRef,
    ) -> ObjectRef:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = self._load_item_stage_by_object_id(
                connection,
                stage_head_ref.object_id,
            )
            if head.to_ref() != stage_head_ref:
                raise FactoryControlIntegrityError("factory item stage material authority drifted")
            material_ref = self._load_item_stage_material_ref(
                connection,
                head.object_id,
            )
            if material_ref is None:
                raise FactoryControlNotFoundError("factory item stage material was not found")
            connection.rollback()
            return material_ref
        finally:
            connection.close()

    def get_item_stage_head(
        self,
        item_id: str,
        stage: FactoryItemStageV2,
    ) -> FactoryItemStageHeadV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM factory_item_stage_current_heads
                WHERE item_id = ? AND stage = ?
                """,
                (item_id, stage.value),
            ).fetchone()
            if row is None:
                historical = connection.execute(
                    """
                    SELECT 1 FROM factory_item_stage_records
                    WHERE item_id = ? AND stage = ?
                    LIMIT 1
                    """,
                    (item_id, stage.value),
                ).fetchone()
                if historical is not None:
                    binding = connection.execute(
                        """
                        SELECT binding_object_id
                        FROM factory_item_binding_current_heads
                        WHERE item_id = ?
                        """,
                        (item_id,),
                    ).fetchone()
                    if binding is not None:
                        current_history = connection.execute(
                            """
                            SELECT 1 FROM factory_item_stage_records
                            WHERE item_id = ? AND stage = ?
                              AND binding_object_id = ?
                            LIMIT 1
                            """,
                            (
                                item_id,
                                stage.value,
                                binding["binding_object_id"],
                            ),
                        ).fetchone()
                        if current_history is not None:
                            raise FactoryControlIntegrityError("factory item stage head is missing")
                raise FactoryControlNotFoundError("factory item stage head was not found")
            value = self._load_item_stage_by_object_id(
                connection,
                str(row["stage_head_object_id"]),
            )
            binding = self._load_item_binding_by_object_id(
                connection,
                str(row["binding_object_id"]),
            )
            if (
                value.item_binding_ref != binding.to_ref()
                or value.stage is not stage
                or value.stage_version != row["stage_version"]
                or binding.item_id != item_id
            ):
                raise FactoryControlIntegrityError("factory item stage head drifted")
            connection.rollback()
            return value
        finally:
            connection.close()

    def list_item_stage_heads(
        self,
        item_id: str,
    ) -> tuple[FactoryItemStageHeadV2, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            binding_row = connection.execute(
                """
                SELECT binding_object_id
                FROM factory_item_binding_current_heads
                WHERE item_id = ?
                """,
                (item_id,),
            ).fetchone()
            if binding_row is None:
                raise FactoryControlNotFoundError("factory item binding was not found")
            rows = connection.execute(
                """
                SELECT * FROM factory_item_stage_current_heads
                WHERE item_id = ?
                ORDER BY stage
                """,
                (item_id,),
            ).fetchall()
            values = tuple(
                self._load_item_stage_by_object_id(
                    connection,
                    str(row["stage_head_object_id"]),
                )
                for row in rows
            )
            if any(value.item_binding_ref.object_id != binding_row["binding_object_id"] for value in values):
                raise FactoryControlIntegrityError("factory item stage heads use a stale binding")
            connection.rollback()
            return values
        finally:
            connection.close()

    def commit_dataset_aggregate(
        self,
        result: FactoryDatasetAggregateResultV2,
        *,
        material_ref: ObjectRef,
        idempotency_key: str,
    ) -> FactoryDatasetAggregateResultV2:
        request_sha256 = _request_sha256(
            result.to_ref(),
            material_ref,
        )
        scope = f"commit-dataset-aggregate:{result.dataset_run_ref.object_id}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-dataset-aggregate-result",
            )
            if replay_id is not None:
                replay = self._load_dataset_aggregate_by_object_id(
                    connection,
                    replay_id,
                )
                if (
                    self._load_dataset_aggregate_material_ref(
                        connection,
                        replay.object_id,
                    )
                    != material_ref
                ):
                    raise FactoryControlIntegrityError("dataset aggregate material replay drifted")
                connection.rollback()
                return replay
            dataset = self._load_run_by_object_id(
                connection,
                result.dataset_run_ref.object_id,
            )
            if dataset.to_ref() != result.dataset_run_ref:
                raise FactoryControlConflictError("dataset aggregate does not bind immutable run")
            candidate_refs = set(result.candidate_binding_refs)
            for binding_ref in result.item_binding_refs:
                binding = self._load_item_binding_by_object_id(
                    connection,
                    binding_ref.object_id,
                )
                current = connection.execute(
                    """
                    SELECT binding_object_id
                    FROM factory_item_binding_current_heads
                    WHERE dataset_run_id = ? AND item_id = ?
                    """,
                    (dataset.run_id, binding.item_id),
                ).fetchone()
                binding_dataset = self._load_run_by_object_id(
                    connection,
                    binding.dataset_run_ref.object_id,
                )
                if (
                    binding.to_ref() != binding_ref
                    or binding_dataset.run_id != dataset.run_id
                    or current is None
                    or current["binding_object_id"] != binding.object_id
                ):
                    raise FactoryControlConflictError("dataset aggregate item binding is not current")
                if binding_ref in candidate_refs:
                    stage_row = connection.execute(
                        """
                        SELECT stage_head_object_id
                        FROM factory_item_stage_current_heads
                        WHERE item_id = ? AND stage = ?
                        """,
                        (
                            binding.item_id,
                            FactoryItemStageV2.GRADING_DESIGN.value,
                        ),
                    ).fetchone()
                    if stage_row is None:
                        raise FactoryControlConflictError("candidate aggregate item has no grading authority")
                    stage = self._load_item_stage_by_object_id(
                        connection,
                        str(stage_row["stage_head_object_id"]),
                    )
                    if stage.item_binding_ref != binding.to_ref() or stage.outcome.value != "SUCCEEDED":
                        raise FactoryControlConflictError("candidate aggregate item is not successful")
            connection.execute(
                """
                INSERT INTO factory_dataset_aggregate_results (
                    aggregate_object_id, dataset_run_id,
                    dataset_run_object_id,
                    core_vertical_result_object_id,
                    batch_quality_object_id, outcome,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.object_id,
                    dataset.run_id,
                    result.dataset_run_ref.object_id,
                    result.core_vertical_result_ref.object_id,
                    (result.batch_quality_ref.object_id if result.batch_quality_ref is not None else None),
                    result.outcome.value,
                    result.object_sha256,
                    _json(result),
                ),
            )
            connection.execute(
                """
                INSERT INTO factory_dataset_aggregate_material_refs (
                    aggregate_object_id,
                    material_object_type, material_object_id,
                    material_object_version,
                    material_object_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.object_id,
                    material_ref.object_type,
                    material_ref.object_id,
                    material_ref.object_version,
                    material_ref.object_sha256,
                ),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DATASET_AGGREGATE)
            connection.execute(
                """
                INSERT INTO factory_dataset_aggregate_current_heads (
                    dataset_run_id, aggregate_object_id
                ) VALUES (?, ?)
                """,
                (dataset.run_id, result.object_id),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_DATASET_AGGREGATE_HEAD)
            self._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=dataset.run_id,
                aggregate_version=1,
                event_type="factory-dataset-aggregate-committed",
                object_id=result.object_id,
                created_at=result.audit.created_at.isoformat(),
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_OUTBOX)
            self._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="factory-dataset-aggregate-result",
                response_id=result.object_id,
            )
            self._fault(FactoryControlStoreFaultPoint.AFTER_IDEMPOTENCY)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("dataset aggregate authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_dataset_aggregate(dataset.run_id)

    def get_dataset_aggregate(
        self,
        dataset_run_id: str,
    ) -> FactoryDatasetAggregateResultV2:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT aggregate_object_id
                FROM factory_dataset_aggregate_current_heads
                WHERE dataset_run_id = ?
                """,
                (dataset_run_id,),
            ).fetchone()
            if head is None:
                historical = connection.execute(
                    """
                    SELECT 1
                    FROM factory_dataset_aggregate_results
                    WHERE dataset_run_id = ?
                    """,
                    (dataset_run_id,),
                ).fetchone()
                if historical is not None:
                    raise FactoryControlIntegrityError("dataset aggregate head is missing")
                raise FactoryControlNotFoundError("dataset aggregate was not found")
            result = self._load_dataset_aggregate_by_object_id(
                connection,
                str(head["aggregate_object_id"]),
            )
            if (
                self._load_dataset_aggregate_material_ref(
                    connection,
                    result.object_id,
                )
                is None
            ):
                raise FactoryControlIntegrityError("dataset aggregate material is missing")
            connection.rollback()
            return result
        finally:
            connection.close()

    def get_dataset_aggregate_material_ref(
        self,
        dataset_run_id: str,
    ) -> ObjectRef:
        result = self.get_dataset_aggregate(dataset_run_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            material_ref = self._load_dataset_aggregate_material_ref(
                connection,
                result.object_id,
            )
            if material_ref is None:
                raise FactoryControlIntegrityError("dataset aggregate material is missing")
            connection.rollback()
            return material_ref
        finally:
            connection.close()

    def rebuild_current_heads(self) -> FactoryControlHeadRebuild:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run_rows = connection.execute(
                "SELECT * FROM factory_runs ORDER BY run_id, run_version"
            ).fetchall()
            runs_by_id: dict[str, list[FactoryRunV2]] = {}
            for row in run_rows:
                run = self._parse_run_row(row)
                runs_by_id.setdefault(run.run_id, []).append(run)
            for values in runs_by_id.values():
                self._validate_contiguous_run_history(tuple(values))

            plan_rows = connection.execute(
                "SELECT * FROM dataset_build_plans ORDER BY run_id, plan_version"
            ).fetchall()
            plans_by_run: dict[str, list[DatasetBuildPlanV2]] = {}
            for row in plan_rows:
                plan = self._parse_plan_row(row)
                run_id = str(row["run_id"])
                if not self._run_ref_exists(connection, run_id, plan.run_ref.object_id):
                    raise FactoryControlIntegrityError("immutable plan references an unknown run")
                plans_by_run.setdefault(run_id, []).append(plan)

            connection.execute("DELETE FROM factory_run_current_heads")
            for run_id, values in runs_by_id.items():
                current = values[-1]
                connection.execute(
                    """
                    INSERT INTO factory_run_current_heads (
                        run_id, run_version, run_object_id
                    ) VALUES (?, ?, ?)
                    """,
                    (run_id, current.run_version, current.object_id),
                )

            connection.execute("DELETE FROM plan_current_heads")
            for run_id, plans in plans_by_run.items():
                current_plan = max(plans, key=lambda value: value.plan_version)
                compiled_row = connection.execute(
                    """
                    SELECT * FROM compiled_dataset_build_plans
                    WHERE source_plan_object_id = ?
                    """,
                    (current_plan.object_id,),
                ).fetchone()
                if compiled_row is None:
                    raise FactoryControlIntegrityError("immutable plan has no compiled authority")
                compiled = self._parse_compiled_plan_row(compiled_row)
                if compiled.source_plan_ref != current_plan.to_ref():
                    raise FactoryControlIntegrityError("compiled plan source differs from immutable plan")
                connection.execute(
                    """
                    INSERT INTO plan_current_heads (
                        run_id, plan_id, plan_version, plan_object_id,
                        compiled_plan_object_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        current_plan.plan_id,
                        current_plan.plan_version,
                        current_plan.object_id,
                        compiled.object_id,
                    ),
                )

            planning_rows = connection.execute(
                """
                SELECT * FROM factory_dataset_planning_authorities
                ORDER BY dataset_run_id, planning_version
                """
            ).fetchall()
            planning_by_run: dict[
                str,
                list[FactoryDatasetPlanningAuthorityV2],
            ] = {}
            for row in planning_rows:
                authority = self._load_planning_authority_by_object_id(
                    connection,
                    str(row["authority_object_id"]),
                )
                planning_by_run.setdefault(
                    str(row["dataset_run_id"]),
                    [],
                ).append(authority)
            for planning_history in planning_by_run.values():
                self._validate_planning_authority_history(tuple(planning_history))
            connection.execute("DELETE FROM factory_dataset_planning_current_heads")
            for dataset_run_id, planning_history in planning_by_run.items():
                current_planning = planning_history[-1]
                connection.execute(
                    """
                    INSERT INTO
                        factory_dataset_planning_current_heads (
                            dataset_run_id, planning_version,
                            authority_object_id
                        ) VALUES (?, ?, ?)
                    """,
                    (
                        dataset_run_id,
                        current_planning.planning_version,
                        current_planning.object_id,
                    ),
                )

            promotion_rows = connection.execute(
                """
                SELECT * FROM domain_plan_promotions
                ORDER BY run_id, plan_kind, promotion_version
                """
            ).fetchall()
            promotions: dict[
                tuple[str, PlanKindV2],
                list[DomainPlanMaterial],
            ] = {}
            for row in promotion_rows:
                key = (
                    str(row["run_id"]),
                    _domain_plan_kind(row["plan_kind"]),
                )
                material = self._load_domain_plan_material_by_object_id(
                    connection,
                    str(row["plan_object_id"]),
                )
                if (
                    material.run_id != key[0]
                    or material.plan_kind is not key[1]
                    or material.plan_version != row["promotion_version"]
                    or material.compiled_plan_ref.object_id != row["compiled_object_id"]
                ):
                    raise FactoryControlIntegrityError("domain plan promotion drifted")
                promotions.setdefault(key, []).append(material)
            for domain_values in promotions.values():
                self._validate_domain_promotion_history(tuple(domain_values))

            connection.execute("DELETE FROM domain_plan_current_heads")
            for (
                domain_run_id,
                plan_kind,
            ), domain_values in promotions.items():
                current_domain = domain_values[-1]
                connection.execute(
                    """
                    INSERT INTO domain_plan_current_heads (
                        run_id, plan_kind, plan_version,
                        plan_object_id, compiled_object_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        domain_run_id,
                        plan_kind.value,
                        current_domain.plan_version,
                        current_domain.plan_ref.object_id,
                        current_domain.compiled_plan_ref.object_id,
                    ),
                )
            connection.execute("DELETE FROM domain_result_current_heads")
            result_head_count = 0
            result_rows = connection.execute(
                """
                SELECT result_object_id FROM domain_results
                ORDER BY run_id, plan_kind, source_plan_object_id
                """
            ).fetchall()
            current_plan_ids = {
                (
                    domain_run_id,
                    plan_kind,
                ): domain_values[-1].plan_ref.object_id
                for (
                    domain_run_id,
                    plan_kind,
                ), domain_values in promotions.items()
            }
            for row in result_rows:
                result_material = self._load_domain_result_material(
                    connection,
                    str(row["result_object_id"]),
                )
                current_plan_id = current_plan_ids.get(
                    (
                        result_material.run_id,
                        result_material.plan_kind,
                    )
                )
                if current_plan_id is None or result_material.plan_ref.object_id != current_plan_id:
                    continue
                connection.execute(
                    """
                    INSERT INTO domain_result_current_heads (
                        run_id, plan_kind, plan_object_id,
                        result_object_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        result_material.run_id,
                        result_material.plan_kind.value,
                        result_material.plan_ref.object_id,
                        result_material.result_ref.object_id,
                    ),
                )
                result_head_count += 1

            binding_rows = connection.execute(
                """
                SELECT * FROM factory_item_run_bindings
                ORDER BY dataset_run_id, item_id, binding_version
                """
            ).fetchall()
            bindings: dict[
                tuple[str, str],
                list[FactoryItemRunBindingV2],
            ] = {}
            for row in binding_rows:
                binding = self._parse_item_binding_row(
                    connection,
                    row,
                )
                binding_key = (
                    str(row["dataset_run_id"]),
                    binding.item_id,
                )
                bindings.setdefault(binding_key, []).append(binding)
            for binding_history in bindings.values():
                self._validate_item_binding_history(
                    connection,
                    tuple(binding_history),
                )

            connection.execute("DELETE FROM factory_item_binding_current_heads")
            current_binding_ids: set[str] = set()
            for (
                dataset_run_id,
                item_id,
            ), binding_history in bindings.items():
                current_binding = binding_history[-1]
                item_run = self._load_run_by_object_id(
                    connection,
                    current_binding.item_run_ref.object_id,
                )
                connection.execute(
                    """
                    INSERT INTO factory_item_binding_current_heads (
                        dataset_run_id, item_id, binding_version,
                        binding_object_id, item_run_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_run_id,
                        item_id,
                        current_binding.binding_version,
                        current_binding.object_id,
                        item_run.run_id,
                    ),
                )
                current_binding_ids.add(current_binding.object_id)

            stage_rows = connection.execute(
                """
                SELECT * FROM factory_item_stage_records
                ORDER BY item_id, binding_object_id, stage,
                         stage_version
                """
            ).fetchall()
            stages: dict[
                tuple[str, str, FactoryItemStageV2],
                list[FactoryItemStageHeadV2],
            ] = {}
            for row in stage_rows:
                stage = self._parse_item_stage_row(
                    connection,
                    row,
                )
                stage_key = (
                    str(row["item_id"]),
                    str(row["binding_object_id"]),
                    stage.stage,
                )
                stages.setdefault(stage_key, []).append(stage)
            for stage_history in stages.values():
                self._validate_item_stage_history(tuple(stage_history))

            connection.execute("DELETE FROM factory_item_stage_current_heads")
            item_stage_head_count = 0
            for (
                item_id,
                binding_object_id,
                stage_kind,
            ), stage_history in stages.items():
                if binding_object_id not in current_binding_ids:
                    continue
                current_stage = stage_history[-1]
                connection.execute(
                    """
                    INSERT INTO factory_item_stage_current_heads (
                        item_id, stage, stage_version,
                        stage_head_object_id, binding_object_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        item_id,
                        stage_kind.value,
                        current_stage.stage_version,
                        current_stage.object_id,
                        binding_object_id,
                    ),
                )
                item_stage_head_count += 1

            aggregate_rows = connection.execute(
                """
                SELECT aggregate_object_id, dataset_run_id
                FROM factory_dataset_aggregate_results
                ORDER BY dataset_run_id
                """
            ).fetchall()
            aggregates: dict[
                str,
                FactoryDatasetAggregateResultV2,
            ] = {}
            for row in aggregate_rows:
                aggregate = self._load_dataset_aggregate_by_object_id(
                    connection,
                    str(row["aggregate_object_id"]),
                )
                dataset_run_id = str(row["dataset_run_id"])
                if dataset_run_id in aggregates:
                    raise FactoryControlIntegrityError("dataset aggregate history is ambiguous")
                if (
                    self._load_dataset_aggregate_material_ref(
                        connection,
                        aggregate.object_id,
                    )
                    is None
                ):
                    raise FactoryControlIntegrityError("dataset aggregate material is missing")
                aggregates[dataset_run_id] = aggregate
            connection.execute("DELETE FROM factory_dataset_aggregate_current_heads")
            for dataset_run_id, aggregate in aggregates.items():
                connection.execute(
                    """
                    INSERT INTO
                        factory_dataset_aggregate_current_heads (
                            dataset_run_id, aggregate_object_id
                        ) VALUES (?, ?)
                    """,
                    (
                        dataset_run_id,
                        aggregate.object_id,
                    ),
                )
            connection.commit()
            return FactoryControlHeadRebuild(
                run_head_count=len(runs_by_id),
                plan_head_count=len(plans_by_run),
                planning_authority_head_count=len(planning_by_run),
                domain_plan_head_count=len(promotions),
                domain_result_head_count=result_head_count,
                item_binding_head_count=len(bindings),
                item_stage_head_count=item_stage_head_count,
                dataset_aggregate_head_count=len(aggregates),
            )
        except (ValidationError, sqlite3.DatabaseError) as exc:
            connection.rollback()
            raise FactoryControlIntegrityError(
                "immutable factory control records cannot rebuild current heads"
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _insert_policy(
        self,
        connection: sqlite3.Connection,
        policy: FactoryRunPolicyV2,
    ) -> None:
        existing = connection.execute(
            "SELECT * FROM factory_run_policies WHERE policy_id = ?",
            (policy.policy_id,),
        ).fetchone()
        if existing is not None:
            if self._parse_policy_row(existing) != policy:
                raise FactoryControlConflictError("factory policy identity already exists")
            return
        connection.execute(
            """
            INSERT INTO factory_run_policies (
                policy_id, object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (policy.policy_id, policy.object_id, policy.object_sha256, _json(policy)),
        )

    def _insert_requirement(
        self,
        connection: sqlite3.Connection,
        requirement: EvaluationRequirementSpecV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO evaluation_requirement_specs (
                requirement_spec_id, run_id, requirement_version,
                object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                requirement.requirement_spec_id,
                requirement.run_id,
                requirement.requirement_version,
                requirement.object_id,
                requirement.object_sha256,
                _json(requirement),
            ),
        )

    def _insert_run(self, connection: sqlite3.Connection, run: FactoryRunV2) -> None:
        connection.execute(
            """
            INSERT INTO factory_runs (
                run_id, run_version, status, policy_object_id,
                requirement_object_id, current_plan_object_id,
                compiled_plan_object_id, object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.run_version,
                run.status.value,
                run.policy_ref.object_id,
                run.requirement_spec_ref.object_id,
                run.current_plan_ref.object_id if run.current_plan_ref else None,
                run.compiled_plan_ref.object_id if run.compiled_plan_ref else None,
                run.object_id,
                run.object_sha256,
                _json(run),
            ),
        )

    def _insert_plan(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan: DatasetBuildPlanV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO dataset_build_plans (
                plan_id, plan_version, run_id, predecessor_object_id,
                object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.plan_id,
                plan.plan_version,
                run_id,
                plan.predecessor_plan_ref.object_id if plan.predecessor_plan_ref else None,
                plan.object_id,
                plan.object_sha256,
                _json(plan),
            ),
        )

    def _insert_compiled_plan(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        compiled_plan: CompiledDatasetBuildPlanV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO compiled_dataset_build_plans (
                compiled_plan_id, run_id, source_plan_object_id,
                object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                compiled_plan.compiled_plan_id,
                run_id,
                compiled_plan.source_plan_ref.object_id,
                compiled_plan.object_id,
                compiled_plan.object_sha256,
                _json(compiled_plan),
            ),
        )

    def _insert_domain_plan_material[
        PlanT: DomainPlanObject,
        CompiledT: CompiledDomainPlanObject,
    ](
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan: PlanT,
        compiled_plan: CompiledT,
    ) -> None:
        plan_json = plan.canonical_json()
        compiled_json = compiled_plan.canonical_json()
        connection.execute(
            """
            INSERT INTO domain_plan_materials (
                plan_object_id, run_id, plan_kind, plan_version,
                predecessor_object_id, source_run_object_id,
                object_type, object_version, object_sha256,
                record_json, record_json_sha256,
                compiled_object_id, compiled_object_type,
                compiled_object_version, compiled_object_sha256,
                compiled_record_json, compiled_record_json_sha256,
                compiled_source_plan_object_id,
                compiled_policy_object_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan.object_id,
                run_id,
                plan_kind.value,
                plan.plan_version,
                (plan.predecessor_plan_ref.object_id if plan.predecessor_plan_ref else None),
                plan.run_ref.object_id,
                plan.to_ref().object_type,
                plan.to_ref().object_version,
                plan.object_sha256,
                plan_json.decode(),
                hashlib.sha256(plan_json).hexdigest(),
                compiled_plan.object_id,
                compiled_plan.to_ref().object_type,
                compiled_plan.to_ref().object_version,
                compiled_plan.object_sha256,
                compiled_json.decode(),
                hashlib.sha256(compiled_json).hexdigest(),
                compiled_plan.source_plan_ref.object_id,
                compiled_plan.policy_ref.object_id,
            ),
        )

    def _promote_domain_plan(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        material: DomainPlanMaterial,
    ) -> None:
        previous_row = connection.execute(
            """
            SELECT * FROM domain_plan_promotions
            WHERE run_id = ? AND plan_kind = ?
            ORDER BY promotion_version DESC
            LIMIT 1
            """,
            (run_id, plan_kind.value),
        ).fetchone()
        expected_version = 1
        if previous_row is not None:
            expected_version = int(previous_row["promotion_version"]) + 1
            previous = self._load_domain_plan_material_by_object_id(
                connection,
                str(previous_row["plan_object_id"]),
            )
            if material.predecessor_plan_ref != previous.plan_ref:
                raise FactoryControlConflictError("domain plan promotion predecessor is invalid")
        elif material.predecessor_plan_ref is not None:
            raise FactoryControlConflictError("first domain plan promotion cannot have a predecessor")
        if material.plan_version != expected_version:
            raise FactoryControlConflictError("domain plan promotion version is not contiguous")
        connection.execute(
            """
            INSERT INTO domain_plan_promotions (
                run_id, plan_kind, promotion_version,
                plan_object_id, compiled_object_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                plan_kind.value,
                material.plan_version,
                material.plan_ref.object_id,
                material.compiled_plan_ref.object_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO domain_plan_current_heads (
                run_id, plan_kind, plan_version,
                plan_object_id, compiled_object_id
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, plan_kind) DO UPDATE SET
                plan_version = excluded.plan_version,
                plan_object_id = excluded.plan_object_id,
                compiled_object_id = excluded.compiled_object_id
            """,
            (
                run_id,
                plan_kind.value,
                material.plan_version,
                material.plan_ref.object_id,
                material.compiled_plan_ref.object_id,
            ),
        )
        if previous_row is not None:
            invalidated_kinds = _invalidated_result_kinds(plan_kind)
            connection.execute(
                f"""
                DELETE FROM domain_result_current_heads
                WHERE run_id = ? AND plan_kind IN (
                    {", ".join("?" for _ in invalidated_kinds)}
                )
                """,
                (
                    run_id,
                    *(value.value for value in invalidated_kinds),
                ),
            )
        self._fault(FactoryControlStoreFaultPoint.AFTER_DOMAIN_PLAN_HEAD)

    def _insert_outbox(
        self,
        connection: sqlite3.Connection,
        *,
        aggregate_type: str,
        aggregate_id: str,
        aggregate_version: int,
        event_type: str,
        object_id: str,
        created_at: str,
    ) -> None:
        seed = f"{aggregate_type}|{aggregate_id}|{aggregate_version}|{event_type}|{object_id}"
        event_id = f"factory-control-event://sha256/{hashlib.sha256(seed.encode()).hexdigest()}"
        connection.execute(
            """
            INSERT INTO factory_control_outbox (
                event_id, aggregate_type, aggregate_id, aggregate_version,
                event_type, attributes_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                aggregate_type,
                aggregate_id,
                aggregate_version,
                event_type,
                json.dumps(
                    {"object_id": object_id},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                created_at,
            ),
        )

    def _check_idempotency(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str = "factory-run",
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
            raise FactoryControlConflictError("idempotency key request hash changed")
        if row["response_type"] != response_type:
            raise FactoryControlIntegrityError("idempotency response type is invalid")
        return str(row["response_id"])

    def _insert_idempotency(
        self,
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

    def _validate_plan_successor(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan: DatasetBuildPlanV2,
    ) -> None:
        head = connection.execute(
            "SELECT * FROM plan_current_heads WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if head is None:
            if plan.plan_version != 1 or plan.predecessor_plan_ref is not None:
                raise FactoryControlConflictError("first committed plan must be version one")
            return
        previous_row = connection.execute(
            "SELECT * FROM dataset_build_plans WHERE object_id = ?",
            (head["plan_object_id"],),
        ).fetchone()
        if previous_row is None:
            raise FactoryControlIntegrityError("plan head points to a missing prior plan")
        previous = self._parse_plan_row(previous_row)
        if (
            plan.plan_id != previous.plan_id
            or plan.plan_version != previous.plan_version + 1
            or plan.predecessor_plan_ref != previous.to_ref()
        ):
            raise FactoryControlConflictError("plan successor chain is invalid")

    def _validate_domain_plan_successor(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan: DomainPlanObject,
    ) -> None:
        head = connection.execute(
            """
            SELECT * FROM domain_plan_current_heads
            WHERE run_id = ? AND plan_kind = ?
            """,
            (run_id, plan_kind.value),
        ).fetchone()
        if head is None:
            if plan.plan_version != 1 or plan.predecessor_plan_ref is not None:
                raise FactoryControlConflictError("first domain plan must be version one")
            return
        previous = self._load_domain_plan_material_by_object_id(
            connection,
            str(head["plan_object_id"]),
        )
        if plan.plan_version != previous.plan_version + 1 or plan.predecessor_plan_ref != previous.plan_ref:
            raise FactoryControlConflictError("domain plan successor chain is invalid")

    def _load_domain_plan_material_by_ref(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        reference: ObjectRef,
    ) -> DomainPlanMaterial:
        material = self._load_domain_plan_material_by_object_id(
            connection,
            reference.object_id,
        )
        if material.run_id != run_id or material.plan_kind is not plan_kind or material.plan_ref != reference:
            raise FactoryControlIntegrityError("domain plan material does not match requested authority")
        return material

    @staticmethod
    def _load_domain_plan_private_material_ref(
        connection: sqlite3.Connection,
        plan_object_id: str,
    ) -> ObjectRef | None:
        row = connection.execute(
            """
            SELECT * FROM domain_plan_material_refs
            WHERE plan_object_id = ?
            """,
            (plan_object_id,),
        ).fetchone()
        if row is None:
            return None
        return ObjectRef(
            object_type=str(row["material_object_type"]),
            object_id=str(row["material_object_id"]),
            object_version=str(row["material_object_version"]),
            object_sha256=str(row["material_object_sha256"]),
        )

    def _load_domain_plan_material_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> DomainPlanMaterial:
        row = connection.execute(
            """
            SELECT * FROM domain_plan_materials
            WHERE plan_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("domain plan material is missing")
        plan_payload = _canonical_record(
            str(row["record_json"]),
            str(row["record_json_sha256"]),
            "domain plan",
        )
        compiled_payload = _canonical_record(
            str(row["compiled_record_json"]),
            str(row["compiled_record_json_sha256"]),
            "compiled domain plan",
        )
        plan_ref = ObjectRef(
            object_type=str(row["object_type"]),
            object_id=str(row["plan_object_id"]),
            object_version=str(row["object_version"]),
            object_sha256=str(row["object_sha256"]),
        )
        compiled_ref = ObjectRef(
            object_type=str(row["compiled_object_type"]),
            object_id=str(row["compiled_object_id"]),
            object_version=str(row["compiled_object_version"]),
            object_sha256=str(row["compiled_object_sha256"]),
        )
        predecessor = _optional_ref_from_payload(
            plan_payload,
            "predecessor_plan_ref",
        )
        run_ref = _required_ref_from_payload(
            plan_payload,
            "run_ref",
        )
        source_plan_ref = _required_ref_from_payload(
            compiled_payload,
            "source_plan_ref",
        )
        policy_ref = _required_ref_from_payload(
            compiled_payload,
            "policy_ref",
        )
        if (
            plan_payload.get("object_id") != plan_ref.object_id
            or plan_payload.get("object_sha256") != plan_ref.object_sha256
            or plan_payload.get("plan_version") != row["plan_version"]
            or (predecessor.object_id if predecessor else None) != row["predecessor_object_id"]
            or run_ref.object_id != row["source_run_object_id"]
            or compiled_payload.get("object_id") != compiled_ref.object_id
            or compiled_payload.get("object_sha256") != compiled_ref.object_sha256
            or source_plan_ref != plan_ref
            or source_plan_ref.object_id != row["compiled_source_plan_object_id"]
            or policy_ref.object_id != row["compiled_policy_object_id"]
        ):
            raise FactoryControlIntegrityError("domain plan materialized columns drifted")
        return DomainPlanMaterial(
            run_id=str(row["run_id"]),
            plan_kind=_domain_plan_kind(row["plan_kind"]),
            plan_version=int(row["plan_version"]),
            predecessor_plan_ref=predecessor,
            plan_ref=plan_ref,
            compiled_plan_ref=compiled_ref,
            plan_record_json=str(row["record_json"]).encode(),
            compiled_plan_record_json=str(row["compiled_record_json"]).encode(),
        )

    def _load_domain_result_material(
        self,
        connection: sqlite3.Connection,
        result_object_id: str,
    ) -> DomainResultMaterial:
        row = connection.execute(
            """
            SELECT * FROM domain_results
            WHERE result_object_id = ?
            """,
            (result_object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("domain result material is missing")
        payload = _canonical_record(
            str(row["record_json"]),
            str(row["record_json_sha256"]),
            "domain result",
        )
        plan_ref = _required_ref_from_payload(payload, "plan_ref")
        result_ref = ObjectRef(
            object_type=str(row["object_type"]),
            object_id=str(row["result_object_id"]),
            object_version=str(row["object_version"]),
            object_sha256=str(row["object_sha256"]),
        )
        if (
            payload.get("object_id") != result_ref.object_id
            or payload.get("object_sha256") != result_ref.object_sha256
            or plan_ref.object_id != str(row["source_plan_object_id"])
        ):
            raise FactoryControlIntegrityError("domain result materialized columns drifted")
        plan = self._load_domain_plan_material_by_object_id(
            connection,
            plan_ref.object_id,
        )
        plan_kind = _domain_plan_kind(row["plan_kind"])
        if plan.run_id != str(row["run_id"]) or plan.plan_kind is not plan_kind or plan.plan_ref != plan_ref:
            raise FactoryControlIntegrityError("domain result source plan is invalid")
        return DomainResultMaterial(
            run_id=str(row["run_id"]),
            plan_kind=plan_kind,
            plan_ref=plan_ref,
            result_ref=result_ref,
            result_record_json=str(row["record_json"]).encode(),
        )

    @staticmethod
    def _validate_domain_promotion_history(
        values: tuple[DomainPlanMaterial, ...],
    ) -> None:
        if tuple(value.plan_version for value in values) != tuple(range(1, len(values) + 1)):
            raise FactoryControlIntegrityError("domain plan promotion history has a version gap")
        for index, value in enumerate(values):
            expected = None if index == 0 else values[index - 1].plan_ref
            if value.predecessor_plan_ref != expected:
                raise FactoryControlIntegrityError("domain plan promotion predecessor drifted")

    def _load_dataset_request_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryDatasetRunRequestV2:
        row = connection.execute(
            """
            SELECT * FROM factory_dataset_run_requests
            WHERE request_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("immutable factory dataset request is missing")
        return self._parse_dataset_request_row(connection, row)

    def _parse_dataset_request_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> FactoryDatasetRunRequestV2:
        value = _parse(
            FactoryDatasetRunRequestV2,
            str(row["record_json"]),
            "immutable factory dataset request",
        )
        run = self._load_current_run(
            connection,
            value.dataset_run_id,
        )
        if (
            row["request_object_id"] != value.object_id
            or row["dataset_run_id"] != value.dataset_run_id
            or row["requirement_object_id"] != value.requirement_spec_ref.object_id
            or row["policy_object_id"] != value.factory_policy_ref.object_id
            or row["manifest_object_id"] != value.manifest_ref.object_id
            or row["object_sha256"] != value.object_sha256
            or run.requirement_spec_ref != value.requirement_spec_ref
            or run.policy_ref != value.factory_policy_ref
        ):
            raise FactoryControlIntegrityError("immutable factory dataset request columns drifted")
        return value

    def _load_planning_authority_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryDatasetPlanningAuthorityV2:
        row = connection.execute(
            """
            SELECT * FROM factory_dataset_planning_authorities
            WHERE authority_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("immutable factory planning authority is missing")
        value = _parse(
            FactoryDatasetPlanningAuthorityV2,
            str(row["record_json"]),
            "immutable factory planning authority",
        )
        run = self._load_run_by_object_id(
            connection,
            value.dataset_run_ref.object_id,
        )
        if (
            run.to_ref() != value.dataset_run_ref
            or row["authority_object_id"] != value.object_id
            or row["dataset_run_id"] != run.run_id
            or row["dataset_run_object_id"] != value.dataset_run_ref.object_id
            or row["planning_version"] != value.planning_version
            or row["predecessor_authority_object_id"]
            != (value.predecessor_authority_ref.object_id if value.predecessor_authority_ref else None)
            or row["plan_object_id"] != value.plan_ref.object_id
            or row["route_decision_object_id"] != value.route_decision_ref.object_id
            or row["invocation_result_object_id"] != value.invocation_result_ref.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable factory planning authority columns drifted")
        return value

    @staticmethod
    def _validate_planning_authority_history(
        values: tuple[FactoryDatasetPlanningAuthorityV2, ...],
    ) -> None:
        if tuple(value.planning_version for value in values) != tuple(range(1, len(values) + 1)):
            raise FactoryControlIntegrityError("factory planning authority history has a version gap")
        for index, value in enumerate(values):
            expected = None if index == 0 else values[index - 1].to_ref()
            if value.predecessor_authority_ref != expected:
                raise FactoryControlIntegrityError("factory planning authority predecessor drifted")

    def _load_item_binding_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryItemRunBindingV2:
        row = connection.execute(
            """
            SELECT * FROM factory_item_run_bindings
            WHERE binding_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("immutable factory item binding is missing")
        return self._parse_item_binding_row(connection, row)

    def _parse_item_binding_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> FactoryItemRunBindingV2:
        value = _parse(
            FactoryItemRunBindingV2,
            str(row["record_json"]),
            "immutable factory item binding",
        )
        dataset = self._load_run_by_object_id(
            connection,
            value.dataset_run_ref.object_id,
        )
        item_run = self._load_run_by_object_id(
            connection,
            value.item_run_ref.object_id,
        )
        if (
            dataset.to_ref() != value.dataset_run_ref
            or item_run.to_ref() != value.item_run_ref
            or row["binding_object_id"] != value.object_id
            or row["dataset_run_id"] != dataset.run_id
            or row["dataset_run_object_id"] != value.dataset_run_ref.object_id
            or row["item_run_id"] != item_run.run_id
            or row["item_run_object_id"] != value.item_run_ref.object_id
            or row["item_id"] != value.item_id
            or row["binding_version"] != value.binding_version
            or row["predecessor_binding_object_id"]
            != (value.predecessor_binding_ref.object_id if value.predecessor_binding_ref else None)
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable factory item binding columns drifted")
        return value

    def _load_item_stage_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryItemStageHeadV2:
        row = connection.execute(
            """
            SELECT * FROM factory_item_stage_records
            WHERE stage_head_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("immutable factory item stage is missing")
        return self._parse_item_stage_row(connection, row)

    @staticmethod
    def _load_item_stage_material_ref(
        connection: sqlite3.Connection,
        stage_head_object_id: str,
    ) -> ObjectRef | None:
        row = connection.execute(
            """
            SELECT * FROM factory_item_stage_material_refs
            WHERE stage_head_object_id = ?
            """,
            (stage_head_object_id,),
        ).fetchone()
        if row is None:
            return None
        return ObjectRef(
            object_type=str(row["material_object_type"]),
            object_id=str(row["material_object_id"]),
            object_version=str(row["material_object_version"]),
            object_sha256=str(row["material_object_sha256"]),
        )

    def _parse_item_stage_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> FactoryItemStageHeadV2:
        value = _parse(
            FactoryItemStageHeadV2,
            str(row["record_json"]),
            "immutable factory item stage",
        )
        binding = self._load_item_binding_by_object_id(
            connection,
            value.item_binding_ref.object_id,
        )
        dataset = self._load_run_by_object_id(
            connection,
            binding.dataset_run_ref.object_id,
        )
        item_run = self._load_run_by_object_id(
            connection,
            binding.item_run_ref.object_id,
        )
        if (
            value.item_binding_ref != binding.to_ref()
            or value.item_run_ref != binding.item_run_ref
            or row["stage_head_object_id"] != value.object_id
            or row["dataset_run_id"] != dataset.run_id
            or row["item_id"] != binding.item_id
            or row["item_run_id"] != item_run.run_id
            or row["binding_object_id"] != binding.object_id
            or row["stage"] != value.stage.value
            or row["stage_version"] != value.stage_version
            or row["predecessor_head_object_id"]
            != (value.predecessor_head_ref.object_id if value.predecessor_head_ref else None)
            or row["outcome"] != value.outcome.value
            or row["result_object_id"] != value.result_ref.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable factory item stage columns drifted")
        return value

    def _load_dataset_aggregate_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryDatasetAggregateResultV2:
        row = connection.execute(
            """
            SELECT * FROM factory_dataset_aggregate_results
            WHERE aggregate_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("immutable dataset aggregate is missing")
        value = _parse(
            FactoryDatasetAggregateResultV2,
            str(row["record_json"]),
            "immutable dataset aggregate",
        )
        dataset = self._load_run_by_object_id(
            connection,
            value.dataset_run_ref.object_id,
        )
        if (
            dataset.to_ref() != value.dataset_run_ref
            or row["aggregate_object_id"] != value.object_id
            or row["dataset_run_id"] != dataset.run_id
            or row["dataset_run_object_id"] != value.dataset_run_ref.object_id
            or row["core_vertical_result_object_id"] != value.core_vertical_result_ref.object_id
            or row["batch_quality_object_id"]
            != (value.batch_quality_ref.object_id if value.batch_quality_ref is not None else None)
            or row["outcome"] != value.outcome.value
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable dataset aggregate columns drifted")
        return value

    @staticmethod
    def _load_dataset_aggregate_material_ref(
        connection: sqlite3.Connection,
        object_id: str,
    ) -> ObjectRef | None:
        row = connection.execute(
            """
            SELECT *
            FROM factory_dataset_aggregate_material_refs
            WHERE aggregate_object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            return None
        return ObjectRef(
            object_type=str(row["material_object_type"]),
            object_id=str(row["material_object_id"]),
            object_version=str(row["material_object_version"]),
            object_sha256=str(row["material_object_sha256"]),
        )

    def _validate_item_binding_history(
        self,
        connection: sqlite3.Connection,
        values: tuple[FactoryItemRunBindingV2, ...],
    ) -> None:
        if tuple(value.binding_version for value in values) != tuple(range(1, len(values) + 1)):
            raise FactoryControlIntegrityError("factory item binding history has a version gap")
        dataset_run_ids = {
            self._load_run_by_object_id(
                connection,
                value.dataset_run_ref.object_id,
            ).run_id
            for value in values
        }
        if (
            len(dataset_run_ids) != 1
            or len({value.item_id for value in values}) != 1
            or len({value.item_run_ref.object_id for value in values}) != len(values)
        ):
            raise FactoryControlIntegrityError("factory item binding history crosses authority")
        for index, value in enumerate(values):
            expected = None if index == 0 else values[index - 1].to_ref()
            if value.predecessor_binding_ref != expected:
                raise FactoryControlIntegrityError("factory item binding predecessor drifted")

    @staticmethod
    def _validate_item_stage_history(
        values: tuple[FactoryItemStageHeadV2, ...],
    ) -> None:
        if tuple(value.stage_version for value in values) != tuple(range(1, len(values) + 1)):
            raise FactoryControlIntegrityError("factory item stage history has a version gap")
        if (
            len({value.item_binding_ref for value in values}) != 1
            or len({value.stage for value in values}) != 1
        ):
            raise FactoryControlIntegrityError("factory item stage history crosses authority")
        for index, value in enumerate(values):
            expected = None if index == 0 else values[index - 1].to_ref()
            if value.predecessor_head_ref != expected:
                raise FactoryControlIntegrityError("factory item stage predecessor drifted")

    def _load_current_run(
        self,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> FactoryRunV2:
        head = connection.execute(
            "SELECT * FROM factory_run_current_heads WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if head is None:
            immutable = connection.execute(
                "SELECT 1 FROM factory_runs WHERE run_id = ? LIMIT 1",
                (run_id,),
            ).fetchone()
            if immutable is not None:
                raise FactoryControlIntegrityError("factory run head is missing")
            raise FactoryControlNotFoundError(f"factory run not found: {run_id}")
        row = connection.execute(
            """
            SELECT * FROM factory_runs
            WHERE run_id = ? AND run_version = ?
            """,
            (run_id, head["run_version"]),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("factory run head points to a missing snapshot")
        run = self._parse_run_row(row)
        if head["run_object_id"] != run.object_id:
            raise FactoryControlIntegrityError("factory run head differs from immutable snapshot")
        latest = connection.execute(
            "SELECT MAX(run_version) FROM factory_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if latest is None or latest[0] != run.run_version:
            raise FactoryControlIntegrityError("factory run head is not the latest snapshot")
        return run

    def _load_planner_assessment_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> PlannerAssessmentV2:
        row = connection.execute(
            """
            SELECT * FROM planner_assessments
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError(
                "planner assessment is missing",
            )
        value = _parse(
            PlannerAssessmentV2,
            str(row["record_json"]),
            "planner assessment",
        )
        source_run = self._load_run_by_object_id(
            connection,
            value.run_ref.object_id,
        )
        if (
            row["assessment_id"] != value.assessment_id
            or row["run_id"] != source_run.run_id
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError(
                "planner assessment materialized columns drifted",
            )
        return value

    def _load_run_by_object_id(
        self,
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryRunV2:
        row = connection.execute(
            "SELECT * FROM factory_runs WHERE object_id = ?",
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("idempotency points to a missing factory run")
        return self._parse_run_row(row)

    def _parse_policy_row(self, row: sqlite3.Row) -> FactoryRunPolicyV2:
        value = _parse(FactoryRunPolicyV2, str(row["record_json"]), "factory policy")
        if (
            row["policy_id"] != value.policy_id
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("factory policy materialized columns drifted")
        return value

    def _parse_requirement_row(
        self,
        row: sqlite3.Row,
    ) -> EvaluationRequirementSpecV2:
        value = _parse(
            EvaluationRequirementSpecV2,
            str(row["record_json"]),
            "factory requirement",
        )
        if (
            row["requirement_spec_id"] != value.requirement_spec_id
            or row["run_id"] != value.run_id
            or row["requirement_version"] != value.requirement_version
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("factory requirement materialized columns drifted")
        return value

    def _parse_run_row(self, row: sqlite3.Row) -> FactoryRunV2:
        value = _parse(FactoryRunV2, str(row["record_json"]), "immutable factory run")
        if (
            row["run_id"] != value.run_id
            or row["run_version"] != value.run_version
            or row["status"] != value.status.value
            or row["policy_object_id"] != value.policy_ref.object_id
            or row["requirement_object_id"] != value.requirement_spec_ref.object_id
            or row["current_plan_object_id"]
            != (value.current_plan_ref.object_id if value.current_plan_ref else None)
            or row["compiled_plan_object_id"]
            != (value.compiled_plan_ref.object_id if value.compiled_plan_ref else None)
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable factory run materialized columns drifted")
        return value

    def _parse_plan_row(self, row: sqlite3.Row) -> DatasetBuildPlanV2:
        value = _parse(DatasetBuildPlanV2, str(row["record_json"]), "immutable factory plan")
        if (
            row["plan_id"] != value.plan_id
            or row["plan_version"] != value.plan_version
            or row["predecessor_object_id"]
            != (value.predecessor_plan_ref.object_id if value.predecessor_plan_ref else None)
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable factory plan materialized columns drifted")
        return value

    def _parse_compiled_plan_row(
        self,
        row: sqlite3.Row,
    ) -> CompiledDatasetBuildPlanV2:
        value = _parse(
            CompiledDatasetBuildPlanV2,
            str(row["record_json"]),
            "immutable compiled factory plan",
        )
        if (
            row["compiled_plan_id"] != value.compiled_plan_id
            or row["source_plan_object_id"] != value.source_plan_ref.object_id
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise FactoryControlIntegrityError("immutable compiled plan materialized columns drifted")
        return value

    @staticmethod
    def _validate_contiguous_run_history(values: tuple[FactoryRunV2, ...]) -> None:
        if tuple(value.run_version for value in values) != tuple(range(len(values))):
            raise FactoryControlIntegrityError("immutable factory run history has a version gap")
        if len({value.run_id for value in values}) != 1:
            raise FactoryControlIntegrityError("immutable factory run history crosses run IDs")

    @staticmethod
    def _run_ref_exists(
        connection: sqlite3.Connection,
        run_id: str,
        object_id: str,
    ) -> bool:
        return (
            connection.execute(
                """
                SELECT 1 FROM factory_runs
                WHERE run_id = ? AND object_id = ?
                """,
                (run_id, object_id),
            ).fetchone()
            is not None
        )

    def _fault(self, point: FactoryControlStoreFaultPoint) -> None:
        if self._fault_injector is not None:
            self._fault_injector.maybe_raise(point)


def _json(value: ContractModelV2) -> str:
    return value.canonical_json().decode()


def _parse[RecordT: ContractModelV2](
    model_type: type[RecordT],
    payload: str,
    label: str,
) -> RecordT:
    try:
        value = model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise FactoryControlIntegrityError(f"{label} JSON is invalid") from exc
    if _json(value) != payload:
        raise FactoryControlIntegrityError(f"{label} JSON is not canonical")
    return value


def _request_sha256(*values: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_record(
    payload: str,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    encoded = payload.encode()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise FactoryControlIntegrityError(f"{label} record hash drifted")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FactoryControlIntegrityError(f"{label} record JSON is invalid") from exc
    if not isinstance(value, dict):
        raise FactoryControlIntegrityError(f"{label} record JSON is not an object")
    canonical = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != payload:
        raise FactoryControlIntegrityError(f"{label} record JSON is not canonical")
    return value


def _required_ref_from_payload(
    payload: dict[str, object],
    field_name: str,
) -> ObjectRef:
    try:
        return ObjectRef.model_validate(payload[field_name])
    except (KeyError, ValidationError) as exc:
        raise FactoryControlIntegrityError("domain plan material has an invalid required ref") from exc


def _optional_ref_from_payload(
    payload: dict[str, object],
    field_name: str,
) -> ObjectRef | None:
    value = payload.get(field_name)
    if value is None:
        return None
    try:
        return ObjectRef.model_validate(value)
    except ValidationError as exc:
        raise FactoryControlIntegrityError("domain plan material has an invalid optional ref") from exc


def _object_ref_from_head(
    connection: sqlite3.Connection,
    head: sqlite3.Row,
) -> ObjectRef:
    row = connection.execute(
        """
        SELECT object_type, object_version, object_sha256
        FROM domain_plan_materials
        WHERE plan_object_id = ?
        """,
        (head["plan_object_id"],),
    ).fetchone()
    if row is None:
        raise FactoryControlIntegrityError("domain plan head points to missing material")
    return ObjectRef(
        object_type=str(row["object_type"]),
        object_id=str(head["plan_object_id"]),
        object_version=str(row["object_version"]),
        object_sha256=str(row["object_sha256"]),
    )


def _domain_plan_kind(value: object) -> PlanKindV2:
    try:
        plan_kind = PlanKindV2(str(value))
    except ValueError as exc:
        raise FactoryControlIntegrityError("domain plan kind is invalid") from exc
    if plan_kind is PlanKindV2.GLOBAL_BUILD:
        raise FactoryControlIntegrityError("global plan cannot use domain plan storage")
    return plan_kind


def _invalidated_result_kinds(
    plan_kind: PlanKindV2,
) -> tuple[PlanKindV2, ...]:
    if plan_kind in {
        PlanKindV2.TRACE_CLEANING,
        PlanKindV2.TASK_REWRITE,
    }:
        return (
            plan_kind,
            PlanKindV2.ATTACHMENT_GENERATION,
            PlanKindV2.CRITERIA_RUBRIC,
            PlanKindV2.GRADING_DESIGN,
        )
    if plan_kind is PlanKindV2.ATTACHMENT_GENERATION:
        return (
            PlanKindV2.ATTACHMENT_GENERATION,
            PlanKindV2.CRITERIA_RUBRIC,
            PlanKindV2.GRADING_DESIGN,
        )
    if plan_kind is PlanKindV2.CRITERIA_RUBRIC:
        return (
            PlanKindV2.CRITERIA_RUBRIC,
            PlanKindV2.GRADING_DESIGN,
        )
    return (plan_kind,)


__all__ = [
    "CompiledDomainPlanObject",
    "DomainPlanMaterial",
    "DomainPlanObject",
    "DomainResultMaterial",
    "DomainResultObject",
    "FactoryControlConcurrencyError",
    "FactoryControlConflictError",
    "FactoryControlHeadRebuild",
    "FactoryControlInjectedCrash",
    "FactoryControlIntegrityError",
    "FactoryControlNotFoundError",
    "FactoryControlStore",
    "FactoryControlStoreError",
    "FactoryControlStoreFaultInjector",
    "FactoryControlStoreFaultPoint",
    "StaticFactoryControlStoreFaultInjector",
]
