from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from eval_factory.approval.persistence import (
    UserDecisionPersistenceService,
)
from eval_factory.approval.revalidation import (
    AdjustmentProducerResult,
    DirectedRevalidationCompiler,
    RevalidationItemSource,
    RevalidationStageSource,
    UserPlanApplicationCompilation,
    UserPlanApplicationCompiler,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationPlanV2,
    DirectedRevalidationReportOutcomeV2,
    DirectedRevalidationReportV2,
    EnvironmentStrategyAdjustmentResultV2,
    FinalDatasetAdjustmentResultV2,
    LabelPlanAdjustmentResultV2,
    RevalidationWorkItemV2,
    RevalidationWorkOutcomeV2,
    RevalidationWorkResultV2,
    UserPlanApplicationPolicyV2,
    UserPlanApplicationV2,
    directed_revalidation_plan_v2_ref,
    environment_strategy_adjustment_result_v2_ref,
    final_dataset_adjustment_result_v2_ref,
    label_plan_adjustment_result_v2_ref,
    revalidation_work_item_v2_ref,
    revalidation_work_result_v2_ref,
    user_plan_application_policy_v2_ref,
    user_plan_application_v2_ref,
    validate_directed_revalidation_plan_v2_identity,
    validate_directed_revalidation_report_v2_identity,
    validate_revalidation_work_item_v2_identity,
    validate_revalidation_work_result_v2_identity,
    validate_user_plan_application_policy_v2_identity,
    validate_user_plan_application_v2_identity,
)
from eval_factory.contracts.approval_decision_v2 import (
    user_decision_adjustment_effect_v2_ref,
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration_v2 import (
    resolved_job_work_graph_v2_ref,
)
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import (
    JobRecord,
    StageResultRecord,
    StageRunRecord,
)


class UserPlanApplicationConflictError(ValueError):
    pass


class UserPlanApplicationPersistenceService:
    def __init__(self, store: JobStore) -> None:
        self.store = store

    def apply(
        self,
        *,
        decision_commit_id: str,
        producer_result: AdjustmentProducerResult,
        item_sources: tuple[RevalidationItemSource, ...],
        policy: UserPlanApplicationPolicyV2,
        idempotency_key: str,
        audit: ContractAudit,
        job_stage_sources: tuple[RevalidationStageSource, ...] = (),
    ) -> UserPlanApplicationCompilation:
        scope = f"apply-user-plan:{decision_commit_id}"
        with self.store._transaction() as connection:
            commit = UserDecisionPersistenceService(self.store)._load_commit(connection, decision_commit_id)
            job_spec = self.store._get_job_spec(connection, commit.job_id)
            graph = self.store._get_job_work_graph(
                connection,
                commit.job_id,
            )
            self._assert_current_heads(connection, commit.job_id)
            compiled = UserPlanApplicationCompiler().compile(
                decision_commit=commit,
                producer_result=producer_result,
                job_spec=job_spec,
                resolved_job_work_graph=graph,
                item_sources=item_sources,
                policy=policy,
                audit=audit,
                job_stage_sources=job_stage_sources,
            )
            request_sha256 = _request_sha256(
                {
                    "decision_commit_ref": (user_decision_commit_result_v2_ref(commit)),
                    "producer_refs": _producer_refs(producer_result),
                    "item_sources": _item_source_payload(item_sources),
                    "job_stage_sources": _stage_source_payload(job_stage_sources),
                    "policy_ref": user_plan_application_policy_v2_ref(policy),
                }
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "USER_PLAN_APPLICATION":
                    raise IdempotencyConflictError("user plan replay response type is corrupt")
                stored = self._load_compilation(connection, response_id)
                if user_plan_application_v2_ref(compiled.application) != user_plan_application_v2_ref(
                    stored.application
                ) or directed_revalidation_plan_v2_ref(compiled.plan) != directed_revalidation_plan_v2_ref(
                    stored.plan
                ):
                    raise ImmutableResultError(
                        "replayed user plan application differs from current authority"
                    )
                self._assert_current_heads(
                    connection,
                    stored.application.job_id,
                )
                return stored
            existing = connection.execute(
                """
                SELECT application_id
                FROM user_plan_applications
                WHERE decision_commit_id = ?
                """,
                (decision_commit_id,),
            ).fetchone()
            if existing is not None:
                raise UserPlanApplicationConflictError(
                    "adjusted decision already has a user plan application"
                )
            self._persist_policy(connection, policy)
            application = compiled.application
            plan = compiled.plan
            connection.execute(
                """
                INSERT INTO user_plan_applications (
                    application_id, job_id, decision_commit_id, policy_id,
                    checkpoint, application_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application.application_id,
                    application.job_id,
                    decision_commit_id,
                    policy.policy_id,
                    application.checkpoint.value,
                    application.application_sha256,
                    self.store._record_json(application),
                ),
            )
            connection.execute(
                """
                INSERT INTO directed_revalidation_plans (
                    plan_id, application_id, resolved_job_work_graph_id,
                    plan_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    application.application_id,
                    plan.resolved_job_work_graph_ref.object_id,
                    plan.plan_sha256,
                    self.store._record_json(plan),
                ),
            )
            for work in plan.work_items:
                connection.execute(
                    """
                    INSERT INTO directed_revalidation_work_items (
                        work_item_id, plan_id, application_id, item_id,
                        stage, work_sha256, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        work.work_item_id,
                        plan.plan_id,
                        application.application_id,
                        work.item_id,
                        work.stage.value,
                        work.work_sha256,
                        self.store._record_json(work),
                    ),
                )
            for ref in application.invalidated_object_refs:
                self._insert_validity_event(
                    connection,
                    application=application,
                    ref=ref,
                    event_kind="INVALIDATED",
                    report_id=None,
                )
                self._delete_current_head(
                    connection,
                    job_id=application.job_id,
                    ref=ref,
                )
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                application.job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=application.job_id,
                aggregate_version=job.row_version,
                event_type="user-plan-application-created",
                attributes=_attributes(
                    application_id=application.application_id,
                    checkpoint=application.checkpoint.value,
                    invalidated_count=len(application.invalidated_object_refs),
                    work_count=plan.work_item_count,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_PLAN_APPLICATION",
                response_id=application.application_id,
                created_at=self.store._clock(),
            )
            return compiled

    def record_work_result(
        self,
        *,
        application_id: str,
        work_item_id: str,
        stage_run_id: str,
        policy: UserPlanApplicationPolicyV2,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> RevalidationWorkResultV2:
        scope = f"record-revalidation-result:{work_item_id}"
        with self.store._transaction() as connection:
            application = self._load_application(
                connection,
                application_id,
            )
            self._assert_current_heads(connection, application.job_id)
            plan = self._load_plan_for_application(
                connection,
                application_id,
            )
            work = self._load_work_item(connection, work_item_id)
            stage_run = self.store._get_record(
                connection,
                "stage_runs",
                "stage_run_id",
                stage_run_id,
                StageRunRecord,
            )
            stage_result = self._load_stage_result_for_run(
                connection,
                stage_run_id,
            )
            stored_policy = self._load_policy(
                connection,
                application.policy_ref.object_id,
            )
            if (
                user_plan_application_policy_v2_ref(policy) != application.policy_ref
                or policy != stored_policy
            ):
                raise ImmutableResultError("revalidation result policy differs from the application")
            request_sha256 = _request_sha256(
                {
                    "application_ref": user_plan_application_v2_ref(application),
                    "work_item_ref": revalidation_work_item_v2_ref(work),
                    "stage_result_id": stage_result.stage_result_id,
                    "stage_result_sha256": stage_result.result_sha256,
                    "policy_ref": user_plan_application_policy_v2_ref(policy),
                }
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "REVALIDATION_WORK_RESULT":
                    raise IdempotencyConflictError("revalidation result replay response type is corrupt")
                return self._load_work_result_by_id(
                    connection,
                    response_id,
                )
            if (
                self._load_optional_work_result(
                    connection,
                    work_item_id,
                )
                is not None
            ):
                raise UserPlanApplicationConflictError("revalidation work already has a result")
            dependency_results = tuple(
                self._load_required_work_result(
                    connection,
                    dependency.object_id,
                )
                for dependency in work.depends_on_work_item_refs
            )
            result = DirectedRevalidationCompiler().record_result(
                application=application,
                plan=plan,
                work_item=work,
                stage_run=stage_run,
                stage_result=stage_result,
                dependency_results=dependency_results,
                policy=stored_policy,
                audit=audit,
            )
            connection.execute(
                """
                INSERT INTO directed_revalidation_work_results (
                    result_id, work_item_id, stage_result_id, outcome,
                    result_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.result_id,
                    work_item_id,
                    stage_result.stage_result_id,
                    result.outcome.value,
                    result.result_sha256,
                    self.store._record_json(result),
                ),
            )
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                application.job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=application.job_id,
                aggregate_version=job.row_version,
                event_type="directed-revalidation-work-recorded",
                attributes=_attributes(
                    application_id=application.application_id,
                    outcome=result.outcome.value,
                    stage=work.stage.value,
                    work_item_id=work.work_item_id,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="REVALIDATION_WORK_RESULT",
                response_id=result.result_id,
                created_at=self.store._clock(),
            )
            return result

    def complete(
        self,
        *,
        application_id: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> DirectedRevalidationReportV2:
        scope = f"complete-revalidation:{application_id}"
        with self.store._transaction() as connection:
            application = self._load_application(
                connection,
                application_id,
            )
            self._assert_current_heads(connection, application.job_id)
            plan = self._load_plan_for_application(
                connection,
                application_id,
            )
            results = self._list_work_results(
                connection,
                plan,
            )
            request_sha256 = _request_sha256(
                {
                    "application_ref": user_plan_application_v2_ref(application),
                    "plan_ref": directed_revalidation_plan_v2_ref(plan),
                    "work_result_refs": tuple(revalidation_work_result_v2_ref(result) for result in results),
                }
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "DIRECTED_REVALIDATION_REPORT":
                    raise IdempotencyConflictError("revalidation report replay response type is corrupt")
                return self._load_report_by_id(connection, response_id)
            existing = self._load_optional_report(
                connection,
                application_id,
            )
            if existing is not None:
                raise UserPlanApplicationConflictError("user plan application already has a report")
            report = DirectedRevalidationCompiler().compile_report(
                application=application,
                plan=plan,
                work_results=results,
                audit=audit,
            )
            connection.execute(
                """
                INSERT INTO directed_revalidation_reports (
                    report_id, application_id, plan_id, outcome,
                    report_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    application_id,
                    plan.plan_id,
                    report.outcome.value,
                    report.report_sha256,
                    self.store._record_json(report),
                ),
            )
            event_kind = (
                "CURRENT"
                if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
                else report.outcome.value
            )
            event_refs = (
                report.replacement_current_refs
                if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
                else report.invalidated_prior_refs
            )
            for ref in event_refs:
                self._insert_validity_event(
                    connection,
                    application=application,
                    ref=ref,
                    event_kind=event_kind,
                    report_id=report.report_id,
                )
            if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE:
                self._replace_current_heads(
                    connection,
                    application.job_id,
                )
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                application.job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=application.job_id,
                aggregate_version=job.row_version,
                event_type="directed-revalidation-completed",
                attributes=_attributes(
                    application_id=application.application_id,
                    outcome=report.outcome.value,
                    report_id=report.report_id,
                    replacement_count=len(report.replacement_current_refs),
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="DIRECTED_REVALIDATION_REPORT",
                response_id=report.report_id,
                created_at=self.store._clock(),
            )
            return report

    def get_application(
        self,
        application_id: str,
    ) -> UserPlanApplicationV2:
        with self.store._connect() as connection:
            return self._load_application(connection, application_id)

    def get_application_for_decision(
        self,
        decision_commit_id: str,
    ) -> UserPlanApplicationV2:
        with self.store._connect() as connection:
            row = connection.execute(
                """
                SELECT application_id
                FROM user_plan_applications
                WHERE decision_commit_id = ?
                """,
                (decision_commit_id,),
            ).fetchone()
            if row is None:
                raise UserPlanApplicationConflictError("user plan application not found for decision")
            return self._load_application(
                connection,
                str(row["application_id"]),
            )

    def get_plan(
        self,
        application_id: str,
    ) -> DirectedRevalidationPlanV2:
        with self.store._connect() as connection:
            return self._load_plan_for_application(
                connection,
                application_id,
            )

    def get_work_item(
        self,
        work_item_id: str,
    ) -> RevalidationWorkItemV2:
        with self.store._connect() as connection:
            return self._load_work_item(connection, work_item_id)

    def list_ready_work(
        self,
        application_id: str,
    ) -> tuple[RevalidationWorkItemV2, ...]:
        with self.store._connect() as connection:
            return self._list_ready_work_in_transaction(
                connection,
                application_id,
            )

    def _list_ready_work_in_transaction(
        self,
        connection: sqlite3.Connection,
        application_id: str,
    ) -> tuple[RevalidationWorkItemV2, ...]:
        plan = self._load_plan_for_application(
            connection,
            application_id,
        )
        if (
            self._load_optional_report(
                connection,
                application_id,
            )
            is not None
        ):
            return ()
        results = {result.work_item_ref: result for result in self._list_work_results(connection, plan)}
        return tuple(
            work
            for work in plan.work_items
            if revalidation_work_item_v2_ref(work) not in results
            and all(
                dependency in results and results[dependency].outcome is RevalidationWorkOutcomeV2.SUCCEEDED
                for dependency in work.depends_on_work_item_refs
            )
        )

    def get_work_result(
        self,
        work_item_id: str,
    ) -> RevalidationWorkResultV2:
        with self.store._connect() as connection:
            return self._load_required_work_result(
                connection,
                work_item_id,
            )

    def get_report(
        self,
        application_id: str,
    ) -> DirectedRevalidationReportV2:
        with self.store._connect() as connection:
            report = self._load_optional_report(
                connection,
                application_id,
            )
            if report is None:
                raise RecordNotFoundError(f"DirectedRevalidationReportV2 not found: {application_id}")
            return report

    def list_current_heads(
        self,
        job_id: str,
    ) -> tuple[ObjectRef, ...]:
        with self.store._connect() as connection:
            return self._assert_current_heads(connection, job_id)

    def rebuild_current_heads(
        self,
        job_id: str,
    ) -> tuple[ObjectRef, ...]:
        with self.store._transaction() as connection:
            self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            self._replace_current_heads(connection, job_id)
            return self._load_current_heads(connection, job_id)

    def _persist_policy(
        self,
        connection: sqlite3.Connection,
        policy: UserPlanApplicationPolicyV2,
    ) -> None:
        try:
            validate_user_plan_application_policy_v2_identity(policy)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("user plan application policy is malformed") from exc
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM user_plan_application_policies
            WHERE policy_id = ?
            """,
            (policy.policy_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO user_plan_application_policies (
                    policy_id, policy_sha256, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.policy_id,
                    policy.policy_sha256,
                    self.store._record_json(policy),
                ),
            )
            return
        stored = self._parse_policy_row(row)
        if stored != policy:
            raise ImmutableResultError("stored user plan application policy differs from input")

    def _load_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> UserPlanApplicationPolicyV2:
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM user_plan_application_policies
            WHERE policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("stored application is missing its policy")
        return self._parse_policy_row(row)

    @staticmethod
    def _parse_policy_row(
        row: sqlite3.Row,
    ) -> UserPlanApplicationPolicyV2:
        try:
            policy = UserPlanApplicationPolicyV2.model_validate_json(str(row["record_json"]))
            validate_user_plan_application_policy_v2_identity(policy)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored user plan application policy is malformed") from exc
        if str(row["policy_sha256"]) != policy.policy_sha256:
            raise ImmutableResultError("stored user plan application policy columns are inconsistent")
        return policy

    def _load_compilation(
        self,
        connection: sqlite3.Connection,
        application_id: str,
    ) -> UserPlanApplicationCompilation:
        return UserPlanApplicationCompilation(
            application=self._load_application(
                connection,
                application_id,
            ),
            plan=self._load_plan_for_application(
                connection,
                application_id,
            ),
        )

    def _load_application(
        self,
        connection: sqlite3.Connection,
        application_id: str,
    ) -> UserPlanApplicationV2:
        row = connection.execute(
            """
            SELECT application_id, job_id, decision_commit_id, policy_id,
                   checkpoint, application_sha256, record_json
            FROM user_plan_applications
            WHERE application_id = ?
            """,
            (application_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"UserPlanApplicationV2 not found: {application_id}")
        try:
            application = UserPlanApplicationV2.model_validate_json(str(row["record_json"]))
            validate_user_plan_application_v2_identity(application)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored user plan application is malformed") from exc
        if (
            str(row["application_id"]) != application.application_id
            or str(row["job_id"]) != application.job_id
            or str(row["decision_commit_id"]) != application.decision_commit_ref.object_id
            or str(row["policy_id"]) != application.policy_ref.object_id
            or str(row["checkpoint"]) != application.checkpoint.value
            or str(row["application_sha256"]) != application.application_sha256
        ):
            raise ImmutableResultError("stored user plan application columns are inconsistent")
        policy = self._load_policy(
            connection,
            application.policy_ref.object_id,
        )
        if user_plan_application_policy_v2_ref(policy) != (application.policy_ref):
            raise ImmutableResultError("stored application policy binding is stale")
        commit = UserDecisionPersistenceService(self.store)._load_commit(
            connection,
            application.decision_commit_ref.object_id,
        )
        if (
            user_decision_commit_result_v2_ref(commit) != application.decision_commit_ref
            or commit.job_id != application.job_id
            or commit.decision_record_ref != application.decision_record_ref
            or commit.adjustment_effect is None
            or user_decision_adjustment_effect_v2_ref(commit.adjustment_effect)
            != application.adjustment_effect_ref
        ):
            raise ImmutableResultError("stored application decision authority is stale")
        return application

    def _load_plan_for_application(
        self,
        connection: sqlite3.Connection,
        application_id: str,
    ) -> DirectedRevalidationPlanV2:
        row = connection.execute(
            """
            SELECT plan_id
            FROM directed_revalidation_plans
            WHERE application_id = ?
            """,
            (application_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("stored application is missing its revalidation plan")
        return self._load_plan_by_id(
            connection,
            str(row["plan_id"]),
        )

    def _load_plan_by_id(
        self,
        connection: sqlite3.Connection,
        plan_id: str,
    ) -> DirectedRevalidationPlanV2:
        row = connection.execute(
            """
            SELECT plan_id, application_id, resolved_job_work_graph_id,
                   plan_sha256, record_json
            FROM directed_revalidation_plans
            WHERE plan_id = ?
            """,
            (plan_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"DirectedRevalidationPlanV2 not found: {plan_id}")
        try:
            plan = DirectedRevalidationPlanV2.model_validate_json(str(row["record_json"]))
            validate_directed_revalidation_plan_v2_identity(plan)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored directed revalidation plan is malformed") from exc
        if (
            str(row["plan_id"]) != plan.plan_id
            or str(row["application_id"]) != plan.application_ref.object_id
            or str(row["resolved_job_work_graph_id"]) != plan.resolved_job_work_graph_ref.object_id
            or str(row["plan_sha256"]) != plan.plan_sha256
        ):
            raise ImmutableResultError("stored directed revalidation plan columns are inconsistent")
        application = self._load_application(
            connection,
            plan.application_ref.object_id,
        )
        if user_plan_application_v2_ref(application) != plan.application_ref:
            raise ImmutableResultError("stored revalidation plan application binding is stale")
        graph = self.store._get_job_work_graph_by_id(
            connection,
            plan.resolved_job_work_graph_ref.object_id,
        )
        if (
            graph.job_id != application.job_id
            or resolved_job_work_graph_v2_ref(graph) != plan.resolved_job_work_graph_ref
        ):
            raise ImmutableResultError("stored revalidation plan work graph binding is stale")
        rows = connection.execute(
            """
            SELECT work_item_id
            FROM directed_revalidation_work_items
            WHERE plan_id = ?
            """,
            (plan.plan_id,),
        ).fetchall()
        observed_ids = tuple(str(value["work_item_id"]) for value in rows)
        expected_ids = tuple(ref.object_id for ref in plan.work_item_refs)
        if len(observed_ids) != len(expected_ids) or set(observed_ids) != set(expected_ids):
            raise ImmutableResultError("stored revalidation work inventory differs from the plan")
        work_items = tuple(
            self._load_work_item(
                connection,
                work_item_id,
            )
            for work_item_id in expected_ids
        )
        if work_items != plan.work_items:
            raise ImmutableResultError("stored revalidation work rows differ from the plan")
        return plan

    def _load_work_item(
        self,
        connection: sqlite3.Connection,
        work_item_id: str,
    ) -> RevalidationWorkItemV2:
        row = connection.execute(
            """
            SELECT work_item_id, plan_id, application_id, item_id,
                   stage, work_sha256, record_json
            FROM directed_revalidation_work_items
            WHERE work_item_id = ?
            """,
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"RevalidationWorkItemV2 not found: {work_item_id}")
        try:
            work = RevalidationWorkItemV2.model_validate_json(str(row["record_json"]))
            validate_revalidation_work_item_v2_identity(work)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored revalidation work item is malformed") from exc
        if (
            str(row["work_item_id"]) != work.work_item_id
            or str(row["application_id"]) != work.application_ref.object_id
            or (str(row["item_id"]) if row["item_id"] is not None else None) != work.item_id
            or str(row["stage"]) != work.stage.value
            or str(row["work_sha256"]) != work.work_sha256
        ):
            raise ImmutableResultError("stored revalidation work item columns are inconsistent")
        plan_row = connection.execute(
            """
            SELECT application_id
            FROM directed_revalidation_plans
            WHERE plan_id = ?
            """,
            (str(row["plan_id"]),),
        ).fetchone()
        if plan_row is None or str(plan_row["application_id"]) != work.application_ref.object_id:
            raise ImmutableResultError("stored revalidation work plan binding is stale")
        return work

    def _load_stage_result_for_run(
        self,
        connection: sqlite3.Connection,
        stage_run_id: str,
    ) -> StageResultRecord:
        row = connection.execute(
            """
            SELECT stage_result_id, result_sha256, record_json
            FROM stage_results
            WHERE stage_run_id = ?
            """,
            (stage_run_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"StageResultRecord not found for run: {stage_run_id}")
        result = self.store._load_record(row, StageResultRecord)
        if (
            str(row["stage_result_id"]) != result.stage_result_id
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ImmutableResultError("stored StageResult columns are inconsistent")
        return result

    def _load_optional_work_result(
        self,
        connection: sqlite3.Connection,
        work_item_id: str,
    ) -> RevalidationWorkResultV2 | None:
        row = connection.execute(
            """
            SELECT result_id
            FROM directed_revalidation_work_results
            WHERE work_item_id = ?
            """,
            (work_item_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_work_result_by_id(
            connection,
            str(row["result_id"]),
        )

    def _load_required_work_result(
        self,
        connection: sqlite3.Connection,
        work_item_id: str,
    ) -> RevalidationWorkResultV2:
        result = self._load_optional_work_result(
            connection,
            work_item_id,
        )
        if result is None:
            raise RecordNotFoundError(f"RevalidationWorkResultV2 not found: {work_item_id}")
        return result

    def _load_work_result_by_id(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> RevalidationWorkResultV2:
        row = connection.execute(
            """
            SELECT result_id, work_item_id, stage_result_id,
                   outcome, result_sha256, record_json
            FROM directed_revalidation_work_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"RevalidationWorkResultV2 not found: {result_id}")
        try:
            result = RevalidationWorkResultV2.model_validate_json(str(row["record_json"]))
            validate_revalidation_work_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored revalidation work result is malformed") from exc
        if (
            str(row["result_id"]) != result.result_id
            or str(row["work_item_id"]) != result.work_item_ref.object_id
            or str(row["stage_result_id"]) != result.stage_result_ref.object_id
            or str(row["outcome"]) != result.outcome.value
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ImmutableResultError("stored revalidation work result columns are inconsistent")
        work = self._load_work_item(
            connection,
            result.work_item_ref.object_id,
        )
        if revalidation_work_item_v2_ref(work) != result.work_item_ref:
            raise ImmutableResultError("stored work result work binding is stale")
        stage_result = self.store._get_record(
            connection,
            "stage_results",
            "stage_result_id",
            result.stage_result_ref.object_id,
            StageResultRecord,
        )
        if stage_result.result_sha256 != result.stage_result_ref.object_sha256:
            raise ImmutableResultError("stored work result StageResult binding is stale")
        return result

    def _list_work_results(
        self,
        connection: sqlite3.Connection,
        plan: DirectedRevalidationPlanV2,
    ) -> tuple[RevalidationWorkResultV2, ...]:
        results: list[RevalidationWorkResultV2] = []
        for work in plan.work_items:
            result = self._load_optional_work_result(
                connection,
                work.work_item_id,
            )
            if result is not None:
                results.append(result)
        return tuple(results)

    def _load_optional_report(
        self,
        connection: sqlite3.Connection,
        application_id: str,
    ) -> DirectedRevalidationReportV2 | None:
        row = connection.execute(
            """
            SELECT report_id
            FROM directed_revalidation_reports
            WHERE application_id = ?
            """,
            (application_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_report_by_id(
            connection,
            str(row["report_id"]),
        )

    def _load_report_by_id(
        self,
        connection: sqlite3.Connection,
        report_id: str,
    ) -> DirectedRevalidationReportV2:
        row = connection.execute(
            """
            SELECT report_id, application_id, plan_id, outcome,
                   report_sha256, record_json
            FROM directed_revalidation_reports
            WHERE report_id = ?
            """,
            (report_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"DirectedRevalidationReportV2 not found: {report_id}")
        try:
            report = DirectedRevalidationReportV2.model_validate_json(str(row["record_json"]))
            validate_directed_revalidation_report_v2_identity(report)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored directed revalidation report is malformed") from exc
        if (
            str(row["report_id"]) != report.report_id
            or str(row["application_id"]) != report.application_ref.object_id
            or str(row["plan_id"]) != report.plan_ref.object_id
            or str(row["outcome"]) != report.outcome.value
            or str(row["report_sha256"]) != report.report_sha256
        ):
            raise ImmutableResultError("stored directed revalidation report columns are inconsistent")
        application = self._load_application(
            connection,
            report.application_ref.object_id,
        )
        plan = self._load_plan_by_id(
            connection,
            report.plan_ref.object_id,
        )
        if (
            user_plan_application_v2_ref(application) != report.application_ref
            or directed_revalidation_plan_v2_ref(plan) != report.plan_ref
            or self._list_work_results(connection, plan) != report.work_results
        ):
            raise ImmutableResultError("stored report authority differs from immutable rows")
        return report

    def _insert_validity_event(
        self,
        connection: sqlite3.Connection,
        *,
        application: UserPlanApplicationV2,
        ref: ObjectRef,
        event_kind: str,
        report_id: str | None,
    ) -> None:
        digest = _request_sha256(
            {
                "application_ref": user_plan_application_v2_ref(application),
                "report_id": report_id,
                "event_kind": event_kind,
                "object_ref": ref,
            }
        )
        connection.execute(
            """
            INSERT INTO object_validity_events (
                validity_event_id, job_id, application_id, report_id,
                event_kind, object_type, object_id, object_version,
                object_sha256, object_ref_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"object-validity-event://sha256/{digest}",
                application.job_id,
                application.application_id,
                report_id,
                event_kind,
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
                self.store._record_json(ref),
            ),
        )

    @staticmethod
    def _delete_current_head(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        ref: ObjectRef,
    ) -> None:
        connection.execute(
            """
            DELETE FROM object_current_heads
            WHERE job_id = ? AND object_type = ? AND object_id = ?
              AND object_version = ? AND object_sha256 = ?
            """,
            (
                job_id,
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )

    def _expected_current_heads(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> dict[ObjectRef, str]:
        invalidated: set[ObjectRef] = set()
        applications: dict[str, UserPlanApplicationV2] = {}
        application_rows = connection.execute(
            """
            SELECT application_id
            FROM user_plan_applications
            WHERE job_id = ?
            ORDER BY application_id
            """,
            (job_id,),
        ).fetchall()
        for row in application_rows:
            application = self._load_application(
                connection,
                str(row["application_id"]),
            )
            applications[application.application_id] = application
            invalidated.update(application.invalidated_object_refs)
        replacements: dict[ObjectRef, str] = {}
        reports: dict[str, DirectedRevalidationReportV2] = {}
        report_rows = connection.execute(
            """
            SELECT report_id
            FROM directed_revalidation_reports
            WHERE application_id IN (
                SELECT application_id
                FROM user_plan_applications
                WHERE job_id = ?
            )
            ORDER BY report_id
            """,
            (job_id,),
        ).fetchall()
        for row in report_rows:
            report = self._load_report_by_id(
                connection,
                str(row["report_id"]),
            )
            reports[report.report_id] = report
            if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE:
                for ref in report.replacement_current_refs:
                    replacements[ref] = report.report_id
        self._validate_validity_events(
            connection,
            job_id=job_id,
            applications=applications,
            reports=reports,
        )
        return {ref: report_id for ref, report_id in replacements.items() if ref not in invalidated}

    def _validate_validity_events(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        applications: dict[str, UserPlanApplicationV2],
        reports: dict[str, DirectedRevalidationReportV2],
    ) -> None:
        expected: set[tuple[str, str | None, str, ObjectRef]] = set()
        for application in applications.values():
            expected.update(
                (
                    application.application_id,
                    None,
                    "INVALIDATED",
                    ref,
                )
                for ref in application.invalidated_object_refs
            )
        for report in reports.values():
            event_kind = (
                "CURRENT"
                if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
                else report.outcome.value
            )
            refs = (
                report.replacement_current_refs
                if report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
                else report.invalidated_prior_refs
            )
            expected.update(
                (
                    report.application_ref.object_id,
                    report.report_id,
                    event_kind,
                    ref,
                )
                for ref in refs
            )
        rows = connection.execute(
            """
            SELECT validity_event_id, application_id, report_id, event_kind,
                   object_type, object_id, object_version, object_sha256,
                   object_ref_json
            FROM object_validity_events
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchall()
        observed: set[tuple[str, str | None, str, ObjectRef]] = set()
        for row in rows:
            try:
                ref = ObjectRef.model_validate_json(str(row["object_ref_json"]))
            except ValidationError as exc:
                raise ImmutableResultError("stored object validity event is malformed") from exc
            application_id = str(row["application_id"])
            report_id = str(row["report_id"]) if row["report_id"] is not None else None
            event_kind = str(row["event_kind"])
            if (
                application_id not in applications
                or (report_id is not None and report_id not in reports)
                or str(row["object_type"]) != ref.object_type
                or str(row["object_id"]) != ref.object_id
                or str(row["object_version"]) != ref.object_version
                or str(row["object_sha256"]) != ref.object_sha256
            ):
                raise ImmutableResultError("stored object validity event columns are inconsistent")
            digest = _request_sha256(
                {
                    "application_ref": user_plan_application_v2_ref(applications[application_id]),
                    "report_id": report_id,
                    "event_kind": event_kind,
                    "object_ref": ref,
                }
            )
            if str(row["validity_event_id"]) != (f"object-validity-event://sha256/{digest}"):
                raise ImmutableResultError("stored object validity event identity is stale")
            observed.add(
                (
                    application_id,
                    report_id,
                    event_kind,
                    ref,
                )
            )
        if len(rows) != len(observed) or observed != expected:
            raise ImmutableResultError("stored object validity events differ from immutable authority")

    def _replace_current_heads(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> None:
        expected = self._expected_current_heads(connection, job_id)
        connection.execute(
            "DELETE FROM object_current_heads WHERE job_id = ?",
            (job_id,),
        )
        for ref, report_id in sorted(
            expected.items(),
            key=lambda value: _ref_key(value[0]),
        ):
            connection.execute(
                """
                INSERT INTO object_current_heads (
                    job_id, object_type, object_id, object_version,
                    object_sha256, report_id, object_ref_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                    report_id,
                    self.store._record_json(ref),
                ),
            )

    def _assert_current_heads(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[ObjectRef, ...]:
        observed = self._load_current_heads(connection, job_id)
        expected = tuple(
            sorted(
                self._expected_current_heads(connection, job_id),
                key=_ref_key,
            )
        )
        if observed != expected:
            raise ImmutableResultError("stored object current heads differ from immutable reports")
        return observed

    def _load_current_heads(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[ObjectRef, ...]:
        rows = connection.execute(
            """
            SELECT object_type, object_id, object_version, object_sha256,
                   object_ref_json
            FROM object_current_heads
            WHERE job_id = ?
            ORDER BY object_type, object_id, object_version, object_sha256
            """,
            (job_id,),
        ).fetchall()
        values: list[ObjectRef] = []
        for row in rows:
            try:
                ref = ObjectRef.model_validate_json(str(row["object_ref_json"]))
            except ValidationError as exc:
                raise ImmutableResultError("stored object current head is malformed") from exc
            if (
                str(row["object_type"]) != ref.object_type
                or str(row["object_id"]) != ref.object_id
                or str(row["object_version"]) != ref.object_version
                or str(row["object_sha256"]) != ref.object_sha256
            ):
                raise ImmutableResultError("stored object current head columns are inconsistent")
            values.append(ref)
        return tuple(values)


def _producer_refs(
    value: AdjustmentProducerResult,
) -> tuple[ObjectRef, ...]:
    if isinstance(value, LabelPlanAdjustmentResultV2):
        return (label_plan_adjustment_result_v2_ref(value),)
    if isinstance(value, EnvironmentStrategyAdjustmentResultV2):
        return (environment_strategy_adjustment_result_v2_ref(value),)
    if isinstance(value, FinalDatasetAdjustmentResultV2):
        return (final_dataset_adjustment_result_v2_ref(value),)
    adjustment = value.adjustment_result
    application = value.application_result
    return (
        ObjectRef(
            object_type="task-rewrite-adjustment-result",
            object_id=adjustment.result_id,
            object_version=adjustment.policy_version,
            object_sha256=adjustment.result_sha256,
        ),
        ObjectRef(
            object_type="task-rewrite-application-result",
            object_id=application.result_id,
            object_version=application.policy_version,
            object_sha256=application.result_sha256,
        ),
    )


def _item_source_payload(
    values: tuple[RevalidationItemSource, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "item_id": value.item_id,
            "source_trace_ref": value.source_trace_ref,
            "authority_refs": value.authority_refs,
            "environment_requirement_ids": (value.environment_requirement_ids),
            "stages": _stage_source_payload(value.stages),
        }
        for value in values
    )


def _stage_source_payload(
    values: tuple[RevalidationStageSource, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "stage": value.stage.value,
            "output_refs": value.output_refs,
        }
        for value in values
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


__all__ = [
    "UserPlanApplicationConflictError",
    "UserPlanApplicationPersistenceService",
]
