from __future__ import annotations

import sqlite3
from dataclasses import replace

from pydantic import ValidationError

from env_mock_agent.facade import ExecutionTelemetryV2
from eval_factory.approval.application_persistence import (
    UserPlanApplicationPersistenceService,
)
from eval_factory.approval.persistence import UserDecisionPersistenceService
from eval_factory.contracts.approval_application_v2 import (
    directed_revalidation_report_v2_ref,
    user_plan_application_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.approval_v2 import user_approval_policy_ref
from eval_factory.contracts.attachment_v2 import (
    attachment_reconstruction_result_v2_ref,
)
from eval_factory.contracts.batch_quality_v2 import batch_quality_report_v2_ref
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.labeling_v2 import label_decision_ref
from eval_factory.contracts.orchestration import ItemStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    WorkControlPolicyV2,
    WorkLeaseV2,
    WorkUnitScopeV2,
    dataset_job_spec_v2_ref,
    resolved_job_work_graph_v2_ref,
    resolved_work_unit_v2_ref,
    work_control_policy_v2_ref,
    work_lease_v2_ref,
)
from eval_factory.contracts.production_release_v2 import (
    ProductionPublishedItemProjectionV2,
    validate_production_published_item_projection_v2_identity,
)
from eval_factory.contracts.quality_v2 import item_quality_compilation_result_ref
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    EvaluationItemReleaseSubjectV2,
    ItemReleaseProjectionV2,
    ReleaseProjectionPhaseV2,
    ReleaseProjectionPolicyV2,
    ReleaseProjectionResultV2,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
    validate_evaluation_item_release_subject_v2_identity,
    validate_evaluation_item_v2_identity,
    validate_item_release_projection_v2_identity,
    validate_query_spec_identity,
    validate_release_decision_v2_identity,
    validate_release_projection_policy_v2_identity,
    validate_release_projection_result_v2_identity,
)
from eval_factory.contracts.release_publication_v2 import (
    NonProductionRegistryEntryV2,
    PublishedItemProjectionV2,
    ReleasePublicationResultV2,
    validate_nonproduction_registry_entry_v2_identity,
    validate_published_item_projection_v2_identity,
    validate_release_publication_result_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_ref,
    task_draft_ref,
    task_prompt_safety_gate_ref,
)
from eval_factory.dataset.release import (
    EvaluationItemReleaseSource,
    ReleaseProjectionCompiler,
    ReleaseProjectionPendingError,
    ReleaseProjectionPolicyError,
)
from eval_factory.orchestration.job_store import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import (
    ItemRecord,
    JobRecord,
    StageResultRecord,
    stage_result_record_ref,
)


class ReleaseProjectionConflictError(ValueError):
    pass


class ReleaseProjectionIntegrityError(ImmutableResultError):
    pass


class ReleaseProjectionPersistenceService:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.compiler = ReleaseProjectionCompiler()

    def request_release(
        self,
        *,
        source: EvaluationItemReleaseSource,
        policy: ReleaseProjectionPolicyV2,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> ReleaseProjectionResultV2:
        scope = f"request-release:{source.item.item_id}"
        with self.store._transaction() as connection:
            current_source = self._load_authoritative_source(connection, source)
            request_sha256 = _source_request_sha256(
                current_source,
                policy=policy,
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "RELEASE_PROJECTION_CANDIDATE":
                    raise IdempotencyConflictError("release candidate replay response type is corrupt")
                stored = self._load_result(connection, response_id)
                if stored.phase is not ReleaseProjectionPhaseV2.CANDIDATE:
                    raise ReleaseProjectionIntegrityError(
                        "release candidate replay resolved to a non-candidate result"
                    )
                previous = self._load_previous_result(connection, stored)
                self.compiler.validate_current(
                    stored,
                    source=current_source,
                    policy=policy,
                    previous_result=previous,
                    audit=stored.audit,
                )
                self._assert_current_result(connection, stored)
                return stored

            current = self._load_current_result_optional(
                connection,
                current_source.item.item_id,
            )
            if current is not None and self._is_current_for_source(
                connection,
                current,
                source=current_source,
                policy=policy,
            ):
                raise ReleaseProjectionConflictError(
                    "the current release projection already represents this authority"
                )
            compiled = self.compiler.compile_candidate(
                source=current_source,
                policy=policy,
                previous_result=current,
                audit=audit,
            ).result
            self._assert_quality_stage_witnesses(
                connection,
                source=current_source,
            )
            self._persist_result(
                connection,
                result=compiled,
                policy=policy,
                stage_result_id=None,
            )
            projected_item = self._project_item(
                connection,
                current_source.item,
                compiled.item_projection.item_status,
            )
            self._publish_current_projection(
                connection,
                result=compiled,
                expected_current=current,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="ITEM",
                aggregate_id=compiled.release_subject.item_id,
                aggregate_version=projected_item.row_version,
                event_type="item-release-candidate-projected",
                attributes=_attributes(
                    pending_checkpoint_count=len(compiled.item_projection.pending_checkpoints),
                    projection_id=compiled.item_projection.projection_id,
                    release_state=compiled.item_projection.release_state.value,
                    result_id=compiled.result_id,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="RELEASE_PROJECTION_CANDIDATE",
                response_id=compiled.result_id,
                created_at=self.store._clock(),
            )
            return compiled

    def finalize_job(
        self,
        *,
        sources: tuple[EvaluationItemReleaseSource, ...],
        policy: ReleaseProjectionPolicyV2,
        lease: WorkLeaseV2,
        work_control_policy: WorkControlPolicyV2,
        holder_ref: ObjectRef,
        expected_lease_version: int,
        idempotency_key: str,
        audit: ContractAudit,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> tuple[ReleaseProjectionResultV2, ...]:
        if not sources:
            raise ReleaseProjectionConflictError("release finalization requires at least one Item source")
        job_ids = {source.item.job_id for source in sources}
        if len(job_ids) != 1:
            raise ReleaseProjectionConflictError("release finalization sources must belong to one Job")
        job_id = next(iter(job_ids))
        scope = f"finalize-release-job:{job_id}"
        with self.store._transaction() as connection:
            current_sources = tuple(self._load_authoritative_source(connection, source) for source in sources)
            current_sources = tuple(sorted(current_sources, key=lambda value: value.item.item_id))
            request_sha256 = _request_sha256(
                {
                    "sources": [_source_authority_payload(source) for source in current_sources],
                    "policy_ref": policy.to_ref(),
                    "work_lease_ref": work_lease_v2_ref(lease),
                    "work_control_policy_ref": work_control_policy_v2_ref(work_control_policy),
                    "holder_ref": holder_ref,
                    "expected_lease_version": expected_lease_version,
                }
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, _response_id = prior
                if response_type != "RELEASE_JOB_FINALIZATION":
                    raise IdempotencyConflictError("release finalization replay response type is corrupt")
                return self._load_terminal_results_for_sources(
                    connection,
                    current_sources,
                    lease=lease,
                )

            candidates = self._load_exact_job_candidates(
                connection,
                current_sources,
            )
            stored_lease, stored_work_policy = self._require_release_lease(
                connection,
                job_id=job_id,
                lease=lease,
                work_control_policy=work_control_policy,
            )
            terminals: list[ReleaseProjectionResultV2] = []
            for source, candidate in zip(
                current_sources,
                candidates,
                strict=True,
            ):
                terminal = self.compiler.finalize(
                    candidate=candidate,
                    source=source,
                    policy=policy,
                    audit=audit,
                ).result
                self._persist_result(
                    connection,
                    result=terminal,
                    policy=policy,
                    stage_result_id=None,
                )
                projected_item = self._project_item(
                    connection,
                    source.item,
                    terminal.item_projection.item_status,
                )
                self._publish_current_projection(
                    connection,
                    result=terminal,
                    expected_current=candidate,
                )
                self.store._append_outbox(
                    connection,
                    aggregate_type="ITEM",
                    aggregate_id=source.item.item_id,
                    aggregate_version=projected_item.row_version,
                    event_type="item-release-terminal-projected",
                    attributes=_attributes(
                        projection_id=terminal.item_projection.projection_id,
                        release_state=terminal.item_projection.release_state.value,
                        result_id=terminal.result_id,
                    ),
                )
                terminals.append(terminal)

            output_refs = tuple(release_projection_result_v2_ref(result) for result in terminals)
            lease_event = self.store._complete_stage_work_lease(
                lease=stored_lease,
                policy=stored_work_policy,
                holder_ref=holder_ref,
                expected_lease_version=expected_lease_version,
                status=StageRunStatus.SUCCEEDED,
                output_refs=output_refs,
                failure=None,
                checkpoint_ref=None,
                metrics_ref=None,
                audit=audit,
                idempotency_key=f"{idempotency_key}:release-stage",
                connection=connection,
                control_authorizations=frozenset(),
                metrics_telemetry=telemetry,
            )
            if stored_lease.stage_run_ref is None:
                raise ReleaseProjectionIntegrityError("release lease is missing its controlled StageRun")
            stage_result = self._load_stage_result_for_run(
                connection,
                stored_lease.stage_run_ref.object_id,
            )
            for terminal in terminals:
                cursor = connection.execute(
                    """
                    UPDATE release_projection_results
                    SET stage_result_id = ?
                    WHERE result_id = ? AND stage_result_id IS NULL
                    """,
                    (stage_result.stage_result_id, terminal.result_id),
                )
                if cursor.rowcount != 1:
                    raise ConcurrencyConflictError("concurrent release StageResult binding")
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job_id,
                aggregate_version=job.row_version,
                event_type="release-job-finalized",
                attributes=_attributes(
                    approved_count=sum(
                        result.release_decision.state is ReleaseStateV2.APPROVED for result in terminals
                    ),
                    item_count=len(terminals),
                    rejected_count=sum(
                        result.release_decision.state is ReleaseStateV2.REJECTED for result in terminals
                    ),
                    stage_result_id=stage_result.stage_result_id,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="RELEASE_JOB_FINALIZATION",
                response_id=lease_event.work_lease_event_id,
                created_at=self.store._clock(),
            )
            return tuple(terminals)

    def get_result(self, result_id: str) -> ReleaseProjectionResultV2:
        with self.store._connect() as connection:
            return self._load_result(connection, result_id)

    def get_current_result(self, item_id: str) -> ReleaseProjectionResultV2:
        with self.store._connect() as connection:
            result = self._load_current_result_optional(connection, item_id)
            if result is None:
                raise RecordNotFoundError(f"current release projection not found: {item_id}")
            return result

    def list_item_results(
        self,
        item_id: str,
    ) -> tuple[ReleaseProjectionResultV2, ...]:
        with self.store._connect() as connection:
            rows = connection.execute(
                """
                SELECT results.result_id
                FROM release_projection_results AS results
                JOIN item_release_projection_events AS projections
                  ON projections.projection_id = results.projection_id
                WHERE results.item_id = ?
                ORDER BY projections.projection_revision
                """,
                (item_id,),
            ).fetchall()
            return tuple(self._load_result(connection, str(row["result_id"])) for row in rows)

    def rebuild_item_projection(
        self,
        item_id: str,
    ) -> ReleaseProjectionResultV2:
        with self.store._transaction() as connection:
            item = self._load_item(connection, item_id)
            rows = connection.execute(
                """
                SELECT projection_id
                FROM item_release_projection_events
                WHERE item_id = ?
                ORDER BY projection_revision
                """,
                (item_id,),
            ).fetchall()
            if not rows:
                raise RecordNotFoundError(f"release projection history not found: {item_id}")
            results: list[ReleaseProjectionResultV2] = []
            previous: ReleaseProjectionResultV2 | None = None
            for expected_revision, row in enumerate(rows, start=1):
                projection = self._load_projection(
                    connection,
                    str(row["projection_id"]),
                )
                if (
                    projection.projection_revision != expected_revision
                    or projection.job_id != item.job_id
                    or projection.item_id != item.item_id
                    or projection.previous_projection_ref
                    != (previous.item_projection_ref if previous is not None else None)
                ):
                    raise ReleaseProjectionIntegrityError("release projection history is not contiguous")
                result_row = connection.execute(
                    """
                    SELECT result_id
                    FROM release_projection_results
                    WHERE projection_id = ?
                    """,
                    (projection.projection_id,),
                ).fetchone()
                if result_row is None:
                    raise ReleaseProjectionIntegrityError("release projection event is missing its result")
                result = self._load_result(
                    connection,
                    str(result_row["result_id"]),
                )
                if result.item_projection != projection or result.previous_result_ref != (
                    previous.to_ref() if previous is not None else None
                ):
                    raise ReleaseProjectionIntegrityError("release result history is not contiguous")
                results.append(result)
                previous = result

            current = results[-1]
            connection.execute(
                "DELETE FROM item_release_current_projections WHERE item_id = ?",
                (item_id,),
            )
            self._insert_current_projection(connection, current)
            if item.status is not current.item_projection.item_status:
                self._rewrite_item_projection(
                    connection,
                    item,
                    current.item_projection.item_status,
                )
            return current

    def _load_authoritative_source(
        self,
        connection: sqlite3.Connection,
        source: EvaluationItemReleaseSource,
    ) -> EvaluationItemReleaseSource:
        job_spec = self.store._get_job_spec(connection, source.item.job_id)
        graph = self.store._get_job_work_graph(connection, source.item.job_id)
        item = self._load_item(connection, source.item.item_id)
        if (
            job_spec != source.job_spec
            or graph != source.resolved_job_work_graph
            or item.job_id != source.item.job_id
        ):
            raise ReleaseProjectionIntegrityError(
                "release source differs from persisted Job, Item, or work graph"
            )

        decisions = tuple(
            UserDecisionPersistenceService(self.store)._load_commit(
                connection,
                user_decision_commit_result_v2_ref(value).object_id,
            )
            for value in source.decision_commits
        )
        if tuple(user_decision_commit_result_v2_ref(value) for value in decisions) != tuple(
            user_decision_commit_result_v2_ref(value) for value in source.decision_commits
        ):
            raise ReleaseProjectionIntegrityError("release decision commits differ from persisted authority")

        application_service = UserPlanApplicationPersistenceService(self.store)
        application_rows = connection.execute(
            """
            SELECT application_id
            FROM user_plan_applications
            WHERE job_id = ?
            ORDER BY application_id
            """,
            (item.job_id,),
        ).fetchall()
        applications = tuple(
            sorted(
                (
                    application
                    for application in (
                        application_service._load_application(
                            connection,
                            str(row["application_id"]),
                        )
                        for row in application_rows
                    )
                    if item.item_id in application.affected_item_ids
                ),
                key=lambda value: (
                    user_plan_application_v2_ref(value).object_type,
                    user_plan_application_v2_ref(value).object_id,
                    user_plan_application_v2_ref(value).object_version,
                    user_plan_application_v2_ref(value).object_sha256,
                ),
            )
        )
        application_refs = tuple(user_plan_application_v2_ref(value) for value in applications)
        if application_refs != source.relevant_revalidation_application_refs:
            raise ReleaseProjectionIntegrityError("release revalidation application inventory is not exact")
        reports = tuple(
            sorted(
                (
                    report
                    for report in (
                        application_service._load_optional_report(
                            connection,
                            application.application_id,
                        )
                        for application in applications
                    )
                    if report is not None
                ),
                key=lambda value: (
                    directed_revalidation_report_v2_ref(value).object_type,
                    directed_revalidation_report_v2_ref(value).object_id,
                    directed_revalidation_report_v2_ref(value).object_version,
                    directed_revalidation_report_v2_ref(value).object_sha256,
                ),
            )
        )
        if tuple(directed_revalidation_report_v2_ref(value) for value in reports) != tuple(
            directed_revalidation_report_v2_ref(value) for value in source.revalidation_reports
        ):
            raise ReleaseProjectionIntegrityError("release revalidation report inventory is not exact")
        current_heads = application_service._assert_current_heads(
            connection,
            item.job_id,
        )
        if current_heads != source.current_head_refs:
            raise ReleaseProjectionIntegrityError("release current-head inventory is not exact")
        return replace(
            source,
            job_spec=job_spec,
            item=item,
            resolved_job_work_graph=graph,
            decision_commits=decisions,
            relevant_revalidation_application_refs=application_refs,
            revalidation_reports=reports,
            current_head_refs=current_heads,
        )

    def _is_current_for_source(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
        *,
        source: EvaluationItemReleaseSource,
        policy: ReleaseProjectionPolicyV2,
    ) -> bool:
        previous = self._load_previous_result(connection, result)
        try:
            self.compiler.validate_current(
                result,
                source=source,
                policy=policy,
                previous_result=previous,
                audit=result.audit,
            )
        except (ReleaseProjectionPolicyError, ReleaseProjectionPendingError):
            return False
        return True

    def _load_previous_result(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> ReleaseProjectionResultV2 | None:
        if result.previous_result_ref is None:
            return None
        previous = self._load_result(
            connection,
            result.previous_result_ref.object_id,
        )
        if previous.to_ref() != result.previous_result_ref:
            raise ReleaseProjectionIntegrityError("release result predecessor binding is stale")
        return previous

    def _assert_quality_stage_witnesses(
        self,
        connection: sqlite3.Connection,
        *,
        source: EvaluationItemReleaseSource,
    ) -> None:
        item_unit = next(
            (
                unit
                for unit in source.resolved_job_work_graph.work_units
                if unit.stage is StageNameV2.ITEM_QUALITY and unit.item_id == source.item.item_id
            ),
            None,
        )
        batch_unit = next(
            (
                unit
                for unit in source.resolved_job_work_graph.work_units
                if unit.stage is StageNameV2.BATCH_QUALITY and unit.scope is WorkUnitScopeV2.JOB
            ),
            None,
        )
        if item_unit is None or batch_unit is None:
            raise ReleaseProjectionIntegrityError("release source is missing Item or Batch quality work")
        expected = (
            (
                item_unit.resolved_work_unit_id,
                item_quality_compilation_result_ref(source.item_quality),
            ),
            (
                batch_unit.resolved_work_unit_id,
                batch_quality_report_v2_ref(source.batch_quality),
            ),
        )
        report_results = tuple(
            result for report in source.revalidation_reports for result in report.work_results
        )
        for work_unit_id, expected_ref in expected:
            if self._has_successful_stage_output(
                connection,
                work_unit_id=work_unit_id,
                expected_ref=expected_ref,
            ):
                continue
            if any(
                result.outcome.value == "SUCCEEDED"
                and expected_ref in result.output_refs
                and self._stored_stage_result_matches_ref(
                    connection,
                    result.stage_result_ref,
                )
                for result in report_results
            ):
                continue
            raise ReleaseProjectionIntegrityError(
                "release quality authority lacks a successful StageResult witness"
            )

    def _has_successful_stage_output(
        self,
        connection: sqlite3.Connection,
        *,
        work_unit_id: str,
        expected_ref: ObjectRef,
    ) -> bool:
        rows = connection.execute(
            """
            SELECT results.record_json
            FROM work_leases AS leases
            JOIN controlled_stage_runs AS controlled
              ON controlled.work_lease_id = leases.work_lease_id
            JOIN stage_results AS results
              ON results.stage_run_id = controlled.stage_run_id
            WHERE leases.resolved_work_unit_id = ?
            ORDER BY leases.attempt DESC
            """,
            (work_unit_id,),
        ).fetchall()
        for row in rows:
            try:
                result = StageResultRecord.model_validate_json(str(row["record_json"]))
            except ValidationError as exc:
                raise ReleaseProjectionIntegrityError("stored quality StageResult is malformed") from exc
            if result.status is StageRunStatus.SUCCEEDED and expected_ref in result.output_refs:
                return True
        return False

    def _stored_stage_result_matches_ref(
        self,
        connection: sqlite3.Connection,
        ref: ObjectRef,
    ) -> bool:
        try:
            result = self.store._get_record(
                connection,
                "stage_results",
                "stage_result_id",
                ref.object_id,
                StageResultRecord,
            )
        except RecordNotFoundError:
            return False
        return stage_result_record_ref(result) == ref

    def _load_exact_job_candidates(
        self,
        connection: sqlite3.Connection,
        sources: tuple[EvaluationItemReleaseSource, ...],
    ) -> tuple[ReleaseProjectionResultV2, ...]:
        source_ids = tuple(source.item.item_id for source in sources)
        if source_ids != tuple(sorted(set(source_ids))):
            raise ReleaseProjectionConflictError(
                "release finalization Item sources must be sorted and unique"
            )
        active_rows = connection.execute(
            """
            SELECT item_id
            FROM items
            WHERE job_id = ? AND status IN (?, ?)
            ORDER BY item_id
            """,
            (
                sources[0].item.job_id,
                ItemStatus.RUNNING.value,
                ItemStatus.NEEDS_REVIEW.value,
            ),
        ).fetchall()
        active_item_ids = tuple(str(row["item_id"]) for row in active_rows)
        if source_ids != active_item_ids:
            raise ReleaseProjectionConflictError(
                "release finalization must cover every non-terminal Job Item"
            )
        rows = connection.execute(
            """
            SELECT result_id, item_id
            FROM item_release_current_projections
            WHERE job_id = ?
            ORDER BY item_id
            """,
            (sources[0].item.job_id,),
        ).fetchall()
        observed_ids = tuple(str(row["item_id"]) for row in rows)
        if observed_ids != source_ids:
            raise ReleaseProjectionConflictError(
                "release finalization must cover every current Item projection"
            )
        candidates = tuple(self._load_result(connection, str(row["result_id"])) for row in rows)
        if any(result.phase is not ReleaseProjectionPhaseV2.CANDIDATE for result in candidates):
            raise ReleaseProjectionConflictError(
                "release finalization requires only current candidate projections"
            )
        return candidates

    def _require_release_lease(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        lease: WorkLeaseV2,
        work_control_policy: WorkControlPolicyV2,
    ) -> tuple[WorkLeaseV2, WorkControlPolicyV2]:
        stored_lease = self.store._get_work_lease(
            connection,
            lease.work_lease_id,
        )
        stored_policy = self.store._get_work_control_policy_by_id(
            connection,
            work_control_policy.work_control_policy_id,
        )
        if stored_lease != lease or stored_policy != work_control_policy:
            raise ReleaseProjectionConflictError("release work lease or control policy is stale")
        unit, _owner_ref = self.store._get_work_unit(
            connection,
            stored_lease.work_unit_ref.object_id,
        )
        if (
            unit.job_id != job_id
            or unit.stage is not StageNameV2.RELEASE
            or unit.scope is not WorkUnitScopeV2.JOB
            or resolved_work_unit_v2_ref(unit) != stored_lease.work_unit_ref
        ):
            raise ReleaseProjectionConflictError("release lease does not own the Job RELEASE work unit")
        return stored_lease, stored_policy

    def _load_terminal_results_for_sources(
        self,
        connection: sqlite3.Connection,
        sources: tuple[EvaluationItemReleaseSource, ...],
        *,
        lease: WorkLeaseV2,
    ) -> tuple[ReleaseProjectionResultV2, ...]:
        results = tuple(
            self._load_current_result_required(
                connection,
                source.item.item_id,
            )
            for source in sources
        )
        if any(result.phase is not ReleaseProjectionPhaseV2.TERMINAL for result in results):
            raise ReleaseProjectionIntegrityError(
                "release finalization replay has a non-terminal current result"
            )
        if lease.stage_run_ref is None:
            raise ReleaseProjectionIntegrityError("release finalization replay lease is missing its StageRun")
        stage_result = self._load_stage_result_for_run(
            connection,
            lease.stage_run_ref.object_id,
        )
        rows = connection.execute(
            """
            SELECT result_id, stage_result_id
            FROM release_projection_results
            WHERE result_id IN ({})
            """.format(",".join("?" for _ in results)),
            tuple(result.result_id for result in results),
        ).fetchall()
        if len(rows) != len(results) or {str(row["stage_result_id"]) for row in rows} != {
            stage_result.stage_result_id
        }:
            raise ReleaseProjectionIntegrityError(
                "release finalization replay StageResult binding is corrupt"
            )
        return results

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
            raise ReleaseProjectionIntegrityError("release StageRun is missing its StageResult")
        try:
            result = StageResultRecord.model_validate_json(str(row["record_json"]))
        except ValidationError as exc:
            raise ReleaseProjectionIntegrityError("stored release StageResult is malformed") from exc
        if (
            str(row["stage_result_id"]) != result.stage_result_id
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored release StageResult columns are inconsistent")
        return result

    def _persist_result(
        self,
        connection: sqlite3.Connection,
        *,
        result: ReleaseProjectionResultV2,
        policy: ReleaseProjectionPolicyV2,
        stage_result_id: str | None,
    ) -> None:
        self._persist_policy(connection, policy)
        self._persist_query(connection, result)
        self._persist_subject(connection, result)
        self._persist_decision(connection, result)
        self._persist_evaluation_item(connection, result)
        self._persist_projection(connection, result)
        existing = connection.execute(
            """
            SELECT result_id
            FROM release_projection_results
            WHERE result_id = ?
            """,
            (result.result_id,),
        ).fetchone()
        if existing is not None:
            stored = self._load_result(connection, result.result_id)
            if stored != result:
                raise ReleaseProjectionIntegrityError("stored release projection result differs from input")
            return
        connection.execute(
            """
            INSERT INTO release_projection_results (
                result_id, job_id, item_id, phase, release_subject_id,
                query_spec_id, release_decision_id, evaluation_item_id,
                projection_id, previous_result_id, stage_result_id,
                result_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.result_id,
                result.release_subject.job_id,
                result.release_subject.item_id,
                result.phase.value,
                result.release_subject.release_subject_id,
                result.query_spec.query_spec_id,
                result.release_decision.release_decision_id,
                result.evaluation_item.evaluation_item_id,
                result.item_projection.projection_id,
                (result.previous_result_ref.object_id if result.previous_result_ref is not None else None),
                stage_result_id,
                result.result_sha256,
                self.store._record_json(result),
            ),
        )

    def _persist_policy(
        self,
        connection: sqlite3.Connection,
        policy: ReleaseProjectionPolicyV2,
    ) -> None:
        try:
            validate_release_projection_policy_v2_identity(policy)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("release projection policy is malformed") from exc
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM release_projection_policies
            WHERE policy_id = ?
            """,
            (policy.release_projection_policy_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO release_projection_policies (
                    policy_id, policy_sha256, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.release_projection_policy_id,
                    policy.policy_sha256,
                    self.store._record_json(policy),
                ),
            )
            return
        stored = self._parse_policy_row(row)
        if stored != policy:
            raise ReleaseProjectionIntegrityError("stored release projection policy differs from input")

    def _persist_query(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        row = connection.execute(
            """
            SELECT query_spec_id
            FROM evaluation_query_specs
            WHERE query_spec_id = ?
            """,
            (result.query_spec.query_spec_id,),
        ).fetchone()
        if row is not None:
            stored = self._load_query(
                connection,
                result.query_spec.query_spec_id,
                expected_job_id=result.release_subject.job_id,
                expected_item_id=result.release_subject.item_id,
            )
            if stored != result.query_spec:
                raise ReleaseProjectionIntegrityError("stored QuerySpec differs from input")
            return
        connection.execute(
            """
            INSERT INTO evaluation_query_specs (
                query_spec_id, job_id, item_id, task_draft_id,
                prompt_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.query_spec.query_spec_id,
                result.release_subject.job_id,
                result.release_subject.item_id,
                result.query_spec.task_draft_ref.object_id,
                result.query_spec.prompt_sha256,
                self.store._record_json(result.query_spec),
            ),
        )

    def _persist_subject(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        subject = result.release_subject
        row = connection.execute(
            """
            SELECT release_subject_id
            FROM evaluation_item_release_subjects
            WHERE release_subject_id = ?
            """,
            (subject.release_subject_id,),
        ).fetchone()
        if row is not None:
            if (
                self._load_subject(
                    connection,
                    subject.release_subject_id,
                )
                != subject
            ):
                raise ReleaseProjectionIntegrityError("stored release subject differs from input")
            return
        connection.execute(
            """
            INSERT INTO evaluation_item_release_subjects (
                release_subject_id, job_id, item_id, policy_id,
                query_spec_id, release_subject_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                subject.release_subject_id,
                subject.job_id,
                subject.item_id,
                subject.policy_ref.object_id,
                subject.components.query_spec_ref.object_id,
                subject.release_subject_sha256,
                self.store._record_json(subject),
            ),
        )

    def _persist_decision(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        decision = result.release_decision
        row = connection.execute(
            """
            SELECT release_decision_id
            FROM release_decisions_v2
            WHERE release_decision_id = ?
            """,
            (decision.release_decision_id,),
        ).fetchone()
        if row is not None:
            if (
                self._load_decision(
                    connection,
                    decision.release_decision_id,
                )
                != decision
            ):
                raise ReleaseProjectionIntegrityError("stored ReleaseDecision differs from input")
            return
        connection.execute(
            """
            INSERT INTO release_decisions_v2 (
                release_decision_id, job_id, item_id, release_subject_id,
                chain_id, previous_decision_id, action, state,
                decision_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.release_decision_id,
                result.release_subject.job_id,
                result.release_subject.item_id,
                result.release_subject.release_subject_id,
                decision.chain_id,
                (
                    decision.previous_decision_ref.object_id
                    if decision.previous_decision_ref is not None
                    else None
                ),
                decision.action.value,
                decision.state.value,
                decision.decision_sha256,
                self.store._record_json(decision),
            ),
        )

    def _persist_evaluation_item(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        item = result.evaluation_item
        row = connection.execute(
            """
            SELECT evaluation_item_id
            FROM evaluation_items_v2
            WHERE evaluation_item_id = ?
            """,
            (item.evaluation_item_id,),
        ).fetchone()
        if row is not None:
            if (
                self._load_evaluation_item(
                    connection,
                    item.evaluation_item_id,
                )
                != item
            ):
                raise ReleaseProjectionIntegrityError("stored EvaluationItem differs from input")
            return
        connection.execute(
            """
            INSERT INTO evaluation_items_v2 (
                evaluation_item_id, job_id, item_id, release_decision_id,
                item_version, item_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.evaluation_item_id,
                result.release_subject.job_id,
                result.release_subject.item_id,
                item.release_decision_ref.object_id,
                item.item_version,
                item.item_sha256,
                self.store._record_json(item),
            ),
        )

    def _persist_projection(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        projection = result.item_projection
        row = connection.execute(
            """
            SELECT projection_id
            FROM item_release_projection_events
            WHERE projection_id = ?
            """,
            (projection.projection_id,),
        ).fetchone()
        if row is not None:
            if (
                self._load_projection(
                    connection,
                    projection.projection_id,
                )
                != projection
            ):
                raise ReleaseProjectionIntegrityError("stored Item release projection differs from input")
            return
        connection.execute(
            """
            INSERT INTO item_release_projection_events (
                projection_id, job_id, item_id, projection_revision,
                previous_projection_id, release_subject_id,
                evaluation_item_id, release_decision_id, release_state,
                item_status, projection_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                projection.projection_id,
                projection.job_id,
                projection.item_id,
                projection.projection_revision,
                (
                    projection.previous_projection_ref.object_id
                    if projection.previous_projection_ref is not None
                    else None
                ),
                projection.release_subject_ref.object_id,
                projection.evaluation_item_ref.object_id,
                projection.release_decision_ref.object_id,
                projection.release_state.value,
                projection.item_status.value,
                projection.projection_sha256,
                self.store._record_json(projection),
            ),
        )

    def _publish_current_projection(
        self,
        connection: sqlite3.Connection,
        *,
        result: ReleaseProjectionResultV2,
        expected_current: ReleaseProjectionResultV2 | None,
    ) -> None:
        row = connection.execute(
            """
            SELECT projection_id, result_id
            FROM item_release_current_projections
            WHERE item_id = ?
            """,
            (result.release_subject.item_id,),
        ).fetchone()
        if expected_current is None:
            if row is not None:
                raise ReleaseProjectionConflictError("release current projection was created concurrently")
            self._insert_current_projection(connection, result)
            return
        if (
            row is None
            or str(row["projection_id"]) != expected_current.item_projection.projection_id
            or str(row["result_id"]) != expected_current.result_id
            or result.previous_result_ref != expected_current.to_ref()
            or result.item_projection.previous_projection_ref != expected_current.item_projection_ref
        ):
            raise ReleaseProjectionConflictError("release current projection predecessor is stale")
        cursor = connection.execute(
            """
            UPDATE item_release_current_projections
            SET job_id = ?, projection_id = ?, result_id = ?,
                projection_revision = ?, release_state = ?,
                item_status = ?, record_json = ?
            WHERE item_id = ? AND projection_id = ? AND result_id = ?
            """,
            (
                result.release_subject.job_id,
                result.item_projection.projection_id,
                result.result_id,
                result.item_projection.projection_revision,
                result.item_projection.release_state.value,
                result.item_projection.item_status.value,
                self.store._record_json(result.item_projection),
                result.release_subject.item_id,
                expected_current.item_projection.projection_id,
                expected_current.result_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrencyConflictError("concurrent release current-projection update")

    def _insert_current_projection(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        projection = result.item_projection
        connection.execute(
            """
            INSERT INTO item_release_current_projections (
                item_id, job_id, projection_id, result_id,
                projection_revision, release_state, item_status, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                projection.item_id,
                projection.job_id,
                projection.projection_id,
                result.result_id,
                projection.projection_revision,
                projection.release_state.value,
                projection.item_status.value,
                self.store._record_json(projection),
            ),
        )

    def _project_item(
        self,
        connection: sqlite3.Connection,
        item: ItemRecord,
        target: ItemStatus,
    ) -> ItemRecord:
        current = self._load_item(connection, item.item_id)
        if current != item:
            raise ConcurrencyConflictError(f"release Item authority is stale: {item.item_id}")
        if current.status is target:
            return current
        if current.status not in {ItemStatus.RUNNING, ItemStatus.NEEDS_REVIEW} or target not in {
            ItemStatus.RUNNING,
            ItemStatus.NEEDS_REVIEW,
            ItemStatus.APPROVED,
            ItemStatus.REJECTED,
        }:
            raise ReleaseProjectionConflictError(
                f"illegal release Item projection: {current.status} -> {target}"
            )
        return self._rewrite_item_projection(connection, current, target)

    def _rewrite_item_projection(
        self,
        connection: sqlite3.Connection,
        current: ItemRecord,
        target: ItemStatus,
    ) -> ItemRecord:
        updated = ItemRecord.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": target,
                "row_version": current.row_version + 1,
                "updated_at": self.store._clock(),
            }
        )
        self.store._update_record(
            connection,
            "items",
            "item_id",
            current.item_id,
            updated.status,
            updated.row_version,
            updated,
            current.row_version,
        )
        return updated

    def _load_item(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> ItemRecord:
        row = connection.execute(
            """
            SELECT item_id, job_id, status, row_version,
                   idempotency_key, record_json
            FROM items
            WHERE item_id = ?
            """,
            (item_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ItemRecord not found: {item_id}")
        return self.store._load_item_row(row)

    def _load_current_result_required(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> ReleaseProjectionResultV2:
        result = self._load_current_result_optional(connection, item_id)
        if result is None:
            raise RecordNotFoundError(f"current release projection not found: {item_id}")
        return result

    def _load_current_result_optional(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> ReleaseProjectionResultV2 | None:
        row = connection.execute(
            """
            SELECT item_id, job_id, projection_id, result_id,
                   projection_revision, release_state, item_status, record_json
            FROM item_release_current_projections
            WHERE item_id = ?
            """,
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            projection = ItemReleaseProjectionV2.model_validate_json(str(row["record_json"]))
            validate_item_release_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored current release projection is malformed") from exc
        if (
            str(row["item_id"]) != projection.item_id
            or str(row["job_id"]) != projection.job_id
            or str(row["projection_id"]) != projection.projection_id
            or int(row["projection_revision"]) != projection.projection_revision
            or str(row["release_state"]) != projection.release_state.value
            or str(row["item_status"]) != projection.item_status.value
        ):
            raise ReleaseProjectionIntegrityError(
                "stored current release projection columns are inconsistent"
            )
        result = self._load_result(
            connection,
            str(row["result_id"]),
        )
        if result.item_projection != projection:
            raise ReleaseProjectionIntegrityError("current release projection differs from immutable result")
        item = self._load_item(connection, item_id)
        if item.status is projection.item_status:
            return result
        if item.status is ItemStatus.RELEASED and projection.item_status is ItemStatus.APPROVED:
            self._assert_publication_successor(
                connection,
                approved=result,
            )
            return result
        else:
            raise ReleaseProjectionIntegrityError(
                "Item status differs from current ReleaseDecision projection"
            )

    def _assert_publication_successor(
        self,
        connection: sqlite3.Connection,
        *,
        approved: ReleaseProjectionResultV2,
    ) -> None:
        row = connection.execute(
            """
            SELECT projection_id, job_id, result_id, item_status, record_json
            FROM item_publication_current_projections
            WHERE item_id = ?
            """,
            (approved.release_subject.item_id,),
        ).fetchone()
        if row is None:
            self._assert_production_successor(
                connection,
                approved=approved,
            )
            return
        try:
            projection = PublishedItemProjectionV2.model_validate_json(str(row["record_json"]))
            validate_published_item_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError(
                "stored current publication projection is malformed"
            ) from exc
        if (
            str(row["projection_id"]) != projection.projection_id
            or str(row["job_id"]) != projection.job_id
            or str(row["item_status"]) != projection.item_status.value
            or projection.approved_result_ref != release_projection_result_v2_ref(approved)
            or projection.approved_projection_ref != approved.item_projection_ref
        ):
            raise ReleaseProjectionIntegrityError("publication successor differs from approved projection")
        result_row = connection.execute(
            """
            SELECT result_id, job_id, manifest_id, registry_entry_id,
                   policy_id, result_sha256, record_json
            FROM release_publication_results
            WHERE result_id = ?
            """,
            (str(row["result_id"]),),
        ).fetchone()
        if result_row is None:
            raise ReleaseProjectionIntegrityError("publication successor is missing its result")
        try:
            publication = ReleasePublicationResultV2.model_validate_json(str(result_row["record_json"]))
            validate_release_publication_result_v2_identity(publication)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored publication result is malformed") from exc
        if (
            str(result_row["result_id"]) != publication.result_id
            or str(result_row["job_id"]) != publication.release_manifest.job_id
            or str(result_row["manifest_id"]) != publication.release_manifest.manifest_id
            or str(result_row["registry_entry_id"]) != publication.registry_entry.registry_entry_id
            or str(result_row["policy_id"]) != publication.policy_ref.object_id
            or str(result_row["result_sha256"]) != publication.result_sha256
            or projection not in publication.item_projections
            or projection.approved_result_ref not in publication.approved_source_result_refs
        ):
            raise ReleaseProjectionIntegrityError(
                "publication result does not contain the exact current successor"
            )
        projection_row = connection.execute(
            """
            SELECT job_id, item_id, approved_result_id, manifest_id,
                   receipt_id, registry_entry_id, publish_decision_id,
                   published_evaluation_item_id, item_status,
                   projection_sha256, record_json
            FROM item_publication_projection_events
            WHERE projection_id = ?
            """,
            (projection.projection_id,),
        ).fetchone()
        if projection_row is None:
            raise ReleaseProjectionIntegrityError("publication successor is missing its immutable event")
        try:
            event = PublishedItemProjectionV2.model_validate_json(str(projection_row["record_json"]))
            validate_published_item_projection_v2_identity(event)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored publication projection event is malformed") from exc
        if (
            event != projection
            or str(projection_row["job_id"]) != event.job_id
            or str(projection_row["item_id"]) != event.item_id
            or str(projection_row["approved_result_id"]) != event.approved_result_ref.object_id
            or str(projection_row["manifest_id"]) != event.release_manifest_ref.object_id
            or str(projection_row["receipt_id"]) != event.export_receipt_ref.object_id
            or str(projection_row["registry_entry_id"]) != event.registry_entry_ref.object_id
            or str(projection_row["publish_decision_id"]) != event.publish_decision_ref.object_id
            or str(projection_row["published_evaluation_item_id"])
            != event.published_evaluation_item_ref.object_id
            or str(projection_row["item_status"]) != event.item_status.value
            or str(projection_row["projection_sha256"]) != event.projection_sha256
        ):
            raise ReleaseProjectionIntegrityError("publication projection event columns are inconsistent")
        index = publication.item_projections.index(projection)
        if (
            self._load_decision(
                connection,
                projection.publish_decision_ref.object_id,
            )
            != publication.published_decisions[index]
            or self._load_evaluation_item(
                connection,
                projection.published_evaluation_item_ref.object_id,
            )
            != publication.published_items[index]
        ):
            raise ReleaseProjectionIntegrityError("publication successor decision or Item is stale")
        registry_row = connection.execute(
            """
            SELECT manifest_id, job_id, channel, registry, published_at,
                   entry_sha256, record_json
            FROM nonproduction_registry_entries
            WHERE registry_entry_id = ?
            """,
            (projection.registry_entry_ref.object_id,),
        ).fetchone()
        if registry_row is None:
            raise ReleaseProjectionIntegrityError("publication successor is missing its registry entry")
        try:
            registry = NonProductionRegistryEntryV2.model_validate_json(str(registry_row["record_json"]))
            validate_nonproduction_registry_entry_v2_identity(registry)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored publication registry entry is malformed") from exc
        if (
            registry != publication.registry_entry
            or str(registry_row["manifest_id"]) != registry.release_manifest_ref.object_id
            or str(registry_row["job_id"]) != registry.job_id
            or str(registry_row["channel"]) != registry.channel.value
            or str(registry_row["registry"]) != registry.registry
            or str(registry_row["published_at"]) != registry.published_at
            or str(registry_row["entry_sha256"]) != registry.entry_sha256
        ):
            raise ReleaseProjectionIntegrityError("publication registry entry columns are inconsistent")

    def _assert_production_successor(
        self,
        connection: sqlite3.Connection,
        *,
        approved: ReleaseProjectionResultV2,
    ) -> None:
        row = connection.execute(
            """
            SELECT job_id, projection_id, result_id, acceptance_id,
                   item_status, record_json
            FROM production_item_publication_current
            WHERE item_id = ?
            """,
            (approved.release_subject.item_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("RELEASED Item is missing its publication successor")
        try:
            projection = ProductionPublishedItemProjectionV2.model_validate_json(str(row["record_json"]))
            validate_production_published_item_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError(
                "stored production successor projection is malformed"
            ) from exc
        if (
            str(row["job_id"]) != projection.job_id
            or str(row["projection_id"]) != projection.projection_id
            or str(row["item_status"]) != projection.item_status.value
            or projection.approved_result_ref != release_projection_result_v2_ref(approved)
            or projection.approved_projection_ref != approved.item_projection_ref
            or projection.source_nonproduction_publication_ref is not None
        ):
            raise ReleaseProjectionIntegrityError("production successor differs from approved projection")
        acceptance = connection.execute(
            """
            SELECT job_id, result_id, registry_entry_id,
                   attestation_authority_id
            FROM production_release_acceptances
            WHERE acceptance_id = ?
            """,
            (str(row["acceptance_id"]),),
        ).fetchone()
        if (
            acceptance is None
            or str(acceptance["job_id"]) != projection.job_id
            or str(acceptance["result_id"]) != str(row["result_id"])
            or str(acceptance["registry_entry_id"]) != projection.registry_entry_ref.object_id
            or str(acceptance["attestation_authority_id"]) != projection.attestation_authority_ref.object_id
        ):
            raise ReleaseProjectionIntegrityError("production successor acceptance is inconsistent")
        decision = self._load_decision(
            connection,
            projection.publish_decision_ref.object_id,
        )
        item = self._load_evaluation_item(
            connection,
            projection.published_evaluation_item_ref.object_id,
        )
        prior_decision = approved.release_decision
        prior_item = approved.evaluation_item
        if (
            decision.previous_decision_ref != release_decision_v2_ref(prior_decision)
            or decision.chain_id != prior_decision.chain_id
            or decision.item_id != prior_decision.item_id
            or decision.channel is not ReleaseChannel.PRODUCTION
            or decision.action is not ReleaseActionV2.PUBLISH
            or decision.state is not ReleaseStateV2.RELEASED
            or decision.production_attestation_ref is None
            or _unchanged_release_decision_fields(decision)
            != _unchanged_release_decision_fields(prior_decision)
            or item.item_version != decision.item_version
            or item.release_decision_ref != release_decision_v2_ref(decision)
            or _unchanged_evaluation_item_fields(item) != _unchanged_evaluation_item_fields(prior_item)
        ):
            raise ReleaseProjectionIntegrityError("production successor decision or Item is stale")

    def _assert_current_result(
        self,
        connection: sqlite3.Connection,
        result: ReleaseProjectionResultV2,
    ) -> None:
        current = self._load_current_result_required(
            connection,
            result.release_subject.item_id,
        )
        if current != result:
            raise ReleaseProjectionIntegrityError("stored release result is not the current Item projection")

    def _load_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> ReleaseProjectionResultV2:
        row = connection.execute(
            """
            SELECT result_id, job_id, item_id, phase, release_subject_id,
                   query_spec_id, release_decision_id, evaluation_item_id,
                   projection_id, previous_result_id, stage_result_id,
                   result_sha256, record_json
            FROM release_projection_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ReleaseProjectionResultV2 not found: {result_id}")
        try:
            result = ReleaseProjectionResultV2.model_validate_json(str(row["record_json"]))
            validate_release_projection_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored release projection result is malformed") from exc
        expected_previous = (
            result.previous_result_ref.object_id if result.previous_result_ref is not None else None
        )
        if (
            str(row["result_id"]) != result.result_id
            or str(row["job_id"]) != result.release_subject.job_id
            or str(row["item_id"]) != result.release_subject.item_id
            or str(row["phase"]) != result.phase.value
            or str(row["release_subject_id"]) != result.release_subject.release_subject_id
            or str(row["query_spec_id"]) != result.query_spec.query_spec_id
            or str(row["release_decision_id"]) != result.release_decision.release_decision_id
            or str(row["evaluation_item_id"]) != result.evaluation_item.evaluation_item_id
            or str(row["projection_id"]) != result.item_projection.projection_id
            or (str(row["previous_result_id"]) if row["previous_result_id"] is not None else None)
            != expected_previous
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored release projection result columns are inconsistent")
        policy = self._load_policy(
            connection,
            result.policy_ref.object_id,
        )
        query = self._load_query(
            connection,
            result.query_spec.query_spec_id,
            expected_job_id=result.release_subject.job_id,
            expected_item_id=result.release_subject.item_id,
        )
        subject = self._load_subject(
            connection,
            result.release_subject.release_subject_id,
        )
        decision = self._load_decision(
            connection,
            result.release_decision.release_decision_id,
        )
        item = self._load_evaluation_item(
            connection,
            result.evaluation_item.evaluation_item_id,
        )
        projection = self._load_projection(
            connection,
            result.item_projection.projection_id,
        )
        if (
            policy.to_ref() != result.policy_ref
            or query != result.query_spec
            or subject != result.release_subject
            or decision != result.release_decision
            or item != result.evaluation_item
            or projection != result.item_projection
        ):
            raise ReleaseProjectionIntegrityError("stored release result differs from immutable child rows")
        if result.phase is ReleaseProjectionPhaseV2.CANDIDATE:
            if row["stage_result_id"] is not None:
                raise ReleaseProjectionIntegrityError("candidate release result cannot bind a StageResult")
        elif row["stage_result_id"] is None:
            raise ReleaseProjectionIntegrityError(
                "terminal release result is missing its StageResult binding"
            )
        else:
            stage_result = self.store._get_record(
                connection,
                "stage_results",
                "stage_result_id",
                str(row["stage_result_id"]),
                StageResultRecord,
            )
            if (
                stage_result.status is not StageRunStatus.SUCCEEDED
                or result.to_ref() not in stage_result.output_refs
            ):
                raise ReleaseProjectionIntegrityError("terminal release result StageResult binding is stale")
        return result

    def _load_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> ReleaseProjectionPolicyV2:
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM release_projection_policies
            WHERE policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its projection policy")
        return self._parse_policy_row(row)

    @staticmethod
    def _parse_policy_row(
        row: sqlite3.Row,
    ) -> ReleaseProjectionPolicyV2:
        try:
            policy = ReleaseProjectionPolicyV2.model_validate_json(str(row["record_json"]))
            validate_release_projection_policy_v2_identity(policy)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored release projection policy is malformed") from exc
        if str(row["policy_sha256"]) != policy.policy_sha256:
            raise ReleaseProjectionIntegrityError("stored release projection policy columns are inconsistent")
        return policy

    def _load_query(
        self,
        connection: sqlite3.Connection,
        query_spec_id: str,
        *,
        expected_job_id: str,
        expected_item_id: str,
    ) -> QuerySpec:
        row = connection.execute(
            """
            SELECT query_spec_id, job_id, item_id, task_draft_id,
                   prompt_sha256, record_json
            FROM evaluation_query_specs
            WHERE query_spec_id = ?
            """,
            (query_spec_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its QuerySpec")
        try:
            query = QuerySpec.model_validate_json(str(row["record_json"]))
            validate_query_spec_identity(query)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored QuerySpec is malformed") from exc
        if (
            str(row["query_spec_id"]) != query.query_spec_id
            or str(row["job_id"]) != expected_job_id
            or str(row["item_id"]) != expected_item_id
            or str(row["task_draft_id"]) != query.task_draft_ref.object_id
            or str(row["prompt_sha256"]) != query.prompt_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored QuerySpec columns are inconsistent")
        return query

    def _load_subject(
        self,
        connection: sqlite3.Connection,
        subject_id: str,
    ) -> EvaluationItemReleaseSubjectV2:
        row = connection.execute(
            """
            SELECT release_subject_id, job_id, item_id, policy_id,
                   query_spec_id, release_subject_sha256, record_json
            FROM evaluation_item_release_subjects
            WHERE release_subject_id = ?
            """,
            (subject_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its subject")
        try:
            subject = EvaluationItemReleaseSubjectV2.model_validate_json(str(row["record_json"]))
            validate_evaluation_item_release_subject_v2_identity(subject)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored release subject is malformed") from exc
        if (
            str(row["release_subject_id"]) != subject.release_subject_id
            or str(row["job_id"]) != subject.job_id
            or str(row["item_id"]) != subject.item_id
            or str(row["policy_id"]) != subject.policy_ref.object_id
            or str(row["query_spec_id"]) != subject.components.query_spec_ref.object_id
            or str(row["release_subject_sha256"]) != subject.release_subject_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored release subject columns are inconsistent")
        return subject

    def _load_decision(
        self,
        connection: sqlite3.Connection,
        decision_id: str,
    ) -> ReleaseDecisionV2:
        row = connection.execute(
            """
            SELECT release_decision_id, job_id, item_id, release_subject_id,
                   chain_id, previous_decision_id, action, state,
                   decision_sha256, record_json
            FROM release_decisions_v2
            WHERE release_decision_id = ?
            """,
            (decision_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its ReleaseDecision")
        try:
            decision = ReleaseDecisionV2.model_validate_json(str(row["record_json"]))
            validate_release_decision_v2_identity(decision)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored ReleaseDecision is malformed") from exc
        expected_previous = (
            decision.previous_decision_ref.object_id if decision.previous_decision_ref is not None else None
        )
        if (
            str(row["release_decision_id"]) != decision.release_decision_id
            or str(row["item_id"]) != decision.item_id
            or str(row["chain_id"]) != decision.chain_id
            or (str(row["previous_decision_id"]) if row["previous_decision_id"] is not None else None)
            != expected_previous
            or str(row["action"]) != decision.action.value
            or str(row["state"]) != decision.state.value
            or str(row["decision_sha256"]) != decision.decision_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored ReleaseDecision columns are inconsistent")
        subject = self._load_subject(
            connection,
            str(row["release_subject_id"]),
        )
        if (
            subject.job_id != str(row["job_id"])
            or subject.item_id != decision.item_id
            or subject.release_subject_sha256 != decision.release_subject_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored ReleaseDecision subject binding is stale")
        return decision

    def _load_evaluation_item(
        self,
        connection: sqlite3.Connection,
        evaluation_item_id: str,
    ) -> EvaluationItemV2:
        row = connection.execute(
            """
            SELECT evaluation_item_id, job_id, item_id, release_decision_id,
                   item_version, item_sha256, record_json
            FROM evaluation_items_v2
            WHERE evaluation_item_id = ?
            """,
            (evaluation_item_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its EvaluationItem")
        try:
            item = EvaluationItemV2.model_validate_json(str(row["record_json"]))
            validate_evaluation_item_v2_identity(item)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored EvaluationItem is malformed") from exc
        if (
            str(row["evaluation_item_id"]) != item.evaluation_item_id
            or str(row["release_decision_id"]) != item.release_decision_ref.object_id
            or str(row["item_version"]) != item.item_version
            or str(row["item_sha256"]) != item.item_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored EvaluationItem columns are inconsistent")
        decision = self._load_decision(
            connection,
            item.release_decision_ref.object_id,
        )
        if (
            release_decision_v2_ref(decision) != item.release_decision_ref
            or str(row["item_id"]) != decision.item_id
        ):
            raise ReleaseProjectionIntegrityError("stored EvaluationItem decision binding is stale")
        return item

    def _load_projection(
        self,
        connection: sqlite3.Connection,
        projection_id: str,
    ) -> ItemReleaseProjectionV2:
        row = connection.execute(
            """
            SELECT projection_id, job_id, item_id, projection_revision,
                   previous_projection_id, release_subject_id,
                   evaluation_item_id, release_decision_id, release_state,
                   item_status, projection_sha256, record_json
            FROM item_release_projection_events
            WHERE projection_id = ?
            """,
            (projection_id,),
        ).fetchone()
        if row is None:
            raise ReleaseProjectionIntegrityError("release result is missing its Item projection event")
        try:
            projection = ItemReleaseProjectionV2.model_validate_json(str(row["record_json"]))
            validate_item_release_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionIntegrityError("stored Item release projection is malformed") from exc
        expected_previous = (
            projection.previous_projection_ref.object_id
            if projection.previous_projection_ref is not None
            else None
        )
        if (
            str(row["projection_id"]) != projection.projection_id
            or str(row["job_id"]) != projection.job_id
            or str(row["item_id"]) != projection.item_id
            or int(row["projection_revision"]) != projection.projection_revision
            or (str(row["previous_projection_id"]) if row["previous_projection_id"] is not None else None)
            != expected_previous
            or str(row["release_subject_id"]) != projection.release_subject_ref.object_id
            or str(row["evaluation_item_id"]) != projection.evaluation_item_ref.object_id
            or str(row["release_decision_id"]) != projection.release_decision_ref.object_id
            or str(row["release_state"]) != projection.release_state.value
            or str(row["item_status"]) != projection.item_status.value
            or str(row["projection_sha256"]) != projection.projection_sha256
        ):
            raise ReleaseProjectionIntegrityError("stored Item release projection columns are inconsistent")
        return projection


def _unchanged_release_decision_fields(
    value: ReleaseDecisionV2,
) -> dict[str, object]:
    return value.model_dump(
        mode="python",
        exclude={
            "release_decision_id",
            "previous_decision_ref",
            "item_version",
            "channel",
            "registry",
            "production_attestation_ref",
            "action",
            "state",
            "idempotency_key",
            "actor",
            "decided_at",
            "decision_sha256",
            "audit",
        },
    )


def _unchanged_evaluation_item_fields(
    value: EvaluationItemV2,
) -> dict[str, object]:
    return value.model_dump(
        mode="python",
        exclude={
            "evaluation_item_id",
            "item_version",
            "release_decision_ref",
            "item_sha256",
            "audit",
        },
    )


def _source_request_sha256(
    source: EvaluationItemReleaseSource,
    *,
    policy: ReleaseProjectionPolicyV2,
) -> str:
    return _request_sha256(
        {
            "source": _source_authority_payload(source),
            "policy_ref": policy.to_ref(),
        }
    )


def _source_authority_payload(
    source: EvaluationItemReleaseSource,
) -> dict[str, object]:
    return {
        "dataset_job_spec_ref": dataset_job_spec_v2_ref(source.job_spec),
        "item_id": source.item.item_id,
        "resolved_job_work_graph_ref": resolved_job_work_graph_v2_ref(source.resolved_job_work_graph),
        "source_trace_refs": source.source_trace_refs,
        "label_decision_refs": tuple(label_decision_ref(value) for value in source.label_decisions),
        "task_draft_ref": task_draft_ref(source.task_draft),
        "task_prompt_safety_gate_ref": task_prompt_safety_gate_ref(source.task_prompt_safety_gate),
        "r4_task_contract_set_ref": r4_task_contract_set_ref(source.task_contract_set),
        "attachment_reconstruction_result_ref": (
            attachment_reconstruction_result_v2_ref(source.attachment_result)
        ),
        "item_quality_result_ref": item_quality_compilation_result_ref(source.item_quality),
        "batch_quality_report_ref": batch_quality_report_v2_ref(source.batch_quality),
        "approval_policy_ref": user_approval_policy_ref(source.approval_policy),
        "decision_commit_refs": tuple(
            user_decision_commit_result_v2_ref(value) for value in source.decision_commits
        ),
        "relevant_application_refs": (source.relevant_revalidation_application_refs),
        "revalidation_report_refs": tuple(
            directed_revalidation_report_v2_ref(value) for value in source.revalidation_reports
        ),
        "current_head_refs": source.current_head_refs,
        "not_required_checkpoints": tuple(value.value for value in source.not_required_checkpoints),
    }


__all__ = [
    "ReleaseProjectionConflictError",
    "ReleaseProjectionIntegrityError",
    "ReleaseProjectionPersistenceService",
]
