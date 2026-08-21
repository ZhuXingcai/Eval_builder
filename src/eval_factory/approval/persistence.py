from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from eval_factory.approval.decisions import (
    AuthenticatedUserContext,
    UserApprovalRequestRevisionCompiler,
    UserDecisionCompiler,
)
from eval_factory.approval.requests import ApprovalCheckpointSource
from eval_factory.contracts.approval import (
    EnvironmentScopeDecision,
    QueryPackagingChoice,
    TypedAdjustment,
    UserApprovalPolicy,
    UserApprovalRequest,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserApprovalRequestRevisionV2,
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitResultV2,
    UserDecisionHandlingPolicyV2,
    user_approval_request_revision_v2_ref,
    user_decision_adjustment_effect_v2_ref,
    user_decision_handling_policy_v2_ref,
    validate_user_approval_request_revision_v2_identity,
    validate_user_decision_commit_result_v2_identity,
    validate_user_decision_record_identity,
)
from eval_factory.contracts.approval_v2 import (
    ApprovalRequestGenerationPolicyV2,
    UserApprovalRequestCompilationResultV2,
    user_approval_request_carried_sha256,
    user_approval_request_compilation_result_v2_ref,
    user_approval_request_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    dataset_job_spec_v2_ref,
)
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import JobRecord


class UserDecisionConflictError(ValueError):
    pass


class UserDecisionPersistenceService:
    def __init__(self, store: JobStore) -> None:
        self.store = store

    def commit(
        self,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        request_compilation: UserApprovalRequestCompilationResultV2,
        checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
        request_ref: ObjectRef,
        handling_policy: UserDecisionHandlingPolicyV2,
        authentication: AuthenticatedUserContext,
        decision: UserDecision,
        adjustments: tuple[TypedAdjustment, ...],
        adjustment_effect: UserDecisionAdjustmentEffectV2 | None,
        environment_decisions: tuple[EnvironmentScopeDecision, ...],
        query_packaging: QueryPackagingChoice | None,
        reason: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> UserDecisionCommitResultV2:
        with self.store._transaction() as connection:
            return self._commit_in_transaction(
                connection,
                job_spec=job_spec,
                approval_policy=approval_policy,
                generation_policy=generation_policy,
                request_compilation=request_compilation,
                checkpoint_sources=checkpoint_sources,
                request_ref=request_ref,
                handling_policy=handling_policy,
                authentication=authentication,
                decision=decision,
                adjustments=adjustments,
                adjustment_effect=adjustment_effect,
                environment_decisions=environment_decisions,
                query_packaging=query_packaging,
                reason=reason,
                idempotency_key=idempotency_key,
                audit=audit,
            )

    def _commit_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        job_spec: DatasetJobSpecV2,
        approval_policy: UserApprovalPolicy,
        generation_policy: ApprovalRequestGenerationPolicyV2,
        request_compilation: UserApprovalRequestCompilationResultV2,
        checkpoint_sources: tuple[ApprovalCheckpointSource, ...],
        request_ref: ObjectRef,
        handling_policy: UserDecisionHandlingPolicyV2,
        authentication: AuthenticatedUserContext,
        decision: UserDecision,
        adjustments: tuple[TypedAdjustment, ...],
        adjustment_effect: UserDecisionAdjustmentEffectV2 | None,
        environment_decisions: tuple[EnvironmentScopeDecision, ...],
        query_packaging: QueryPackagingChoice | None,
        reason: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> UserDecisionCommitResultV2:
        request_sha256 = _decision_submission_sha256(
            job_spec=job_spec,
            request_compilation=request_compilation,
            request_ref=request_ref,
            handling_policy=handling_policy,
            authentication=authentication,
            decision=decision,
            adjustments=adjustments,
            adjustment_effect=adjustment_effect,
            environment_decisions=environment_decisions,
            query_packaging=query_packaging,
            reason=reason,
        )
        scope = f"commit-user-decision:{job_spec.job_id}"
        prior = self.store._idempotent_response(
            connection,
            scope=scope,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        if prior is not None:
            response_type, response_id = prior
            if response_type != "USER_DECISION_COMMIT":
                raise IdempotencyConflictError("user decision replay response type is corrupt")
            return self._load_commit(connection, response_id)

        job = self._load_job(connection, job_spec)
        existing = connection.execute(
            """
            SELECT decision_record_id
            FROM user_decision_records
            WHERE request_id = ?
            """,
            (request_ref.object_id,),
        ).fetchone()
        if existing is not None:
            raise UserDecisionConflictError(
                f"approval request {request_ref.object_id!r} already has a decision"
            )
        result = UserDecisionCompiler().compile(
            job_spec=job_spec,
            approval_policy=approval_policy,
            generation_policy=generation_policy,
            request_compilation=request_compilation,
            checkpoint_sources=checkpoint_sources,
            request_ref=request_ref,
            handling_policy=handling_policy,
            authentication=authentication,
            decision=decision,
            adjustments=adjustments,
            adjustment_effect=adjustment_effect,
            environment_decisions=environment_decisions,
            query_packaging=query_packaging,
            reason=reason,
            idempotency_key=idempotency_key,
            decided_at=self.store._clock(),
            audit=audit,
        )
        self._persist_compilation(
            connection,
            request_compilation,
        )
        if result.request_revision is not None:
            self._insert_revision(connection, result.request_revision)
        record = result.decision_record
        connection.execute(
            """
            INSERT INTO user_decision_records (
                decision_record_id, job_id, request_id,
                authenticated_user, decision, record_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.decision_record_id,
                result.job_id,
                record.request_ref.object_id,
                record.authenticated_user,
                record.decision.value,
                record.record_sha256,
                self.store._record_json(record),
            ),
        )
        connection.execute(
            """
            INSERT INTO user_decision_commits (
                commit_id, job_id, request_id, decision_record_id,
                outcome, commit_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.commit_id,
                result.job_id,
                result.request_ref.object_id,
                record.decision_record_id,
                result.outcome.value,
                result.commit_sha256,
                self.store._record_json(result),
            ),
        )
        self.store._append_outbox(
            connection,
            aggregate_type="JOB",
            aggregate_id=result.job_id,
            aggregate_version=job.row_version,
            event_type="user-decision-committed",
            attributes=_attributes(
                checkpoint=record.checkpoint.value,
                commit_id=result.commit_id,
                decision_record_id=record.decision_record_id,
                outcome=result.outcome.value,
            ),
        )
        self.store._record_idempotency(
            connection,
            scope=scope,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            response_type="USER_DECISION_COMMIT",
            response_id=result.commit_id,
            created_at=record.decided_at,
        )
        return result

    def materialize_request_revision(
        self,
        *,
        pending_revision: UserApprovalRequestRevisionV2,
        successor_request: UserApprovalRequest,
        handling_policy: UserDecisionHandlingPolicyV2,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> UserApprovalRequestRevisionV2:
        request_sha256 = _request_sha256(
            {
                "pending_revision_ref": user_approval_request_revision_v2_ref(pending_revision),
                "successor_request_ref": user_approval_request_ref(successor_request),
                "handling_policy_ref": user_decision_handling_policy_v2_ref(handling_policy),
            }
        )
        scope = f"materialize-user-request-revision:{pending_revision.job_id}"
        with self.store._transaction() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                response_type, response_id = prior
                if response_type != "USER_APPROVAL_REQUEST_REVISION":
                    raise IdempotencyConflictError("request revision replay response type is corrupt")
                return self._load_revision(connection, response_id)
            stored_pending = self._load_revision(
                connection,
                pending_revision.revision_id,
            )
            materialized = UserApprovalRequestRevisionCompiler().materialize(
                pending_revision=stored_pending,
                successor_request=successor_request,
                handling_policy=handling_policy,
                audit=audit,
            )
            self._insert_revision(connection, materialized)
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                materialized.job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=materialized.job_id,
                aggregate_version=job.row_version,
                event_type="user-approval-request-revision-materialized",
                attributes=_attributes(
                    checkpoint=materialized.checkpoint.value,
                    revision_id=materialized.revision_id,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_APPROVAL_REQUEST_REVISION",
                response_id=materialized.revision_id,
                created_at=self.store._clock(),
            )
            return materialized

    def get_commit(
        self,
        commit_id: str,
    ) -> UserDecisionCommitResultV2:
        with self.store._connect() as connection:
            return self._load_commit(connection, commit_id)

    def get_decision_for_request(
        self,
        request_id: str,
    ) -> UserDecisionRecord:
        with self.store._connect() as connection:
            row = connection.execute(
                """
                SELECT decision_record_id, job_id, request_id,
                       authenticated_user, decision, record_sha256, record_json
                FROM user_decision_records
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"UserDecisionRecord not found for request: {request_id}")
            record = self._parse_decision_row(row)
            request = self._load_request(
                connection,
                request_id,
                expected_job_id=str(row["job_id"]),
            )
            if record.request_ref != user_approval_request_ref(request):
                raise ImmutableResultError("stored decision request differs from request authority")
            return record

    def get_request_revision(
        self,
        revision_id: str,
    ) -> UserApprovalRequestRevisionV2:
        with self.store._connect() as connection:
            return self._load_revision(connection, revision_id)

    def list_job_commits(
        self,
        job_id: str,
    ) -> tuple[UserDecisionCommitResultV2, ...]:
        with self.store._connect() as connection:
            rows = connection.execute(
                """
                SELECT commit_id
                FROM user_decision_commits
                WHERE job_id = ?
                ORDER BY commit_id
                """,
                (job_id,),
            ).fetchall()
            return tuple(self._load_commit(connection, str(row["commit_id"])) for row in rows)

    def _load_job(
        self,
        connection: sqlite3.Connection,
        job_spec: DatasetJobSpecV2,
    ) -> JobRecord:
        row = connection.execute(
            """
            SELECT record_json, job_spec_json
            FROM jobs
            WHERE job_id = ?
            """,
            (job_spec.job_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"JobRecord not found: {job_spec.job_id}")
        record = JobRecord.model_validate_json(str(row["record_json"]))
        stored_spec = DatasetJobSpecV2.model_validate_json(str(row["job_spec_json"]))
        if stored_spec != job_spec:
            raise ImmutableResultError("stored DatasetJobSpec differs from decision input")
        if dataset_job_spec_v2_ref(stored_spec) != dataset_job_spec_v2_ref(job_spec):
            raise ImmutableResultError("stored DatasetJobSpec identity is stale")
        return record

    def _persist_compilation(
        self,
        connection: sqlite3.Connection,
        compilation: UserApprovalRequestCompilationResultV2,
    ) -> None:
        existing = connection.execute(
            """
            SELECT job_id, checkpoint, result_sha256, record_json
            FROM user_approval_request_compilations
            WHERE result_id = ?
            """,
            (compilation.result_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO user_approval_request_compilations (
                    result_id, job_id, checkpoint, result_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    compilation.result_id,
                    compilation.job_id,
                    compilation.checkpoint.value,
                    compilation.result_sha256,
                    self.store._record_json(compilation),
                ),
            )
        else:
            try:
                stored = UserApprovalRequestCompilationResultV2.model_validate_json(
                    str(existing["record_json"])
                )
                user_approval_request_compilation_result_v2_ref(stored)
            except (ValidationError, ValueError) as exc:
                raise ImmutableResultError("stored approval request compilation is malformed") from exc
            if (
                stored != compilation
                or str(existing["job_id"]) != compilation.job_id
                or str(existing["checkpoint"]) != compilation.checkpoint.value
                or str(existing["result_sha256"]) != compilation.result_sha256
            ):
                raise ImmutableResultError("stored approval request compilation differs from input")
        for request in compilation.requests:
            self._persist_request(
                connection,
                compilation=compilation,
                request=request,
            )

    def _persist_request(
        self,
        connection: sqlite3.Connection,
        *,
        compilation: UserApprovalRequestCompilationResultV2,
        request: UserApprovalRequest,
    ) -> None:
        existing = connection.execute(
            """
            SELECT job_id, compilation_result_id, checkpoint,
                   request_sha256, record_json
            FROM user_approval_requests
            WHERE request_id = ?
            """,
            (request.request_id,),
        ).fetchone()
        request_sha256 = user_approval_request_carried_sha256(request)
        if existing is None:
            connection.execute(
                """
                INSERT INTO user_approval_requests (
                    request_id, job_id, compilation_result_id,
                    checkpoint, request_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request.request_id,
                    compilation.job_id,
                    compilation.result_id,
                    request.checkpoint.value,
                    request_sha256,
                    self.store._record_json(request),
                ),
            )
            return
        try:
            stored = UserApprovalRequest.model_validate_json(str(existing["record_json"]))
            user_approval_request_ref(stored)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored approval request is malformed") from exc
        if (
            stored != request
            or str(existing["job_id"]) != compilation.job_id
            or str(existing["compilation_result_id"]) != compilation.result_id
            or str(existing["checkpoint"]) != request.checkpoint.value
            or str(existing["request_sha256"]) != request_sha256
        ):
            raise ImmutableResultError("stored approval request differs from input")

    def _load_compilation(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> UserApprovalRequestCompilationResultV2:
        row = connection.execute(
            """
            SELECT job_id, checkpoint, result_sha256, record_json
            FROM user_approval_request_compilations
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("stored user decision commit is missing its request compilation")
        try:
            result = UserApprovalRequestCompilationResultV2.model_validate_json(str(row["record_json"]))
            user_approval_request_compilation_result_v2_ref(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored approval request compilation is malformed") from exc
        if (
            str(row["job_id"]) != result.job_id
            or str(row["checkpoint"]) != result.checkpoint.value
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ImmutableResultError("stored approval request compilation columns are inconsistent")
        return result

    def _load_request(
        self,
        connection: sqlite3.Connection,
        request_id: str,
        *,
        expected_job_id: str | None = None,
        expected_compilation_result_id: str | None = None,
    ) -> UserApprovalRequest:
        row = connection.execute(
            """
            SELECT job_id, compilation_result_id, checkpoint,
                   request_sha256, record_json
            FROM user_approval_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("stored user decision commit is missing its approval request")
        try:
            request = UserApprovalRequest.model_validate_json(str(row["record_json"]))
            observed_ref = user_approval_request_ref(request)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored approval request is malformed") from exc
        if (
            (expected_job_id is not None and str(row["job_id"]) != expected_job_id)
            or (
                expected_compilation_result_id is not None
                and str(row["compilation_result_id"]) != expected_compilation_result_id
            )
            or str(row["checkpoint"]) != request.checkpoint.value
            or str(row["request_sha256"]) != observed_ref.object_sha256
        ):
            raise ImmutableResultError("stored approval request columns are inconsistent")
        return request

    def _insert_revision(
        self,
        connection: sqlite3.Connection,
        revision: UserApprovalRequestRevisionV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO user_approval_request_revisions (
                revision_id, job_id, source_request_id,
                revision_number, state, revision_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                revision.job_id,
                revision.source_request_ref.object_id,
                revision.revision_number,
                revision.state.value,
                revision.revision_sha256,
                self.store._record_json(revision),
            ),
        )

    def _load_revision(
        self,
        connection: sqlite3.Connection,
        revision_id: str,
    ) -> UserApprovalRequestRevisionV2:
        row = connection.execute(
            """
            SELECT revision_id, job_id, source_request_id,
                   revision_number, state, revision_sha256, record_json
            FROM user_approval_request_revisions
            WHERE revision_id = ?
            """,
            (revision_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"UserApprovalRequestRevisionV2 not found: {revision_id}")
        try:
            revision = UserApprovalRequestRevisionV2.model_validate_json(str(row["record_json"]))
            validate_user_approval_request_revision_v2_identity(revision)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored request revision is malformed") from exc
        if (
            str(row["revision_id"]) != revision.revision_id
            or str(row["job_id"]) != revision.job_id
            or str(row["source_request_id"]) != revision.source_request_ref.object_id
            or int(row["revision_number"]) != revision.revision_number
            or str(row["state"]) != revision.state.value
            or str(row["revision_sha256"]) != revision.revision_sha256
        ):
            raise ImmutableResultError("stored request revision columns are inconsistent")
        return revision

    def _load_commit(
        self,
        connection: sqlite3.Connection,
        commit_id: str,
    ) -> UserDecisionCommitResultV2:
        row = connection.execute(
            """
            SELECT commit_id, job_id, request_id, decision_record_id,
                   outcome, commit_sha256, record_json
            FROM user_decision_commits
            WHERE commit_id = ?
            """,
            (commit_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"UserDecisionCommitResultV2 not found: {commit_id}")
        try:
            result = UserDecisionCommitResultV2.model_validate_json(str(row["record_json"]))
            validate_user_decision_commit_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored user decision commit is malformed") from exc
        if (
            str(row["commit_id"]) != result.commit_id
            or str(row["job_id"]) != result.job_id
            or str(row["request_id"]) != result.request_ref.object_id
            or str(row["decision_record_id"]) != result.decision_record.decision_record_id
            or str(row["outcome"]) != result.outcome.value
            or str(row["commit_sha256"]) != result.commit_sha256
        ):
            raise ImmutableResultError("stored user decision commit columns are inconsistent")
        decision_row = connection.execute(
            """
            SELECT decision_record_id, job_id, request_id,
                   authenticated_user, decision, record_sha256, record_json
            FROM user_decision_records
            WHERE decision_record_id = ?
            """,
            (result.decision_record.decision_record_id,),
        ).fetchone()
        if decision_row is None:
            raise ImmutableResultError("stored user decision commit is missing its decision record")
        if (
            str(decision_row["job_id"]) != result.job_id
            or self._parse_decision_row(decision_row) != result.decision_record
        ):
            raise ImmutableResultError("stored decision row differs from the commit result")
        compilation = self._load_compilation(
            connection,
            result.request_compilation_result_ref.object_id,
        )
        if (
            compilation.job_id != result.job_id
            or user_approval_request_compilation_result_v2_ref(compilation)
            != result.request_compilation_result_ref
        ):
            raise ImmutableResultError("stored request compilation differs from the decision commit")
        request = self._load_request(
            connection,
            result.request_ref.object_id,
            expected_job_id=result.job_id,
            expected_compilation_result_id=compilation.result_id,
        )
        if (
            user_approval_request_ref(request) != result.request_ref
            or result.request_ref not in compilation.request_refs
            or result.decision_record.checkpoint is not request.checkpoint
            or result.decision_record.approval_policy_ref != request.approval_policy_ref
            or result.decision_record.subject_refs != request.subject_refs
            or result.decision_record.plan_ref != request.plan_ref
            or result.decision_record.projection_ref != request.projection_ref
            or result.decision_record.authenticated_user != request.requested_by
        ):
            raise ImmutableResultError("stored approval request differs from the decision commit")
        if result.request_revision is not None:
            stored_revision = self._load_revision(
                connection,
                result.request_revision.revision_id,
            )
            if stored_revision != result.request_revision:
                raise ImmutableResultError("stored request revision differs from the decision commit")
        return result

    @staticmethod
    def _parse_decision_row(
        row: sqlite3.Row,
    ) -> UserDecisionRecord:
        try:
            record = UserDecisionRecord.model_validate_json(str(row["record_json"]))
            validate_user_decision_record_identity(record)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored user decision record is malformed") from exc
        if (
            str(row["decision_record_id"]) != record.decision_record_id
            or str(row["request_id"]) != record.request_ref.object_id
            or str(row["authenticated_user"]) != record.authenticated_user
            or str(row["decision"]) != record.decision.value
            or str(row["record_sha256"]) != record.record_sha256
        ):
            raise ImmutableResultError("stored user decision record columns are inconsistent")
        return record


def _decision_submission_sha256(
    *,
    job_spec: DatasetJobSpecV2,
    request_compilation: UserApprovalRequestCompilationResultV2,
    request_ref: ObjectRef,
    handling_policy: UserDecisionHandlingPolicyV2,
    authentication: AuthenticatedUserContext,
    decision: UserDecision,
    adjustments: tuple[TypedAdjustment, ...],
    adjustment_effect: UserDecisionAdjustmentEffectV2 | None,
    environment_decisions: tuple[EnvironmentScopeDecision, ...],
    query_packaging: QueryPackagingChoice | None,
    reason: str,
) -> str:
    return _request_sha256(
        {
            "dataset_job_spec_ref": dataset_job_spec_v2_ref(job_spec),
            "request_compilation_result_ref": (
                user_approval_request_compilation_result_v2_ref(request_compilation)
            ),
            "request_ref": request_ref,
            "handling_policy_ref": user_decision_handling_policy_v2_ref(handling_policy),
            "authenticated_user": authentication.authenticated_user,
            "authentication_context_ref": (authentication.authentication_context_ref),
            "decision": decision.value,
            "adjustments": [value.model_dump(mode="json", exclude_none=False) for value in adjustments],
            "adjustment_effect_ref": (
                user_decision_adjustment_effect_v2_ref(adjustment_effect)
                if adjustment_effect is not None
                else None
            ),
            "environment_decisions": [
                value.model_dump(mode="json", exclude_none=False) for value in environment_decisions
            ],
            "query_packaging": (query_packaging.value if query_packaging is not None else None),
            "reason": reason,
        }
    )


__all__ = [
    "UserDecisionConflictError",
    "UserDecisionPersistenceService",
]
