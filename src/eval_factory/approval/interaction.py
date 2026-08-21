from __future__ import annotations

import hashlib
import sqlite3
from typing import cast

from pydantic import ValidationError

from eval_factory.approval.application_persistence import (
    UserPlanApplicationPersistenceService,
)
from eval_factory.approval.decisions import AuthenticatedUserContext
from eval_factory.approval.interaction_models import (
    UserCheckpointSourceContextV2,
    user_checkpoint_source_context_v2_ref,
)
from eval_factory.approval.interaction_store import (
    UserCheckpointMaterialStore,
)
from eval_factory.approval.persistence import (
    UserDecisionPersistenceService,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalRequest,
    UserDecision,
)
from eval_factory.contracts.approval_application_v2 import (
    directed_revalidation_report_v2_ref,
    revalidation_work_item_v2_ref,
    user_plan_application_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserApprovalRequestRevisionStateV2,
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_approval_request_revision_v2_ref,
    user_decision_commit_result_v2_ref,
    user_decision_handling_policy_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    UserApprovalRequestCompilationOutcomeV2,
    user_approval_request_compilation_result_v2_ref,
    user_approval_request_ref,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointDecisionResultV2,
    UserCheckpointDecisionSubmissionV2,
    UserCheckpointInteractionPageV2,
    UserCheckpointInteractionPolicyV2,
    UserCheckpointInteractionStateV2,
    UserCheckpointInteractionV2,
    UserCheckpointOpenOutcomeV2,
    UserCheckpointOpenResultV2,
    UserCheckpointPresentationV2,
    UserCheckpointResumeDispositionV2,
    UserCheckpointResumeResultV2,
    UserCheckpointShowResultV2,
    user_checkpoint_decision_result_v2_ref,
    user_checkpoint_interaction_policy_v2_ref,
    user_checkpoint_interaction_v2_ref,
    user_checkpoint_open_result_v2_ref,
    user_checkpoint_presentation_v2_ref,
    user_checkpoint_resume_result_v2_ref,
    validate_user_checkpoint_decision_result_v2_identity,
    validate_user_checkpoint_interaction_policy_v2_identity,
    validate_user_checkpoint_interaction_v2_identity,
    validate_user_checkpoint_open_result_v2_identity,
    validate_user_checkpoint_resume_result_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import dataset_job_spec_v2_ref
from eval_factory.orchestration.job_store import (
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import JobRecord


class UserCheckpointInteractionError(ValueError):
    pass


class UserCheckpointInteractionConflictError(UserCheckpointInteractionError):
    pass


class UserCheckpointInteractionIntegrityError(UserCheckpointInteractionError):
    pass


class UserCheckpointInteractionNotResumableError(UserCheckpointInteractionError):
    pass


class UserCheckpointAuthenticationError(UserCheckpointInteractionError):
    pass


class UserCheckpointInteractionService:
    def __init__(
        self,
        store: JobStore,
        material_store: UserCheckpointMaterialStore,
    ) -> None:
        self.store = store
        self.material_store = material_store
        self.decisions = UserDecisionPersistenceService(store)

    def open_checkpoint(
        self,
        *,
        source_context: UserCheckpointSourceContextV2,
        policy: UserCheckpointInteractionPolicyV2,
        expected_job_version: int,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> UserCheckpointOpenResultV2:
        context = UserCheckpointSourceContextV2.model_validate(source_context.model_dump(mode="python"))
        context_ref = user_checkpoint_source_context_v2_ref(context)
        parsed_policy = UserCheckpointInteractionPolicyV2.model_validate(policy.model_dump(mode="python"))
        validate_user_checkpoint_interaction_policy_v2_identity(parsed_policy)
        policy_ref = user_checkpoint_interaction_policy_v2_ref(parsed_policy)
        compilation = context.request_compilation
        if compilation.request_count > parsed_policy.max_requests_per_interaction:
            raise UserCheckpointInteractionError("checkpoint request count exceeds interaction policy")
        if (
            parsed_policy.max_source_context_bytes > self.material_store.max_source_context_bytes
            or parsed_policy.max_presentation_bytes > self.material_store.max_presentation_bytes
        ):
            raise UserCheckpointInteractionError("checkpoint material store limits are narrower than policy")
        request_sha256 = _request_sha256(
            {
                "source_context_ref": context_ref,
                "interaction_policy_ref": policy_ref,
                "expected_job_version": expected_job_version,
            }
        )
        scope = f"open-user-checkpoint:{context.job_spec.job_id}"
        outcome = {
            UserApprovalRequestCompilationOutcomeV2.DISABLED: (UserCheckpointOpenOutcomeV2.DISABLED),
            UserApprovalRequestCompilationOutcomeV2.NOT_REQUIRED: (UserCheckpointOpenOutcomeV2.NOT_REQUIRED),
        }.get(compilation.outcome)
        if outcome is not None:
            return self._open_empty(
                context=context,
                policy=parsed_policy,
                outcome=outcome,
                expected_job_version=expected_job_version,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                scope=scope,
                audit=audit,
            )
        if compilation.outcome is not UserApprovalRequestCompilationOutcomeV2.REQUESTED:
            raise UserCheckpointInteractionError("checkpoint compilation outcome is unsupported")

        presentations = context.presentations()
        if len(presentations) != compilation.request_count:
            raise UserCheckpointInteractionError("checkpoint presentation coverage is incomplete")
        self.material_store.put_source_context(context)
        for presentation in presentations:
            self.material_store.put_presentation(presentation)

        with self.store._transaction() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "USER_CHECKPOINT_OPEN_RESULT":
                    raise IdempotencyConflictError("checkpoint open replay response type is corrupt")
                return self._load_open_result(connection, prior[1])
            self._validate_current_context(connection, context)
            current = self._load_current_optional(
                connection,
                context.job_spec.job_id,
            )
            job = self._load_open_job(
                connection,
                job_id=context.job_spec.job_id,
                expected_job_version=expected_job_version,
                current=current,
            )
            blocked_successor = current is not None and current.state in {
                UserCheckpointInteractionStateV2.WAITING_REVISION,
                UserCheckpointInteractionStateV2.DEFERRED,
            }
            if blocked_successor:
                assert current is not None
                if current.policy_ref != policy_ref:
                    raise UserCheckpointInteractionConflictError(
                        "checkpoint successor must preserve the interaction policy"
                    )
            selected_request, selected_presentation = self._select_open_request(
                connection,
                context=context,
                presentations=presentations,
                current=current,
            )
            additional_interactions = 2 if blocked_successor else 1
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM user_checkpoint_interactions
                    WHERE job_id = ?
                    """,
                    (job.job_id,),
                ).fetchone()[0]
            )
            if count + additional_interactions > parsed_policy.max_interactions_per_job:
                raise UserCheckpointInteractionError("checkpoint interaction count limit exceeded")
            self._persist_policy(connection, parsed_policy)
            self.decisions._persist_compilation(connection, compilation)
            predecessor = current
            if blocked_successor:
                if current is None or current.decision_commit_ref is None:
                    raise UserCheckpointInteractionIntegrityError(
                        "checkpoint successor has no decision authority"
                    )
                superseded = UserCheckpointInteractionV2.create(
                    job_id=current.job_id,
                    chain_id=current.chain_id,
                    interaction_version=current.interaction_version + 1,
                    predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
                    checkpoint=current.checkpoint,
                    request_compilation_ref=current.request_compilation_ref,
                    request_ref=current.request_ref,
                    presentation_ref=current.presentation_ref,
                    source_context_ref=current.source_context_ref,
                    state=UserCheckpointInteractionStateV2.SUPERSEDED,
                    decision_commit_ref=current.decision_commit_ref,
                    application_ref=current.application_ref,
                    job_status=JobStatus.BLOCKED,
                    occurred_at=self.store._clock(),
                    policy_ref=current.policy_ref,
                    audit=audit,
                )
                self._insert_interaction(connection, superseded)
                predecessor = superseded
            interaction = UserCheckpointInteractionV2.create(
                job_id=job.job_id,
                chain_id=_chain_id(job.job_id),
                interaction_version=(predecessor.interaction_version + 1 if predecessor is not None else 1),
                predecessor_interaction_ref=(
                    user_checkpoint_interaction_v2_ref(predecessor) if predecessor is not None else None
                ),
                checkpoint=compilation.checkpoint,
                request_compilation_ref=(user_approval_request_compilation_result_v2_ref(compilation)),
                request_ref=selected_presentation.request_ref,
                presentation_ref=user_checkpoint_presentation_v2_ref(selected_presentation),
                source_context_ref=context_ref,
                state=UserCheckpointInteractionStateV2.PENDING_DECISION,
                decision_commit_ref=None,
                application_ref=None,
                job_status=JobStatus.BLOCKED,
                occurred_at=self.store._clock(),
                policy_ref=policy_ref,
                audit=audit,
            )
            if selected_presentation.request_ref != user_approval_request_ref(selected_request):
                raise UserCheckpointInteractionIntegrityError(
                    "selected presentation/request authority is inconsistent"
                )
            self._insert_interaction(connection, interaction)
            self._replace_head(connection, interaction)
            blocked = job
            if job.status is JobStatus.RUNNING:
                blocked = self.store._transition_job_in_transaction(
                    connection,
                    job_id=job.job_id,
                    target=JobStatus.BLOCKED,
                    expected_version=job.row_version,
                    reason="user-checkpoint",
                )
            result = UserCheckpointOpenResultV2.create(
                job_id=job.job_id,
                outcome=UserCheckpointOpenOutcomeV2.PAUSED,
                interaction=interaction,
                job_status=blocked.status,
                job_version=blocked.row_version,
                policy_ref=policy_ref,
                audit=audit,
            )
            self._persist_result(connection, result)
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job.job_id,
                aggregate_version=blocked.row_version,
                event_type="user-checkpoint-opened",
                attributes=_attributes(
                    checkpoint=interaction.checkpoint.value,
                    interaction_id=interaction.interaction_id,
                    request_id=interaction.request_ref.object_id,
                    superseded=blocked_successor,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_CHECKPOINT_OPEN_RESULT",
                response_id=result.result_id,
                created_at=interaction.occurred_at,
            )
            return result

    def decide(
        self,
        *,
        interaction_id: str,
        submission: UserCheckpointDecisionSubmissionV2,
        authentication: AuthenticatedUserContext,
        expected_job_version: int,
        audit: ContractAudit,
    ) -> UserCheckpointDecisionResultV2:
        parsed = UserCheckpointDecisionSubmissionV2.model_validate(submission.model_dump(mode="python"))
        initial = self.get_interaction(interaction_id)
        if parsed.interaction_ref != user_checkpoint_interaction_v2_ref(initial):
            raise UserCheckpointInteractionConflictError("decision submission targets another interaction")
        request_sha256 = _request_sha256(
            {
                "interaction_ref": parsed.interaction_ref,
                "decision": parsed.decision,
                "adjustments": parsed.adjustments,
                "adjustment_effect": parsed.adjustment_effect,
                "environment_decisions": parsed.environment_decisions,
                "query_packaging": parsed.query_packaging,
                "reason": parsed.reason,
                "authenticated_user": authentication.authenticated_user,
                "authentication_context_ref": (authentication.authentication_context_ref),
                "expected_job_version": expected_job_version,
            }
        )
        scope = f"decide-user-checkpoint:{initial.job_id}"
        with self.store._transaction() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "USER_CHECKPOINT_DECISION_RESULT":
                    raise IdempotencyConflictError("checkpoint decision replay response type is corrupt")
                return self._load_decision_result(connection, prior[1])
            current = self._load_current_optional(connection, initial.job_id)
            if (
                current is None
                or current.interaction_id != interaction_id
                or current.state is not UserCheckpointInteractionStateV2.PENDING_DECISION
            ):
                raise UserCheckpointInteractionConflictError(
                    "decision requires the current pending interaction"
                )
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                current.job_id,
                JobRecord,
            )
            self.store._require_version(
                job.row_version,
                expected_job_version,
                current.job_id,
            )
            if job.status is not JobStatus.BLOCKED:
                raise UserCheckpointInteractionConflictError("decision requires a BLOCKED Job")
            context = self.material_store.get_source_context(current.source_context_ref)
            presentation = self.material_store.get_presentation(current.presentation_ref)
            if (
                context.job_spec.job_id != current.job_id
                or context.request_compilation.result_id != current.request_compilation_ref.object_id
                or presentation.request_ref != current.request_ref
            ):
                raise UserCheckpointInteractionIntegrityError("checkpoint decision material binding is stale")
            request = next(
                (
                    value
                    for value in context.request_compilation.requests
                    if value.request_id == current.request_ref.object_id
                ),
                None,
            )
            if (
                request is None
                or authentication.authenticated_user != request.requested_by
                or authentication.authentication_context_ref.object_type != "authenticated-user-context"
                or authentication.authentication_context_ref.object_version != "v2"
            ):
                raise UserCheckpointAuthenticationError(
                    "checkpoint authentication does not match requesting user"
                )
            commit = self.decisions._commit_in_transaction(
                connection,
                job_spec=context.job_spec,
                approval_policy=context.approval_policy,
                generation_policy=context.generation_policy,
                request_compilation=context.request_compilation,
                checkpoint_sources=context.sources(),
                request_ref=current.request_ref,
                handling_policy=context.handling_policy,
                authentication=authentication,
                decision=parsed.decision,
                adjustments=parsed.adjustments,
                adjustment_effect=parsed.adjustment_effect,
                environment_decisions=parsed.environment_decisions,
                query_packaging=parsed.query_packaging,
                reason=parsed.reason,
                idempotency_key=parsed.idempotency_key,
                audit=audit,
            )
            (
                state,
                disposition,
                target_status,
            ) = _decision_mapping(
                current.checkpoint,
                parsed.decision,
            )
            self._require_interaction_capacity(
                connection,
                job_id=current.job_id,
                policy_ref=current.policy_ref,
            )
            resulting_job = job
            if target_status is JobStatus.FAILED:
                resulting_job = self.store._transition_job_in_transaction(
                    connection,
                    job_id=job.job_id,
                    target=JobStatus.FAILED,
                    expected_version=job.row_version,
                    reason="user-checkpoint-rejected",
                )
            outcome_interaction = UserCheckpointInteractionV2.create(
                job_id=current.job_id,
                chain_id=current.chain_id,
                interaction_version=current.interaction_version + 1,
                predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
                checkpoint=current.checkpoint,
                request_compilation_ref=current.request_compilation_ref,
                request_ref=current.request_ref,
                presentation_ref=current.presentation_ref,
                source_context_ref=current.source_context_ref,
                state=state,
                decision_commit_ref=user_decision_commit_result_v2_ref(commit),
                application_ref=None,
                job_status=resulting_job.status,
                occurred_at=self.store._clock(),
                policy_ref=current.policy_ref,
                audit=audit,
            )
            self._insert_interaction(connection, outcome_interaction)
            next_interaction = self._next_pending_interaction(
                connection,
                context=context,
                current=outcome_interaction,
                commit_decision=parsed.decision,
                audit=audit,
            )
            head = next_interaction or outcome_interaction
            self._replace_head(connection, head)
            if next_interaction is not None:
                disposition = UserCheckpointResumeDispositionV2.NEXT_REQUEST
            result = UserCheckpointDecisionResultV2.create(
                predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
                interaction=outcome_interaction,
                next_interaction=next_interaction,
                decision_commit=commit,
                resume_disposition=disposition,
                job_status=resulting_job.status,
                job_version=resulting_job.row_version,
                audit=audit,
            )
            self._persist_decision_result(connection, result)
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=current.job_id,
                aggregate_version=resulting_job.row_version,
                event_type="user-checkpoint-decision-mapped",
                attributes=_attributes(
                    checkpoint=current.checkpoint.value,
                    interaction_id=outcome_interaction.interaction_id,
                    outcome=commit.outcome.value,
                    state=head.state.value,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_CHECKPOINT_DECISION_RESULT",
                response_id=result.result_id,
                created_at=commit.decision_record.decided_at,
            )
            return result

    def resume_checkpoint(
        self,
        *,
        job_id: str,
        expected_job_version: int,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> UserCheckpointResumeResultV2:
        initial = self.get_current_interaction(job_id)
        request_sha256 = _request_sha256(
            {
                "job_id": job_id,
                "expected_job_version": expected_job_version,
            }
        )
        scope = f"resume-user-checkpoint:{job_id}"
        with self.store._transaction() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "USER_CHECKPOINT_RESUME_RESULT":
                    raise IdempotencyConflictError("checkpoint resume replay response type is corrupt")
                return self._load_resume_result(connection, prior[1])
            current = self._load_current_optional(connection, job_id)
            if current is None or current.interaction_id != initial.interaction_id:
                raise UserCheckpointInteractionConflictError(
                    "checkpoint resume requires the current interaction"
                )
            if current.state not in {
                UserCheckpointInteractionStateV2.RESUMABLE,
                UserCheckpointInteractionStateV2.WAITING_REVALIDATION,
            }:
                raise UserCheckpointInteractionNotResumableError("checkpoint interaction is not resumable")
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            self.store._require_version(
                job.row_version,
                expected_job_version,
                job_id,
            )
            if job.status is not JobStatus.BLOCKED:
                raise UserCheckpointInteractionConflictError("checkpoint resume requires a BLOCKED Job")
            if current.decision_commit_ref is None:
                raise UserCheckpointInteractionIntegrityError("resumable interaction has no decision commit")
            commit = self.decisions._load_commit(
                connection,
                current.decision_commit_ref.object_id,
            )
            if user_decision_commit_result_v2_ref(commit) != (current.decision_commit_ref):
                raise UserCheckpointInteractionIntegrityError("resume decision commit binding is stale")
            self.material_store.verify_source_context(current.source_context_ref)
            self.material_store.verify_presentation(current.presentation_ref)
            application_ref: ObjectRef | None = None
            report_ref: ObjectRef | None = None
            incomplete_refs: tuple[ObjectRef, ...] = ()
            ready_refs: tuple[ObjectRef, ...] = ()
            interaction_policy = self._load_policy(
                connection,
                current.policy_ref,
            )
            if current.state is UserCheckpointInteractionStateV2.WAITING_REVALIDATION:
                row = connection.execute(
                    """
                    SELECT application_id
                    FROM user_plan_applications
                    WHERE decision_commit_id = ?
                    """,
                    (commit.commit_id,),
                ).fetchone()
                if row is None:
                    raise UserCheckpointInteractionNotResumableError(
                        "adjusted checkpoint requires an applied R7-06 plan"
                    )
                application_service = UserPlanApplicationPersistenceService(self.store)
                application = application_service._load_application(
                    connection,
                    str(row["application_id"]),
                )
                application_ref = user_plan_application_v2_ref(application)
                ready_refs = tuple(
                    revalidation_work_item_v2_ref(value)
                    for value in (
                        application_service._list_ready_work_in_transaction(
                            connection,
                            application.application_id,
                        )
                    )
                )
                report = application_service._load_optional_report(
                    connection,
                    application.application_id,
                )
                report_ref = directed_revalidation_report_v2_ref(report) if report is not None else None
            else:
                incomplete_refs = self.store._list_incomplete_work_refs_in_transaction(
                    connection,
                    job_id,
                )
            if (
                len(incomplete_refs) > interaction_policy.max_ready_work_refs
                or len(ready_refs) > interaction_policy.max_ready_work_refs
            ):
                raise UserCheckpointInteractionError(
                    "checkpoint resume work inventory exceeds interaction policy"
                )
            interaction_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM user_checkpoint_interactions
                    WHERE job_id = ?
                    """,
                    (job_id,),
                ).fetchone()[0]
            )
            if interaction_count + 1 > interaction_policy.max_interactions_per_job:
                raise UserCheckpointInteractionError("checkpoint interaction count limit exceeded")
            running = self.store._transition_job_in_transaction(
                connection,
                job_id=job_id,
                target=JobStatus.RUNNING,
                expected_version=job.row_version,
                reason="user-checkpoint-resume",
            )
            resumed = UserCheckpointInteractionV2.create(
                job_id=current.job_id,
                chain_id=current.chain_id,
                interaction_version=current.interaction_version + 1,
                predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
                checkpoint=current.checkpoint,
                request_compilation_ref=current.request_compilation_ref,
                request_ref=current.request_ref,
                presentation_ref=current.presentation_ref,
                source_context_ref=current.source_context_ref,
                state=UserCheckpointInteractionStateV2.RESUMED,
                decision_commit_ref=current.decision_commit_ref,
                application_ref=application_ref,
                job_status=running.status,
                occurred_at=self.store._clock(),
                policy_ref=current.policy_ref,
                audit=audit,
            )
            self._insert_interaction(connection, resumed)
            self._replace_head(connection, resumed)
            result = UserCheckpointResumeResultV2.create(
                predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
                interaction=resumed,
                decision_commit_ref=current.decision_commit_ref,
                application_ref=application_ref,
                revalidation_report_ref=report_ref,
                incomplete_work_refs=incomplete_refs,
                revalidation_ready_work_refs=ready_refs,
                job_version=running.row_version,
                audit=audit,
            )
            self._persist_resume_result(connection, result)
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job_id,
                aggregate_version=running.row_version,
                event_type="user-checkpoint-resumed",
                attributes=_attributes(
                    checkpoint=current.checkpoint.value,
                    interaction_id=resumed.interaction_id,
                    revalidation_required=application_ref is not None,
                ),
            )
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_CHECKPOINT_RESUME_RESULT",
                response_id=result.result_id,
                created_at=resumed.occurred_at,
            )
            return result

    def _next_pending_interaction(
        self,
        connection: sqlite3.Connection,
        *,
        context: UserCheckpointSourceContextV2,
        current: UserCheckpointInteractionV2,
        commit_decision: UserDecision,
        audit: ContractAudit,
    ) -> UserCheckpointInteractionV2 | None:
        may_continue = commit_decision is UserDecision.ACCEPT or (
            current.checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW
            and commit_decision is UserDecision.REJECT
        )
        if not may_continue:
            return None
        request_ids = tuple(request.request_id for request in context.request_compilation.requests)
        decided_ids = {
            str(row["request_id"])
            for row in connection.execute(
                """
                SELECT request_id
                FROM user_decision_records
                WHERE job_id = ?
                """,
                (current.job_id,),
            ).fetchall()
            if str(row["request_id"]) in request_ids
        }
        next_request = next(
            (
                request
                for request in context.request_compilation.requests
                if request.request_id not in decided_ids
            ),
            None,
        )
        if next_request is None:
            return None
        presentation_by_request = {value.request_ref.object_id: value for value in context.presentations()}
        next_presentation = presentation_by_request.get(next_request.request_id)
        if next_presentation is None:
            raise UserCheckpointInteractionIntegrityError("next checkpoint request has no presentation")
        self.material_store.verify_presentation(user_checkpoint_presentation_v2_ref(next_presentation))
        self._require_interaction_capacity(
            connection,
            job_id=current.job_id,
            policy_ref=current.policy_ref,
        )
        value = UserCheckpointInteractionV2.create(
            job_id=current.job_id,
            chain_id=current.chain_id,
            interaction_version=current.interaction_version + 1,
            predecessor_interaction_ref=(user_checkpoint_interaction_v2_ref(current)),
            checkpoint=current.checkpoint,
            request_compilation_ref=current.request_compilation_ref,
            request_ref=next_presentation.request_ref,
            presentation_ref=user_checkpoint_presentation_v2_ref(next_presentation),
            source_context_ref=current.source_context_ref,
            state=UserCheckpointInteractionStateV2.PENDING_DECISION,
            decision_commit_ref=None,
            application_ref=None,
            job_status=JobStatus.BLOCKED,
            occurred_at=self.store._clock(),
            policy_ref=current.policy_ref,
            audit=audit,
        )
        self._insert_interaction(connection, value)
        return value

    def _persist_decision_result(
        self,
        connection: sqlite3.Connection,
        result: UserCheckpointDecisionResultV2,
    ) -> None:
        ref = user_checkpoint_decision_result_v2_ref(result)
        connection.execute(
            """
            INSERT INTO user_checkpoint_results (
                result_id, job_id, result_type, interaction_id,
                result_sha256, record_json
            ) VALUES (?, ?, 'DECISION', ?, ?, ?)
            """,
            (
                result.result_id,
                result.interaction.job_id,
                result.interaction.interaction_id,
                ref.object_sha256,
                self.store._record_json(result),
            ),
        )

    def _persist_resume_result(
        self,
        connection: sqlite3.Connection,
        result: UserCheckpointResumeResultV2,
    ) -> None:
        ref = user_checkpoint_resume_result_v2_ref(result)
        connection.execute(
            """
            INSERT INTO user_checkpoint_results (
                result_id, job_id, result_type, interaction_id,
                result_sha256, record_json
            ) VALUES (?, ?, 'RESUME', ?, ?, ?)
            """,
            (
                result.result_id,
                result.interaction.job_id,
                result.interaction.interaction_id,
                ref.object_sha256,
                self.store._record_json(result),
            ),
        )

    def _load_decision_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> UserCheckpointDecisionResultV2:
        row = self._load_result_row(connection, result_id, "DECISION")
        try:
            result = UserCheckpointDecisionResultV2.model_validate_json(str(row["record_json"]))
            validate_user_checkpoint_decision_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint decision result is malformed") from exc
        if (
            str(row["job_id"]) != result.interaction.job_id
            or str(row["interaction_id"]) != result.interaction.interaction_id
            or str(row["result_sha256"]) != result.result_sha256
            or self._load_interaction_from_chain(
                connection,
                result.interaction.interaction_id,
            )
            != result.interaction
        ):
            raise ImmutableResultError("stored checkpoint decision result is inconsistent")
        if (
            result.next_interaction is not None
            and self._load_interaction_from_chain(
                connection,
                result.next_interaction.interaction_id,
            )
            != result.next_interaction
        ):
            raise ImmutableResultError("stored next checkpoint interaction differs from decision")
        commit = self.decisions._load_commit(
            connection,
            result.decision_commit.commit_id,
        )
        if commit != result.decision_commit:
            raise ImmutableResultError("stored decision commit differs from checkpoint result")
        return result

    def _load_resume_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> UserCheckpointResumeResultV2:
        row = self._load_result_row(connection, result_id, "RESUME")
        try:
            result = UserCheckpointResumeResultV2.model_validate_json(str(row["record_json"]))
            validate_user_checkpoint_resume_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint resume result is malformed") from exc
        if (
            str(row["job_id"]) != result.interaction.job_id
            or str(row["interaction_id"]) != result.interaction.interaction_id
            or str(row["result_sha256"]) != result.result_sha256
            or self._load_interaction_from_chain(
                connection,
                result.interaction.interaction_id,
            )
            != result.interaction
        ):
            raise ImmutableResultError("stored checkpoint resume result is inconsistent")
        return result

    @staticmethod
    def _load_result_row(
        connection: sqlite3.Connection,
        result_id: str,
        expected_type: str,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT job_id, result_type, interaction_id,
                   result_sha256, record_json
            FROM user_checkpoint_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"checkpoint result not found: {result_id}")
        if str(row["result_type"]) != expected_type:
            raise ImmutableResultError("stored checkpoint result type is inconsistent")
        return cast(sqlite3.Row, row)

    def get_interaction(
        self,
        interaction_id: str,
    ) -> UserCheckpointInteractionV2:
        with self.store._connect() as connection:
            return self._load_interaction_from_chain(
                connection,
                interaction_id,
            )

    def get_current_interaction(
        self,
        job_id: str,
    ) -> UserCheckpointInteractionV2:
        with self.store._connect() as connection:
            interaction = self._load_current_optional(connection, job_id)
            if interaction is None:
                raise RecordNotFoundError(f"current checkpoint interaction not found: {job_id}")
            return interaction

    def get_current_interaction_optional(
        self,
        job_id: str,
    ) -> UserCheckpointInteractionV2 | None:
        with self.store._connect() as connection:
            return self._load_current_optional(connection, job_id)

    def get_result(
        self,
        result_id: str,
    ) -> UserCheckpointOpenResultV2 | UserCheckpointDecisionResultV2 | UserCheckpointResumeResultV2:
        with self.store._connect() as connection:
            row = connection.execute(
                """
                SELECT result_type
                FROM user_checkpoint_results
                WHERE result_id = ?
                """,
                (result_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"checkpoint result not found: {result_id}")
            result_type = str(row["result_type"])
            if result_type == "OPEN":
                return self._load_open_result(connection, result_id)
            if result_type == "DECISION":
                return self._load_decision_result(connection, result_id)
            if result_type == "RESUME":
                return self._load_resume_result(connection, result_id)
            raise ImmutableResultError("stored checkpoint result type is unsupported")

    def rebuild_current_head(
        self,
        job_id: str,
    ) -> UserCheckpointInteractionV2 | None:
        with self.store._transaction() as connection:
            self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            values = self._load_interaction_chain(connection, job_id)
            connection.execute(
                """
                DELETE FROM user_checkpoint_current_heads
                WHERE job_id = ?
                """,
                (job_id,),
            )
            if not values:
                return None
            latest = values[-1]
            self._replace_head(connection, latest)
            return latest

    def list_interactions(
        self,
        job_id: str,
        *,
        offset: int,
        limit: int,
    ) -> UserCheckpointInteractionPageV2:
        if offset < 0 or limit < 1 or limit > 10_000:
            raise UserCheckpointInteractionError("checkpoint interaction page is invalid")
        with self.store._connect() as connection:
            self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            all_values = self._load_interaction_chain(connection, job_id)
            total = len(all_values)
            if offset > total:
                raise UserCheckpointInteractionError("checkpoint interaction page offset exceeds total")
            values = all_values[offset : offset + limit]
            return UserCheckpointInteractionPageV2(
                job_id=job_id,
                total=total,
                offset=offset,
                limit=limit,
                interactions=values,
            )

    def show_interaction(
        self,
        interaction_id: str,
    ) -> UserCheckpointShowResultV2:
        interaction = self.get_interaction(interaction_id)
        presentation = self.material_store.get_presentation(interaction.presentation_ref)
        context = self.material_store.get_source_context(interaction.source_context_ref)
        request = next(
            (
                value
                for value in context.request_compilation.requests
                if value.request_id == interaction.request_ref.object_id
            ),
            None,
        )
        if request is None:
            raise UserCheckpointInteractionIntegrityError("checkpoint request is missing from source context")
        if (
            presentation.job_id != interaction.job_id
            or presentation.request_ref != interaction.request_ref
            or presentation.checkpoint is not interaction.checkpoint
        ):
            raise UserCheckpointInteractionIntegrityError(
                "checkpoint presentation differs from interaction authority"
            )
        return UserCheckpointShowResultV2(
            interaction=interaction,
            request=request,
            presentation=presentation,
        )

    def _open_empty(
        self,
        *,
        context: UserCheckpointSourceContextV2,
        policy: UserCheckpointInteractionPolicyV2,
        outcome: UserCheckpointOpenOutcomeV2,
        expected_job_version: int,
        idempotency_key: str,
        request_sha256: str,
        scope: str,
        audit: ContractAudit,
    ) -> UserCheckpointOpenResultV2:
        with self.store._transaction() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                if prior[0] != "USER_CHECKPOINT_OPEN_RESULT":
                    raise IdempotencyConflictError("checkpoint open replay response type is corrupt")
                return self._load_open_result(connection, prior[1])
            self._validate_current_context(
                connection,
                context,
                material_required=False,
            )
            current = self._load_current_optional(
                connection,
                context.job_spec.job_id,
            )
            if current is not None and current.state is not UserCheckpointInteractionStateV2.RESUMED:
                raise UserCheckpointInteractionConflictError(
                    "empty checkpoint cannot replace unresolved successor authority"
                )
            job = self._load_open_job(
                connection,
                job_id=context.job_spec.job_id,
                expected_job_version=expected_job_version,
                current=current,
            )
            policy_ref = user_checkpoint_interaction_policy_v2_ref(policy)
            self._persist_policy(connection, policy)
            result = UserCheckpointOpenResultV2.create(
                job_id=job.job_id,
                outcome=outcome,
                interaction=None,
                job_status=job.status,
                job_version=job.row_version,
                policy_ref=policy_ref,
                audit=audit,
            )
            self._persist_result(connection, result)
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="USER_CHECKPOINT_OPEN_RESULT",
                response_id=result.result_id,
                created_at=self.store._clock(),
            )
            return result

    def _validate_current_context(
        self,
        connection: sqlite3.Connection,
        context: UserCheckpointSourceContextV2,
        *,
        material_required: bool = True,
    ) -> None:
        stored_spec = self.store._get_job_spec(
            connection,
            context.job_spec.job_id,
        )
        if stored_spec != context.job_spec or dataset_job_spec_v2_ref(stored_spec) != dataset_job_spec_v2_ref(
            context.job_spec
        ):
            raise UserCheckpointInteractionIntegrityError(
                "stored Job spec differs from checkpoint source context"
            )
        if material_required:
            expected_ref = user_checkpoint_source_context_v2_ref(context)
            material = self.material_store.get_source_context(expected_ref)
            if user_checkpoint_source_context_v2_ref(material) != expected_ref:
                raise UserCheckpointInteractionIntegrityError(
                    "checkpoint source context differs from immutable material"
                )

    def _select_open_request(
        self,
        connection: sqlite3.Connection,
        *,
        context: UserCheckpointSourceContextV2,
        presentations: tuple[UserCheckpointPresentationV2, ...],
        current: UserCheckpointInteractionV2 | None,
    ) -> tuple[UserApprovalRequest, UserCheckpointPresentationV2]:
        presentation_by_request = {value.request_ref.object_id: value for value in presentations}
        selected: UserApprovalRequest | None
        if current is not None and current.state is UserCheckpointInteractionStateV2.WAITING_REVISION:
            selected = self._materialized_successor_request(
                connection,
                context=context,
                current=current,
            )
        elif current is not None and current.state is UserCheckpointInteractionStateV2.DEFERRED:
            selected = self._deferred_successor_request(
                connection,
                context=context,
                current=current,
            )
        else:
            selected = next(
                (
                    request
                    for request in context.request_compilation.requests
                    if not self._request_has_decision(
                        connection,
                        request.request_id,
                    )
                ),
                None,
            )
            if selected is None:
                raise UserCheckpointInteractionConflictError(
                    "checkpoint compilation has no undecided request"
                )
        presentation = presentation_by_request.get(selected.request_id)
        if presentation is None:
            raise UserCheckpointInteractionIntegrityError("selected checkpoint request has no presentation")
        return selected, presentation

    def _materialized_successor_request(
        self,
        connection: sqlite3.Connection,
        *,
        context: UserCheckpointSourceContextV2,
        current: UserCheckpointInteractionV2,
    ) -> UserApprovalRequest:
        commit = self._load_current_decision_commit(connection, current)
        pending = commit.request_revision
        if (
            commit.outcome is not UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED
            or pending is None
            or pending.state is not UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED
        ):
            raise UserCheckpointInteractionIntegrityError(
                "waiting revision interaction has no pending request revision"
            )
        if (
            current.checkpoint is not context.request_compilation.checkpoint
            or user_decision_handling_policy_v2_ref(context.handling_policy) != commit.decision_policy_ref
        ):
            raise UserCheckpointInteractionConflictError("checkpoint successor decision policy is stale")
        pending_ref = user_approval_request_revision_v2_ref(pending)
        rows = connection.execute(
            """
            SELECT revision_id
            FROM user_approval_request_revisions
            WHERE source_request_id = ?
            ORDER BY revision_number
            """,
            (current.request_ref.object_id,),
        ).fetchall()
        materialized = tuple(
            revision
            for row in rows
            for revision in (
                self.decisions._load_revision(
                    connection,
                    str(row["revision_id"]),
                ),
            )
            if (
                revision.state is UserApprovalRequestRevisionStateV2.MATERIALIZED
                and revision.predecessor_revision_ref == pending_ref
            )
        )
        if not materialized:
            raise UserCheckpointInteractionConflictError(
                "checkpoint successor requires a materialized request revision"
            )
        if len(materialized) != 1:
            raise UserCheckpointInteractionIntegrityError(
                "checkpoint successor has conflicting materialized revisions"
            )
        successor_ref = materialized[0].successor_request_ref
        selected = next(
            (
                request
                for request in context.request_compilation.requests
                if user_approval_request_ref(request) == successor_ref
            ),
            None,
        )
        if selected is None:
            raise UserCheckpointInteractionConflictError(
                "checkpoint compilation does not contain the materialized successor"
            )
        if self._request_has_decision(connection, selected.request_id):
            raise UserCheckpointInteractionConflictError(
                "materialized checkpoint successor is already decided"
            )
        return selected

    def _deferred_successor_request(
        self,
        connection: sqlite3.Connection,
        *,
        context: UserCheckpointSourceContextV2,
        current: UserCheckpointInteractionV2,
    ) -> UserApprovalRequest:
        commit = self._load_current_decision_commit(connection, current)
        if (
            commit.outcome is not UserDecisionCommitOutcomeV2.DEFERRED
            or current.checkpoint is not ApprovalCheckpoint.FINAL_DATASET_REVIEW
            or context.request_compilation.checkpoint is not ApprovalCheckpoint.FINAL_DATASET_REVIEW
            or user_decision_handling_policy_v2_ref(context.handling_policy) != commit.decision_policy_ref
        ):
            raise UserCheckpointInteractionConflictError("deferred checkpoint successor authority is stale")
        prior_context = self.material_store.get_source_context(current.source_context_ref)
        prior_request = next(
            (
                request
                for request in prior_context.request_compilation.requests
                if request.request_id == current.request_ref.object_id
            ),
            None,
        )
        if prior_request is None:
            raise UserCheckpointInteractionIntegrityError("deferred checkpoint source request is missing")
        candidates = tuple(
            request
            for request in context.request_compilation.requests
            if (
                request.request_id != prior_request.request_id
                and request.checkpoint is prior_request.checkpoint
                and request.requested_by == prior_request.requested_by
                and request.approval_policy_ref == prior_request.approval_policy_ref
                and request.subject_refs == prior_request.subject_refs
                and request.plan_ref == prior_request.plan_ref
                and not self._request_has_decision(
                    connection,
                    request.request_id,
                )
            )
        )
        if len(candidates) != 1:
            raise UserCheckpointInteractionConflictError(
                "deferred checkpoint requires one new current successor request"
            )
        return candidates[0]

    def _load_current_decision_commit(
        self,
        connection: sqlite3.Connection,
        current: UserCheckpointInteractionV2,
    ) -> UserDecisionCommitResultV2:
        if current.decision_commit_ref is None:
            raise UserCheckpointInteractionIntegrityError("checkpoint interaction has no decision commit")
        commit = self.decisions._load_commit(
            connection,
            current.decision_commit_ref.object_id,
        )
        if (
            user_decision_commit_result_v2_ref(commit) != current.decision_commit_ref
            or commit.request_ref != current.request_ref
        ):
            raise UserCheckpointInteractionIntegrityError(
                "checkpoint interaction decision authority is stale"
            )
        return commit

    @staticmethod
    def _request_has_decision(
        connection: sqlite3.Connection,
        request_id: str,
    ) -> bool:
        return (
            connection.execute(
                """
                SELECT 1
                FROM user_decision_records
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            is not None
        )

    def _load_open_job(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        expected_job_version: int,
        current: UserCheckpointInteractionV2 | None,
    ) -> JobRecord:
        job = self.store._get_record(
            connection,
            "jobs",
            "job_id",
            job_id,
            JobRecord,
        )
        self.store._require_version(
            job.row_version,
            expected_job_version,
            job_id,
        )
        if current is None or current.state is UserCheckpointInteractionStateV2.RESUMED:
            expected_status = JobStatus.RUNNING
        elif current.state in {
            UserCheckpointInteractionStateV2.WAITING_REVISION,
            UserCheckpointInteractionStateV2.DEFERRED,
        }:
            expected_status = JobStatus.BLOCKED
        else:
            raise UserCheckpointInteractionConflictError(
                "current checkpoint interaction cannot accept a successor"
            )
        if job.status is not expected_status:
            raise UserCheckpointInteractionConflictError(
                f"checkpoint open requires a {expected_status.value} Job"
            )
        return job

    def _persist_policy(
        self,
        connection: sqlite3.Connection,
        policy: UserCheckpointInteractionPolicyV2,
    ) -> None:
        ref = user_checkpoint_interaction_policy_v2_ref(policy)
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM user_checkpoint_interaction_policies
            WHERE policy_id = ?
            """,
            (policy.policy_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO user_checkpoint_interaction_policies (
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
        try:
            stored = UserCheckpointInteractionPolicyV2.model_validate_json(str(row["record_json"]))
            stored_ref = user_checkpoint_interaction_policy_v2_ref(stored)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint interaction policy is malformed") from exc
        if str(row["policy_sha256"]) != stored.policy_sha256 or stored_ref != ref or stored != policy:
            raise ImmutableResultError("stored checkpoint interaction policy differs")

    def _load_policy(
        self,
        connection: sqlite3.Connection,
        policy_ref: ObjectRef,
    ) -> UserCheckpointInteractionPolicyV2:
        row = connection.execute(
            """
            SELECT policy_sha256, record_json
            FROM user_checkpoint_interaction_policies
            WHERE policy_id = ?
            """,
            (policy_ref.object_id,),
        ).fetchone()
        if row is None:
            raise ImmutableResultError("checkpoint interaction is missing its policy")
        try:
            policy = UserCheckpointInteractionPolicyV2.model_validate_json(str(row["record_json"]))
            observed_ref = user_checkpoint_interaction_policy_v2_ref(policy)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint interaction policy is malformed") from exc
        if str(row["policy_sha256"]) != policy.policy_sha256 or observed_ref != policy_ref:
            raise ImmutableResultError("stored checkpoint interaction policy columns are inconsistent")
        return policy

    def _require_interaction_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        policy_ref: ObjectRef,
    ) -> None:
        policy = self._load_policy(connection, policy_ref)
        count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM user_checkpoint_interactions
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()[0]
        )
        if count + 1 > policy.max_interactions_per_job:
            raise UserCheckpointInteractionError("checkpoint interaction count limit exceeded")

    def _insert_interaction(
        self,
        connection: sqlite3.Connection,
        interaction: UserCheckpointInteractionV2,
    ) -> None:
        validate_user_checkpoint_interaction_v2_identity(interaction)
        connection.execute(
            """
            INSERT INTO user_checkpoint_interactions (
                interaction_id, job_id, chain_id, interaction_version,
                checkpoint, state, request_id, decision_commit_id,
                interaction_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction.interaction_id,
                interaction.job_id,
                interaction.chain_id,
                interaction.interaction_version,
                interaction.checkpoint.value,
                interaction.state.value,
                interaction.request_ref.object_id,
                (
                    interaction.decision_commit_ref.object_id
                    if interaction.decision_commit_ref is not None
                    else None
                ),
                interaction.interaction_sha256,
                self.store._record_json(interaction),
            ),
        )

    def _replace_head(
        self,
        connection: sqlite3.Connection,
        interaction: UserCheckpointInteractionV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO user_checkpoint_current_heads (
                job_id, interaction_id, chain_id,
                interaction_version, state
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                interaction_id = excluded.interaction_id,
                chain_id = excluded.chain_id,
                interaction_version = excluded.interaction_version,
                state = excluded.state
            """,
            (
                interaction.job_id,
                interaction.interaction_id,
                interaction.chain_id,
                interaction.interaction_version,
                interaction.state.value,
            ),
        )

    def _persist_result(
        self,
        connection: sqlite3.Connection,
        result: UserCheckpointOpenResultV2,
    ) -> None:
        ref = user_checkpoint_open_result_v2_ref(result)
        connection.execute(
            """
            INSERT INTO user_checkpoint_results (
                result_id, job_id, result_type, interaction_id,
                result_sha256, record_json
            ) VALUES (?, ?, 'OPEN', ?, ?, ?)
            """,
            (
                result.result_id,
                result.job_id,
                (result.interaction.interaction_id if result.interaction is not None else None),
                ref.object_sha256,
                self.store._record_json(result),
            ),
        )

    def _load_open_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> UserCheckpointOpenResultV2:
        row = connection.execute(
            """
            SELECT job_id, result_type, interaction_id,
                   result_sha256, record_json
            FROM user_checkpoint_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"checkpoint open result not found: {result_id}")
        try:
            result = UserCheckpointOpenResultV2.model_validate_json(str(row["record_json"]))
            validate_user_checkpoint_open_result_v2_identity(result)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint open result is malformed") from exc
        interaction_id = result.interaction.interaction_id if result.interaction is not None else None
        if (
            str(row["job_id"]) != result.job_id
            or str(row["result_type"]) != "OPEN"
            or row["interaction_id"] != interaction_id
            or str(row["result_sha256"]) != result.result_sha256
        ):
            raise ImmutableResultError("stored checkpoint open result columns are inconsistent")
        if result.interaction is not None:
            stored = self._load_interaction_from_chain(
                connection,
                result.interaction.interaction_id,
            )
            if stored != result.interaction:
                raise ImmutableResultError("stored checkpoint interaction differs from open result")
        return result

    def _load_interaction(
        self,
        connection: sqlite3.Connection,
        interaction_id: str,
    ) -> UserCheckpointInteractionV2:
        row = connection.execute(
            """
            SELECT interaction_id, job_id, chain_id, interaction_version,
                   checkpoint, state, request_id, decision_commit_id,
                   interaction_sha256, record_json
            FROM user_checkpoint_interactions
            WHERE interaction_id = ?
            """,
            (interaction_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"checkpoint interaction not found: {interaction_id}")
        try:
            interaction = UserCheckpointInteractionV2.model_validate_json(str(row["record_json"]))
            validate_user_checkpoint_interaction_v2_identity(interaction)
        except (ValidationError, ValueError) as exc:
            raise ImmutableResultError("stored checkpoint interaction is malformed") from exc
        decision_id = (
            interaction.decision_commit_ref.object_id if interaction.decision_commit_ref is not None else None
        )
        if (
            str(row["interaction_id"]) != interaction.interaction_id
            or str(row["job_id"]) != interaction.job_id
            or str(row["chain_id"]) != interaction.chain_id
            or int(row["interaction_version"]) != interaction.interaction_version
            or str(row["checkpoint"]) != interaction.checkpoint.value
            or str(row["state"]) != interaction.state.value
            or str(row["request_id"]) != interaction.request_ref.object_id
            or row["decision_commit_id"] != decision_id
            or str(row["interaction_sha256"]) != interaction.interaction_sha256
        ):
            raise ImmutableResultError("stored checkpoint interaction columns are inconsistent")
        self.material_store.verify_source_context(interaction.source_context_ref)
        self.material_store.verify_presentation(interaction.presentation_ref)
        if interaction.interaction_version > 1:
            predecessor_row = connection.execute(
                """
                SELECT interaction_id, job_id, chain_id,
                       interaction_version, interaction_sha256
                FROM user_checkpoint_interactions
                WHERE chain_id = ? AND interaction_version = ?
                """,
                (
                    interaction.chain_id,
                    interaction.interaction_version - 1,
                ),
            ).fetchone()
            if predecessor_row is None:
                raise ImmutableResultError("checkpoint interaction chain has a version gap")
            predecessor_ref = ObjectRef(
                object_type="user-checkpoint-interaction",
                object_id=str(predecessor_row["interaction_id"]),
                object_version="v2",
                object_sha256=str(predecessor_row["interaction_sha256"]),
            )
            if (
                str(predecessor_row["job_id"]) != interaction.job_id
                or str(predecessor_row["chain_id"]) != interaction.chain_id
                or int(predecessor_row["interaction_version"]) != interaction.interaction_version - 1
                or interaction.predecessor_interaction_ref != predecessor_ref
            ):
                raise ImmutableResultError("checkpoint interaction predecessor chain is inconsistent")
        if interaction.decision_commit_ref is not None:
            commit = self.decisions._load_commit(
                connection,
                interaction.decision_commit_ref.object_id,
            )
            if (
                user_decision_commit_result_v2_ref(commit) != interaction.decision_commit_ref
                or commit.job_id != interaction.job_id
                or commit.request_ref != interaction.request_ref
                or commit.decision_record.checkpoint is not interaction.checkpoint
            ):
                raise ImmutableResultError("checkpoint interaction decision binding is stale")
        if interaction.application_ref is not None:
            application = UserPlanApplicationPersistenceService(self.store)._load_application(
                connection,
                interaction.application_ref.object_id,
            )
            if (
                user_plan_application_v2_ref(application) != interaction.application_ref
                or application.job_id != interaction.job_id
                or application.decision_commit_ref != interaction.decision_commit_ref
                or application.checkpoint is not interaction.checkpoint
            ):
                raise ImmutableResultError("checkpoint interaction application binding is stale")
        return interaction

    def _load_interaction_chain(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> tuple[UserCheckpointInteractionV2, ...]:
        rows = connection.execute(
            """
            SELECT interaction_id
            FROM user_checkpoint_interactions
            WHERE job_id = ?
            ORDER BY interaction_version
            """,
            (job_id,),
        ).fetchall()
        values = tuple(
            self._load_interaction(
                connection,
                str(row["interaction_id"]),
            )
            for row in rows
        )
        for index, value in enumerate(values):
            if value.interaction_version != index + 1:
                raise ImmutableResultError("checkpoint interaction chain has a version gap")
            expected_predecessor = (
                None if index == 0 else user_checkpoint_interaction_v2_ref(values[index - 1])
            )
            if (
                value.chain_id != values[0].chain_id
                or value.predecessor_interaction_ref != expected_predecessor
            ):
                raise ImmutableResultError("checkpoint interaction chain is inconsistent")
        return values

    def _load_interaction_from_chain(
        self,
        connection: sqlite3.Connection,
        interaction_id: str,
    ) -> UserCheckpointInteractionV2:
        interaction = self._load_interaction(
            connection,
            interaction_id,
        )
        chain = self._load_interaction_chain(
            connection,
            interaction.job_id,
        )
        stored = next(
            (value for value in chain if value.interaction_id == interaction_id),
            None,
        )
        if stored is None or stored != interaction:
            raise ImmutableResultError("checkpoint interaction is missing from its immutable chain")
        return stored

    def _load_current_optional(
        self,
        connection: sqlite3.Connection,
        job_id: str,
    ) -> UserCheckpointInteractionV2 | None:
        row = connection.execute(
            """
            SELECT interaction_id, chain_id, interaction_version, state
            FROM user_checkpoint_current_heads
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            orphan = connection.execute(
                """
                SELECT 1
                FROM user_checkpoint_interactions
                WHERE job_id = ?
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if orphan is not None:
                raise ImmutableResultError("checkpoint current head is missing")
            return None
        values = self._load_interaction_chain(connection, job_id)
        if not values:
            raise ImmutableResultError("checkpoint current head has no immutable interaction")
        interaction = values[-1]
        if (
            interaction.job_id != job_id
            or str(row["interaction_id"]) != interaction.interaction_id
            or str(row["chain_id"]) != interaction.chain_id
            or int(row["interaction_version"]) != interaction.interaction_version
            or str(row["state"]) != interaction.state.value
        ):
            raise ImmutableResultError("checkpoint current head differs from immutable interaction")
        return interaction


def _decision_mapping(
    checkpoint: ApprovalCheckpoint,
    decision: UserDecision,
) -> tuple[
    UserCheckpointInteractionStateV2,
    UserCheckpointResumeDispositionV2,
    JobStatus,
]:
    if decision is UserDecision.ACCEPT:
        return (
            UserCheckpointInteractionStateV2.RESUMABLE,
            UserCheckpointResumeDispositionV2.DIRECT,
            JobStatus.BLOCKED,
        )
    if decision is UserDecision.ADJUST:
        return (
            UserCheckpointInteractionStateV2.WAITING_REVALIDATION,
            UserCheckpointResumeDispositionV2.AFTER_REVALIDATION,
            JobStatus.BLOCKED,
        )
    if decision is UserDecision.REQUEST_MORE_EXAMPLES:
        return (
            UserCheckpointInteractionStateV2.WAITING_REVISION,
            UserCheckpointResumeDispositionV2.WAITING_REVISION,
            JobStatus.BLOCKED,
        )
    if decision is UserDecision.DEFER:
        return (
            UserCheckpointInteractionStateV2.DEFERRED,
            UserCheckpointResumeDispositionV2.DEFERRED,
            JobStatus.BLOCKED,
        )
    if checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW:
        return (
            UserCheckpointInteractionStateV2.RESUMABLE,
            UserCheckpointResumeDispositionV2.DIRECT,
            JobStatus.BLOCKED,
        )
    return (
        UserCheckpointInteractionStateV2.TERMINATED,
        UserCheckpointResumeDispositionV2.TERMINATED,
        JobStatus.FAILED,
    )


def _chain_id(job_id: str) -> str:
    digest = hashlib.sha256(job_id.encode()).hexdigest()
    return f"user-checkpoint-chain://sha256/{digest}"


__all__ = [
    "UserCheckpointAuthenticationError",
    "UserCheckpointInteractionConflictError",
    "UserCheckpointInteractionError",
    "UserCheckpointInteractionIntegrityError",
    "UserCheckpointInteractionNotResumableError",
    "UserCheckpointInteractionService",
]
