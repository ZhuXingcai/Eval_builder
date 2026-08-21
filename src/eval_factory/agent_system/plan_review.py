from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Self, cast

from pydantic import Field, model_validator

from eval_factory.agent_system.delivery_planning import (
    DatasetDeliveryPlanCompiler,
)
from eval_factory.agent_system.plan_adapters import (
    AttachmentGenerationPlanAdapter,
    CompiledReviewablePlanV2,
    CriteriaRubricPlanAdapter,
    DatasetDeliveryPlanAdapter,
    GlobalBuildPlanAdapter,
    GradingDesignPlanAdapter,
    ReviewablePlanAdapter,
    ReviewablePlanAdapterError,
    ReviewablePlanAdapterRegistry,
    ReviewablePlanV2,
    erased_adapter,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.store import (
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentGenerationPlanV2,
    CompiledDatasetBuildPlanV2,
    CriteriaRubricPlanV2,
    DatasetBuildPlanV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    GradingDesignPlanV2,
    PlanDecisionKindV2,
    PlanDecisionV2,
    PlanKindV2,
    PlanReviewPresentationV2,
    PlanReviewRequestV2,
    PlanReviewResultV2,
    PlanReviewStateV2,
    PlanRevisionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.dataset_runtime_v2 import (
    DatasetDeliveryPlanV2,
)


class PlanReviewError(RuntimeError):
    pass


class PlanReviewConflictError(PlanReviewError):
    pass


class PlanReviewIntegrityError(PlanReviewError):
    pass


class PlanReviewNotResumableError(PlanReviewError):
    pass


class PlanReviewDecisionSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-decision-submission/v1"] = (
        "eval-factory/plan-review-decision-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    decision: PlanDecisionKindV2
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def reject_edit(self) -> Self:
        if self.decision is PlanDecisionKindV2.EDIT:
            raise ValueError("edit decisions require a typed plan revision")
        return self


class PlanReviewEditSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-edit-submission/v1"] = (
        "eval-factory/plan-review-edit-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: DatasetBuildPlanV2
    changed_paths: tuple[str, ...] = Field(min_length=1, max_length=256)
    invalidated_object_refs: tuple[ObjectRef, ...] = Field(default=(), max_length=100_000)
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        keys = tuple(_ref_key(value) for value in self.invalidated_object_refs)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("invalidated_object_refs must be sorted and unique")
        return self


class AttachmentPlanReviewEditSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-plan-review-edit-submission/v1"] = (
        "eval-factory/attachment-plan-review-edit-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: AttachmentGenerationPlanV2
    changed_paths: tuple[str, ...] = Field(min_length=1, max_length=256)
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        return self


class CriteriaRubricPlanReviewEditSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/criteria-rubric-plan-review-edit-submission/v1"] = (
        "eval-factory/criteria-rubric-plan-review-edit-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: CriteriaRubricPlanV2
    changed_paths: tuple[str, ...] = Field(min_length=1, max_length=256)
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        return self


class GradingDesignPlanReviewEditSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/grading-design-plan-review-edit-submission/v1"] = (
        "eval-factory/grading-design-plan-review-edit-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: GradingDesignPlanV2
    changed_paths: tuple[str, ...] = Field(
        min_length=1,
        max_length=256,
    )
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        return self


class DatasetDeliveryPlanReviewEditSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/dataset-delivery-plan-review-edit-submission/v1"] = (
        "eval-factory/dataset-delivery-plan-review-edit-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    edited_plan: DatasetDeliveryPlanV2
    changed_paths: tuple[str, ...] = Field(
        min_length=1,
        max_length=256,
    )
    decided_by: str = Field(min_length=3, max_length=256)
    reason_code: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=3, max_length=256)

    @model_validator(mode="after")
    def validate_edit(self) -> Self:
        if tuple(sorted(set(self.changed_paths))) != self.changed_paths:
            raise ValueError("changed_paths must be sorted and unique")
        return self


class PlanReviewResumeSubmissionV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-resume-submission/v1"] = (
        "eval-factory/plan-review-resume-submission/v1"
    )
    expected_plan_version: int = Field(ge=1, le=1_000_000)
    resumed_by: str = Field(min_length=3, max_length=256)
    idempotency_key: str = Field(min_length=3, max_length=256)


class PlanReviewViewV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-view/v1"] = "eval-factory/plan-review-view/v1"
    run_id: str
    current_run_ref: ObjectRef
    request: PlanReviewRequestV2
    presentation: PlanReviewPresentationV2
    result: PlanReviewResultV2
    plan: ReviewablePlanV2
    decision: PlanDecisionV2 | None
    revision: PlanRevisionV2 | None


class PlanReviewPageV1(ContractModelV2):
    schema_version: Literal["eval-factory/plan-review-page/v1"] = "eval-factory/plan-review-page/v1"
    items: tuple[PlanReviewViewV1, ...]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    total: int = Field(ge=0)


class PlanReviewService:
    def __init__(
        self,
        store: FactoryControlStore,
        *,
        compiler: DatasetBuildPlanCompiler | None = None,
        adapter_registry: ReviewablePlanAdapterRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.compiler = compiler
        self.adapter_registry = adapter_registry or (
            ReviewablePlanAdapterRegistry(
                (
                    erased_adapter(GlobalBuildPlanAdapter(compiler)),
                    erased_adapter(AttachmentGenerationPlanAdapter(None)),
                    erased_adapter(CriteriaRubricPlanAdapter(None)),
                    erased_adapter(GradingDesignPlanAdapter(None)),
                    erased_adapter(DatasetDeliveryPlanAdapter(DatasetDeliveryPlanCompiler())),
                )
            )
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._initialize()

    def open_global_plan(
        self,
        *,
        run_id: str,
        title: str,
        summary_lines: tuple[str, ...],
        editable_paths: tuple[str, ...],
        warning_codes: tuple[str, ...],
        requested_by: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        plan, _ = self.store.get_plan(run_id)
        presentation = PlanReviewPresentationV2.create(
            presentation_id=f"plan-review-presentation://{_slug(run_id)}/{plan.plan_version}",
            plan_ref=plan.to_ref(),
            plan_kind=PlanKindV2.GLOBAL_BUILD,
            title=title,
            summary_lines=summary_lines,
            editable_paths=tuple(sorted(editable_paths)),
            warning_codes=tuple(sorted(warning_codes)),
            audit=audit,
        )
        return self._open_plan(
            run_id=run_id,
            plan=plan,
            plan_kind=PlanKindV2.GLOBAL_BUILD,
            presentation=presentation,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            audit=audit,
        )

    def open_plan(
        self,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        requested_by: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            raise PlanReviewError("global plans require open_global_plan")
        material = self.store.get_domain_plan(run_id, plan_kind)
        adapter = self._adapter(plan_kind, material.plan_ref)
        plan = adapter.parse(material.plan_record_json)
        if plan.to_ref() != material.plan_ref:
            raise PlanReviewIntegrityError("domain plan adapter parsed another authority")
        presentation = adapter.safe_presentation(plan)
        return self._open_plan(
            run_id=run_id,
            plan=plan,
            plan_kind=plan_kind,
            presentation=presentation,
            requested_by=requested_by,
            idempotency_key=idempotency_key,
            audit=audit,
        )

    def _open_plan(
        self,
        *,
        run_id: str,
        plan: ReviewablePlanV2,
        plan_kind: PlanKindV2,
        presentation: PlanReviewPresentationV2,
        requested_by: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        run = self.store.get_run(run_id)
        request_run_ref = run.to_ref()
        if run.status is FactoryRunStatusV2.WAITING_REVIEW and run.pending_review_ref is not None:
            existing = self.show(run.pending_review_ref.object_id)
            request_run_ref = existing.request.run_ref
        request_suffix = (
            str(plan.plan_version)
            if plan_kind is PlanKindV2.GLOBAL_BUILD
            else (f"{plan_kind.value.casefold().replace('_', '-')}/{plan.plan_version}")
        )
        request = PlanReviewRequestV2.create(
            review_request_id=(f"plan-review-request://{_slug(run_id)}/{request_suffix}"),
            run_ref=request_run_ref,
            plan_ref=plan.to_ref(),
            plan_kind=plan_kind,
            plan_version=plan.plan_version,
            presentation_ref=presentation.to_ref(),
            requested_by=requested_by,
            audit=audit,
        )
        pending = PlanReviewResultV2.create(
            result_id=f"plan-review-result://{_slug(request.review_request_id)}/pending",
            review_request_ref=request.to_ref(),
            plan_ref=plan.to_ref(),
            plan_version=plan.plan_version,
            state=PlanReviewStateV2.PENDING_REVIEW,
            decision_ref=None,
            successor_plan_ref=None,
            resume_token_ref=None,
            audit=audit,
        )
        request_sha256 = _request_sha256(request, presentation)
        scope = _open_scope(run_id, plan_kind)
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_result(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay is not None:
                connection.rollback()
                return self.show(request.review_request_id)
            current = self.store._load_current_run(connection, run_id)
            if (
                current != run
                or current.status is not FactoryRunStatusV2.PLANNING
                or not self._is_current_plan(
                    connection,
                    run_id=run_id,
                    plan_kind=plan_kind,
                    plan_ref=plan.to_ref(),
                )
            ):
                raise PlanReviewConflictError("plan review requires the current planning authority")
            existing = connection.execute(
                "SELECT 1 FROM plan_review_requests WHERE plan_object_id = ?",
                (plan.object_id,),
            ).fetchone()
            if existing is not None:
                raise PlanReviewConflictError("current plan already has a review request")
            self._insert_presentation(connection, presentation)
            self._insert_request(connection, run_id, request)
            self._insert_result(connection, pending)
            self._replace_head(connection, pending)
            successor = self._append_run(
                connection,
                current=current,
                status=FactoryRunStatusV2.WAITING_REVIEW,
                pending_review_ref=request.to_ref(),
                audit=audit,
            )
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=successor.run_version,
                event_type="plan-review-opened",
                object_id=request.object_id,
                created_at=self._clock().isoformat(),
            )
            self._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_id=pending.object_id,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PlanReviewConflictError("plan review open authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.show(request.review_request_id)

    def decide(
        self,
        review_request_id: str,
        submission: PlanReviewDecisionSubmissionV1,
        *,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        parsed = PlanReviewDecisionSubmissionV1.model_validate(submission.model_dump(mode="python"))
        initial = self.show(review_request_id)
        if parsed.expected_plan_version != initial.request.plan_version:
            raise PlanReviewConflictError("plan review decision uses a stale plan version")
        request_sha256 = _request_sha256(initial.request.to_ref(), parsed)
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_result(
                connection,
                scope=f"decide-plan-review:{initial.request.review_request_id}",
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
            )
            if replay is not None:
                connection.rollback()
                return self.show(review_request_id)
            run_id, request, current = self._load_current(connection, review_request_id)
            if current.state is not PlanReviewStateV2.PENDING_REVIEW:
                raise PlanReviewConflictError("plan review decision requires a pending review")
            if parsed.decided_by != request.requested_by:
                raise PlanReviewConflictError("plan review decision principal does not match requester")
            state = {
                PlanDecisionKindV2.APPROVE: PlanReviewStateV2.APPROVED,
                PlanDecisionKindV2.REJECT: PlanReviewStateV2.REJECTED,
                PlanDecisionKindV2.DEFER: PlanReviewStateV2.DEFERRED,
                PlanDecisionKindV2.REQUEST_MORE: PlanReviewStateV2.REVISION_REQUESTED,
            }[parsed.decision]
            decision = self._decision(
                request=request,
                expected_plan_version=parsed.expected_plan_version,
                decision=parsed.decision,
                revision_ref=None,
                decided_by=parsed.decided_by,
                reason_code=parsed.reason_code,
                idempotency_key=parsed.idempotency_key,
                audit=audit,
            )
            resume_token = (
                _stable_ref("plan-review-resume-token", decision.object_id)
                if parsed.decision is PlanDecisionKindV2.APPROVE
                else None
            )
            result = PlanReviewResultV2.create(
                result_id=f"plan-review-result://{_slug(review_request_id)}/{state.value.casefold()}",
                review_request_ref=request.to_ref(),
                plan_ref=request.plan_ref,
                plan_version=request.plan_version,
                state=state,
                decision_ref=decision.to_ref(),
                successor_plan_ref=None,
                resume_token_ref=resume_token,
                audit=audit,
            )
            self._insert_decision(connection, decision)
            self._insert_result(connection, result)
            self._replace_head(connection, result)
            run = self.store._load_current_run(connection, run_id)
            self._validate_waiting_run(
                connection,
                run_id=run_id,
                request=request,
                run=run,
            )
            aggregate_version = run.run_version
            if parsed.decision is PlanDecisionKindV2.REJECT:
                run = self._append_run(
                    connection,
                    current=run,
                    status=FactoryRunStatusV2.FAILED,
                    pending_review_ref=None,
                    audit=audit,
                )
                aggregate_version = run.run_version
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=aggregate_version,
                event_type="plan-review-decided",
                object_id=result.object_id,
                created_at=self._clock().isoformat(),
            )
            self._record_idempotency(
                connection,
                scope=f"decide-plan-review:{review_request_id}",
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
                response_id=result.object_id,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PlanReviewConflictError("plan review decision authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.show(review_request_id)

    def edit_global_plan(
        self,
        review_request_id: str,
        submission: PlanReviewEditSubmissionV1,
        *,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        parsed = PlanReviewEditSubmissionV1.model_validate(submission.model_dump(mode="python"))
        return self._edit_plan(
            review_request_id=review_request_id,
            expected_plan_version=parsed.expected_plan_version,
            successor=parsed.edited_plan,
            changed_paths=parsed.changed_paths,
            provided_invalidated_refs=(parsed.invalidated_object_refs),
            decided_by=parsed.decided_by,
            reason_code=parsed.reason_code,
            idempotency_key=parsed.idempotency_key,
            audit=audit,
        )

    def edit_plan(
        self,
        review_request_id: str,
        submission: (
            AttachmentPlanReviewEditSubmissionV1
            | CriteriaRubricPlanReviewEditSubmissionV1
            | DatasetDeliveryPlanReviewEditSubmissionV1
            | GradingDesignPlanReviewEditSubmissionV1
        ),
        *,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        parsed = type(submission).model_validate(submission.model_dump(mode="python"))
        return self._edit_plan(
            review_request_id=review_request_id,
            expected_plan_version=parsed.expected_plan_version,
            successor=parsed.edited_plan,
            changed_paths=parsed.changed_paths,
            provided_invalidated_refs=None,
            decided_by=parsed.decided_by,
            reason_code=parsed.reason_code,
            idempotency_key=parsed.idempotency_key,
            audit=audit,
        )

    def _edit_plan(
        self,
        *,
        review_request_id: str,
        expected_plan_version: int,
        successor: ReviewablePlanV2,
        changed_paths: tuple[str, ...],
        provided_invalidated_refs: tuple[ObjectRef, ...] | None,
        decided_by: str,
        reason_code: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        initial = self.show(review_request_id)
        request = initial.request
        base = self._current_plan(
            initial.run_id,
            request.plan_kind,
        )
        if base.to_ref() != request.plan_ref:
            raise PlanReviewConflictError("review request no longer binds the current base plan")
        run = self.store.get_run(initial.run_id)
        if (
            expected_plan_version != request.plan_version
            or base.plan_version != request.plan_version
            or successor.run_ref != run.to_ref()
        ):
            raise PlanReviewConflictError("edited plan is not a contiguous current successor")
        adapter = self._adapter(
            request.plan_kind,
            request.plan_ref,
        )
        if successor.to_ref().object_type != adapter.object_type:
            raise PlanReviewConflictError("edited plan changes the reviewed plan type")
        if not set(changed_paths).issubset(initial.presentation.editable_paths):
            raise PlanReviewConflictError("edited plan changes a non-editable path")
        try:
            adapter.validate_successor(base, successor)
        except ReviewablePlanAdapterError as exc:
            raise PlanReviewConflictError("edited plan is not a contiguous current successor") from exc
        invalidated_refs = adapter.invalidated_refs(
            base,
            successor,
        )
        if provided_invalidated_refs is not None and provided_invalidated_refs != invalidated_refs:
            raise PlanReviewConflictError("edited plan invalidation does not match adapter authority")
        policy = self.store.get_policy(run.policy_ref.object_id)
        try:
            compiled = adapter.compile(
                successor,
                policy=policy,
                audit=audit,
            )
        except ReviewablePlanAdapterError as exc:
            raise PlanReviewError("plan review edit requires a deterministic compiler") from exc
        request_sha256 = _request_sha256(
            request.to_ref(),
            expected_plan_version,
            successor.to_ref(),
            changed_paths,
            invalidated_refs,
            decided_by,
            reason_code,
            idempotency_key,
            compiled.to_ref(),
        )
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_result(
                connection,
                scope=f"decide-plan-review:{review_request_id}",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay is not None:
                connection.rollback()
                return self.show(review_request_id)
            run_id, stored_request, current = self._load_current(
                connection,
                review_request_id,
            )
            if stored_request != request or current.state is not PlanReviewStateV2.PENDING_REVIEW:
                raise PlanReviewConflictError("plan review edit requires the current pending review")
            waiting_run = self.store._load_current_run(connection, run_id)
            self._validate_waiting_run(
                connection,
                run_id=run_id,
                request=request,
                run=waiting_run,
            )
            if decided_by != request.requested_by:
                raise PlanReviewConflictError("plan review edit principal does not match requester")
            revision = PlanRevisionV2.create(
                revision_id=(f"plan-revision://{_slug(review_request_id)}/{successor.plan_version}"),
                review_request_ref=request.to_ref(),
                base_plan_ref=base.to_ref(),
                successor_plan_ref=successor.to_ref(),
                base_plan_version=base.plan_version,
                successor_plan_version=successor.plan_version,
                changed_paths=changed_paths,
                invalidated_object_refs=invalidated_refs,
                revised_by=decided_by,
                audit=audit,
            )
            decision = self._decision(
                request=request,
                expected_plan_version=expected_plan_version,
                decision=PlanDecisionKindV2.EDIT,
                revision_ref=revision.to_ref(),
                decided_by=decided_by,
                reason_code=reason_code,
                idempotency_key=idempotency_key,
                audit=audit,
            )
            result = PlanReviewResultV2.create(
                result_id=f"plan-review-result://{_slug(review_request_id)}/revision-requested",
                review_request_ref=request.to_ref(),
                plan_ref=base.to_ref(),
                plan_version=base.plan_version,
                state=PlanReviewStateV2.REVISION_REQUESTED,
                decision_ref=decision.to_ref(),
                successor_plan_ref=successor.to_ref(),
                resume_token_ref=_stable_ref("plan-review-resume-token", decision.object_id),
                audit=audit,
            )
            self._insert_revision(
                connection,
                run_id=run_id,
                plan_kind=request.plan_kind,
                revision=revision,
                plan=successor,
                compiled_plan=compiled,
            )
            self._insert_decision(connection, decision)
            self._insert_result(connection, result)
            self._replace_head(connection, result)
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=run.run_version,
                event_type="plan-review-edited",
                object_id=result.object_id,
                created_at=self._clock().isoformat(),
            )
            self._record_idempotency(
                connection,
                scope=f"decide-plan-review:{review_request_id}",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_id=result.object_id,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PlanReviewConflictError("plan review edit authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.show(review_request_id)

    def resume(
        self,
        review_request_id: str,
        submission: PlanReviewResumeSubmissionV1,
        *,
        audit: ContractAudit,
    ) -> PlanReviewViewV1:
        parsed = PlanReviewResumeSubmissionV1.model_validate(submission.model_dump(mode="python"))
        initial = self.show(review_request_id)
        if parsed.expected_plan_version != initial.request.plan_version:
            raise PlanReviewConflictError("plan review resume uses a stale plan version")
        request_sha256 = _request_sha256(
            initial.request.to_ref(),
            initial.decision.to_ref() if initial.decision is not None else None,
            parsed,
        )
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._idempotent_result(
                connection,
                scope=f"resume-plan-review:{review_request_id}",
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
            )
            if replay is not None:
                connection.rollback()
                return self.show(review_request_id)
            run_id, request, current = self._load_current(connection, review_request_id)
            if current.state not in {
                PlanReviewStateV2.APPROVED,
                PlanReviewStateV2.REVISION_REQUESTED,
            }:
                raise PlanReviewNotResumableError("plan review state is not resumable")
            if parsed.resumed_by != request.requested_by:
                raise PlanReviewConflictError("plan review resume principal does not match requester")
            if current.state is PlanReviewStateV2.REVISION_REQUESTED and current.successor_plan_ref is None:
                raise PlanReviewNotResumableError(
                    "request-more review requires a compiled successor before resume"
                )
            if current.resume_token_ref is None or current.decision_ref is None:
                raise PlanReviewIntegrityError("resumable plan review lacks authority refs")
            run = self.store._load_current_run(connection, run_id)
            self._validate_waiting_run(
                connection,
                run_id=run_id,
                request=request,
                run=run,
            )
            next_compiled_plan_ref = run.compiled_plan_ref
            next_run_plan_ref = run.current_plan_ref
            revision = self._load_optional_revision(connection, request.review_request_id)
            if current.state is PlanReviewStateV2.REVISION_REQUESTED:
                if revision is None or revision.successor_plan_ref != current.successor_plan_ref:
                    raise PlanReviewNotResumableError(
                        "request-more review requires an edited successor before resume"
                    )
                material = connection.execute(
                    """
                    SELECT * FROM plan_revision_materials
                    WHERE revision_id = ?
                    """,
                    (revision.revision_id,),
                ).fetchone()
                if material is None:
                    raise PlanReviewIntegrityError("plan revision material is missing")
                adapter = self._adapter(
                    request.plan_kind,
                    revision.successor_plan_ref,
                )
                try:
                    plan = adapter.parse(str(material["plan_record_json"]).encode())
                    compiled = adapter.parse_compiled(str(material["compiled_plan_record_json"]).encode())
                except ReviewablePlanAdapterError as exc:
                    raise PlanReviewIntegrityError("plan revision material is invalid") from exc
                if (
                    plan.canonical_json().decode() != material["plan_record_json"]
                    or compiled.canonical_json().decode() != material["compiled_plan_record_json"]
                    or plan.to_ref() != revision.successor_plan_ref
                    or compiled.source_plan_ref != plan.to_ref()
                ):
                    raise PlanReviewIntegrityError("plan revision material drifted")
                if request.plan_kind is PlanKindV2.GLOBAL_BUILD:
                    if not isinstance(
                        plan,
                        DatasetBuildPlanV2,
                    ) or not isinstance(
                        compiled,
                        CompiledDatasetBuildPlanV2,
                    ):
                        raise PlanReviewIntegrityError("global revision adapter returned another type")
                    next_run_plan_ref = plan.to_ref()
                    next_compiled_plan_ref = compiled.to_ref()
                    self.store._insert_plan(
                        connection,
                        run_id=run_id,
                        plan=plan,
                    )
                    self.store._insert_compiled_plan(
                        connection,
                        run_id=run_id,
                        compiled_plan=compiled,
                    )
                    connection.execute(
                        """
                        UPDATE plan_current_heads
                        SET plan_id = ?, plan_version = ?,
                            plan_object_id = ?,
                            compiled_plan_object_id = ?
                        WHERE run_id = ?
                        """,
                        (
                            plan.plan_id,
                            plan.plan_version,
                            plan.object_id,
                            compiled.object_id,
                            run_id,
                        ),
                    )
                else:
                    domain_material = self.store._load_domain_plan_material_by_ref(
                        connection,
                        run_id=run_id,
                        plan_kind=request.plan_kind,
                        reference=plan.to_ref(),
                    )
                    if (
                        domain_material.plan_record_json != plan.canonical_json()
                        or domain_material.compiled_plan_record_json != compiled.canonical_json()
                    ):
                        raise PlanReviewIntegrityError("domain revision material drifted")
                    self.store._promote_domain_plan(
                        connection,
                        run_id=run_id,
                        plan_kind=request.plan_kind,
                        material=domain_material,
                    )
            target_plan_ref = current.successor_plan_ref or current.plan_ref
            target_version = (
                current.plan_version + 1 if current.successor_plan_ref is not None else current.plan_version
            )
            resumed = PlanReviewResultV2.create(
                result_id=f"plan-review-result://{_slug(review_request_id)}/resumed",
                review_request_ref=request.to_ref(),
                plan_ref=target_plan_ref,
                plan_version=target_version,
                state=PlanReviewStateV2.RESUMED,
                decision_ref=current.decision_ref,
                successor_plan_ref=None,
                resume_token_ref=current.resume_token_ref,
                audit=audit,
            )
            self._insert_result(connection, resumed)
            self._replace_head(connection, resumed)
            successor_run = self._append_run(
                connection,
                current=run,
                status=FactoryRunStatusV2.PLANNING,
                pending_review_ref=None,
                current_plan_ref=next_run_plan_ref,
                compiled_plan_ref=next_compiled_plan_ref,
                audit=audit,
            )
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=successor_run.run_version,
                event_type="plan-review-resumed",
                object_id=resumed.object_id,
                created_at=self._clock().isoformat(),
            )
            self._record_idempotency(
                connection,
                scope=f"resume-plan-review:{review_request_id}",
                idempotency_key=parsed.idempotency_key,
                request_sha256=request_sha256,
                response_id=resumed.object_id,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise PlanReviewConflictError("plan review resume authority raced") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.show(review_request_id)

    def show(self, review_request_id: str) -> PlanReviewViewV1:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            run_id, request, result = self._load_current(connection, review_request_id)
            presentation = self._load_presentation(
                connection,
                request.presentation_ref.object_id,
            )
            decision = self._load_optional_decision(connection, request.review_request_id)
            revision = self._load_optional_revision(connection, request.review_request_id)
            plan = self._load_plan(
                connection,
                request=request,
                reference=(result.successor_plan_ref or result.plan_ref),
            )
            current_run = self.store._load_current_run(connection, run_id)
            connection.rollback()
            return PlanReviewViewV1(
                run_id=run_id,
                current_run_ref=current_run.to_ref(),
                request=request,
                presentation=presentation,
                result=result,
                plan=plan,
                decision=decision,
                revision=revision,
            )
        finally:
            connection.close()

    def show_by_result_ref(
        self,
        reference: ObjectRef,
    ) -> PlanReviewViewV1:
        if reference.object_type != "plan-review-result":
            raise PlanReviewIntegrityError(
                "plan review result ref has the wrong object type",
            )
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT review_request_id
                FROM plan_review_current_heads
                WHERE object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise FactoryControlNotFoundError(
                    "current plan review result was not found",
                )
            review_request_id = str(row["review_request_id"])
            _run_id, _request, result = self._load_current(
                connection,
                review_request_id,
            )
            if result.to_ref() != reference:
                raise PlanReviewIntegrityError(
                    "plan review result differs from its reference",
                )
            connection.rollback()
        finally:
            connection.close()
        return self.show(review_request_id)

    def export(self, review_request_id: str) -> PlanReviewViewV1:
        return self.show(review_request_id)

    def require_resumed_plan(
        self,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan_ref: ObjectRef,
    ) -> PlanReviewViewV1:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            raise PlanReviewNotResumableError(
                "domain execution requires a domain plan review",
            )
        return self.require_resumed_review(
            run_id=run_id,
            plan_kind=plan_kind,
            plan_ref=plan_ref,
        )

    def require_resumed_review(
        self,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan_ref: ObjectRef,
    ) -> PlanReviewViewV1:
        current_plan = (
            self.store.get_plan(run_id)[0]
            if plan_kind is PlanKindV2.GLOBAL_BUILD
            else self._current_plan(run_id, plan_kind)
        )
        if current_plan.to_ref() != plan_ref:
            raise PlanReviewConflictError(
                "review execution plan is not current",
            )
        page = self.list_reviews(
            run_id=run_id,
            state=PlanReviewStateV2.RESUMED,
            limit=500,
        )
        matches = tuple(
            view
            for view in page.items
            if (view.request.plan_kind is plan_kind and view.plan.to_ref() == plan_ref)
        )
        if len(matches) != 1:
            raise PlanReviewNotResumableError(
                "execution requires one resumed plan review",
            )
        return matches[0]

    def list_reviews(
        self,
        *,
        run_id: str | None = None,
        state: PlanReviewStateV2 | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> PlanReviewPageV1:
        if offset < 0 or limit < 1 or limit > 500:
            raise PlanReviewError("plan review pagination is outside bounds")
        connection = self.store._connect()
        try:
            clauses: list[str] = []
            parameters: list[object] = []
            if run_id is not None:
                clauses.append("requests.run_id = ?")
                parameters.append(run_id)
            if state is not None:
                clauses.append("results.state = ?")
                parameters.append(state.value)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT requests.review_request_id
                FROM plan_review_requests AS requests
                JOIN plan_review_current_heads AS heads
                  ON heads.review_request_id = requests.review_request_id
                JOIN plan_review_results AS results
                  ON results.result_id = heads.result_id
                {where}
                ORDER BY requests.review_request_id
                """,
                tuple(parameters),
            ).fetchall()
        finally:
            connection.close()
        selected = rows[offset : offset + limit]
        return PlanReviewPageV1(
            items=tuple(self.show(str(row["review_request_id"])) for row in selected),
            offset=offset,
            limit=limit,
            total=len(rows),
        )

    def _initialize(self) -> None:
        connection = self.store._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plan_review_presentations (
                    presentation_id TEXT PRIMARY KEY,
                    object_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS plan_revision_materials (
                    revision_id TEXT PRIMARY KEY,
                    plan_object_id TEXT NOT NULL UNIQUE,
                    compiled_plan_object_id TEXT NOT NULL UNIQUE,
                    plan_record_json TEXT NOT NULL,
                    compiled_plan_record_json TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _append_run(
        self,
        connection: sqlite3.Connection,
        *,
        current: FactoryRunV2,
        status: FactoryRunStatusV2,
        pending_review_ref: ObjectRef | None,
        current_plan_ref: ObjectRef | None = None,
        compiled_plan_ref: ObjectRef | None = None,
        audit: ContractAudit,
    ) -> FactoryRunV2:
        successor = FactoryRunV2.create(
            run_id=current.run_id,
            run_version=current.run_version + 1,
            status=status,
            policy_ref=current.policy_ref,
            requirement_spec_ref=current.requirement_spec_ref,
            current_plan_ref=(current.current_plan_ref if current_plan_ref is None else current_plan_ref),
            compiled_plan_ref=(current.compiled_plan_ref if compiled_plan_ref is None else compiled_plan_ref),
            active_task_refs=current.active_task_refs,
            result_refs=current.result_refs,
            pending_review_ref=pending_review_ref,
            planner_assessment_ref=current.planner_assessment_ref,
            completion_ref=current.completion_ref,
            delivery_manifest_ref=current.delivery_manifest_ref,
            transition_count=current.transition_count + 1,
            model_requests_used=current.model_requests_used,
            model_tokens_used=current.model_tokens_used,
            cost_micro_usd_used=current.cost_micro_usd_used,
            audit=audit,
        )
        self.store._insert_run(connection, successor)
        changed = connection.execute(
            """
            UPDATE factory_run_current_heads
            SET run_version = ?, run_object_id = ?
            WHERE run_id = ? AND run_version = ?
            """,
            (
                successor.run_version,
                successor.object_id,
                current.run_id,
                current.run_version,
            ),
        ).rowcount
        if changed != 1:
            raise PlanReviewConflictError("Factory run head changed during plan review")
        return successor

    def _validate_waiting_run(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        request: PlanReviewRequestV2,
        run: FactoryRunV2,
    ) -> None:
        if run.status is not FactoryRunStatusV2.WAITING_REVIEW or run.pending_review_ref != request.to_ref():
            raise PlanReviewConflictError("Factory run is not waiting on this review")
        source_row = connection.execute(
            """
            SELECT * FROM factory_runs
            WHERE run_id = ? AND run_version = ?
            """,
            (run_id, run.run_version - 1),
        ).fetchone()
        if source_row is None:
            raise PlanReviewIntegrityError("plan review source run is missing")
        source = self.store._parse_run_row(source_row)
        if source.to_ref() != request.run_ref:
            raise PlanReviewIntegrityError("plan review request binds another run snapshot")

    def _decision(
        self,
        *,
        request: PlanReviewRequestV2,
        expected_plan_version: int,
        decision: PlanDecisionKindV2,
        revision_ref: ObjectRef | None,
        decided_by: str,
        reason_code: str,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> PlanDecisionV2:
        return PlanDecisionV2.create(
            decision_id=f"plan-decision://{_slug(request.review_request_id)}/{decision.value.casefold()}",
            review_request_ref=request.to_ref(),
            expected_plan_version=expected_plan_version,
            decision=decision,
            revision_ref=revision_ref,
            decided_by=decided_by,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
            decided_at=self._clock(),
            audit=audit,
        )

    def _insert_presentation(
        self,
        connection: sqlite3.Connection,
        value: PlanReviewPresentationV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO plan_review_presentations (
                presentation_id, object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                value.presentation_id,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    def _insert_request(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        value: PlanReviewRequestV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO plan_review_requests (
                review_request_id, run_id, plan_object_id, plan_version,
                object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.review_request_id,
                run_id,
                value.plan_ref.object_id,
                value.plan_version,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    def _insert_revision(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        revision: PlanRevisionV2,
        plan: ReviewablePlanV2,
        compiled_plan: CompiledReviewablePlanV2,
    ) -> None:
        review_request_id = self._business_request_id(
            connection,
            revision.review_request_ref,
        )
        connection.execute(
            """
            INSERT INTO plan_revisions (
                revision_id, review_request_id, object_id,
                object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                review_request_id,
                revision.object_id,
                revision.object_sha256,
                revision.canonical_json().decode(),
            ),
        )
        connection.execute(
            """
            INSERT INTO plan_revision_materials (
                revision_id, plan_object_id, compiled_plan_object_id,
                plan_record_json, compiled_plan_record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                revision.revision_id,
                plan.object_id,
                compiled_plan.object_id,
                plan.canonical_json().decode(),
                compiled_plan.canonical_json().decode(),
            ),
        )
        if plan_kind is not PlanKindV2.GLOBAL_BUILD:
            self.store._insert_domain_plan_material(
                connection,
                run_id=run_id,
                plan_kind=plan_kind,
                plan=plan,
                compiled_plan=compiled_plan,
            )

    def _insert_decision(
        self,
        connection: sqlite3.Connection,
        value: PlanDecisionV2,
    ) -> None:
        review_request_id = self._business_request_id(
            connection,
            value.review_request_ref,
        )
        connection.execute(
            """
            INSERT INTO plan_decisions (
                decision_id, review_request_id, object_id,
                object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                value.decision_id,
                review_request_id,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    def _insert_result(
        self,
        connection: sqlite3.Connection,
        value: PlanReviewResultV2,
    ) -> None:
        review_request_id = self._business_request_id(
            connection,
            value.review_request_ref,
        )
        connection.execute(
            """
            INSERT INTO plan_review_results (
                result_id, review_request_id, state,
                object_id, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                value.result_id,
                review_request_id,
                value.state.value,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    def _replace_head(
        self,
        connection: sqlite3.Connection,
        value: PlanReviewResultV2,
    ) -> None:
        review_request_id = self._business_request_id(
            connection,
            value.review_request_ref,
        )
        connection.execute(
            """
            INSERT INTO plan_review_current_heads (
                review_request_id, result_id, state, object_id
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(review_request_id) DO UPDATE SET
                result_id = excluded.result_id,
                state = excluded.state,
                object_id = excluded.object_id
            """,
            (
                review_request_id,
                value.result_id,
                value.state.value,
                value.object_id,
            ),
        )

    @staticmethod
    def _business_request_id(
        connection: sqlite3.Connection,
        reference: ObjectRef,
    ) -> str:
        row = connection.execute(
            """
            SELECT review_request_id, object_sha256
            FROM plan_review_requests
            WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if row is None or row["object_sha256"] != reference.object_sha256:
            raise PlanReviewIntegrityError("plan review request ref is not persisted")
        return str(row["review_request_id"])

    def _load_current(
        self,
        connection: sqlite3.Connection,
        review_request_id: str,
    ) -> tuple[str, PlanReviewRequestV2, PlanReviewResultV2]:
        request_row = connection.execute(
            """
            SELECT * FROM plan_review_requests
            WHERE review_request_id = ? OR object_id = ?
            """,
            (review_request_id, review_request_id),
        ).fetchone()
        if request_row is None:
            raise FactoryControlNotFoundError(f"plan review not found: {review_request_id}")
        request = PlanReviewRequestV2.model_validate_json(str(request_row["record_json"]))
        if (
            request_row["review_request_id"] != request.review_request_id
            or request_row["plan_object_id"] != request.plan_ref.object_id
            or request_row["plan_version"] != request.plan_version
            or request_row["object_id"] != request.object_id
            or request_row["object_sha256"] != request.object_sha256
        ):
            raise PlanReviewIntegrityError("plan review request materialized columns drifted")
        head = connection.execute(
            "SELECT * FROM plan_review_current_heads WHERE review_request_id = ?",
            (request.review_request_id,),
        ).fetchone()
        if head is None:
            raise PlanReviewIntegrityError("plan review current head is missing")
        result_row = connection.execute(
            "SELECT * FROM plan_review_results WHERE result_id = ?",
            (head["result_id"],),
        ).fetchone()
        if result_row is None:
            raise PlanReviewIntegrityError("plan review head points to a missing result")
        result = PlanReviewResultV2.model_validate_json(str(result_row["record_json"]))
        if (
            result.review_request_ref != request.to_ref()
            or head["state"] != result.state.value
            or head["object_id"] != result.object_id
            or result_row["state"] != result.state.value
            or result_row["object_id"] != result.object_id
            or result_row["object_sha256"] != result.object_sha256
        ):
            raise PlanReviewIntegrityError("plan review current result drifted")
        return str(request_row["run_id"]), request, result

    @staticmethod
    def _load_presentation(
        connection: sqlite3.Connection,
        object_id: str,
    ) -> PlanReviewPresentationV2:
        row = connection.execute(
            "SELECT * FROM plan_review_presentations WHERE object_id = ?",
            (object_id,),
        ).fetchone()
        if row is None:
            raise PlanReviewIntegrityError("plan review presentation is missing")
        value = PlanReviewPresentationV2.model_validate_json(str(row["record_json"]))
        if (
            row["presentation_id"] != value.presentation_id
            or row["object_id"] != value.object_id
            or row["object_sha256"] != value.object_sha256
        ):
            raise PlanReviewIntegrityError("plan review presentation drifted")
        return value

    @staticmethod
    def _load_optional_decision(
        connection: sqlite3.Connection,
        review_request_id: str,
    ) -> PlanDecisionV2 | None:
        row = connection.execute(
            "SELECT * FROM plan_decisions WHERE review_request_id = ?",
            (review_request_id,),
        ).fetchone()
        if row is None:
            return None
        value = PlanDecisionV2.model_validate_json(str(row["record_json"]))
        if row["object_id"] != value.object_id or row["object_sha256"] != value.object_sha256:
            raise PlanReviewIntegrityError("plan review decision drifted")
        return value

    @staticmethod
    def _load_optional_revision(
        connection: sqlite3.Connection,
        review_request_id: str,
    ) -> PlanRevisionV2 | None:
        row = connection.execute(
            "SELECT * FROM plan_revisions WHERE review_request_id = ?",
            (review_request_id,),
        ).fetchone()
        if row is None:
            return None
        value = PlanRevisionV2.model_validate_json(str(row["record_json"]))
        if row["object_id"] != value.object_id or row["object_sha256"] != value.object_sha256:
            raise PlanReviewIntegrityError("plan review revision drifted")
        return value

    def _load_plan(
        self,
        connection: sqlite3.Connection,
        *,
        request: PlanReviewRequestV2,
        reference: ObjectRef,
    ) -> ReviewablePlanV2:
        adapter = self._adapter(
            request.plan_kind,
            reference,
        )
        payload: bytes | None = None
        if request.plan_kind is PlanKindV2.GLOBAL_BUILD:
            row = connection.execute(
                """
                SELECT record_json
                FROM dataset_build_plans
                WHERE object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is not None:
                payload = str(row["record_json"]).encode()
        else:
            try:
                material = self.store._load_domain_plan_material_by_ref(
                    connection,
                    run_id=self._request_run_id(
                        connection,
                        request,
                    ),
                    plan_kind=request.plan_kind,
                    reference=reference,
                )
            except FactoryControlIntegrityError as exc:
                raise PlanReviewIntegrityError("domain plan material failed integrity validation") from exc
            payload = material.plan_record_json
        if payload is None:
            row = connection.execute(
                """
                SELECT plan_record_json AS record_json
                FROM plan_revision_materials
                WHERE plan_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is not None:
                payload = str(row["record_json"]).encode()
        if payload is None:
            raise PlanReviewIntegrityError("plan review plan material is missing")
        try:
            value = cast(
                ReviewablePlanV2,
                adapter.parse(payload),
            )
        except ReviewablePlanAdapterError as exc:
            raise PlanReviewIntegrityError("plan review plan material is invalid") from exc
        if value.to_ref() != reference or value.canonical_json() != payload:
            raise PlanReviewIntegrityError("plan review plan material drifted")
        return value

    def _current_plan(
        self,
        run_id: str,
        plan_kind: PlanKindV2,
    ) -> ReviewablePlanV2:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            plan, _ = self.store.get_plan(run_id)
            return plan
        material = self.store.get_domain_plan(run_id, plan_kind)
        adapter = self._adapter(plan_kind, material.plan_ref)
        try:
            domain_plan = cast(
                ReviewablePlanV2,
                adapter.parse(material.plan_record_json),
            )
        except ReviewablePlanAdapterError as exc:
            raise PlanReviewIntegrityError("current domain plan material is invalid") from exc
        if (
            domain_plan.to_ref() != material.plan_ref
            or domain_plan.canonical_json() != material.plan_record_json
        ):
            raise PlanReviewIntegrityError("current domain plan material drifted")
        return domain_plan

    def _adapter(
        self,
        plan_kind: PlanKindV2,
        reference: ObjectRef,
    ) -> ReviewablePlanAdapter[Any, Any]:
        try:
            return self.adapter_registry.resolve(
                plan_kind,
                reference.object_type,
            )
        except ReviewablePlanAdapterError as exc:
            raise PlanReviewIntegrityError("reviewable plan adapter is unavailable") from exc

    def _is_current_plan(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        plan_kind: PlanKindV2,
        plan_ref: ObjectRef,
    ) -> bool:
        if plan_kind is PlanKindV2.GLOBAL_BUILD:
            run = self.store._load_current_run(
                connection,
                run_id,
            )
            return run.current_plan_ref == plan_ref
        row = connection.execute(
            """
            SELECT plan_object_id
            FROM domain_plan_current_heads
            WHERE run_id = ? AND plan_kind = ?
            """,
            (run_id, plan_kind.value),
        ).fetchone()
        return row is not None and row["plan_object_id"] == plan_ref.object_id

    @staticmethod
    def _request_run_id(
        connection: sqlite3.Connection,
        request: PlanReviewRequestV2,
    ) -> str:
        row = connection.execute(
            """
            SELECT run_id
            FROM plan_review_requests
            WHERE review_request_id = ?
            """,
            (request.review_request_id,),
        ).fetchone()
        if row is None:
            raise PlanReviewIntegrityError("plan review request run binding is missing")
        return str(row["run_id"])

    def _idempotent_result(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
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
            raise PlanReviewConflictError("plan review idempotency request changed")
        if row["response_type"] != "plan-review-result":
            raise PlanReviewIntegrityError("plan review idempotency response type drifted")
        result = connection.execute(
            "SELECT 1 FROM plan_review_results WHERE object_id = ?",
            (row["response_id"],),
        ).fetchone()
        if result is None:
            raise PlanReviewIntegrityError("plan review idempotency result is missing")
        return str(row["response_id"])

    @staticmethod
    def _record_idempotency(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO factory_control_idempotency (
                scope, idempotency_key, request_sha256,
                response_type, response_id
            ) VALUES (?, ?, ?, 'plan-review-result', ?)
            """,
            (scope, idempotency_key, request_sha256, response_id),
        )


def _request_sha256(*values: object) -> str:
    payload = canonical_value_v2(values)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _stable_ref(object_type: str, seed: str) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _slug(value: str) -> str:
    return value.rsplit("://", 1)[-1].replace("/", "-")


def _open_scope(
    run_id: str,
    plan_kind: PlanKindV2,
) -> str:
    if plan_kind is PlanKindV2.GLOBAL_BUILD:
        return f"open-plan-review:{run_id}"
    return f"open-plan-review:{run_id}:{plan_kind.value}"


__all__ = [
    "AttachmentPlanReviewEditSubmissionV1",
    "CriteriaRubricPlanReviewEditSubmissionV1",
    "DatasetDeliveryPlanReviewEditSubmissionV1",
    "GradingDesignPlanReviewEditSubmissionV1",
    "PlanReviewConflictError",
    "PlanReviewDecisionSubmissionV1",
    "PlanReviewEditSubmissionV1",
    "PlanReviewError",
    "PlanReviewIntegrityError",
    "PlanReviewNotResumableError",
    "PlanReviewPageV1",
    "PlanReviewResumeSubmissionV1",
    "PlanReviewService",
    "PlanReviewViewV1",
]
