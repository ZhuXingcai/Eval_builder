from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from eval_factory.agent_system.plan_review import (
    AttachmentPlanReviewEditSubmissionV1,
    CriteriaRubricPlanReviewEditSubmissionV1,
    DatasetDeliveryPlanReviewEditSubmissionV1,
    GradingDesignPlanReviewEditSubmissionV1,
    PlanReviewConflictError,
    PlanReviewError,
    PlanReviewIntegrityError,
    PlanReviewNotResumableError,
    PlanReviewPageV1,
    PlanReviewService,
    PlanReviewViewV1,
)
from eval_factory.agent_system.store import FactoryControlNotFoundError
from eval_factory.console_api.contracts import (
    PlanReviewApiContractV1,
    PlanReviewDecisionCommandV1,
    PlanReviewEditCommandV1,
    PlanReviewResumeCommandV1,
    compile_plan_review_edit,
    plan_review_api_contract,
)
from eval_factory.contracts.agent_system_v2 import PlanReviewStateV2
from eval_factory.contracts.core import ContractAudit
from eval_factory.contracts.core_v2 import ContractModelV2

_PRINCIPAL_HEADER = "X-Eval-Factory-Principal"
_MAX_COMMAND_BYTES = 8 * 1024 * 1024


def create_app(service: PlanReviewService) -> FastAPI:
    app = FastAPI(
        title="Eval Dataset Factory Plan Console API",
        version="1.0.0",
        description="Strict shared PlanReview API for CLI/Web HITL parity.",
    )

    @app.exception_handler(HTTPException)
    async def http_handler(
        _: Request,
        exception: HTTPException,
    ) -> JSONResponse:
        detail: dict[str, object] = exception.detail if isinstance(exception.detail, dict) else {}
        code = detail.get("error_code")
        return _error(
            exception.status_code,
            (str(code) if isinstance(code, str) else "HTTP_ERROR"),
            "console request was rejected",
        )

    @app.exception_handler(FactoryControlNotFoundError)
    async def not_found_handler(_: Request, __: FactoryControlNotFoundError) -> JSONResponse:
        return _error(404, "NOT_FOUND", "requested plan review was not found")

    @app.exception_handler(PlanReviewNotResumableError)
    async def not_resumable_handler(_: Request, __: PlanReviewNotResumableError) -> JSONResponse:
        return _error(409, "PLAN_NOT_RESUMABLE", "plan review is not ready to resume")

    @app.exception_handler(PlanReviewConflictError)
    async def conflict_handler(_: Request, __: PlanReviewConflictError) -> JSONResponse:
        return _error(409, "PLAN_REVIEW_CONFLICT", "plan review authority conflicts")

    @app.exception_handler(PlanReviewIntegrityError)
    async def integrity_handler(_: Request, __: PlanReviewIntegrityError) -> JSONResponse:
        return _error(500, "INTEGRITY_ERROR", "plan review evidence failed integrity validation")

    @app.exception_handler(PlanReviewError)
    async def policy_handler(_: Request, __: PlanReviewError) -> JSONResponse:
        return _error(422, "POLICY_ERROR", "plan review request violates current policy")

    @app.get("/api/plan-reviews/contract", response_model=PlanReviewApiContractV1)
    def contract() -> PlanReviewApiContractV1:
        return plan_review_api_contract()

    @app.get("/api/plan-reviews", response_model=PlanReviewPageV1)
    def list_reviews(
        run_id: str | None = None,
        state: PlanReviewStateV2 | None = PlanReviewStateV2.PENDING_REVIEW,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> PlanReviewPageV1:
        return service.list_reviews(
            run_id=run_id,
            state=state,
            offset=offset,
            limit=limit,
        )

    @app.get("/api/plan-reviews/show", response_model=PlanReviewViewV1)
    def show(review_id: str) -> PlanReviewViewV1:
        return service.show(review_id)

    @app.get("/api/plan-reviews/export", response_model=PlanReviewViewV1)
    def export(review_id: str) -> PlanReviewViewV1:
        return service.export(review_id)

    @app.post(
        "/api/plan-reviews/decision",
        response_model=PlanReviewViewV1,
        openapi_extra=_request_body_schema(PlanReviewDecisionCommandV1),
    )
    async def decide(
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> PlanReviewViewV1:
        command = await _parse_command(request, PlanReviewDecisionCommandV1)
        if command.submission.decided_by != principal:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "PRINCIPAL_MISMATCH"},
            )
        return service.decide(
            command.review_id,
            command.submission,
            audit=_audit(service, command.review_id, principal),
        )

    @app.post(
        "/api/plan-reviews/edit",
        response_model=PlanReviewViewV1,
        openapi_extra=_request_body_schema(PlanReviewEditCommandV1),
    )
    async def edit(
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> PlanReviewViewV1:
        command = await _parse_command(request, PlanReviewEditCommandV1)
        if command.submission.decided_by != principal:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "PRINCIPAL_MISMATCH"},
            )
        audit = _audit(service, command.review_id, principal)
        submission = compile_plan_review_edit(
            service.show(command.review_id),
            command.submission,
            audit=audit,
        )
        if isinstance(
            submission,
            (
                AttachmentPlanReviewEditSubmissionV1,
                CriteriaRubricPlanReviewEditSubmissionV1,
                DatasetDeliveryPlanReviewEditSubmissionV1,
                GradingDesignPlanReviewEditSubmissionV1,
            ),
        ):
            return service.edit_plan(
                command.review_id,
                submission,
                audit=audit,
            )
        return service.edit_global_plan(
            command.review_id,
            submission,
            audit=audit,
        )

    @app.post(
        "/api/plan-reviews/resume",
        response_model=PlanReviewViewV1,
        openapi_extra=_request_body_schema(PlanReviewResumeCommandV1),
    )
    async def resume(
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> PlanReviewViewV1:
        command = await _parse_command(request, PlanReviewResumeCommandV1)
        if command.submission.resumed_by != principal:
            raise HTTPException(
                status_code=403,
                detail={"error_code": "PRINCIPAL_MISMATCH"},
            )
        return service.resume(
            command.review_id,
            command.submission,
            audit=_audit(service, command.review_id, principal),
        )

    return app


async def _parse_command[CommandT: ContractModelV2](
    request: Request,
    model_type: type[CommandT],
) -> CommandT:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_COMMAND_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"error_code": "COMMAND_TOO_LARGE"},
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error_code": "INVALID_CONTENT_LENGTH"},
            ) from exc
    payload = await request.body()
    if len(payload) > _MAX_COMMAND_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error_code": "COMMAND_TOO_LARGE"},
        )
    try:
        return model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "INVALID_COMMAND"},
        ) from exc


def _request_body_schema(
    model_type: type[ContractModelV2],
) -> dict[str, object]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": model_type.model_json_schema(),
                }
            },
        }
    }


def _audit(
    service: PlanReviewService,
    review_id: str,
    principal: str,
) -> ContractAudit:
    current = service.show(review_id)
    return ContractAudit(
        created_at=datetime.now(UTC),
        created_by=principal,
        governing_versions=current.request.audit.governing_versions,
        input_refs=(current.result.to_ref(),),
    )


def _error(
    status_code: int,
    error_code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "schema_version": "eval-factory/console-api-error/v1",
            "error_code": error_code,
            "message": message,
        },
    )


__all__ = ["create_app"]
