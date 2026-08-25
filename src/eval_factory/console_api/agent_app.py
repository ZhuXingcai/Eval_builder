from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from eval_factory.agent_system.graph import (
    FactoryGraphDriftError,
    FactoryGraphError,
)
from eval_factory.agent_system.graph_journal import (
    FactoryGraphJournalConflictError,
    FactoryGraphJournalError,
    FactoryGraphJournalIntegrityError,
    FactoryGraphJournalNotFoundError,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.store import (
    FactoryControlConcurrencyError,
    FactoryControlConflictError,
    FactoryControlIntegrityError,
)
from eval_factory.console_api.agent_contracts import (
    AgentShellApiContractV1,
    AgentShellCloseSessionCommandV1,
    AgentShellCreateSessionCommandV1,
    AgentShellDeliverySummaryV1,
    AgentShellEventPageV1,
    AgentShellMemberProjectionV1,
    AgentShellPostMessageCommandV1,
    AgentShellProjectionV1,
    AgentShellReconcileCommandV1,
    AgentShellSessionPageV1,
    AgentShellSourceAdmissionCommandV1,
    AgentShellSourceAdmissionContractV1,
    AgentShellSourceAdmissionV1,
    AgentShellTeamSummaryV1,
    AgentShellWorkspaceDescriptorV1,
    AgentShellWorkspaceKindV1,
    agent_shell_api_contract,
)
from eval_factory.console_api.agent_service import AgentShellService
from eval_factory.console_api.app import (
    _PRINCIPAL_HEADER,
    _error,
    _parse_command,
    _request_body_schema,
    create_app,
)
from eval_factory.console_api.composition_types import (
    AgentShellCompositionBlockedError,
    AgentShellCompositionConflictError,
    AgentShellCompositionError,
    AgentShellCompositionIntegrityError,
    AgentShellCompositionNotConfiguredError,
    AgentShellCompositionNotFoundError,
)
from eval_factory.console_api.source_admission import (
    HarnessSourceAdmissionService,
)
from eval_factory.console_api.sse import (
    AgentShellSseCursorError,
    resolve_sse_cursor,
    stream_session_events,
)
from eval_factory.harness import (
    HarnessSessionConcurrencyError,
    HarnessSessionConflictError,
    HarnessSessionIntegrityError,
    HarnessSessionNotFoundError,
)
from eval_factory.harness.source_admission import (
    HarnessSourceAdmissionConflictError,
    HarnessSourceAdmissionError,
    HarnessSourceAdmissionIntegrityError,
    HarnessSourceAdmissionLimitError,
    HarnessSourceAdmissionMultipartError,
    HarnessSourceAdmissionNotFoundError,
    HarnessSourceAdmissionValidationError,
    SourceUploadPart,
)
from eval_factory.team import (
    TeamConcurrencyError,
    TeamIdempotencyConflictError,
    TeamIntegrityError,
    TeamNotFoundError,
    TeamStoreError,
)

_SOURCE_COMMAND_PART_BYTES = 512 * 1024


def create_agent_app(
    plan_review_service: PlanReviewService,
    agent_shell_service: AgentShellService,
    source_admission_service: HarnessSourceAdmissionService | None = None,
) -> FastAPI:
    app = create_app(plan_review_service)
    app.title = "Eval Dataset Factory Agent API"
    app.version = "1.3.0"
    app.description = (
        "Conversation-first Harness session API with committed sequence SSE, "
        "bounded source admission, and owner-backed service composition."
    )

    @app.exception_handler(HarnessSessionNotFoundError)
    async def harness_not_found(
        _: Request,
        __: HarnessSessionNotFoundError,
    ) -> JSONResponse:
        return _error(404, "HARNESS_SESSION_NOT_FOUND", "session was not found")

    @app.exception_handler(HarnessSessionConcurrencyError)
    async def harness_stale(
        _: Request,
        __: HarnessSessionConcurrencyError,
    ) -> JSONResponse:
        return _error(409, "HARNESS_SESSION_STALE", "session authority is stale")

    @app.exception_handler(HarnessSessionConflictError)
    async def harness_conflict(
        _: Request,
        __: HarnessSessionConflictError,
    ) -> JSONResponse:
        return _error(409, "HARNESS_SESSION_CONFLICT", "session command conflicts")

    @app.exception_handler(HarnessSessionIntegrityError)
    async def harness_integrity(
        _: Request,
        __: HarnessSessionIntegrityError,
    ) -> JSONResponse:
        return _error(
            500,
            "HARNESS_SESSION_INTEGRITY",
            "session evidence failed integrity validation",
        )

    @app.exception_handler(HarnessSourceAdmissionNotFoundError)
    async def source_not_found(
        _: Request,
        __: HarnessSourceAdmissionNotFoundError,
    ) -> JSONResponse:
        return _error(
            404,
            "SOURCE_ADMISSION_NOT_FOUND",
            "source admission was not found",
        )

    @app.exception_handler(HarnessSourceAdmissionLimitError)
    async def source_limit(
        _: Request,
        exception: HarnessSourceAdmissionLimitError,
    ) -> JSONResponse:
        return _error(413, exception.code, "source upload exceeds its limits")

    @app.exception_handler(HarnessSourceAdmissionConflictError)
    async def source_conflict(
        _: Request,
        exception: HarnessSourceAdmissionConflictError,
    ) -> JSONResponse:
        return _error(409, exception.code, "source admission conflicts")

    @app.exception_handler(HarnessSourceAdmissionValidationError)
    async def source_invalid(
        _: Request,
        exception: HarnessSourceAdmissionValidationError,
    ) -> JSONResponse:
        return _error(422, exception.code, "source upload is invalid")

    @app.exception_handler(HarnessSourceAdmissionIntegrityError)
    async def source_integrity(
        _: Request,
        exception: HarnessSourceAdmissionIntegrityError,
    ) -> JSONResponse:
        return _error(
            500,
            exception.code,
            "source admission failed integrity validation",
        )

    @app.exception_handler(HarnessSourceAdmissionError)
    async def source_error(
        _: Request,
        exception: HarnessSourceAdmissionError,
    ) -> JSONResponse:
        return _error(422, exception.code, "source admission was rejected")

    @app.exception_handler(AgentShellCompositionNotConfiguredError)
    async def composition_not_configured(
        _: Request,
        exception: AgentShellCompositionNotConfiguredError,
    ) -> JSONResponse:
        return _error(
            503,
            exception.code,
            "Agent Shell composition is unavailable",
        )

    @app.exception_handler(AgentShellCompositionNotFoundError)
    @app.exception_handler(FactoryGraphJournalNotFoundError)
    async def composition_not_found(
        _: Request,
        exception: AgentShellCompositionNotFoundError,
    ) -> JSONResponse:
        return _error(
            404,
            exception.code,
            "Agent Shell projection was not found",
        )

    @app.exception_handler(AgentShellCompositionConflictError)
    @app.exception_handler(FactoryGraphDriftError)
    @app.exception_handler(FactoryGraphJournalConflictError)
    @app.exception_handler(FactoryControlConflictError)
    @app.exception_handler(FactoryControlConcurrencyError)
    @app.exception_handler(TeamConcurrencyError)
    @app.exception_handler(TeamIdempotencyConflictError)
    async def composition_conflict(
        _: Request,
        __: Exception,
    ) -> JSONResponse:
        return _error(
            409,
            "AGENT_SHELL_COMPOSITION_CONFLICT",
            "Agent Shell authority conflicts",
        )

    @app.exception_handler(AgentShellCompositionBlockedError)
    async def composition_blocked(
        _: Request,
        exception: AgentShellCompositionBlockedError,
    ) -> JSONResponse:
        return _error(
            422,
            exception.code,
            "Agent Shell composition is blocked",
        )

    @app.exception_handler(AgentShellCompositionIntegrityError)
    @app.exception_handler(FactoryGraphJournalIntegrityError)
    @app.exception_handler(FactoryControlIntegrityError)
    @app.exception_handler(TeamIntegrityError)
    async def composition_integrity(
        _: Request,
        __: Exception,
    ) -> JSONResponse:
        return _error(
            500,
            "AGENT_SHELL_COMPOSITION_INTEGRITY",
            "Agent Shell authority failed integrity validation",
        )

    @app.exception_handler(TeamNotFoundError)
    async def team_not_found(
        _: Request,
        __: TeamNotFoundError,
    ) -> JSONResponse:
        return _error(
            404,
            "AGENT_SHELL_TEAM_NOT_FOUND",
            "Team projection was not found",
        )

    @app.exception_handler(AgentShellCompositionError)
    @app.exception_handler(FactoryGraphError)
    @app.exception_handler(FactoryGraphJournalError)
    @app.exception_handler(TeamStoreError)
    async def composition_error(
        _: Request,
        __: Exception,
    ) -> JSONResponse:
        return _error(
            422,
            "AGENT_SHELL_COMPOSITION_ERROR",
            "Agent Shell request was rejected",
        )

    @app.get("/api/harness/contract", response_model=AgentShellApiContractV1)
    def contract() -> AgentShellApiContractV1:
        return agent_shell_api_contract()

    @app.get(
        "/api/harness/source-admission/contract",
        response_model=AgentShellSourceAdmissionContractV1,
    )
    def source_contract() -> AgentShellSourceAdmissionContractV1:
        return _require_source_admission(
            source_admission_service,
        ).contract()

    @app.get("/api/harness/sessions", response_model=AgentShellSessionPageV1)
    def list_sessions(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> AgentShellSessionPageV1:
        return agent_shell_service.list_sessions(offset=offset, limit=limit)

    @app.post(
        "/api/harness/sessions",
        response_model=AgentShellProjectionV1,
        openapi_extra=_request_body_schema(
            AgentShellCreateSessionCommandV1,
        ),
    )
    async def create_session(
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> AgentShellProjectionV1:
        command = await _parse_command(
            request,
            AgentShellCreateSessionCommandV1,
        )
        return agent_shell_service.create_session(
            command,
            principal=principal,
        )

    @app.get(
        "/api/harness/sessions/{session_id}",
        response_model=AgentShellProjectionV1,
    )
    def get_session(session_id: str) -> AgentShellProjectionV1:
        return agent_shell_service.get_session(session_id)

    @app.delete(
        "/api/harness/sessions/{session_id}",
        response_model=AgentShellProjectionV1,
        openapi_extra=_request_body_schema(
            AgentShellCloseSessionCommandV1,
        ),
    )
    async def close_session(
        session_id: str,
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> AgentShellProjectionV1:
        command = await _parse_command(
            request,
            AgentShellCloseSessionCommandV1,
        )
        return agent_shell_service.close_session(
            session_id,
            command,
            principal=principal,
        )

    @app.post(
        "/api/harness/sessions/{session_id}/messages",
        response_model=AgentShellProjectionV1,
        openapi_extra=_request_body_schema(
            AgentShellPostMessageCommandV1,
        ),
    )
    async def post_message(
        session_id: str,
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> AgentShellProjectionV1:
        command = await _parse_command(
            request,
            AgentShellPostMessageCommandV1,
        )
        return await agent_shell_service.post_message(
            session_id,
            command,
            principal=principal,
        )

    @app.post(
        "/api/harness/sessions/{session_id}/reconcile",
        response_model=AgentShellProjectionV1,
        openapi_extra=_request_body_schema(
            AgentShellReconcileCommandV1,
        ),
    )
    async def reconcile_session(
        session_id: str,
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> AgentShellProjectionV1:
        command = await _parse_command(
            request,
            AgentShellReconcileCommandV1,
        )
        return await agent_shell_service.reconcile(
            session_id,
            command,
            principal=principal,
        )

    @app.get(
        "/api/harness/sessions/{session_id}/team",
        response_model=AgentShellTeamSummaryV1,
    )
    def get_team(session_id: str) -> AgentShellTeamSummaryV1:
        return agent_shell_service.team(session_id)

    @app.get(
        "/api/harness/sessions/{session_id}/members/{member_id}",
        response_model=AgentShellMemberProjectionV1,
    )
    def get_member(
        session_id: str,
        member_id: str,
    ) -> AgentShellMemberProjectionV1:
        return agent_shell_service.member(session_id, member_id)

    @app.get(
        "/api/harness/sessions/{session_id}/workspaces/{kind}",
        response_model=AgentShellWorkspaceDescriptorV1,
    )
    def get_workspace(
        session_id: str,
        kind: AgentShellWorkspaceKindV1,
    ) -> AgentShellWorkspaceDescriptorV1:
        return agent_shell_service.workspace(session_id, kind)

    @app.get(
        "/api/harness/sessions/{session_id}/delivery",
        response_model=AgentShellDeliverySummaryV1,
    )
    def get_delivery(
        session_id: str,
    ) -> AgentShellDeliverySummaryV1:
        return agent_shell_service.delivery(session_id)

    @app.get(
        "/api/harness/sessions/{session_id}/sources",
        response_model=AgentShellSourceAdmissionV1,
    )
    def get_sources(session_id: str) -> AgentShellSourceAdmissionV1:
        return _require_source_admission(
            source_admission_service,
        ).get(session_id)

    @app.post(
        "/api/harness/sessions/{session_id}/sources",
        response_model=AgentShellSourceAdmissionV1,
        openapi_extra=_source_request_body_schema(),
    )
    async def admit_sources(
        session_id: str,
        request: Request,
        principal: str = Header(alias=_PRINCIPAL_HEADER),
    ) -> AgentShellSourceAdmissionV1:
        service = _require_source_admission(source_admission_service)
        _validate_source_content_length(
            request,
            maximum=service.limits.max_request_bytes,
        )
        try:
            async with request.form(
                max_files=service.limits.max_source_files + 1,
                max_fields=8,
                max_part_size=_SOURCE_COMMAND_PART_BYTES,
            ) as form:
                values = tuple(form.multi_items())
                command_values = tuple(value for key, value in values if key == "command")
                manifest_values = tuple(value for key, value in values if key == "manifest")
                trace_values = tuple(value for key, value in values if key == "traces")
                if (
                    any(key not in {"command", "manifest", "traces"} for key, _ in values)
                    or len(command_values) != 1
                    or not isinstance(command_values[0], str)
                    or len(manifest_values) != 1
                    or not isinstance(manifest_values[0], UploadFile)
                    or not trace_values
                    or any(not isinstance(value, UploadFile) for value in trace_values)
                ):
                    raise HarnessSourceAdmissionMultipartError(
                        "source multipart fields are invalid",
                    )
                try:
                    command = AgentShellSourceAdmissionCommandV1.model_validate_json(
                        command_values[0],
                    )
                except ValidationError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={"error_code": "INVALID_COMMAND"},
                    ) from exc
                manifest_upload = manifest_values[0]
                assert isinstance(manifest_upload, UploadFile)
                uploads = tuple(value for value in trace_values if isinstance(value, UploadFile))
                return await run_in_threadpool(
                    service.admit,
                    session_id,
                    command,
                    manifest=SourceUploadPart(
                        filename=manifest_upload.filename or "",
                        stream=manifest_upload.file,
                    ),
                    traces=tuple(
                        SourceUploadPart(
                            filename=upload.filename or "",
                            stream=upload.file,
                        )
                        for upload in uploads
                    ),
                    principal=principal,
                )
        except HarnessSourceAdmissionError:
            raise
        except (
            HarnessSessionConcurrencyError,
            HarnessSessionConflictError,
            HarnessSessionIntegrityError,
            HarnessSessionNotFoundError,
        ):
            raise
        except HTTPException:
            raise
        except StarletteHTTPException as exc:
            raise HarnessSourceAdmissionMultipartError(
                "source multipart parsing failed",
            ) from exc

    @app.get(
        "/api/harness/sessions/{session_id}/events",
        response_model=AgentShellEventPageV1,
    )
    def list_events(
        session_id: str,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> AgentShellEventPageV1:
        return agent_shell_service.list_events(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    @app.get("/api/harness/sessions/{session_id}/stream")
    async def stream(
        session_id: str,
        request: Request,
        after_sequence: int | None = Query(default=None, ge=0),
        follow: bool = True,
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
        ),
    ) -> StreamingResponse:
        projection = agent_shell_service.get_session(session_id)
        try:
            cursor = resolve_sse_cursor(
                last_event_id=last_event_id,
                after_sequence=after_sequence,
                watermark=projection.reconnect_cursor,
            )
        except AgentShellSseCursorError:
            return StreamingResponse(
                iter((b'event: error\ndata: {"error_code":"INVALID_SSE_CURSOR"}\n\n',)),
                status_code=409,
                media_type="text/event-stream",
            )
        return StreamingResponse(
            stream_session_events(
                service=agent_shell_service,
                session_id=session_id,
                request=request,
                after_sequence=cursor,
                follow=follow,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _require_source_admission(
    service: HarnessSourceAdmissionService | None,
) -> HarnessSourceAdmissionService:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail={"error_code": "SOURCE_ADMISSION_NOT_CONFIGURED"},
        )
    return service


def _validate_source_content_length(
    request: Request,
    *,
    maximum: int,
) -> None:
    raw = request.headers.get("content-length")
    if raw is None:
        raise HTTPException(
            status_code=411,
            detail={"error_code": "CONTENT_LENGTH_REQUIRED"},
        )
    try:
        observed = int(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error_code": "INVALID_CONTENT_LENGTH"},
        ) from exc
    if observed < 1 or observed > maximum:
        raise HarnessSourceAdmissionLimitError(
            "source multipart request exceeds its byte limit",
        )


def _source_request_body_schema() -> dict[str, object]:
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["command", "manifest", "traces"],
                        "properties": {
                            "command": {
                                "type": "string",
                                "contentMediaType": "application/json",
                                "contentSchema": (AgentShellSourceAdmissionCommandV1.model_json_schema()),
                            },
                            "manifest": {
                                "type": "string",
                                "format": "binary",
                            },
                            "traces": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary",
                                },
                                "minItems": 1,
                                "maxItems": 500,
                            },
                        },
                    },
                },
            },
        },
    }


__all__ = ["create_agent_app"]
