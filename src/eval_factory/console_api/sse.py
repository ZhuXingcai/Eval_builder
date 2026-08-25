from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from time import monotonic

from fastapi import Request

from eval_factory.console_api.agent_contracts import AgentShellEventV1
from eval_factory.console_api.agent_service import AgentShellService


class AgentShellSseCursorError(ValueError):
    pass


def resolve_sse_cursor(
    *,
    last_event_id: str | None,
    after_sequence: int | None,
    watermark: int,
) -> int:
    header_cursor: int | None = None
    if last_event_id is not None:
        if not last_event_id.isascii() or not last_event_id.isdecimal():
            raise AgentShellSseCursorError(
                "Last-Event-ID must be an integer sequence",
            )
        header_cursor = int(last_event_id)
        if str(header_cursor) != last_event_id:
            raise AgentShellSseCursorError(
                "Last-Event-ID must use canonical decimal form",
            )
    if header_cursor is not None and after_sequence is not None and header_cursor != after_sequence:
        raise AgentShellSseCursorError(
            "Last-Event-ID and after_sequence disagree",
        )
    cursor = (
        header_cursor if header_cursor is not None else (after_sequence if after_sequence is not None else 0)
    )
    if cursor < 0 or cursor > watermark:
        raise AgentShellSseCursorError(
            "SSE cursor is outside committed session history",
        )
    return cursor


async def stream_session_events(
    *,
    service: AgentShellService,
    session_id: str,
    request: Request,
    after_sequence: int,
    follow: bool,
    poll_interval_seconds: float = 0.1,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[bytes]:
    if poll_interval_seconds <= 0 or heartbeat_seconds <= 0:
        raise ValueError("SSE timing intervals must be positive")
    cursor = after_sequence
    last_heartbeat = monotonic()
    while True:
        page = service.list_events(
            session_id,
            after_sequence=cursor,
            limit=500,
        )
        for event in page.events:
            cursor = event.sequence
            yield encode_session_event(event)
        if page.next_sequence is not None:
            continue
        if not follow:
            return
        if await request.is_disconnected():
            return
        now = monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            yield encode_heartbeat()
            last_heartbeat = now
        await asyncio.sleep(poll_interval_seconds)


def encode_session_event(event: AgentShellEventV1) -> bytes:
    return (f"id: {event.sequence}\nevent: session-event\ndata: {event.model_dump_json()}\n\n").encode()


def encode_heartbeat() -> bytes:
    payload = json.dumps(
        {
            "schema_version": "eval-factory/agent-shell-heartbeat/v1",
            "connection": "alive",
            "timestamp": datetime.now(UTC).isoformat(),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"event: heartbeat\ndata: {payload}\n\n".encode()


__all__ = [
    "AgentShellSseCursorError",
    "encode_heartbeat",
    "encode_session_event",
    "resolve_sse_cursor",
    "stream_session_events",
]
