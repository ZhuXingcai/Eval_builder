from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RuntimeName(StrEnum):
    PROVIDER = "provider"
    CLAUDE_AGENT_SDK = "claude_agent_sdk"
    CLAUDE_CODE_CLI = "claude_code_cli"
    PI_RPC = "pi_rpc"
    FAKE = "fake"


class RuntimeEventType(StrEnum):
    RUNTIME_STARTED = "runtime_started"
    MESSAGE_DELTA = "message_delta"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_FINISHED = "tool_finished"
    ARTIFACT_WRITTEN = "artifact_written"
    WARNING = "warning"
    USAGE = "usage"
    RUNTIME_FINISHED = "runtime_finished"
    RUNTIME_FAILED = "runtime_failed"


class RuntimeErrorCode(StrEnum):
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    AUTH_UNAVAILABLE = "auth_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    TOOL_DENIED = "tool_denied"
    PATH_ESCAPE = "path_escape"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROTOCOL_ERROR = "protocol_error"
    PROCESS_ERROR = "process_error"


class RuntimeCredentialStatus(StrEnum):
    READY = "READY"
    NOT_REQUIRED = "NOT_REQUIRED"
    UNKNOWN = "UNKNOWN"
    MISSING = "MISSING"
    INVALID = "INVALID"


class RuntimeSandboxEnforcement(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RuntimeCapabilities(BaseModel):
    name: RuntimeName
    available: bool
    version: str | None = None
    models: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    supports_resume: bool = False
    supports_cancel: bool = True
    supports_event_streaming: bool = True
    supports_tool_progress: bool = False
    credential_status: RuntimeCredentialStatus = RuntimeCredentialStatus.UNKNOWN
    sandbox_enforcement: RuntimeSandboxEnforcement = RuntimeSandboxEnforcement.NONE
    reason: str = ""


class ModelProfile(BaseModel):
    provider: str
    model: str
    thinking: str | None = None
    effort: str | None = None
    extra: dict[str, object] = Field(default_factory=dict)


class RuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    artifact_id: str
    role: str
    workspace: str
    prompt: str
    system_policy: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    disallowed_tools: list[str] = Field(default_factory=list)
    model_profile: ModelProfile
    max_turns: int = Field(default=50, ge=1, le=300)
    max_tool_events: int = Field(default=300, ge=1, le=10_000)
    timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    max_budget_usd: float | None = Field(default=None, gt=0)
    credential_profile: str | None = None
    source_evidence_refs: list[str] = Field(default_factory=list)
    forbidden_output_refs: list[str] = Field(default_factory=list)


class RuntimeSessionRef(BaseModel):
    runtime: RuntimeName
    run_handle: str
    workspace: str
    model_profile: ModelProfile
    external_session_id: str | None = None
    session_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RuntimeResumeRequest(BaseModel):
    request: RuntimeRequest
    session: RuntimeSessionRef
    prompt: str


class RuntimeUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: int = 0
    turns: int = 0
    duration_ms: int | None = None


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: RuntimeEventType
    run_id: str
    artifact_id: str
    runtime: RuntimeName
    sequence: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    raw_type: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    usage: RuntimeUsage | None = None
    session: RuntimeSessionRef | None = None
    error_code: RuntimeErrorCode | None = None
    run_handle: str | None = None
