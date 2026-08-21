from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    ReconstructionMode,
    RelativePath,
    Sha256,
)
from env_mock_agent.facade.routing_v2 import CapabilityToken
from env_mock_agent.facade.telemetry_v2 import ExecutionTelemetryV2

ARTIFACT_EXECUTION_POLICY_VERSION: Literal["artifact-execution/r5-06-v1"] = "artifact-execution/r5-06-v1"


class AttachmentExecutionRouteKindV2(StrEnum):
    PROVIDER = "PROVIDER"
    RUNTIME = "RUNTIME"


class AttachmentExecutionStatusV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class AttachmentExecutionFailureCodeV2(StrEnum):
    SELECTED_IMPLEMENTATION_MISSING = "SELECTED_IMPLEMENTATION_MISSING"
    SELECTED_IMPLEMENTATION_MISMATCH = "SELECTED_IMPLEMENTATION_MISMATCH"
    MATERIAL_NOT_FOUND = "MATERIAL_NOT_FOUND"
    MATERIAL_MISMATCH = "MATERIAL_MISMATCH"
    WORLD_LEDGER_STALE = "WORLD_LEDGER_STALE"
    WORLD_LEDGER_FACT_MISSING = "WORLD_LEDGER_FACT_MISSING"
    WORLD_LEDGER_MUTATION_ATTEMPT = "WORLD_LEDGER_MUTATION_ATTEMPT"
    PROVIDER_EXECUTION_FAILED = "PROVIDER_EXECUTION_FAILED"
    RUNTIME_EXECUTION_FAILED = "RUNTIME_EXECUTION_FAILED"
    RUNTIME_TIMEOUT = "RUNTIME_TIMEOUT"
    RUNTIME_AUTH_UNAVAILABLE = "RUNTIME_AUTH_UNAVAILABLE"
    RUNTIME_TOOL_DENIED = "RUNTIME_TOOL_DENIED"
    RUNTIME_PROTOCOL_ERROR = "RUNTIME_PROTOCOL_ERROR"
    OUTPUT_MISSING = "OUTPUT_MISSING"
    OUTPUT_PATH_VIOLATION = "OUTPUT_PATH_VIOLATION"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    RESOURCE_GRANT_INVALID = "RESOURCE_GRANT_INVALID"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    RESOURCE_RENDERER_UNAVAILABLE = "RESOURCE_RENDERER_UNAVAILABLE"
    MODEL_GRANT_INVALID = "MODEL_GRANT_INVALID"
    MODEL_PROFILE_MISMATCH = "MODEL_PROFILE_MISMATCH"
    MODEL_LIMIT_EXCEEDED = "MODEL_LIMIT_EXCEEDED"


class WorldLedgerFactLockV2(FacadeModel):
    schema_version: Literal["env-mock-agent/world-ledger-fact-lock/v2"] = (
        "env-mock-agent/world-ledger-fact-lock/v2"
    )
    fact_id: Identifier
    value_ref: FacadeObjectRef
    value_sha256: Sha256
    source_refs: tuple[FacadeObjectRef, ...] = ()

    @model_validator(mode="after")
    def validate_lock(self) -> WorldLedgerFactLockV2:
        _require_ref(
            self.value_ref,
            "world-fact-value",
            "v2",
            "value_ref",
        )
        if self.value_ref.object_sha256 != self.value_sha256:
            raise ValueError("value_ref hash must match value_sha256")
        _require_sorted_unique_refs("source refs", self.source_refs)
        return self


class WorldLedgerSnapshotRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/world-ledger-snapshot-request/v2"] = (
        "env-mock-agent/world-ledger-snapshot-request/v2"
    )
    snapshot_request_id: Identifier
    world_ledger_ref: FacadeObjectRef
    required_fact_ids: tuple[Identifier, ...]
    policy_version: Literal["artifact-execution/r5-06-v1"] = ARTIFACT_EXECUTION_POLICY_VERSION
    idempotency_key: Identifier
    snapshot_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> WorldLedgerSnapshotRequestV2:
        _require_ref_type(
            self.world_ledger_ref,
            "world-ledger",
            "world_ledger_ref",
        )
        _require_sorted_unique("required fact IDs", self.required_fact_ids)
        return self


class WorldLedgerSnapshotV2(FacadeModel):
    schema_version: Literal["env-mock-agent/world-ledger-snapshot/v2"] = (
        "env-mock-agent/world-ledger-snapshot/v2"
    )
    world_ledger_snapshot_id: Identifier
    snapshot_request_ref: FacadeObjectRef
    world_ledger_ref: FacadeObjectRef
    fact_locks: tuple[WorldLedgerFactLockV2, ...]
    ledger_sha256: Sha256
    policy_version: Literal["artifact-execution/r5-06-v1"] = ARTIFACT_EXECUTION_POLICY_VERSION
    snapshotted_at: datetime
    world_ledger_snapshot_sha256: Sha256

    @model_validator(mode="after")
    def validate_snapshot(self) -> WorldLedgerSnapshotV2:
        _require_ref(
            self.snapshot_request_ref,
            "world-ledger-snapshot-request",
            "v2",
            "snapshot_request_ref",
        )
        _require_ref_type(
            self.world_ledger_ref,
            "world-ledger",
            "world_ledger_ref",
        )
        fact_ids = tuple(item.fact_id for item in self.fact_locks)
        _require_sorted_unique("fact lock IDs", fact_ids)
        if self.snapshotted_at.tzinfo is None:
            raise ValueError("snapshotted_at must be timezone-aware")
        return self


class AttachmentExecutionSourceSpanV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-execution-source-span/v2"] = (
        "env-mock-agent/attachment-execution-source-span/v2"
    )
    span_id: Identifier
    source_trace_id: Identifier
    raw_sha256: Sha256
    approximate: bool = False


class AttachmentExecutionEvidenceGrantV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-execution-evidence-grant/v2"] = (
        "env-mock-agent/attachment-execution-evidence-grant/v2"
    )
    evidence_ref_id: Identifier
    subject_ref: FacadeObjectRef
    source_spans: tuple[AttachmentExecutionSourceSpanV2, ...] = Field(min_length=1)
    polarity: Literal[
        "POSITIVE",
        "NEGATIVE",
        "UNCERTAINTY",
        "CONTRADICTION",
    ]
    capability: Identifier
    capability_complete: bool

    @model_validator(mode="after")
    def validate_grant(self) -> AttachmentExecutionEvidenceGrantV2:
        span_keys = tuple(
            (
                item.source_trace_id,
                item.span_id,
                item.raw_sha256,
                item.approximate,
            )
            for item in self.source_spans
        )
        _require_sorted_unique("source spans", span_keys)
        return self


class AttachmentExecutionRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-execution-request/v2"] = (
        "env-mock-agent/attachment-execution-request/v2"
    )
    execution_request_id: Identifier
    execution_plan_ref: FacadeObjectRef
    execution_group_id: Identifier
    artifact_id: Identifier
    build_spec_ref: FacadeObjectRef
    producer_task_view_ref: FacadeObjectRef
    content_contract_ref: FacadeObjectRef
    render_contract_ref: FacadeObjectRef
    provider_payload_ref: FacadeObjectRef | None = None
    model_profile_ref: FacadeObjectRef | None = None
    selected_route_kind: AttachmentExecutionRouteKindV2
    selected_route_id: Identifier
    selected_route_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    relative_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    mode: ReconstructionMode
    runtime_role: Identifier
    required_runtime_tools: tuple[CapabilityToken, ...]
    evidence_grants: tuple[AttachmentExecutionEvidenceGrantV2, ...]
    world_ledger_snapshot_ref: FacadeObjectRef
    locked_fact_ids: tuple[Identifier, ...]
    dependency_result_refs: tuple[FacadeObjectRef, ...]
    attempt: int = Field(ge=1)
    retry_of_result_ref: FacadeObjectRef | None = None
    policy_version: Literal["artifact-execution/r5-06-v1"] = ARTIFACT_EXECUTION_POLICY_VERSION
    idempotency_key: Identifier
    execution_request_sha256: Sha256

    @field_validator("selected_route_kind", mode="before")
    @classmethod
    def parse_route_kind(
        cls,
        value: object,
    ) -> AttachmentExecutionRouteKindV2:
        if isinstance(value, AttachmentExecutionRouteKindV2):
            return value
        if isinstance(value, str):
            return AttachmentExecutionRouteKindV2(value)
        raise TypeError("selected_route_kind must be an AttachmentExecutionRouteKindV2")

    @field_validator("mode", mode="before")
    @classmethod
    def parse_mode(cls, value: object) -> ReconstructionMode:
        if isinstance(value, ReconstructionMode):
            return value
        if isinstance(value, str):
            return ReconstructionMode(value)
        raise TypeError("mode must be a ReconstructionMode")

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentExecutionRequestV2:
        for ref, expected_type, expected_version, field_name in (
            (
                self.execution_plan_ref,
                "artifact-execution-plan",
                "v2",
                "execution_plan_ref",
            ),
            (
                self.build_spec_ref,
                "artifact-build-spec",
                "v2",
                "build_spec_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
                "producer_task_view_ref",
            ),
            (
                self.content_contract_ref,
                "artifact-content-contract",
                "v2",
                "content_contract_ref",
            ),
            (
                self.render_contract_ref,
                "artifact-render-contract",
                "v2",
                "render_contract_ref",
            ),
            (
                self.world_ledger_snapshot_ref,
                "world-ledger-snapshot",
                "v2",
                "world_ledger_snapshot_ref",
            ),
        ):
            _require_ref(ref, expected_type, expected_version, field_name)
        if self.provider_payload_ref is not None:
            _require_ref(
                self.provider_payload_ref,
                "attachment-provider-payload",
                "v2",
                "provider_payload_ref",
            )
        if self.model_profile_ref is not None:
            _require_ref_type(
                self.model_profile_ref,
                "model-profile",
                "model_profile_ref",
            )
        provider_route = self.selected_route_kind is AttachmentExecutionRouteKindV2.PROVIDER
        if provider_route:
            if self.provider_payload_ref is None or self.model_profile_ref is not None:
                raise ValueError("provider route requires payload and no model profile")
        elif self.provider_payload_ref is not None or self.model_profile_ref is None:
            raise ValueError("runtime route requires model profile and no provider payload")
        if self.mode is ReconstructionMode.BLOCKED:
            raise ValueError("execution request cannot use BLOCKED mode")
        _require_sorted_unique("required runtime tools", self.required_runtime_tools)
        evidence_ids = tuple(item.evidence_ref_id for item in self.evidence_grants)
        _require_sorted_unique("evidence grant IDs", evidence_ids)
        _require_sorted_unique("locked fact IDs", self.locked_fact_ids)
        _require_sorted_unique_refs(
            "dependency result refs",
            self.dependency_result_refs,
        )
        for ref in self.dependency_result_refs:
            _require_ref(
                ref,
                "attachment-execution-result",
                "v2",
                "dependency_result_refs",
            )
        if self.attempt == 1 and self.retry_of_result_ref is not None:
            raise ValueError("first attempt cannot reference a retry result")
        if self.attempt > 1 and self.retry_of_result_ref is None:
            raise ValueError("later attempt requires retry_of_result_ref")
        if self.retry_of_result_ref is not None:
            _require_ref(
                self.retry_of_result_ref,
                "attachment-execution-result",
                "v2",
                "retry_of_result_ref",
            )
        return self


class AttachmentExecutionResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-execution-result/v2"] = (
        "env-mock-agent/attachment-execution-result/v2"
    )
    execution_result_id: Identifier
    execution_request_ref: FacadeObjectRef
    artifact_id: Identifier
    attempt: int = Field(ge=1)
    status: AttachmentExecutionStatusV2
    world_ledger_snapshot_ref: FacadeObjectRef
    selected_route_kind: AttachmentExecutionRouteKindV2
    selected_route_id: Identifier
    worker_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
    )
    output_ref: FacadeObjectRef | None = None
    output_sha256: Sha256 | None = None
    retryable: bool
    failure_code: AttachmentExecutionFailureCodeV2 | None = None
    telemetry: ExecutionTelemetryV2
    policy_version: Literal["artifact-execution/r5-06-v1"] = ARTIFACT_EXECUTION_POLICY_VERSION
    execution_result_sha256: Sha256

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> AttachmentExecutionStatusV2:
        if isinstance(value, AttachmentExecutionStatusV2):
            return value
        if isinstance(value, str):
            return AttachmentExecutionStatusV2(value)
        raise TypeError("status must be an AttachmentExecutionStatusV2")

    @field_validator("selected_route_kind", mode="before")
    @classmethod
    def parse_route_kind(
        cls,
        value: object,
    ) -> AttachmentExecutionRouteKindV2:
        if isinstance(value, AttachmentExecutionRouteKindV2):
            return value
        if isinstance(value, str):
            return AttachmentExecutionRouteKindV2(value)
        raise TypeError("selected_route_kind must be an AttachmentExecutionRouteKindV2")

    @field_validator("failure_code", mode="before")
    @classmethod
    def parse_failure_code(
        cls,
        value: object,
    ) -> AttachmentExecutionFailureCodeV2 | None:
        if value is None or isinstance(value, AttachmentExecutionFailureCodeV2):
            return value
        if isinstance(value, str):
            return AttachmentExecutionFailureCodeV2(value)
        raise TypeError("failure_code must be an AttachmentExecutionFailureCodeV2")

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentExecutionResultV2:
        _require_ref(
            self.execution_request_ref,
            "attachment-execution-request",
            "v2",
            "execution_request_ref",
        )
        _require_ref(
            self.world_ledger_snapshot_ref,
            "world-ledger-snapshot",
            "v2",
            "world_ledger_snapshot_ref",
        )
        if self.status is AttachmentExecutionStatusV2.SUCCEEDED:
            if (
                self.output_ref is None
                or self.output_sha256 is None
                or self.output_ref.object_sha256 != self.output_sha256
            ):
                raise ValueError("successful execution requires matching output ref and hash")
            if self.failure_code is not None or self.retryable:
                raise ValueError("successful execution cannot contain failure or retryability")
        else:
            if self.output_ref is not None or self.output_sha256 is not None:
                raise ValueError("failed execution cannot contain output")
            if self.failure_code is None:
                raise ValueError("failed execution requires failure code")
            expected_retryable = self.status is AttachmentExecutionStatusV2.RETRYABLE_FAILURE
            if self.retryable is not expected_retryable:
                raise ValueError("retryable must match execution status")
        return self


class AttachmentExecutionFacade(Protocol):
    async def snapshot_world(
        self,
        request: WorldLedgerSnapshotRequestV2,
    ) -> WorldLedgerSnapshotV2: ...

    async def execute(
        self,
        request: AttachmentExecutionRequestV2,
    ) -> AttachmentExecutionResultV2: ...


def world_ledger_fact_id(key: str) -> str:
    normalized = key.strip()
    if not normalized:
        raise ValueError("world ledger fact key cannot be empty")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"world-fact://sha256/{digest}"


def world_ledger_snapshot_request_carried_sha256(
    request: WorldLedgerSnapshotRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={
                "snapshot_request_id",
                "snapshot_request_sha256",
            },
            exclude_none=False,
        )
    )


def world_ledger_snapshot_request_ref(
    request: WorldLedgerSnapshotRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="world-ledger-snapshot-request",
        object_id=request.snapshot_request_id,
        object_version="v2",
        object_sha256=request.snapshot_request_sha256,
    )


def world_ledger_snapshot_carried_sha256(
    snapshot: WorldLedgerSnapshotV2,
) -> str:
    return _payload_sha256(
        snapshot.model_dump(
            mode="json",
            exclude={
                "world_ledger_snapshot_id",
                "world_ledger_snapshot_sha256",
            },
            exclude_none=False,
        )
    )


def world_ledger_snapshot_ref(
    snapshot: WorldLedgerSnapshotV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="world-ledger-snapshot",
        object_id=snapshot.world_ledger_snapshot_id,
        object_version="v2",
        object_sha256=snapshot.world_ledger_snapshot_sha256,
    )


def attachment_execution_request_carried_sha256(
    request: AttachmentExecutionRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={
                "execution_request_id",
                "execution_request_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_execution_request_ref(
    request: AttachmentExecutionRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-execution-request",
        object_id=request.execution_request_id,
        object_version="v2",
        object_sha256=request.execution_request_sha256,
    )


def attachment_execution_result_carried_sha256(
    result: AttachmentExecutionResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={
                "execution_result_id",
                "execution_result_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_execution_result_ref(
    result: AttachmentExecutionResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-execution-result",
        object_id=result.execution_result_id,
        object_version="v2",
        object_sha256=result.execution_result_sha256,
    )


def validate_world_ledger_snapshot_request_identity(
    request: WorldLedgerSnapshotRequestV2,
) -> None:
    if world_ledger_snapshot_request_carried_sha256(request) != request.snapshot_request_sha256:
        raise ValueError("world ledger snapshot request identity is stale")


def validate_world_ledger_snapshot_identity(
    snapshot: WorldLedgerSnapshotV2,
) -> None:
    if world_ledger_snapshot_carried_sha256(snapshot) != snapshot.world_ledger_snapshot_sha256:
        raise ValueError("world ledger snapshot identity is stale")


def validate_attachment_execution_request_identity(
    request: AttachmentExecutionRequestV2,
) -> None:
    if attachment_execution_request_carried_sha256(request) != request.execution_request_sha256:
        raise ValueError("attachment execution request identity is stale")


def validate_attachment_execution_result_identity(
    result: AttachmentExecutionResultV2,
) -> None:
    if attachment_execution_result_carried_sha256(result) != result.execution_result_sha256:
        raise ValueError("attachment execution result identity is stale")


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref(
    ref: FacadeObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_ref_type(
    ref: FacadeObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _ref_key(ref: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[FacadeObjectRef, ...],
) -> None:
    _require_sorted_unique(label, tuple(_ref_key(ref) for ref in refs))


def _require_sorted_unique(
    label: str,
    values: tuple[object, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{label} must be sorted")
