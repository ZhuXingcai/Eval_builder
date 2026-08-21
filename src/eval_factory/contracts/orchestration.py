from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    FailureRecord,
    Identifier,
    ObjectRef,
    Sha256,
)


class StageName(StrEnum):
    TRACE_INDEX = "trace_index"
    SAFETY = "safety"
    LABEL = "label"
    TASK_AUTHORING = "task_authoring"
    ATTACHMENT = "attachment"
    ITEM_QUALITY = "item_quality"
    BATCH_QUALITY = "batch_quality"
    HUMAN_REVIEW = "human_review"
    RELEASE = "release"


class ResourceBudget(ContractModel):
    schema_version: Literal["eval-factory/resource-budget/v1"] = "eval-factory/resource-budget/v1"
    max_model_requests: int = Field(ge=0)
    max_model_tokens: int = Field(ge=0)
    max_processes: int = Field(ge=0)
    max_renderers: int = Field(ge=0)
    max_network_requests: int = Field(ge=0)
    max_storage_bytes: int = Field(ge=0)


class ConcurrencyLimit(ContractModel):
    schema_version: Literal["eval-factory/concurrency-limit/v1"] = "eval-factory/concurrency-limit/v1"
    model_requests: int = Field(ge=1)
    processes: int = Field(ge=1)
    renderers: int = Field(ge=1)
    network_requests: int = Field(ge=1)
    artifacts_per_item: int = Field(ge=1)
    items: int = Field(ge=1)


class TraceSourceRef(ContractModel):
    schema_version: Literal["eval-factory/trace-source-ref/v1"] = "eval-factory/trace-source-ref/v1"
    source_trace_id: Identifier
    source_uri: str = Field(min_length=3, max_length=1024)
    raw_sha256: Sha256
    adapter_name: Identifier
    adapter_version: str = Field(min_length=1, max_length=128)
    processing_class: Literal["RESTRICTED_TRACE_RAW"]


class ExportTarget(ContractModel):
    schema_version: Literal["eval-factory/export-target/v1"] = "eval-factory/export-target/v1"
    profile: Literal["LH", "GENERIC"]
    profile_version: str = Field(min_length=1, max_length=128)
    channel: Literal["CANARY", "INTERNAL_REVIEW", "PRODUCTION"]
    registry: Identifier


class DatasetJobSpec(ContractModel):
    schema_version: Literal["eval-factory/dataset-job-spec/v1"] = "eval-factory/dataset-job-spec/v1"
    job_id: Identifier
    traces: tuple[TraceSourceRef, ...] = Field(min_length=1)
    requested_stages: tuple[StageName, ...] = Field(min_length=1)
    privacy_profile: Identifier
    model_profiles: tuple[Identifier, ...] = ()
    budget: ResourceBudget
    concurrency: ConcurrencyLimit
    selection_spec_ref: ObjectRef | None = None
    export_target: ExportTarget
    idempotency_key: Identifier
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_stage_order(self) -> DatasetJobSpec:
        if len(set(self.requested_stages)) != len(self.requested_stages):
            raise ValueError("requested_stages must be unique")
        if self.requested_stages[0] is not StageName.TRACE_INDEX:
            raise ValueError("requested_stages must begin with trace_index")
        canonical = list(StageName)
        positions = [canonical.index(stage) for stage in self.requested_stages]
        if positions != sorted(positions):
            raise ValueError("requested_stages must follow canonical stage order")
        return self


class JobStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ItemStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    RUNNING = "RUNNING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RELEASED = "RELEASED"
    REVOKED = "REVOKED"
    FAILED = "FAILED"


class StageRunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"
    SUCCEEDED = "SUCCEEDED"
    CANCELLED = "CANCELLED"


class StageRun(ContractModel):
    schema_version: Literal["eval-factory/stage-run/v1"] = "eval-factory/stage-run/v1"
    stage_run_id: Identifier
    job_id: Identifier
    item_id: Identifier | None = None
    stage: StageName
    attempt: int = Field(ge=1)
    status: StageRunStatus
    principal_ref: ObjectRef
    input_refs: tuple[ObjectRef, ...]
    retry_of: ObjectRef | None = None
    idempotency_key: Identifier
    started_at: datetime | None = None
    ended_at: datetime | None = None


class StageResult(ContractModel):
    schema_version: Literal["eval-factory/stage-result/v1"] = "eval-factory/stage-result/v1"
    stage_result_id: Identifier
    stage_run_ref: ObjectRef
    status: StageRunStatus
    output_refs: tuple[ObjectRef, ...] = ()
    failure: FailureRecord | None = None
    checkpoint_ref: ObjectRef | None = None
    metrics_ref: ObjectRef | None = None
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_outcome(self) -> StageResult:
        if self.status is StageRunStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful StageResult cannot contain failure")
        elif self.failure is None:
            raise ValueError("non-success StageResult requires failure")
        return self
