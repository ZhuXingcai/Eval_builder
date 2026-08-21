from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    FailureRecord,
    Identifier,
    ObjectRef,
    Sha256,
    TypedAttribute,
)
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_sha256_v2,
    canonical_value_v2,
)
from eval_factory.contracts.model_control_v2 import WorkModelReservationStateV2
from eval_factory.contracts.orchestration import ItemStatus, JobStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2, WorkLeaseStateV2
from eval_factory.contracts.resource_v2 import (
    NonModelResourceVectorV2,
    WorkResourceReservationStateV2,
)


class JobRecord(ContractModelV2):
    schema_version: Literal["eval-factory/job-record/v1"] = "eval-factory/job-record/v1"
    job_id: Identifier
    job_spec_sha256: Sha256
    status: JobStatus = JobStatus.CREATED
    row_version: int = Field(ge=0)
    idempotency_key: Identifier
    created_at: datetime
    updated_at: datetime


class ItemRecord(ContractModelV2):
    schema_version: Literal["eval-factory/item-record/v1"] = "eval-factory/item-record/v1"
    item_id: Identifier
    job_id: Identifier
    status: ItemStatus = ItemStatus.CANDIDATE
    row_version: int = Field(ge=0)
    idempotency_key: Identifier
    created_at: datetime
    updated_at: datetime


class StageRunRecord(ContractModelV2):
    schema_version: Literal["eval-factory/stage-run-record/v1"] = "eval-factory/stage-run-record/v1"
    stage_run_id: Identifier
    job_id: Identifier
    item_id: Identifier | None = None
    stage: StageNameV2
    attempt: int = Field(ge=1)
    status: StageRunStatus = StageRunStatus.PENDING
    principal_ref: ObjectRef
    input_refs: tuple[ObjectRef, ...]
    retry_of_stage_run_id: Identifier | None = None
    row_version: int = Field(ge=0)
    idempotency_key: Identifier
    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def validate_retry(self) -> StageRunRecord:
        if self.attempt == 1 and self.retry_of_stage_run_id is not None:
            raise ValueError("first stage attempt cannot retry another run")
        if self.attempt > 1 and self.retry_of_stage_run_id is None:
            raise ValueError("later stage attempt requires retry_of_stage_run_id")
        return self


TERMINAL_STAGE_STATUSES = frozenset(
    {
        StageRunStatus.BLOCKED_POLICY,
        StageRunStatus.BLOCKED_CAPABILITY,
        StageRunStatus.RETRYABLE_FAILURE,
        StageRunStatus.TERMINAL_FAILURE,
        StageRunStatus.SUCCEEDED,
        StageRunStatus.CANCELLED,
    }
)


class StageResultRecord(ContractModelV2):
    schema_version: Literal["eval-factory/stage-result-record/v1"] = "eval-factory/stage-result-record/v1"
    stage_result_id: Identifier
    stage_run_id: Identifier
    stage_run_version: int = Field(ge=0)
    status: StageRunStatus
    output_refs: tuple[ObjectRef, ...] = ()
    failure: FailureRecord | None = None
    checkpoint_ref: ObjectRef | None = None
    metrics_ref: ObjectRef | None = None
    result_sha256: Sha256
    created_at: datetime

    @model_validator(mode="after")
    def validate_outcome(self) -> StageResultRecord:
        if self.status not in TERMINAL_STAGE_STATUSES:
            raise ValueError("StageResultRecord requires a terminal stage status")
        if self.status is StageRunStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful stage result cannot contain failure")
        elif self.failure is None:
            raise ValueError("non-success stage result requires failure")
        return self


class OutboxEvent(ContractModelV2):
    schema_version: Literal["eval-factory/outbox-event/v1"] = "eval-factory/outbox-event/v1"
    event_id: Identifier
    aggregate_type: Literal["JOB", "ITEM", "STAGE_RUN"]
    aggregate_id: Identifier
    aggregate_version: int = Field(ge=0)
    event_type: Identifier
    attributes: tuple[TypedAttribute, ...] = ()
    created_at: datetime
    published_at: datetime | None = None


class WorkLeaseHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/work-lease-head-record/v1"] = (
        "eval-factory/work-lease-head-record/v1"
    )
    work_lease_id: Identifier
    work_unit_id: Identifier
    state: WorkLeaseStateV2
    lease_version: int = Field(ge=0)
    fencing_token: int = Field(ge=1)
    expires_at: datetime


class ResourcePoolHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/resource-pool-head-record/v1"] = (
        "eval-factory/resource-pool-head-record/v1"
    )
    resource_pool_policy_id: Identifier
    active_capacity: NonModelResourceVectorV2
    row_version: int = Field(ge=0)


class JobResourceHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/job-resource-head-record/v1"] = (
        "eval-factory/job-resource-head-record/v1"
    )
    job_resource_policy_id: Identifier
    active_capacity: NonModelResourceVectorV2
    reserved_budget: NonModelResourceVectorV2
    consumed_budget: NonModelResourceVectorV2
    row_version: int = Field(ge=0)


class WorkResourceReservationHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/work-resource-reservation-head-record/v1"] = (
        "eval-factory/work-resource-reservation-head-record/v1"
    )
    work_resource_reservation_id: Identifier
    state: WorkResourceReservationStateV2
    reservation_version: int = Field(ge=0)


class ModelRatePoolHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/model-rate-pool-head-record/v1"] = (
        "eval-factory/model-rate-pool-head-record/v1"
    )
    model_rate_pool_policy_id: Identifier
    # A provider-reported overrun creates explicit debt that later refill repays.
    request_credits: int
    token_credits: int
    active_concurrency: int = Field(ge=0)
    blocked_until: datetime | None = None
    effective_at: datetime
    bucket_version: int = Field(ge=0)
    row_version: int = Field(ge=0)


class JobModelHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/job-model-head-record/v1"] = "eval-factory/job-model-head-record/v1"
    job_model_policy_id: Identifier
    active_concurrency: int = Field(ge=0)
    reserved_requests: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    consumed_requests: int = Field(ge=0)
    consumed_tokens: int = Field(ge=0)
    row_version: int = Field(ge=0)


class WorkModelReservationHeadRecord(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-reservation-head-record/v1"] = (
        "eval-factory/work-model-reservation-head-record/v1"
    )
    work_model_reservation_id: Identifier
    state: WorkModelReservationStateV2
    reservation_version: int = Field(ge=0)


def stage_run_record_carried_sha256(record: StageRunRecord) -> str:
    payload = {
        "stage_run_id": record.stage_run_id,
        "job_id": record.job_id,
        "item_id": record.item_id,
        "stage": record.stage.value,
        "attempt": record.attempt,
        "principal_ref": record.principal_ref,
        "input_refs": record.input_refs,
        "retry_of_stage_run_id": record.retry_of_stage_run_id,
        "idempotency_key": record.idempotency_key,
    }
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_result_record_carried_sha256(record: StageResultRecord) -> str:
    encoded = json.dumps(
        canonical_value_v2(
            record.model_dump(
                mode="python",
                exclude={"schema_version", "result_sha256"},
            )
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def item_record_ref(record: ItemRecord) -> ObjectRef:
    return ObjectRef(
        object_type="item-record",
        object_id=record.item_id,
        object_version="record/v1",
        object_sha256=canonical_sha256_v2(record),
    )


def stage_run_record_ref(record: StageRunRecord) -> ObjectRef:
    return ObjectRef(
        object_type="stage-run",
        object_id=record.stage_run_id,
        object_version="identity/v1",
        object_sha256=stage_run_record_carried_sha256(record),
    )


def stage_result_record_ref(record: StageResultRecord) -> ObjectRef:
    return ObjectRef(
        object_type="stage-result",
        object_id=record.stage_result_id,
        object_version="record/v1",
        object_sha256=record.result_sha256,
    )
