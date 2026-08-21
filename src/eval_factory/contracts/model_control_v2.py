from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

R6_MODEL_CONTROL_POLICY_VERSION: Literal["model-control/r6-03-v1"] = "model-control/r6-03-v1"


class ModelAdmissionOutcomeV2(StrEnum):
    ADMITTED = "ADMITTED"
    WAITING_RATE = "WAITING_RATE"
    WAITING_CONCURRENCY = "WAITING_CONCURRENCY"
    WAITING_PROVIDER = "WAITING_PROVIDER"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    UNSATISFIABLE_DEMAND = "UNSATISFIABLE_DEMAND"


class ModelRateDimensionV2(StrEnum):
    REQUESTS = "REQUESTS"
    TOKENS = "TOKENS"
    CONCURRENCY = "CONCURRENCY"
    PROVIDER_BACKPRESSURE = "PROVIDER_BACKPRESSURE"


class ModelDemandModeV2(StrEnum):
    DIRECT_REQUEST = "DIRECT_REQUEST"
    OPAQUE_RUNTIME_ENVELOPE = "OPAQUE_RUNTIME_ENVELOPE"


class ModelUsageSourceV2(StrEnum):
    REPORTED = "REPORTED"
    CONSERVATIVE_ALLOWANCE = "CONSERVATIVE_ALLOWANCE"


class ProviderBackpressureKindV2(StrEnum):
    RATE_LIMITED = "RATE_LIMITED"
    OVERLOADED = "OVERLOADED"


class WorkModelEventKindV2(StrEnum):
    ADMITTED = "ADMITTED"
    COMPLETED_RELEASED = "COMPLETED_RELEASED"
    BACKPRESSURED_RELEASED = "BACKPRESSURED_RELEASED"
    USAGE_OVERRUN_RELEASED = "USAGE_OVERRUN_RELEASED"
    RELEASE_PENDING_ACK = "RELEASE_PENDING_ACK"
    ACKNOWLEDGED_RELEASED = "ACKNOWLEDGED_RELEASED"
    QUARANTINED = "QUARANTINED"


class WorkModelReservationStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    RELEASE_PENDING_ACK = "RELEASE_PENDING_ACK"
    RELEASED = "RELEASED"
    QUARANTINED = "QUARANTINED"


class WorkModelCancellationReasonV2(StrEnum):
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    USAGE_OVERRUN = "USAGE_OVERRUN"


class WorkModelCancellationOutcomeV2(StrEnum):
    CANCELLED = "CANCELLED"
    ALREADY_FINISHED = "ALREADY_FINISHED"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    FAILED = "FAILED"


class ModelRatePoolPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/model-rate-pool-policy/v2"] = (
        "eval-factory/model-rate-pool-policy/v2"
    )
    model_rate_pool_policy_id: Identifier
    provider_bucket_ref: ObjectRef
    allowed_model_profile_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    requests_per_minute: int = Field(ge=1)
    tokens_per_minute: int = Field(ge=1)
    request_burst: int = Field(ge=1)
    token_burst: int = Field(ge=1)
    max_concurrent_requests: int = Field(ge=1)
    minimum_backpressure_seconds: int = Field(ge=1)
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    model_rate_pool_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(self.provider_bucket_ref, "provider-rate-bucket", "v1", "provider_bucket_ref")
        profiles = _sorted_unique_refs(self.allowed_model_profile_refs, "allowed_model_profile_refs")
        if profiles != self.allowed_model_profile_refs:
            raise ValueError("allowed_model_profile_refs must be sorted and unique")
        for ref in profiles:
            _require_model_profile_ref(ref, "allowed_model_profile_refs")
        refs = (self.provider_bucket_ref, *profiles)
        _require_model_audit(self.audit, refs)
        validate_model_rate_pool_policy_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        provider_bucket_ref: ObjectRef,
        allowed_model_profile_refs: tuple[ObjectRef, ...],
        requests_per_minute: int,
        tokens_per_minute: int,
        request_burst: int,
        token_burst: int,
        max_concurrent_requests: int,
        minimum_backpressure_seconds: int,
        audit: ContractAudit,
    ) -> ModelRatePoolPolicyV2:
        value = cls(
            model_rate_pool_policy_id="model-rate-pool-policy://pending",
            provider_bucket_ref=provider_bucket_ref,
            allowed_model_profile_refs=_sorted_unique_refs(
                allowed_model_profile_refs,
                "allowed_model_profile_refs",
            ),
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            request_burst=request_burst,
            token_burst=token_burst,
            max_concurrent_requests=max_concurrent_requests,
            minimum_backpressure_seconds=minimum_backpressure_seconds,
            model_rate_pool_policy_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "model_rate_pool_policy_id",
            "model_rate_pool_policy_sha256",
            "model-rate-pool-policy",
        )


class JobModelPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/job-model-policy/v2"] = "eval-factory/job-model-policy/v2"
    job_model_policy_id: Identifier
    dataset_job_spec_ref: ObjectRef
    resolved_job_work_graph_ref: ObjectRef
    model_rate_pool_policy_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    model_profile_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    max_concurrent_requests: int = Field(ge=1)
    max_model_requests: int = Field(ge=0)
    max_model_tokens: int = Field(ge=0)
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    job_model_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_ref(self.dataset_job_spec_ref, "dataset-job-spec", "v2", "dataset_job_spec_ref")
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        pools = _sorted_unique_refs(
            self.model_rate_pool_policy_refs,
            "model_rate_pool_policy_refs",
        )
        profiles = _sorted_unique_refs(self.model_profile_refs, "model_profile_refs")
        if pools != self.model_rate_pool_policy_refs:
            raise ValueError("model_rate_pool_policy_refs must be sorted and unique")
        if profiles != self.model_profile_refs:
            raise ValueError("model_profile_refs must be sorted and unique")
        for ref in pools:
            _require_ref(ref, "model-rate-pool-policy", "v2", "model_rate_pool_policy_refs")
        for ref in profiles:
            _require_model_profile_ref(ref, "model_profile_refs")
        refs = (
            self.dataset_job_spec_ref,
            self.resolved_job_work_graph_ref,
            *pools,
            *profiles,
        )
        _require_model_audit(self.audit, refs)
        validate_job_model_policy_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_job_spec_ref: ObjectRef,
        resolved_job_work_graph_ref: ObjectRef,
        model_rate_pool_policy_refs: tuple[ObjectRef, ...],
        model_profile_refs: tuple[ObjectRef, ...],
        max_concurrent_requests: int,
        max_model_requests: int,
        max_model_tokens: int,
        audit: ContractAudit,
    ) -> JobModelPolicyV2:
        value = cls(
            job_model_policy_id="job-model-policy://pending",
            dataset_job_spec_ref=dataset_job_spec_ref,
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            model_rate_pool_policy_refs=_sorted_unique_refs(
                model_rate_pool_policy_refs,
                "model_rate_pool_policy_refs",
            ),
            model_profile_refs=_sorted_unique_refs(model_profile_refs, "model_profile_refs"),
            max_concurrent_requests=max_concurrent_requests,
            max_model_requests=max_model_requests,
            max_model_tokens=max_model_tokens,
            job_model_policy_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "job_model_policy_id",
            "job_model_policy_sha256",
            "job-model-policy",
        )


class WorkModelDemandV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-demand/v2"] = "eval-factory/work-model-demand/v2"
    work_model_demand_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    work_unit_ref: ObjectRef
    job_model_policy_ref: ObjectRef
    model_rate_pool_policy_ref: ObjectRef
    model_profile_ref: ObjectRef
    operation_ref: ObjectRef
    model_execution_profile_ref: ObjectRef
    attempt: int = Field(ge=1)
    mode: ModelDemandModeV2
    request_allowance: int = Field(ge=1)
    input_token_allowance: int = Field(ge=0)
    output_token_allowance: int = Field(ge=1)
    cache_token_allowance: int = Field(ge=0)
    token_allowance: int = Field(ge=1)
    concurrency_units: int = Field(ge=1)
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    work_model_demand_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_demand(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.resolved_job_work_graph_ref,
                "resolved-job-work-graph",
                "v2",
                "resolved_job_work_graph_ref",
            ),
            (self.work_unit_ref, "resolved-work-unit", "v2", "work_unit_ref"),
            (self.job_model_policy_ref, "job-model-policy", "v2", "job_model_policy_ref"),
            (
                self.model_rate_pool_policy_ref,
                "model-rate-pool-policy",
                "v2",
                "model_rate_pool_policy_ref",
            ),
            (
                self.model_execution_profile_ref,
                "model-execution-profile",
                "v1",
                "model_execution_profile_ref",
            ),
        ):
            _require_ref(ref, object_type, version, name)
        _require_model_profile_ref(self.model_profile_ref, "model_profile_ref")
        if self.operation_ref.object_version not in {"v1", "v2"}:
            raise ValueError("operation_ref must reference a versioned operation")
        expected_tokens = (
            self.input_token_allowance + self.output_token_allowance + self.cache_token_allowance
        )
        if self.token_allowance != expected_tokens:
            raise ValueError("token allowance must equal input, output, and cache allowances")
        if self.mode is ModelDemandModeV2.DIRECT_REQUEST and self.request_allowance != 1:
            raise ValueError("DIRECT_REQUEST demand requires exactly one request")
        refs = (
            self.resolved_job_work_graph_ref,
            self.work_unit_ref,
            self.job_model_policy_ref,
            self.model_rate_pool_policy_ref,
            self.model_profile_ref,
            self.operation_ref,
            self.model_execution_profile_ref,
        )
        _require_model_audit(self.audit, refs)
        validate_work_model_demand_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        work_unit_ref: ObjectRef,
        job_model_policy_ref: ObjectRef,
        model_rate_pool_policy_ref: ObjectRef,
        model_profile_ref: ObjectRef,
        operation_ref: ObjectRef,
        model_execution_profile_ref: ObjectRef,
        attempt: int,
        mode: ModelDemandModeV2,
        request_allowance: int,
        input_token_allowance: int,
        output_token_allowance: int,
        cache_token_allowance: int,
        concurrency_units: int,
        audit: ContractAudit,
    ) -> WorkModelDemandV2:
        token_allowance = input_token_allowance + output_token_allowance + cache_token_allowance
        value = cls(
            work_model_demand_id="work-model-demand://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            work_unit_ref=work_unit_ref,
            job_model_policy_ref=job_model_policy_ref,
            model_rate_pool_policy_ref=model_rate_pool_policy_ref,
            model_profile_ref=model_profile_ref,
            operation_ref=operation_ref,
            model_execution_profile_ref=model_execution_profile_ref,
            attempt=attempt,
            mode=mode,
            request_allowance=request_allowance,
            input_token_allowance=input_token_allowance,
            output_token_allowance=output_token_allowance,
            cache_token_allowance=cache_token_allowance,
            token_allowance=token_allowance,
            concurrency_units=concurrency_units,
            work_model_demand_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_model_demand_id",
            "work_model_demand_sha256",
            "work-model-demand",
        )


class ModelAdmissionDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/model-admission-decision/v2"] = (
        "eval-factory/model-admission-decision/v2"
    )
    model_admission_decision_id: Identifier
    job_model_policy_ref: ObjectRef
    work_model_demand_ref: ObjectRef
    outcome: ModelAdmissionOutcomeV2
    constrained_dimensions: tuple[ModelRateDimensionV2, ...]
    eligible_at: datetime | None = None
    pool_head_version_before: int = Field(ge=0)
    job_head_version_before: int = Field(ge=0)
    decided_at: datetime
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    model_admission_decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        _require_ref(self.job_model_policy_ref, "job-model-policy", "v2", "job_model_policy_ref")
        _require_ref(self.work_model_demand_ref, "work-model-demand", "v2", "work_model_demand_ref")
        expected = tuple(sorted(set(self.constrained_dimensions), key=lambda item: item.value))
        if expected != self.constrained_dimensions:
            raise ValueError("constrained_dimensions must be sorted and unique")
        if self.outcome is ModelAdmissionOutcomeV2.ADMITTED:
            if self.constrained_dimensions or self.eligible_at is not None:
                raise ValueError("ADMITTED model decision cannot carry constraints or eligibility")
        elif not self.constrained_dimensions:
            raise ValueError("negative model decision requires constrained dimensions")
        waiting_with_time = self.outcome in {
            ModelAdmissionOutcomeV2.WAITING_RATE,
            ModelAdmissionOutcomeV2.WAITING_PROVIDER,
        }
        if waiting_with_time != (self.eligible_at is not None):
            raise ValueError("only rate/provider waiting decisions require eligible_at")
        if self.eligible_at is not None and self.eligible_at <= self.decided_at:
            raise ValueError("eligible_at must be later than decided_at")
        refs = (self.job_model_policy_ref, self.work_model_demand_ref)
        _require_model_audit(self.audit, refs)
        validate_model_admission_decision_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        job_model_policy_ref: ObjectRef,
        work_model_demand_ref: ObjectRef,
        outcome: ModelAdmissionOutcomeV2,
        constrained_dimensions: tuple[ModelRateDimensionV2, ...],
        eligible_at: datetime | None,
        pool_head_version_before: int,
        job_head_version_before: int,
        decided_at: datetime,
        audit: ContractAudit,
    ) -> ModelAdmissionDecisionV2:
        value = cls(
            model_admission_decision_id="model-admission-decision://pending",
            job_model_policy_ref=job_model_policy_ref,
            work_model_demand_ref=work_model_demand_ref,
            outcome=outcome,
            constrained_dimensions=tuple(sorted(set(constrained_dimensions), key=lambda item: item.value)),
            eligible_at=eligible_at,
            pool_head_version_before=pool_head_version_before,
            job_head_version_before=job_head_version_before,
            decided_at=decided_at,
            model_admission_decision_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "model_admission_decision_id",
            "model_admission_decision_sha256",
            "model-admission-decision",
        )


class WorkModelReservationV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-reservation/v2"] = (
        "eval-factory/work-model-reservation/v2"
    )
    work_model_reservation_id: Identifier
    model_admission_decision_ref: ObjectRef
    admission_outcome: Literal[ModelAdmissionOutcomeV2.ADMITTED] = ModelAdmissionOutcomeV2.ADMITTED
    work_model_demand_ref: ObjectRef
    work_lease_ref: ObjectRef
    work_dispatch_decision_ref: ObjectRef
    holder_ref: ObjectRef
    fencing_token: int = Field(ge=1)
    invocation_handle_ref: ObjectRef
    request_allowance: int = Field(ge=1)
    token_allowance: int = Field(ge=1)
    concurrency_units: int = Field(ge=1)
    acquired_at: datetime
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    work_model_reservation_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_reservation(self) -> Self:
        for ref, object_type, version, name in (
            (
                self.model_admission_decision_ref,
                "model-admission-decision",
                "v2",
                "model_admission_decision_ref",
            ),
            (self.work_model_demand_ref, "work-model-demand", "v2", "work_model_demand_ref"),
            (self.work_lease_ref, "work-lease", "v2", "work_lease_ref"),
            (
                self.work_dispatch_decision_ref,
                "work-dispatch-decision",
                "v2",
                "work_dispatch_decision_ref",
            ),
            (self.holder_ref, "worker-principal", "v1", "holder_ref"),
            (
                self.invocation_handle_ref,
                "model-invocation-handle",
                "v1",
                "invocation_handle_ref",
            ),
        ):
            _require_ref(ref, object_type, version, name)
        refs = (
            self.model_admission_decision_ref,
            self.work_model_demand_ref,
            self.work_lease_ref,
            self.work_dispatch_decision_ref,
            self.holder_ref,
            self.invocation_handle_ref,
        )
        _require_model_audit(self.audit, refs)
        validate_work_model_reservation_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        model_admission_decision_ref: ObjectRef,
        work_model_demand_ref: ObjectRef,
        work_lease_ref: ObjectRef,
        work_dispatch_decision_ref: ObjectRef,
        holder_ref: ObjectRef,
        fencing_token: int,
        invocation_handle_ref: ObjectRef,
        request_allowance: int,
        token_allowance: int,
        concurrency_units: int,
        acquired_at: datetime,
        audit: ContractAudit,
    ) -> WorkModelReservationV2:
        value = cls(
            work_model_reservation_id="work-model-reservation://pending",
            model_admission_decision_ref=model_admission_decision_ref,
            work_model_demand_ref=work_model_demand_ref,
            work_lease_ref=work_lease_ref,
            work_dispatch_decision_ref=work_dispatch_decision_ref,
            holder_ref=holder_ref,
            fencing_token=fencing_token,
            invocation_handle_ref=invocation_handle_ref,
            request_allowance=request_allowance,
            token_allowance=token_allowance,
            concurrency_units=concurrency_units,
            acquired_at=acquired_at,
            work_model_reservation_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_model_reservation_id",
            "work_model_reservation_sha256",
            "work-model-reservation",
        )


class ModelUsageV2(ContractModelV2):
    schema_version: Literal["eval-factory/model-usage/v2"] = "eval-factory/model-usage/v2"
    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)
    source: ModelUsageSourceV2
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION

    @model_validator(mode="after")
    def validate_usage(self) -> Self:
        expected = (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        if self.charged_tokens != expected:
            raise ValueError("charged_tokens must equal all normalized token components")
        return self

    @classmethod
    def conservative(cls, *, requests: int, tokens: int) -> ModelUsageV2:
        return cls(
            requests=requests,
            input_tokens=tokens,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=tokens,
            source=ModelUsageSourceV2.CONSERVATIVE_ALLOWANCE,
        )


class ProviderBackpressureRecordV2(ContractModelV2):
    schema_version: Literal["eval-factory/provider-backpressure-record/v2"] = (
        "eval-factory/provider-backpressure-record/v2"
    )
    provider_backpressure_record_id: Identifier
    work_model_reservation_ref: ObjectRef
    facade_receipt_ref: ObjectRef
    kind: ProviderBackpressureKindV2
    usage: ModelUsageV2
    observed_at: datetime
    eligible_at: datetime
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    provider_backpressure_record_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _require_ref(
            self.work_model_reservation_ref,
            "work-model-reservation",
            "v2",
            "work_model_reservation_ref",
        )
        _require_ref(self.facade_receipt_ref, "model-control-receipt", "v2", "facade_receipt_ref")
        if self.eligible_at <= self.observed_at:
            raise ValueError("backpressure eligible_at must be later than observed_at")
        refs = (self.work_model_reservation_ref, self.facade_receipt_ref)
        _require_model_audit(self.audit, refs)
        validate_provider_backpressure_record_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_model_reservation_ref: ObjectRef,
        facade_receipt_ref: ObjectRef,
        kind: ProviderBackpressureKindV2,
        usage: ModelUsageV2,
        observed_at: datetime,
        eligible_at: datetime,
        audit: ContractAudit,
    ) -> ProviderBackpressureRecordV2:
        value = cls(
            provider_backpressure_record_id="provider-backpressure-record://pending",
            work_model_reservation_ref=work_model_reservation_ref,
            facade_receipt_ref=facade_receipt_ref,
            kind=kind,
            usage=usage,
            observed_at=observed_at,
            eligible_at=eligible_at,
            provider_backpressure_record_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "provider_backpressure_record_id",
            "provider_backpressure_record_sha256",
            "provider-backpressure-record",
        )


class WorkModelCancellationRequestV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-cancellation-request/v2"] = (
        "eval-factory/work-model-cancellation-request/v2"
    )
    work_model_cancellation_request_id: Identifier
    work_model_reservation_ref: ObjectRef
    work_lease_event_ref: ObjectRef
    invocation_handle_ref: ObjectRef
    fencing_token: int = Field(ge=1)
    reason: WorkModelCancellationReasonV2
    requested_at: datetime
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    work_model_cancellation_request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.work_model_reservation_ref,
            "work-model-reservation",
            "v2",
            "work_model_reservation_ref",
        )
        _require_ref(self.work_lease_event_ref, "work-lease-event", "v2", "work_lease_event_ref")
        _require_ref(
            self.invocation_handle_ref,
            "model-invocation-handle",
            "v1",
            "invocation_handle_ref",
        )
        refs = (
            self.work_model_reservation_ref,
            self.work_lease_event_ref,
            self.invocation_handle_ref,
        )
        _require_model_audit(self.audit, refs)
        validate_work_model_cancellation_request_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_model_reservation_ref: ObjectRef,
        work_lease_event_ref: ObjectRef,
        invocation_handle_ref: ObjectRef,
        fencing_token: int,
        reason: WorkModelCancellationReasonV2,
        requested_at: datetime,
        audit: ContractAudit,
    ) -> WorkModelCancellationRequestV2:
        value = cls(
            work_model_cancellation_request_id="work-model-cancellation-request://pending",
            work_model_reservation_ref=work_model_reservation_ref,
            work_lease_event_ref=work_lease_event_ref,
            invocation_handle_ref=invocation_handle_ref,
            fencing_token=fencing_token,
            reason=reason,
            requested_at=requested_at,
            work_model_cancellation_request_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_model_cancellation_request_id",
            "work_model_cancellation_request_sha256",
            "work-model-cancellation-request",
        )


class WorkModelCancellationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-cancellation-result/v2"] = (
        "eval-factory/work-model-cancellation-result/v2"
    )
    work_model_cancellation_result_id: Identifier
    cancellation_request_ref: ObjectRef
    facade_cancellation_result_ref: ObjectRef
    outcome: WorkModelCancellationOutcomeV2
    completed_at: datetime
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    work_model_cancellation_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.cancellation_request_ref,
            "work-model-cancellation-request",
            "v2",
            "cancellation_request_ref",
        )
        _require_ref(
            self.facade_cancellation_result_ref,
            "model-invocation-cancellation-result",
            "v2",
            "facade_cancellation_result_ref",
        )
        refs = (self.cancellation_request_ref, self.facade_cancellation_result_ref)
        _require_model_audit(self.audit, refs)
        validate_work_model_cancellation_result_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        cancellation_request_ref: ObjectRef,
        facade_cancellation_result_ref: ObjectRef,
        outcome: WorkModelCancellationOutcomeV2,
        completed_at: datetime,
        audit: ContractAudit,
    ) -> WorkModelCancellationResultV2:
        value = cls(
            work_model_cancellation_result_id="work-model-cancellation-result://pending",
            cancellation_request_ref=cancellation_request_ref,
            facade_cancellation_result_ref=facade_cancellation_result_ref,
            outcome=outcome,
            completed_at=completed_at,
            work_model_cancellation_result_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_model_cancellation_result_id",
            "work_model_cancellation_result_sha256",
            "work-model-cancellation-result",
        )


class WorkModelEventV2(ContractModelV2):
    schema_version: Literal["eval-factory/work-model-event/v2"] = "eval-factory/work-model-event/v2"
    work_model_event_id: Identifier
    work_model_reservation_ref: ObjectRef
    work_lease_event_ref: ObjectRef | None = None
    event_kind: WorkModelEventKindV2
    reservation_version: int = Field(ge=0)
    bucket_version: int = Field(ge=1)
    effective_at: datetime
    usage: ModelUsageV2
    refunded_request_allowance: int = Field(ge=0)
    refunded_token_allowance: int = Field(ge=0)
    released_concurrency: int = Field(ge=0)
    provider_backpressure_ref: ObjectRef | None = None
    model_cancellation_request_ref: ObjectRef | None = None
    model_cancellation_result_ref: ObjectRef | None = None
    blocked_until: datetime | None = None
    policy_version: Literal["model-control/r6-03-v1"] = R6_MODEL_CONTROL_POLICY_VERSION
    work_model_event_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        _require_ref(
            self.work_model_reservation_ref,
            "work-model-reservation",
            "v2",
            "work_model_reservation_ref",
        )
        admitted = self.event_kind is WorkModelEventKindV2.ADMITTED
        if admitted != (self.work_lease_event_ref is None):
            raise ValueError("only ADMITTED event omits work_lease_event_ref")
        if self.work_lease_event_ref is not None:
            _require_ref(
                self.work_lease_event_ref,
                "work-lease-event",
                "v2",
                "work_lease_event_ref",
            )
        if admitted:
            if (
                self.reservation_version != 0
                or self.released_concurrency
                or self.refunded_request_allowance
                or self.refunded_token_allowance
            ):
                raise ValueError("ADMITTED event cannot release or refund")
        elif self.reservation_version < 1:
            raise ValueError("terminal model event requires positive reservation version")
        backpressured = self.event_kind is WorkModelEventKindV2.BACKPRESSURED_RELEASED
        if backpressured != (self.provider_backpressure_ref is not None):
            raise ValueError("backpressure event requires exactly one backpressure ref")
        if self.provider_backpressure_ref is not None:
            _require_ref(
                self.provider_backpressure_ref,
                "provider-backpressure-record",
                "v2",
                "provider_backpressure_ref",
            )
        pending = self.event_kind is WorkModelEventKindV2.RELEASE_PENDING_ACK
        if pending != (self.model_cancellation_request_ref is not None):
            raise ValueError("release-pending event requires exactly one cancellation request")
        acknowledged = self.event_kind in {
            WorkModelEventKindV2.ACKNOWLEDGED_RELEASED,
            WorkModelEventKindV2.QUARANTINED,
        }
        if acknowledged != (self.model_cancellation_result_ref is not None):
            raise ValueError("acknowledgement event requires exactly one cancellation result")
        for ref, object_type, name in (
            (
                self.model_cancellation_request_ref,
                "work-model-cancellation-request",
                "model_cancellation_request_ref",
            ),
            (
                self.model_cancellation_result_ref,
                "work-model-cancellation-result",
                "model_cancellation_result_ref",
            ),
        ):
            if ref is not None:
                _require_ref(ref, object_type, "v2", name)
        if self.blocked_until is not None and self.blocked_until <= self.effective_at:
            raise ValueError("blocked_until must be later than effective_at")
        refs = tuple(
            ref
            for ref in (
                self.work_model_reservation_ref,
                self.work_lease_event_ref,
                self.provider_backpressure_ref,
                self.model_cancellation_request_ref,
                self.model_cancellation_result_ref,
            )
            if ref is not None
        )
        _require_model_audit(self.audit, refs)
        validate_work_model_event_v2_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        work_model_reservation_ref: ObjectRef,
        work_lease_event_ref: ObjectRef | None,
        event_kind: WorkModelEventKindV2,
        reservation_version: int,
        bucket_version: int,
        effective_at: datetime,
        usage: ModelUsageV2,
        refunded_request_allowance: int,
        refunded_token_allowance: int,
        released_concurrency: int,
        provider_backpressure_ref: ObjectRef | None,
        model_cancellation_request_ref: ObjectRef | None,
        model_cancellation_result_ref: ObjectRef | None,
        blocked_until: datetime | None,
        audit: ContractAudit,
    ) -> WorkModelEventV2:
        value = cls(
            work_model_event_id="work-model-event://pending",
            work_model_reservation_ref=work_model_reservation_ref,
            work_lease_event_ref=work_lease_event_ref,
            event_kind=event_kind,
            reservation_version=reservation_version,
            bucket_version=bucket_version,
            effective_at=effective_at,
            usage=usage,
            refunded_request_allowance=refunded_request_allowance,
            refunded_token_allowance=refunded_token_allowance,
            released_concurrency=released_concurrency,
            provider_backpressure_ref=provider_backpressure_ref,
            model_cancellation_request_ref=model_cancellation_request_ref,
            model_cancellation_result_ref=model_cancellation_result_ref,
            blocked_until=blocked_until,
            work_model_event_sha256="0" * 64,
            audit=audit,
        )
        return _finalize(
            value,
            "work_model_event_id",
            "work_model_event_sha256",
            "work-model-event",
        )


_MODEL_REF_FIELDS: dict[type[ContractModelV2], tuple[str, str, str]] = {
    ModelRatePoolPolicyV2: (
        "model_rate_pool_policy_id",
        "model_rate_pool_policy_sha256",
        "model-rate-pool-policy",
    ),
    JobModelPolicyV2: (
        "job_model_policy_id",
        "job_model_policy_sha256",
        "job-model-policy",
    ),
    WorkModelDemandV2: (
        "work_model_demand_id",
        "work_model_demand_sha256",
        "work-model-demand",
    ),
    ModelAdmissionDecisionV2: (
        "model_admission_decision_id",
        "model_admission_decision_sha256",
        "model-admission-decision",
    ),
    WorkModelReservationV2: (
        "work_model_reservation_id",
        "work_model_reservation_sha256",
        "work-model-reservation",
    ),
    WorkModelEventV2: (
        "work_model_event_id",
        "work_model_event_sha256",
        "work-model-event",
    ),
    ProviderBackpressureRecordV2: (
        "provider_backpressure_record_id",
        "provider_backpressure_record_sha256",
        "provider-backpressure-record",
    ),
    WorkModelCancellationRequestV2: (
        "work_model_cancellation_request_id",
        "work_model_cancellation_request_sha256",
        "work-model-cancellation-request",
    ),
    WorkModelCancellationResultV2: (
        "work_model_cancellation_result_id",
        "work_model_cancellation_result_sha256",
        "work-model-cancellation-result",
    ),
}


def _carried(value: ContractModelV2, id_field: str, sha_field: str) -> str:
    encoded = json.dumps(
        canonical_value_v2(value.model_dump(mode="python", exclude={id_field, sha_field, "audit"})),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    sha_field: str,
    prefix: str,
) -> ModelT:
    digest = _carried(value, id_field, sha_field)
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            sha_field: digest,
        }
    )


def _validate(value: ContractModelV2, id_field: str, sha_field: str, prefix: str) -> None:
    object_id = str(getattr(value, id_field))
    object_sha256 = str(getattr(value, sha_field))
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    observed = _carried(value, id_field, sha_field)
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix.replace('-', ' ')} identity is stale")


def _ref_for(value: ContractModelV2) -> ObjectRef:
    id_field, sha_field, object_type = _MODEL_REF_FIELDS[type(value)]
    _validate(value, id_field, sha_field, object_type)
    return ObjectRef(
        object_type=object_type,
        object_id=str(getattr(value, id_field)),
        object_version="v2",
        object_sha256=str(getattr(value, sha_field)),
    )


def model_rate_pool_policy_v2_ref(value: ModelRatePoolPolicyV2) -> ObjectRef:
    return _ref_for(value)


def job_model_policy_v2_ref(value: JobModelPolicyV2) -> ObjectRef:
    return _ref_for(value)


def work_model_demand_v2_ref(value: WorkModelDemandV2) -> ObjectRef:
    return _ref_for(value)


def model_admission_decision_v2_ref(value: ModelAdmissionDecisionV2) -> ObjectRef:
    return _ref_for(value)


def work_model_reservation_v2_ref(value: WorkModelReservationV2) -> ObjectRef:
    return _ref_for(value)


def work_model_event_v2_ref(value: WorkModelEventV2) -> ObjectRef:
    return _ref_for(value)


def provider_backpressure_record_v2_ref(value: ProviderBackpressureRecordV2) -> ObjectRef:
    return _ref_for(value)


def work_model_cancellation_request_v2_ref(
    value: WorkModelCancellationRequestV2,
) -> ObjectRef:
    return _ref_for(value)


def work_model_cancellation_result_v2_ref(
    value: WorkModelCancellationResultV2,
) -> ObjectRef:
    return _ref_for(value)


def validate_model_rate_pool_policy_v2_identity(value: ModelRatePoolPolicyV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[ModelRatePoolPolicyV2])


def validate_job_model_policy_v2_identity(value: JobModelPolicyV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[JobModelPolicyV2])


def validate_work_model_demand_v2_identity(value: WorkModelDemandV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[WorkModelDemandV2])


def validate_model_admission_decision_v2_identity(value: ModelAdmissionDecisionV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[ModelAdmissionDecisionV2])


def validate_work_model_reservation_v2_identity(value: WorkModelReservationV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[WorkModelReservationV2])


def validate_work_model_event_v2_identity(value: WorkModelEventV2) -> None:
    _validate(value, *_MODEL_REF_FIELDS[WorkModelEventV2])


def validate_provider_backpressure_record_v2_identity(
    value: ProviderBackpressureRecordV2,
) -> None:
    _validate(value, *_MODEL_REF_FIELDS[ProviderBackpressureRecordV2])


def validate_work_model_cancellation_request_v2_identity(
    value: WorkModelCancellationRequestV2,
) -> None:
    _validate(value, *_MODEL_REF_FIELDS[WorkModelCancellationRequestV2])


def validate_work_model_cancellation_result_v2_identity(
    value: WorkModelCancellationResultV2,
) -> None:
    _validate(value, *_MODEL_REF_FIELDS[WorkModelCancellationResultV2])


def _require_ref(ref: ObjectRef, object_type: str, version: str, name: str) -> None:
    if ref.object_type != object_type or ref.object_version != version:
        raise ValueError(f"{name} must reference {object_type} {version}")


def _require_model_profile_ref(ref: ObjectRef, name: str) -> None:
    if ref.object_type != "model-profile" or ref.object_version not in {"v1", "v2"}:
        raise ValueError(f"{name} must reference a versioned model-profile")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _sorted_unique_refs(refs: tuple[ObjectRef, ...], name: str) -> tuple[ObjectRef, ...]:
    if len(set(refs)) != len(refs):
        raise ValueError(f"{name} must be unique")
    return tuple(sorted(refs, key=_ref_key))


def _require_model_audit(audit: ContractAudit, refs: tuple[ObjectRef, ...]) -> None:
    expected = tuple(sorted(refs, key=_ref_key))
    if audit.input_refs != expected:
        raise ValueError("model control audit refs are incomplete")
    bindings = tuple(item for item in audit.governing_versions if item.component == "model-control")
    if len(bindings) != 1 or bindings[0].version != R6_MODEL_CONTROL_POLICY_VERSION:
        raise ValueError("model control audit is missing the current policy")
