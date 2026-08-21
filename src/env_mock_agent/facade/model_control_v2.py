from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    Sha256,
)
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentRepairRequestV2,
    AttachmentRepairResultV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewResultV2,
)
from env_mock_agent.facade.telemetry_v2 import ExecutionTelemetryV2

ATTACHMENT_MODEL_CONTROL_POLICY_VERSION: Literal["attachment-model-control/r6-03-v1"] = (
    "attachment-model-control/r6-03-v1"
)


class ModelDemandModeFacadeV2(StrEnum):
    DIRECT_REQUEST = "DIRECT_REQUEST"
    OPAQUE_RUNTIME_ENVELOPE = "OPAQUE_RUNTIME_ENVELOPE"


class ModelControlUsageSourceV2(StrEnum):
    REPORTED = "REPORTED"
    CONSERVATIVE_ALLOWANCE = "CONSERVATIVE_ALLOWANCE"


class ModelProviderBackpressureKindV2(StrEnum):
    RATE_LIMITED = "RATE_LIMITED"
    OVERLOADED = "OVERLOADED"


class ModelControlReceiptOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BACKPRESSURED = "BACKPRESSURED"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    FAILED = "FAILED"


class ModelInvocationCancellationOutcomeV2(StrEnum):
    CANCELLED = "CANCELLED"
    ALREADY_FINISHED = "ALREADY_FINISHED"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    FAILED = "FAILED"


class ModelInvocationDescriptorV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-invocation-descriptor/v2"] = (
        "env-mock-agent/model-invocation-descriptor/v2"
    )
    invocation_descriptor_id: Identifier
    operation_ref: FacadeObjectRef
    model_profile_ref: FacadeObjectRef
    mode: ModelDemandModeFacadeV2
    request_allowance: int = Field(ge=1)
    input_token_allowance: int = Field(ge=0)
    output_token_allowance: int = Field(ge=1)
    cache_token_allowance: int = Field(ge=0)
    token_allowance: int = Field(ge=1)
    idempotency_key: Identifier
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    invocation_descriptor_sha256: Sha256

    @model_validator(mode="after")
    def validate_descriptor(self) -> Self:
        _require_model_profile_ref(self.model_profile_ref, "model_profile_ref")
        if self.operation_ref.object_version not in {"v1", "v2"}:
            raise ValueError("operation_ref must reference a versioned operation")
        expected = self.input_token_allowance + self.output_token_allowance + self.cache_token_allowance
        if self.token_allowance != expected:
            raise ValueError("token allowance must equal input, output, and cache allowances")
        if self.mode is ModelDemandModeFacadeV2.DIRECT_REQUEST and self.request_allowance != 1:
            raise ValueError("DIRECT_REQUEST descriptor requires exactly one request")
        validate_model_invocation_descriptor_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        operation_ref: FacadeObjectRef,
        model_profile_ref: FacadeObjectRef,
        mode: ModelDemandModeFacadeV2,
        request_allowance: int,
        input_token_allowance: int,
        output_token_allowance: int,
        cache_token_allowance: int,
        idempotency_key: Identifier,
    ) -> ModelInvocationDescriptorV2:
        token_allowance = input_token_allowance + output_token_allowance + cache_token_allowance
        value = cls(
            invocation_descriptor_id="model-invocation-descriptor://pending",
            operation_ref=operation_ref,
            model_profile_ref=model_profile_ref,
            mode=mode,
            request_allowance=request_allowance,
            input_token_allowance=input_token_allowance,
            output_token_allowance=output_token_allowance,
            cache_token_allowance=cache_token_allowance,
            token_allowance=token_allowance,
            idempotency_key=idempotency_key,
            invocation_descriptor_sha256="0" * 64,
        )
        return _finalize(
            value,
            "invocation_descriptor_id",
            "invocation_descriptor_sha256",
            "model-invocation-descriptor",
        )


class ModelControlUsageV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-control-usage/v2"] = "env-mock-agent/model-control-usage/v2"
    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    charged_tokens: int = Field(ge=0)
    source: ModelControlUsageSourceV2

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
    def conservative(cls, *, requests: int, tokens: int) -> ModelControlUsageV2:
        return cls(
            requests=requests,
            input_tokens=tokens,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            charged_tokens=tokens,
            source=ModelControlUsageSourceV2.CONSERVATIVE_ALLOWANCE,
        )


class ModelControlGrantV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-control-grant/v2"] = "env-mock-agent/model-control-grant/v2"
    model_control_grant_id: Identifier
    reservation_ref: FacadeObjectRef
    invocation_handle_ref: FacadeObjectRef
    descriptor_ref: FacadeObjectRef
    model_profile_ref: FacadeObjectRef
    fencing_token: int = Field(ge=1)
    request_allowance: int = Field(ge=1)
    token_allowance: int = Field(ge=1)
    output_token_limit: int = Field(ge=1)
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    model_control_grant_sha256: Sha256

    @model_validator(mode="after")
    def validate_grant(self) -> Self:
        _require_ref(
            self.reservation_ref,
            "work-model-reservation",
            "v2",
            "reservation_ref",
        )
        _require_ref(
            self.invocation_handle_ref,
            "model-invocation-handle",
            "v1",
            "invocation_handle_ref",
        )
        _require_ref(
            self.descriptor_ref,
            "model-invocation-descriptor",
            "v2",
            "descriptor_ref",
        )
        _require_model_profile_ref(self.model_profile_ref, "model_profile_ref")
        if self.output_token_limit > self.token_allowance:
            raise ValueError("output token limit cannot exceed token allowance")
        validate_model_control_grant_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        reservation_ref: FacadeObjectRef,
        invocation_handle_ref: FacadeObjectRef,
        descriptor_ref: FacadeObjectRef,
        model_profile_ref: FacadeObjectRef,
        fencing_token: int,
        request_allowance: int,
        token_allowance: int,
        output_token_limit: int,
    ) -> ModelControlGrantV2:
        value = cls(
            model_control_grant_id="model-control-grant://pending",
            reservation_ref=reservation_ref,
            invocation_handle_ref=invocation_handle_ref,
            descriptor_ref=descriptor_ref,
            model_profile_ref=model_profile_ref,
            fencing_token=fencing_token,
            request_allowance=request_allowance,
            token_allowance=token_allowance,
            output_token_limit=output_token_limit,
            model_control_grant_sha256="0" * 64,
        )
        return _finalize(
            value,
            "model_control_grant_id",
            "model_control_grant_sha256",
            "model-control-grant",
        )

    def validate_descriptor(self, descriptor: ModelInvocationDescriptorV2) -> None:
        if self.descriptor_ref != model_invocation_descriptor_ref(descriptor):
            raise ValueError("model grant does not bind the invocation descriptor")
        if self.model_profile_ref != descriptor.model_profile_ref:
            raise ValueError("model grant profile differs from invocation descriptor")
        if (
            self.request_allowance != descriptor.request_allowance
            or self.token_allowance != descriptor.token_allowance
            or self.output_token_limit != descriptor.output_token_allowance
        ):
            raise ValueError("model grant allowance differs from invocation descriptor")

    def validate_usage(self, usage: ModelControlUsageV2) -> None:
        if usage.requests > self.request_allowance:
            raise ValueError("request usage exceeds model grant")
        if usage.charged_tokens > self.token_allowance:
            raise ValueError("token usage exceeds model grant")
        if usage.output_tokens > self.output_token_limit:
            raise ValueError("output token usage exceeds model grant")


class ModelProviderBackpressureV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-provider-backpressure/v2"] = (
        "env-mock-agent/model-provider-backpressure/v2"
    )
    model_provider_backpressure_id: Identifier
    descriptor_ref: FacadeObjectRef
    kind: ModelProviderBackpressureKindV2
    retry_at: datetime
    observed_at: datetime
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    model_provider_backpressure_sha256: Sha256

    @model_validator(mode="after")
    def validate_backpressure(self) -> Self:
        _require_ref(
            self.descriptor_ref,
            "model-invocation-descriptor",
            "v2",
            "descriptor_ref",
        )
        if self.retry_at <= self.observed_at:
            raise ValueError("retry_at must be later than observed_at")
        validate_model_provider_backpressure_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        descriptor_ref: FacadeObjectRef,
        kind: ModelProviderBackpressureKindV2,
        retry_at: datetime,
        observed_at: datetime,
    ) -> ModelProviderBackpressureV2:
        value = cls(
            model_provider_backpressure_id="model-provider-backpressure://pending",
            descriptor_ref=descriptor_ref,
            kind=kind,
            retry_at=retry_at,
            observed_at=observed_at,
            model_provider_backpressure_sha256="0" * 64,
        )
        return _finalize(
            value,
            "model_provider_backpressure_id",
            "model_provider_backpressure_sha256",
            "model-provider-backpressure",
        )


class ModelControlReceiptV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-control-receipt/v2"] = (
        "env-mock-agent/model-control-receipt/v2"
    )
    model_control_receipt_id: Identifier
    descriptor_ref: FacadeObjectRef
    grant_ref: FacadeObjectRef
    outcome: ModelControlReceiptOutcomeV2
    result_ref: FacadeObjectRef | None = None
    usage: ModelControlUsageV2
    backpressure_ref: FacadeObjectRef | None = None
    failure_code: Identifier | None = None
    completed_at: datetime
    telemetry: ExecutionTelemetryV2
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    model_control_receipt_sha256: Sha256

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        _require_ref(
            self.descriptor_ref,
            "model-invocation-descriptor",
            "v2",
            "descriptor_ref",
        )
        _require_ref(self.grant_ref, "model-control-grant", "v2", "grant_ref")
        if self.outcome is ModelControlReceiptOutcomeV2.SUCCEEDED:
            if self.result_ref is None or self.backpressure_ref is not None or self.failure_code is not None:
                raise ValueError("successful model receipt requires result only")
        elif self.outcome is ModelControlReceiptOutcomeV2.BACKPRESSURED:
            if self.result_ref is not None or self.backpressure_ref is None or self.failure_code is not None:
                raise ValueError("backpressured receipt requires backpressure only")
        elif self.result_ref is not None or self.backpressure_ref is not None or self.failure_code is None:
            raise ValueError("failed or blocked model receipt requires failure_code only")
        if self.backpressure_ref is not None:
            _require_ref(
                self.backpressure_ref,
                "model-provider-backpressure",
                "v2",
                "backpressure_ref",
            )
        validate_model_control_receipt_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        descriptor_ref: FacadeObjectRef,
        grant_ref: FacadeObjectRef,
        outcome: ModelControlReceiptOutcomeV2,
        result_ref: FacadeObjectRef | None,
        usage: ModelControlUsageV2,
        backpressure_ref: FacadeObjectRef | None,
        failure_code: Identifier | None,
        completed_at: datetime,
        telemetry: ExecutionTelemetryV2 | None = None,
    ) -> ModelControlReceiptV2:
        value = cls(
            model_control_receipt_id="model-control-receipt://pending",
            descriptor_ref=descriptor_ref,
            grant_ref=grant_ref,
            outcome=outcome,
            result_ref=result_ref,
            usage=usage,
            backpressure_ref=backpressure_ref,
            failure_code=failure_code,
            completed_at=completed_at,
            telemetry=telemetry or ExecutionTelemetryV2.unavailable(observed_at=completed_at),
            model_control_receipt_sha256="0" * 64,
        )
        return _finalize(
            value,
            "model_control_receipt_id",
            "model_control_receipt_sha256",
            "model-control-receipt",
        )


class ModelInvocationCancellationRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-invocation-cancellation-request/v2"] = (
        "env-mock-agent/model-invocation-cancellation-request/v2"
    )
    cancellation_request_id: Identifier
    reservation_ref: FacadeObjectRef
    invocation_handle_ref: FacadeObjectRef
    fencing_token: int = Field(ge=1)
    reason: Literal["CANCELLED", "EXPIRED", "USAGE_OVERRUN"]
    requested_at: datetime
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    cancellation_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.reservation_ref,
            "work-model-reservation",
            "v2",
            "reservation_ref",
        )
        _require_ref(
            self.invocation_handle_ref,
            "model-invocation-handle",
            "v1",
            "invocation_handle_ref",
        )
        validate_model_invocation_cancellation_request_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        reservation_ref: FacadeObjectRef,
        invocation_handle_ref: FacadeObjectRef,
        fencing_token: int,
        reason: Literal["CANCELLED", "EXPIRED", "USAGE_OVERRUN"],
        requested_at: datetime,
    ) -> ModelInvocationCancellationRequestV2:
        value = cls(
            cancellation_request_id="model-invocation-cancellation-request://pending",
            reservation_ref=reservation_ref,
            invocation_handle_ref=invocation_handle_ref,
            fencing_token=fencing_token,
            reason=reason,
            requested_at=requested_at,
            cancellation_request_sha256="0" * 64,
        )
        return _finalize(
            value,
            "cancellation_request_id",
            "cancellation_request_sha256",
            "model-invocation-cancellation-request",
        )


class ModelInvocationCancellationResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/model-invocation-cancellation-result/v2"] = (
        "env-mock-agent/model-invocation-cancellation-result/v2"
    )
    cancellation_result_id: Identifier
    cancellation_request_ref: FacadeObjectRef
    outcome: ModelInvocationCancellationOutcomeV2
    completed_at: datetime
    policy_version: Literal["attachment-model-control/r6-03-v1"] = ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    cancellation_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.cancellation_request_ref,
            "model-invocation-cancellation-request",
            "v2",
            "cancellation_request_ref",
        )
        validate_model_invocation_cancellation_result_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        cancellation_request_ref: FacadeObjectRef,
        outcome: ModelInvocationCancellationOutcomeV2,
        completed_at: datetime,
    ) -> ModelInvocationCancellationResultV2:
        value = cls(
            cancellation_result_id="model-invocation-cancellation-result://pending",
            cancellation_request_ref=cancellation_request_ref,
            outcome=outcome,
            completed_at=completed_at,
            cancellation_result_sha256="0" * 64,
        )
        return _finalize(
            value,
            "cancellation_result_id",
            "cancellation_result_sha256",
            "model-invocation-cancellation-result",
        )


class ModelControlledStructuredFacade(Protocol):
    async def generate_json_with_model_control(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        system: str,
        prompt: str,
        schema: dict[str, object],
        result_ref: FacadeObjectRef,
    ) -> tuple[dict[str, object] | None, ModelControlReceiptV2]: ...

    async def cancel_model_invocation(
        self,
        request: ModelInvocationCancellationRequestV2,
    ) -> ModelInvocationCancellationResultV2: ...


class ModelControlledAttachmentSemanticFacade(Protocol):
    async def review_with_model_control(
        self,
        *,
        request: AttachmentSemanticReviewRequestV2,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
    ) -> tuple[AttachmentSemanticReviewResultV2 | None, ModelControlReceiptV2]: ...

    async def repair_with_model_control(
        self,
        *,
        request: AttachmentRepairRequestV2,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
    ) -> tuple[AttachmentRepairResultV2 | None, ModelControlReceiptV2]: ...

    async def cancel_model_invocation(
        self,
        request: ModelInvocationCancellationRequestV2,
    ) -> ModelInvocationCancellationResultV2: ...


_REF_FIELDS: dict[type[FacadeModel], tuple[str, str, str]] = {
    ModelInvocationDescriptorV2: (
        "invocation_descriptor_id",
        "invocation_descriptor_sha256",
        "model-invocation-descriptor",
    ),
    ModelControlGrantV2: (
        "model_control_grant_id",
        "model_control_grant_sha256",
        "model-control-grant",
    ),
    ModelProviderBackpressureV2: (
        "model_provider_backpressure_id",
        "model_provider_backpressure_sha256",
        "model-provider-backpressure",
    ),
    ModelControlReceiptV2: (
        "model_control_receipt_id",
        "model_control_receipt_sha256",
        "model-control-receipt",
    ),
    ModelInvocationCancellationRequestV2: (
        "cancellation_request_id",
        "cancellation_request_sha256",
        "model-invocation-cancellation-request",
    ),
    ModelInvocationCancellationResultV2: (
        "cancellation_result_id",
        "cancellation_result_sha256",
        "model-invocation-cancellation-result",
    ),
}


def model_invocation_descriptor_ref(
    value: ModelInvocationDescriptorV2,
) -> FacadeObjectRef:
    return _ref_for(value)


def model_control_grant_ref(value: ModelControlGrantV2) -> FacadeObjectRef:
    return _ref_for(value)


def model_provider_backpressure_ref(
    value: ModelProviderBackpressureV2,
) -> FacadeObjectRef:
    return _ref_for(value)


def model_control_receipt_ref(value: ModelControlReceiptV2) -> FacadeObjectRef:
    return _ref_for(value)


def model_invocation_cancellation_request_ref(
    value: ModelInvocationCancellationRequestV2,
) -> FacadeObjectRef:
    return _ref_for(value)


def model_invocation_cancellation_result_ref(
    value: ModelInvocationCancellationResultV2,
) -> FacadeObjectRef:
    return _ref_for(value)


def validate_model_invocation_descriptor_identity(
    value: ModelInvocationDescriptorV2,
) -> None:
    _validate(value, *_REF_FIELDS[ModelInvocationDescriptorV2])


def validate_model_control_grant_identity(value: ModelControlGrantV2) -> None:
    _validate(value, *_REF_FIELDS[ModelControlGrantV2])


def validate_model_provider_backpressure_identity(
    value: ModelProviderBackpressureV2,
) -> None:
    _validate(value, *_REF_FIELDS[ModelProviderBackpressureV2])


def validate_model_control_receipt_identity(value: ModelControlReceiptV2) -> None:
    _validate(value, *_REF_FIELDS[ModelControlReceiptV2])


def validate_model_invocation_cancellation_request_identity(
    value: ModelInvocationCancellationRequestV2,
) -> None:
    _validate(value, *_REF_FIELDS[ModelInvocationCancellationRequestV2])


def validate_model_invocation_cancellation_result_identity(
    value: ModelInvocationCancellationResultV2,
) -> None:
    _validate(value, *_REF_FIELDS[ModelInvocationCancellationResultV2])


def _carried(value: FacadeModel, id_field: str, sha_field: str) -> str:
    encoded = json.dumps(
        value.model_dump(
            mode="json",
            exclude={id_field, sha_field},
            exclude_none=False,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize[ModelT: FacadeModel](
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


def _validate(value: FacadeModel, id_field: str, sha_field: str, prefix: str) -> None:
    object_id = str(getattr(value, id_field))
    object_sha256 = str(getattr(value, sha_field))
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    observed = _carried(value, id_field, sha_field)
    if object_id != f"{prefix}://sha256/{observed}" or object_sha256 != observed:
        raise ValueError(f"{prefix.replace('-', ' ')} identity is stale")


def _ref_for(value: FacadeModel) -> FacadeObjectRef:
    id_field, sha_field, object_type = _REF_FIELDS[type(value)]
    _validate(value, id_field, sha_field, object_type)
    return FacadeObjectRef(
        object_type=object_type,
        object_id=str(getattr(value, id_field)),
        object_version="v2",
        object_sha256=str(getattr(value, sha_field)),
    )


def _require_ref(ref: FacadeObjectRef, object_type: str, version: str, name: str) -> None:
    if ref.object_type != object_type or ref.object_version != version:
        raise ValueError(f"{name} must reference {object_type} {version}")


def _require_model_profile_ref(ref: FacadeObjectRef, name: str) -> None:
    if ref.object_type != "model-profile" or ref.object_version not in {"v1", "v2"}:
        raise ValueError(f"{name} must reference a versioned model-profile")
