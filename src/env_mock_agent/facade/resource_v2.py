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
from env_mock_agent.facade.execution_v2 import (
    AttachmentExecutionRequestV2,
    AttachmentExecutionResultV2,
)
from env_mock_agent.facade.retrieval_v2 import (
    PublicSourceFetchRequestV2,
    PublicSourceFetchResultV2,
    PublicSourceSearchRequestV2,
    PublicSourceSearchResultV2,
)

ATTACHMENT_RESOURCE_POLICY_VERSION: Literal["attachment-resource/r6-04-v1"] = "attachment-resource/r6-04-v1"


class AttachmentResourceTerminationOutcomeV2(StrEnum):
    TERMINATED = "TERMINATED"
    ALREADY_STOPPED = "ALREADY_STOPPED"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    FAILED = "FAILED"


class AttachmentResourceUsageV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-resource-usage/v2"] = (
        "env-mock-agent/attachment-resource-usage/v2"
    )
    process_starts: int = Field(ge=0)
    renderer_operations: int = Field(ge=0)
    network_requests: int = Field(ge=0)
    retained_storage_bytes: int = Field(ge=0)


class AttachmentResourceGrantV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-resource-grant/v2"] = (
        "env-mock-agent/attachment-resource-grant/v2"
    )
    resource_grant_id: Identifier
    reservation_ref: FacadeObjectRef
    execution_handle_ref: FacadeObjectRef
    operation_ref: FacadeObjectRef
    fencing_token: int = Field(ge=1)
    process_limit: int = Field(ge=0)
    renderer_limit: int = Field(ge=0)
    network_request_limit: int = Field(ge=0)
    storage_byte_limit: int = Field(ge=0)
    policy_version: Literal["attachment-resource/r6-04-v1"] = ATTACHMENT_RESOURCE_POLICY_VERSION
    resource_grant_sha256: Sha256

    @model_validator(mode="after")
    def validate_grant(self) -> Self:
        _require_ref(self.reservation_ref, "work-resource-reservation", "v2", "reservation_ref")
        _require_ref(
            self.execution_handle_ref,
            "resource-execution-handle",
            "v1",
            "execution_handle_ref",
        )
        if (
            self.operation_ref.object_type
            not in {
                "attachment-execution-request",
                "public-source-fetch-request",
                "public-source-search-request",
            }
            or self.operation_ref.object_version != "v2"
        ):
            raise ValueError("operation_ref must reference an approved resource-bound request v2")
        validate_attachment_resource_grant_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        reservation_ref: FacadeObjectRef,
        execution_handle_ref: FacadeObjectRef,
        operation_ref: FacadeObjectRef,
        fencing_token: int,
        process_limit: int,
        renderer_limit: int,
        network_request_limit: int,
        storage_byte_limit: int,
    ) -> AttachmentResourceGrantV2:
        value = cls(
            resource_grant_id="attachment-resource-grant://pending",
            reservation_ref=reservation_ref,
            execution_handle_ref=execution_handle_ref,
            operation_ref=operation_ref,
            fencing_token=fencing_token,
            process_limit=process_limit,
            renderer_limit=renderer_limit,
            network_request_limit=network_request_limit,
            storage_byte_limit=storage_byte_limit,
            resource_grant_sha256="0" * 64,
        )
        return _finalize(
            value,
            "resource_grant_id",
            "resource_grant_sha256",
            "attachment-resource-grant",
        )

    def validate_usage(self, usage: AttachmentResourceUsageV2) -> None:
        if usage.process_starts > self.process_limit:
            raise ValueError("process usage exceeds resource grant")
        if usage.renderer_operations > self.renderer_limit:
            raise ValueError("renderer usage exceeds resource grant")
        if usage.network_requests > self.network_request_limit:
            raise ValueError("network usage exceeds resource grant")
        if usage.retained_storage_bytes > self.storage_byte_limit:
            raise ValueError("storage usage exceeds resource grant")


class ResourceBoundAttachmentExecutionResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/resource-bound-attachment-execution-result/v2"] = (
        "env-mock-agent/resource-bound-attachment-execution-result/v2"
    )
    resource_grant_ref: FacadeObjectRef
    execution_result: AttachmentExecutionResultV2
    usage: AttachmentResourceUsageV2

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.resource_grant_ref,
            "attachment-resource-grant",
            "v2",
            "resource_grant_ref",
        )
        return self


class ResourceBoundPublicSourceSearchResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/resource-bound-public-source-search-result/v2"] = (
        "env-mock-agent/resource-bound-public-source-search-result/v2"
    )
    resource_grant_ref: FacadeObjectRef
    search_result: PublicSourceSearchResultV2
    usage: AttachmentResourceUsageV2


class ResourceBoundPublicSourceFetchResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/resource-bound-public-source-fetch-result/v2"] = (
        "env-mock-agent/resource-bound-public-source-fetch-result/v2"
    )
    resource_grant_ref: FacadeObjectRef
    fetch_result: PublicSourceFetchResultV2
    usage: AttachmentResourceUsageV2


class AttachmentResourceTerminationRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-resource-termination-request/v2"] = (
        "env-mock-agent/attachment-resource-termination-request/v2"
    )
    termination_request_id: Identifier
    reservation_ref: FacadeObjectRef
    execution_handle_ref: FacadeObjectRef
    fencing_token: int = Field(ge=1)
    reason: Literal["CANCELLED", "EXPIRED", "RESOURCE_OVERRUN"]
    requested_at: datetime
    policy_version: Literal["attachment-resource/r6-04-v1"] = ATTACHMENT_RESOURCE_POLICY_VERSION
    termination_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(self.reservation_ref, "work-resource-reservation", "v2", "reservation_ref")
        _require_ref(
            self.execution_handle_ref,
            "resource-execution-handle",
            "v1",
            "execution_handle_ref",
        )
        validate_attachment_resource_termination_request_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        reservation_ref: FacadeObjectRef,
        execution_handle_ref: FacadeObjectRef,
        fencing_token: int,
        reason: Literal["CANCELLED", "EXPIRED", "RESOURCE_OVERRUN"],
        requested_at: datetime,
    ) -> AttachmentResourceTerminationRequestV2:
        value = cls(
            termination_request_id="attachment-resource-termination-request://pending",
            reservation_ref=reservation_ref,
            execution_handle_ref=execution_handle_ref,
            fencing_token=fencing_token,
            reason=reason,
            requested_at=requested_at,
            termination_request_sha256="0" * 64,
        )
        return _finalize(
            value,
            "termination_request_id",
            "termination_request_sha256",
            "attachment-resource-termination-request",
        )


class AttachmentResourceTerminationResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-resource-termination-result/v2"] = (
        "env-mock-agent/attachment-resource-termination-result/v2"
    )
    termination_result_id: Identifier
    termination_request_ref: FacadeObjectRef
    outcome: AttachmentResourceTerminationOutcomeV2
    completed_at: datetime
    policy_version: Literal["attachment-resource/r6-04-v1"] = ATTACHMENT_RESOURCE_POLICY_VERSION
    termination_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.termination_request_ref,
            "attachment-resource-termination-request",
            "v2",
            "termination_request_ref",
        )
        validate_attachment_resource_termination_result_identity(self)
        return self

    @classmethod
    def create(
        cls,
        *,
        termination_request_ref: FacadeObjectRef,
        outcome: AttachmentResourceTerminationOutcomeV2,
        completed_at: datetime,
    ) -> AttachmentResourceTerminationResultV2:
        value = cls(
            termination_result_id="attachment-resource-termination-result://pending",
            termination_request_ref=termination_request_ref,
            outcome=outcome,
            completed_at=completed_at,
            termination_result_sha256="0" * 64,
        )
        return _finalize(
            value,
            "termination_result_id",
            "termination_result_sha256",
            "attachment-resource-termination-result",
        )


class ResourceBoundAttachmentExecutionFacade(Protocol):
    async def execute_with_resources(
        self,
        request: AttachmentExecutionRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundAttachmentExecutionResultV2: ...

    async def terminate_resources(
        self,
        request: AttachmentResourceTerminationRequestV2,
    ) -> AttachmentResourceTerminationResultV2: ...


class ResourceBoundRetrievalFacade(Protocol):
    async def search_with_resources(
        self,
        request: PublicSourceSearchRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundPublicSourceSearchResultV2: ...

    async def fetch_with_resources(
        self,
        request: PublicSourceFetchRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundPublicSourceFetchResultV2: ...


def attachment_resource_grant_ref(value: AttachmentResourceGrantV2) -> FacadeObjectRef:
    validate_attachment_resource_grant_identity(value)
    return FacadeObjectRef(
        object_type="attachment-resource-grant",
        object_id=value.resource_grant_id,
        object_version="v2",
        object_sha256=value.resource_grant_sha256,
    )


def attachment_resource_termination_request_ref(
    value: AttachmentResourceTerminationRequestV2,
) -> FacadeObjectRef:
    validate_attachment_resource_termination_request_identity(value)
    return FacadeObjectRef(
        object_type="attachment-resource-termination-request",
        object_id=value.termination_request_id,
        object_version="v2",
        object_sha256=value.termination_request_sha256,
    )


def attachment_resource_termination_result_ref(
    value: AttachmentResourceTerminationResultV2,
) -> FacadeObjectRef:
    validate_attachment_resource_termination_result_identity(value)
    return FacadeObjectRef(
        object_type="attachment-resource-termination-result",
        object_id=value.termination_result_id,
        object_version="v2",
        object_sha256=value.termination_result_sha256,
    )


def validate_attachment_resource_grant_identity(value: AttachmentResourceGrantV2) -> None:
    _validate(value, "resource_grant_id", "resource_grant_sha256", "attachment-resource-grant")


def validate_attachment_resource_termination_request_identity(
    value: AttachmentResourceTerminationRequestV2,
) -> None:
    _validate(
        value,
        "termination_request_id",
        "termination_request_sha256",
        "attachment-resource-termination-request",
    )


def validate_attachment_resource_termination_result_identity(
    value: AttachmentResourceTerminationResultV2,
) -> None:
    _validate(
        value,
        "termination_result_id",
        "termination_result_sha256",
        "attachment-resource-termination-result",
    )


def _carried(value: FacadeModel, id_field: str, sha_field: str) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={id_field, sha_field},
        exclude_none=False,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finalize[FacadeResourceModelT: FacadeModel](
    value: FacadeResourceModelT,
    id_field: str,
    sha_field: str,
    prefix: str,
) -> FacadeResourceModelT:
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
    digest = _carried(value, id_field, sha_field)
    if object_id != f"{prefix}://sha256/{digest}" or object_sha256 != digest:
        raise ValueError(f"{prefix.replace('-', ' ')} identity is stale")


def _require_ref(
    ref: FacadeObjectRef,
    object_type: str,
    version: str,
    field_name: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != version:
        raise ValueError(f"{field_name} must reference {object_type} {version}")
