from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.task_v2 import (
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
)

PRODUCER_TASK_VIEW_POLICY_VERSION: Literal["producer-task-view/r4-08-v1"] = "producer-task-view/r4-08-v1"


class ProducerTaskViewPolicyError(RuntimeError):
    pass


class ProducerTaskViewProjectionOutcome(StrEnum):
    PROJECTED = "PROJECTED"
    BLOCKED_SAFETY = "BLOCKED_SAFETY"


class ProducerTaskViewProjectionReason(StrEnum):
    PROMPT_SAFETY_PENDING = "PROMPT_SAFETY_PENDING"
    PROMPT_SAFETY_BLOCKED = "PROMPT_SAFETY_BLOCKED"


class ProducerStorageAccessOutcome(StrEnum):
    GRANTED = "GRANTED"
    DENIED_IDENTITY = "DENIED_IDENTITY"
    DENIED_SCOPE = "DENIED_SCOPE"


class ProducerStorageAccessReason(StrEnum):
    PRINCIPAL_MISMATCH = "PRINCIPAL_MISMATCH"
    PURPOSE_MISMATCH = "PURPOSE_MISMATCH"
    SUBJECT_NOT_AUTHORIZED = "SUBJECT_NOT_AUTHORIZED"


class ProducerTaskViewProjectionResult(ContractModel):
    schema_version: Literal["eval-factory/producer-task-view-projection-result/r4-08"] = (
        "eval-factory/producer-task-view-projection-result/r4-08"
    )
    result_id: Identifier
    source_task_draft_sha256: Sha256
    outcome: ProducerTaskViewProjectionOutcome
    producer_task_view: ProducerTaskViewV2 | None = None
    storage_authorization: ProducerStorageAuthorizationV2 | None = None
    unresolved_reasons: frozenset[ProducerTaskViewProjectionReason] = frozenset()
    policy_version: Literal["producer-task-view/r4-08-v1"] = PRODUCER_TASK_VIEW_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ProducerTaskViewProjectionOutcome:
        if isinstance(value, ProducerTaskViewProjectionOutcome):
            return value
        if isinstance(value, str):
            return ProducerTaskViewProjectionOutcome(value)
        raise TypeError("outcome must be a ProducerTaskViewProjectionOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ProducerTaskViewProjectionReason]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item
                if isinstance(item, ProducerTaskViewProjectionReason)
                else ProducerTaskViewProjectionReason(item)
                for item in value
            )
        raise TypeError("unresolved_reasons must be a collection")

    @model_validator(mode="after")
    def validate_result(self) -> ProducerTaskViewProjectionResult:
        if self.outcome is ProducerTaskViewProjectionOutcome.PROJECTED:
            if self.producer_task_view is None or self.storage_authorization is None:
                raise ValueError("PROJECTED result requires producer view and storage authorization")
            if self.unresolved_reasons:
                raise ValueError("PROJECTED result cannot carry unresolved reasons")
        else:
            if self.producer_task_view is not None or self.storage_authorization is not None:
                raise ValueError("BLOCKED_SAFETY result cannot carry producer contracts")
            if not self.unresolved_reasons:
                raise ValueError("BLOCKED_SAFETY result requires unresolved reasons")
        return self


class ProducerStorageAccessRequest(ContractModel):
    schema_version: Literal["eval-factory/producer-storage-access-request/r4-08"] = (
        "eval-factory/producer-storage-access-request/r4-08"
    )
    request_id: Identifier
    authorization_ref: ObjectRef
    producer_principal_id: Identifier
    purpose: Identifier
    requested_subject_ref: ObjectRef
    policy_version: Literal["producer-task-view/r4-08-v1"] = PRODUCER_TASK_VIEW_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> ProducerStorageAccessRequest:
        if (
            self.authorization_ref.object_type != "producer-storage-authorization"
            or self.authorization_ref.object_version != "v2"
        ):
            raise ValueError("authorization_ref must reference ProducerStorageAuthorization v2")
        return self


class ProducerStorageAccessResult(ContractModel):
    schema_version: Literal["eval-factory/producer-storage-access-result/r4-08"] = (
        "eval-factory/producer-storage-access-result/r4-08"
    )
    result_id: Identifier
    request_ref: ObjectRef
    authorization_ref: ObjectRef
    outcome: ProducerStorageAccessOutcome
    authorized_subject_ref: ObjectRef | None = None
    reasons: frozenset[ProducerStorageAccessReason] = frozenset()
    policy_version: Literal["producer-task-view/r4-08-v1"] = PRODUCER_TASK_VIEW_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ProducerStorageAccessOutcome:
        if isinstance(value, ProducerStorageAccessOutcome):
            return value
        if isinstance(value, str):
            return ProducerStorageAccessOutcome(value)
        raise TypeError("outcome must be a ProducerStorageAccessOutcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ProducerStorageAccessReason]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, ProducerStorageAccessReason) else ProducerStorageAccessReason(item)
                for item in value
            )
        raise TypeError("reasons must be a collection")

    @model_validator(mode="after")
    def validate_result(self) -> ProducerStorageAccessResult:
        _require_ref(
            self.request_ref,
            "producer-storage-access-request",
            "request_ref",
        )
        _require_ref(
            self.authorization_ref,
            "producer-storage-authorization",
            "authorization_ref",
        )
        if self.outcome is ProducerStorageAccessOutcome.GRANTED:
            if self.authorized_subject_ref is None:
                raise ValueError("GRANTED result requires authorized subject")
            if self.reasons:
                raise ValueError("GRANTED result cannot carry denial reasons")
        else:
            if self.authorized_subject_ref is not None:
                raise ValueError("denied result cannot carry authorized subject")
            if not self.reasons:
                raise ValueError("denied result requires reasons")
        return self


def producer_storage_access_request_carried_sha256(
    request: ProducerStorageAccessRequest,
) -> str:
    return producer_payload_sha256(
        {
            "authorization_ref": _ref_payload(request.authorization_ref),
            "producer_principal_id": request.producer_principal_id,
            "purpose": request.purpose,
            "requested_subject_ref": _ref_payload(request.requested_subject_ref),
            "policy_version": request.policy_version,
        }
    )


def producer_storage_access_request_ref(
    request: ProducerStorageAccessRequest,
) -> ObjectRef:
    return ObjectRef(
        object_type="producer-storage-access-request",
        object_id=request.request_id,
        object_version="r4-08",
        object_sha256=request.request_sha256,
    )


def producer_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)
