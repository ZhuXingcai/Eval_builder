from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.task import EvaluationFailureClass
from eval_factory.contracts.task_v2 import (
    EvaluatorExecutionModeV2,
    EvaluatorFailureSignalV2,
    EvaluatorModelDomainV2,
    EvaluatorReferenceDataClassV2,
    EvaluatorReferenceGrantV2,
    EvaluatorSpecV2,
    ReferencePolicyV2,
)

EVALUATION_CONTRACT_POLICY_VERSION: Literal["evaluation-contract/r4-06-v1"] = "evaluation-contract/r4-06-v1"
REFERENCE_ACCESS_POLICY_VERSION: Literal["reference-access/r4-06-v1"] = "reference-access/r4-06-v1"


class EvaluationContractPolicyError(RuntimeError):
    pass


class EvaluationContractOutcome(StrEnum):
    COMPILED = "COMPILED"
    BLOCKED_BINDING = "BLOCKED_BINDING"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class EvaluationContractReason(StrEnum):
    MISSING_EVALUATOR_BINDING = "MISSING_EVALUATOR_BINDING"
    MISSING_EVALUATOR_CAPABILITY = "MISSING_EVALUATOR_CAPABILITY"
    REFERENCE_MODE_MISMATCH = "REFERENCE_MODE_MISMATCH"


class EvaluatorAccessPrincipalType(StrEnum):
    EVALUATOR = "EVALUATOR"
    CONTESTANT = "CONTESTANT"
    ATTACHMENT_PRODUCER = "ATTACHMENT_PRODUCER"


class EvaluatorReferenceAccessOutcome(StrEnum):
    GRANTED = "GRANTED"
    NO_REFERENCE_REQUIRED = "NO_REFERENCE_REQUIRED"
    HUMAN_ONLY = "HUMAN_ONLY"
    DENIED_IDENTITY = "DENIED_IDENTITY"
    DENIED_SCOPE = "DENIED_SCOPE"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class EvaluatorReferenceAccessReason(StrEnum):
    PRINCIPAL_TYPE_DENIED = "PRINCIPAL_TYPE_DENIED"
    EVALUATOR_BINDING_NOT_FOUND = "EVALUATOR_BINDING_NOT_FOUND"
    EVALUATOR_PRINCIPAL_MISMATCH = "EVALUATOR_PRINCIPAL_MISMATCH"
    MODEL_PROFILE_MISMATCH = "MODEL_PROFILE_MISMATCH"
    MODEL_AUTHORIZATION_MISSING = "MODEL_AUTHORIZATION_MISSING"
    MODEL_DOMAIN_DENIED = "MODEL_DOMAIN_DENIED"
    MODEL_PURPOSE_DENIED = "MODEL_PURPOSE_DENIED"
    MODEL_DATA_CLASS_DENIED = "MODEL_DATA_CLASS_DENIED"
    REFERENCE_SCOPE_DENIED = "REFERENCE_SCOPE_DENIED"
    MODEL_AUTHORIZATION_NOT_YET_VALID = "MODEL_AUTHORIZATION_NOT_YET_VALID"
    MODEL_AUTHORIZATION_EXPIRED = "MODEL_AUTHORIZATION_EXPIRED"
    EXTERNAL_ENDPOINT_DENIED = "EXTERNAL_ENDPOINT_DENIED"
    RETENTION_POLICY_DENIED = "RETENTION_POLICY_DENIED"
    TRAINING_USE_DENIED = "TRAINING_USE_DENIED"
    AUTHORITY_UNVERIFIED = "AUTHORITY_UNVERIFIED"
    SELF_SIGNED_APPROVAL_DENIED = "SELF_SIGNED_APPROVAL_DENIED"


class EvaluatorBindingDefinition(ContractModel):
    schema_version: Literal["eval-factory/evaluator-binding-definition/r4-06"] = (
        "eval-factory/evaluator-binding-definition/r4-06"
    )
    evaluator_binding_id: Identifier
    evaluator_type: Identifier
    execution_mode: EvaluatorExecutionModeV2
    evaluator_version: str = Field(min_length=1, max_length=128)
    input_contract_ref: ObjectRef
    output_contract_ref: ObjectRef
    evaluator_principal_id: Identifier | None = None
    model_profile_ref: ObjectRef | None = None
    reference_refs: tuple[ObjectRef, ...] = ()
    timeout_seconds: int = Field(gt=0)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def parse_execution_mode(cls, value: object) -> EvaluatorExecutionModeV2:
        return _parse_enum(value, EvaluatorExecutionModeV2, "execution_mode")

    @model_validator(mode="after")
    def validate_definition(self) -> EvaluatorBindingDefinition:
        _require_ref_type(
            self.input_contract_ref,
            "evaluator-input-contract",
            "input_contract_ref",
        )
        _require_ref_type(
            self.output_contract_ref,
            "evaluator-output-contract",
            "output_contract_ref",
        )
        if self.model_profile_ref is not None:
            _require_ref_type(
                self.model_profile_ref,
                "model-profile",
                "model_profile_ref",
            )
        _require_unique_refs(self.reference_refs)
        if self.execution_mode is EvaluatorExecutionModeV2.DETERMINISTIC:
            if self.evaluator_principal_id is None:
                raise ValueError("DETERMINISTIC definition requires evaluator principal")
            if self.model_profile_ref is not None:
                raise ValueError("DETERMINISTIC definition cannot carry model profile")
        elif self.execution_mode is EvaluatorExecutionModeV2.MODEL:
            if self.evaluator_principal_id is None or self.model_profile_ref is None:
                raise ValueError("MODEL definition requires evaluator principal and model profile")
        elif self.evaluator_principal_id is not None or self.model_profile_ref is not None:
            raise ValueError("HUMAN_ONLY definition cannot carry automated principal or model profile")
        return self


class EvaluationContractCompileResult(ContractModel):
    schema_version: Literal["eval-factory/evaluation-contract-compile-result/r4-06"] = (
        "eval-factory/evaluation-contract-compile-result/r4-06"
    )
    result_id: Identifier
    outcome: EvaluationContractOutcome
    evaluator_spec: EvaluatorSpecV2 | None = None
    reference_policy: ReferencePolicyV2 | None = None
    unresolved_reasons: frozenset[EvaluationContractReason] = frozenset()
    policy_version: Literal["evaluation-contract/r4-06-v1"] = EVALUATION_CONTRACT_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> EvaluationContractOutcome:
        return _parse_enum(value, EvaluationContractOutcome, "outcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[EvaluationContractReason]:
        return _parse_enum_set(
            value,
            EvaluationContractReason,
            "unresolved_reasons",
        )

    @model_validator(mode="after")
    def validate_result(self) -> EvaluationContractCompileResult:
        if self.outcome is EvaluationContractOutcome.COMPILED:
            if self.evaluator_spec is None or self.reference_policy is None:
                raise ValueError("COMPILED result requires evaluator spec and reference policy")
            if self.unresolved_reasons:
                raise ValueError("COMPILED result cannot carry unresolved reasons")
        else:
            if self.evaluator_spec is not None or self.reference_policy is not None:
                raise ValueError(f"{self.outcome.value} result cannot carry compiled contracts")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} result requires unresolved reasons")
        return self


class EvaluatorFailureObservation(ContractModel):
    schema_version: Literal["eval-factory/evaluator-failure-observation/r4-06"] = (
        "eval-factory/evaluator-failure-observation/r4-06"
    )
    observation_id: Identifier
    source_observation_ref: ObjectRef
    signals: tuple[EvaluatorFailureSignalV2, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    policy_version: Literal["evaluation-contract/r4-06-v1"] = EVALUATION_CONTRACT_POLICY_VERSION
    observation_sha256: Sha256
    audit: ContractAudit

    @field_validator("signals", mode="before")
    @classmethod
    def parse_signals(
        cls,
        value: object,
    ) -> tuple[EvaluatorFailureSignalV2, ...]:
        if isinstance(value, (tuple, list, set, frozenset)):
            return tuple(
                item if isinstance(item, EvaluatorFailureSignalV2) else EvaluatorFailureSignalV2(item)
                for item in value
            )
        raise TypeError("signals must be a collection")

    @model_validator(mode="after")
    def validate_observation(self) -> EvaluatorFailureObservation:
        _require_ref_type(
            self.source_observation_ref,
            "evaluation-observation",
            "source_observation_ref",
        )
        if len(self.signals) != len(set(self.signals)):
            raise ValueError("failure signals must be unique")
        evidence_ids = tuple(item.evidence_ref_id for item in self.evidence_refs)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("observation evidence IDs must be unique")
        return self


class EvaluatorFailureClassification(ContractModel):
    schema_version: Literal["eval-factory/evaluator-failure-classification/r4-06"] = (
        "eval-factory/evaluator-failure-classification/r4-06"
    )
    classification_id: Identifier
    evaluator_spec_ref: ObjectRef
    observation_ref: ObjectRef
    failure_class: EvaluationFailureClass
    matched_rule_ids: tuple[Identifier, ...] = ()
    policy_version: Literal["evaluation-contract/r4-06-v1"] = EVALUATION_CONTRACT_POLICY_VERSION
    classification_sha256: Sha256
    audit: ContractAudit

    @field_validator("failure_class", mode="before")
    @classmethod
    def parse_failure_class(cls, value: object) -> EvaluationFailureClass:
        return _parse_enum(value, EvaluationFailureClass, "failure_class")

    @model_validator(mode="after")
    def validate_classification(self) -> EvaluatorFailureClassification:
        _require_ref_type(
            self.evaluator_spec_ref,
            "evaluator-spec",
            "evaluator_spec_ref",
        )
        if self.evaluator_spec_ref.object_version != "v2":
            raise ValueError("evaluator_spec_ref must reference EvaluatorSpec v2")
        _require_ref_type(
            self.observation_ref,
            "evaluator-failure-observation",
            "observation_ref",
        )
        if len(self.matched_rule_ids) != len(set(self.matched_rule_ids)):
            raise ValueError("matched rule IDs must be unique")
        return self


class VerifiedModelDomainAuthorization(ContractModel):
    schema_version: Literal["eval-factory/verified-model-domain-authorization/r4-06"] = (
        "eval-factory/verified-model-domain-authorization/r4-06"
    )
    authorization_id: Identifier
    model_profile_ref: ObjectRef
    model_domain_approval_ref: ObjectRef
    verification_ref: ObjectRef
    domain: EvaluatorModelDomainV2
    allowed_purposes: frozenset[Identifier] = Field(min_length=1)
    allowed_data_classes: frozenset[EvaluatorReferenceDataClassV2] = Field(min_length=1)
    approved_reference_refs: tuple[ObjectRef, ...] = ()
    valid_from: datetime
    expires_at: datetime
    internal_endpoint: bool
    retention_policy_satisfied: bool
    training_use_approved: bool
    independent_authority_verified: bool
    non_self_signed: bool
    policy_version: Literal["reference-access/r4-06-v1"] = REFERENCE_ACCESS_POLICY_VERSION
    authorization_sha256: Sha256
    audit: ContractAudit

    @field_validator("domain", mode="before")
    @classmethod
    def parse_domain(cls, value: object) -> EvaluatorModelDomainV2:
        return _parse_enum(value, EvaluatorModelDomainV2, "domain")

    @field_validator("allowed_data_classes", mode="before")
    @classmethod
    def parse_data_classes(
        cls,
        value: object,
    ) -> frozenset[EvaluatorReferenceDataClassV2]:
        return _parse_enum_set(
            value,
            EvaluatorReferenceDataClassV2,
            "allowed_data_classes",
        )

    @model_validator(mode="after")
    def validate_authorization(self) -> VerifiedModelDomainAuthorization:
        _require_ref_type(
            self.model_profile_ref,
            "model-profile",
            "model_profile_ref",
        )
        _require_ref_type(
            self.model_domain_approval_ref,
            "model-domain-approval",
            "model_domain_approval_ref",
        )
        _require_ref_type(
            self.verification_ref,
            "model-domain-verification",
            "verification_ref",
        )
        if (
            self.valid_from.tzinfo is None
            or self.valid_from.utcoffset() is None
            or self.expires_at.tzinfo is None
            or self.expires_at.utcoffset() is None
        ):
            raise ValueError("authorization validity bounds must be timezone-aware")
        if self.valid_from >= self.expires_at:
            raise ValueError("authorization expiry must be after valid_from")
        _require_unique_refs(self.approved_reference_refs)
        return self


class EvaluatorReferenceAccessRequest(ContractModel):
    schema_version: Literal["eval-factory/evaluator-reference-access-request/r4-06"] = (
        "eval-factory/evaluator-reference-access-request/r4-06"
    )
    request_id: Identifier
    principal_type: EvaluatorAccessPrincipalType
    principal_id: Identifier
    purpose: Literal["EVALUATION"] = "EVALUATION"
    evaluator_binding_id: Identifier
    model_profile_ref: ObjectRef | None = None
    policy_version: Literal["reference-access/r4-06-v1"] = REFERENCE_ACCESS_POLICY_VERSION
    request_sha256: Sha256
    audit: ContractAudit

    @field_validator("principal_type", mode="before")
    @classmethod
    def parse_principal_type(cls, value: object) -> EvaluatorAccessPrincipalType:
        return _parse_enum(value, EvaluatorAccessPrincipalType, "principal_type")

    @model_validator(mode="after")
    def validate_request(self) -> EvaluatorReferenceAccessRequest:
        if self.model_profile_ref is not None:
            _require_ref_type(
                self.model_profile_ref,
                "model-profile",
                "model_profile_ref",
            )
        return self


class EvaluatorReferenceAccessResult(ContractModel):
    schema_version: Literal["eval-factory/evaluator-reference-access-result/r4-06"] = (
        "eval-factory/evaluator-reference-access-result/r4-06"
    )
    access_result_id: Identifier
    request_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    outcome: EvaluatorReferenceAccessOutcome
    grant: EvaluatorReferenceGrantV2 | None = None
    reasons: frozenset[EvaluatorReferenceAccessReason] = frozenset()
    policy_version: Literal["reference-access/r4-06-v1"] = REFERENCE_ACCESS_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> EvaluatorReferenceAccessOutcome:
        return _parse_enum(value, EvaluatorReferenceAccessOutcome, "outcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[EvaluatorReferenceAccessReason]:
        return _parse_enum_set(
            value,
            EvaluatorReferenceAccessReason,
            "reasons",
        )

    @model_validator(mode="after")
    def validate_result(self) -> EvaluatorReferenceAccessResult:
        _require_ref_type(
            self.request_ref,
            "evaluator-reference-access-request",
            "request_ref",
        )
        _require_ref_type(
            self.evaluator_spec_ref,
            "evaluator-spec",
            "evaluator_spec_ref",
        )
        _require_ref_type(
            self.reference_policy_ref,
            "reference-policy",
            "reference_policy_ref",
        )
        if self.outcome is EvaluatorReferenceAccessOutcome.GRANTED:
            if self.grant is None:
                raise ValueError("GRANTED access result requires a grant")
            if self.reasons:
                raise ValueError("GRANTED access result cannot carry denial reasons")
        elif self.grant is not None:
            raise ValueError(f"{self.outcome.value} access result cannot carry a grant")
        elif (
            self.outcome
            not in {
                EvaluatorReferenceAccessOutcome.NO_REFERENCE_REQUIRED,
                EvaluatorReferenceAccessOutcome.HUMAN_ONLY,
            }
            and not self.reasons
        ):
            raise ValueError(f"{self.outcome.value} access result requires reasons")
        return self


def evaluator_failure_observation_sha256(
    observation: EvaluatorFailureObservation,
) -> str:
    return evaluation_payload_sha256(
        {
            "source_observation_ref": _ref_payload(observation.source_observation_ref),
            "signals": sorted(item.value for item in observation.signals),
            "evidence_refs": [
                item.model_dump(mode="json", exclude_none=False)
                for item in sorted(
                    observation.evidence_refs,
                    key=lambda item: item.evidence_ref_id,
                )
            ],
            "policy_version": observation.policy_version,
        }
    )


def verified_model_domain_authorization_sha256(
    authorization: VerifiedModelDomainAuthorization,
) -> str:
    return evaluation_payload_sha256(
        {
            "model_profile_ref": _ref_payload(authorization.model_profile_ref),
            "model_domain_approval_ref": _ref_payload(authorization.model_domain_approval_ref),
            "verification_ref": _ref_payload(authorization.verification_ref),
            "domain": authorization.domain.value,
            "allowed_purposes": sorted(authorization.allowed_purposes),
            "allowed_data_classes": sorted(item.value for item in authorization.allowed_data_classes),
            "approved_reference_refs": [
                _ref_payload(ref)
                for ref in sorted(
                    authorization.approved_reference_refs,
                    key=_ref_key,
                )
            ],
            "valid_from": authorization.valid_from,
            "expires_at": authorization.expires_at,
            "internal_endpoint": authorization.internal_endpoint,
            "retention_policy_satisfied": (authorization.retention_policy_satisfied),
            "training_use_approved": authorization.training_use_approved,
            "independent_authority_verified": (authorization.independent_authority_verified),
            "non_self_signed": authorization.non_self_signed,
            "policy_version": authorization.policy_version,
        }
    )


def evaluator_reference_access_request_sha256(
    request: EvaluatorReferenceAccessRequest,
) -> str:
    return evaluation_payload_sha256(
        {
            "principal_type": request.principal_type.value,
            "principal_id": request.principal_id,
            "purpose": request.purpose,
            "evaluator_binding_id": request.evaluator_binding_id,
            "model_profile_ref": (
                _ref_payload(request.model_profile_ref) if request.model_profile_ref is not None else None
            ),
            "policy_version": request.policy_version,
        }
    )


def evaluation_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _require_unique_refs(refs: tuple[ObjectRef, ...]) -> None:
    keys = tuple(_ref_key(ref) for ref in refs)
    if len(keys) != len(set(keys)):
        raise ValueError("reference refs must be unique")


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _parse_enum[T: StrEnum](
    value: object,
    enum_type: type[T],
    field_name: str,
) -> T:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")


def _parse_enum_set[T: StrEnum](
    value: object,
    enum_type: type[T],
    field_name: str,
) -> frozenset[T]:
    if isinstance(value, (frozenset, set, tuple, list)):
        return frozenset(item if isinstance(item, enum_type) else enum_type(item) for item in value)
    raise TypeError(f"{field_name} must be a collection")
