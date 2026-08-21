from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,255}$"),
]


def _validate_relative_path(value: str) -> str:
    if value.startswith("/") or any(part == ".." for part in value.split("/")):
        raise ValueError("path must be relative and cannot traverse parents")
    return value


RelativePath = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9._ -][A-Za-z0-9._/ -]{0,511}$"),
    AfterValidator(_validate_relative_path),
]


class ContractModel(BaseModel):
    """Closed immutable base for every value crossing a factory stage boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    def canonical_json(self) -> bytes:
        value = self.model_dump(mode="json", by_alias=True, exclude_none=False)
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


class ObjectRef(ContractModel):
    schema_version: Literal["eval-factory/object-ref/v1"] = "eval-factory/object-ref/v1"
    object_type: Identifier
    object_id: Identifier
    object_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    object_sha256: Sha256

    def require_matching_hash(self, value: ContractModel) -> None:
        observed = value.canonical_sha256()
        if observed != self.object_sha256:
            raise ValueError(
                f"stale or mismatched object reference: expected {self.object_sha256}, observed {observed}"
            )


class SourceSpanRef(ContractModel):
    schema_version: Literal["eval-factory/source-span-ref/v1"] = "eval-factory/source-span-ref/v1"
    span_id: Identifier
    source_trace_id: Identifier
    raw_sha256: Sha256
    approximate: bool = False


class EvidencePolarity(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNCERTAINTY = "UNCERTAINTY"
    CONTRADICTION = "CONTRADICTION"


class EvidenceRef(ContractModel):
    schema_version: Literal["eval-factory/evidence-ref/v1"] = "eval-factory/evidence-ref/v1"
    evidence_ref_id: Identifier
    subject_ref: ObjectRef
    source_spans: tuple[SourceSpanRef, ...] = Field(min_length=1)
    polarity: EvidencePolarity
    capability: Identifier
    capability_complete: bool


class VersionBinding(ContractModel):
    schema_version: Literal["eval-factory/version-binding/v1"] = "eval-factory/version-binding/v1"
    component: Identifier
    version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    sha256: Sha256 | None = None


ScalarValue = str | int | float | bool | None


class TypedAttribute(ContractModel):
    schema_version: Literal["eval-factory/typed-attribute/v1"] = "eval-factory/typed-attribute/v1"
    key: Identifier
    value: ScalarValue


class ContractAudit(ContractModel):
    schema_version: Literal["eval-factory/contract-audit/v1"] = "eval-factory/contract-audit/v1"
    created_at: datetime
    created_by: Identifier
    governing_versions: tuple[VersionBinding, ...] = Field(min_length=1)
    input_refs: tuple[ObjectRef, ...] = ()


class FailureClass(StrEnum):
    CONTESTANT = "CONTESTANT_FAILURE"
    EVALUATOR = "EVALUATOR_FAILURE"
    ENVIRONMENT = "ENVIRONMENT_FAILURE"
    INDETERMINATE = "INDETERMINATE"
    POLICY = "POLICY_FAILURE"
    CAPABILITY = "CAPABILITY_FAILURE"
    INTERNAL = "INTERNAL_FAILURE"


class FailureRecord(ContractModel):
    schema_version: Literal["eval-factory/failure-record/v1"] = "eval-factory/failure-record/v1"
    failure_class: FailureClass
    code: Identifier
    message: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    retryable: bool
    evidence_refs: tuple[EvidenceRef, ...] = ()
    detail: tuple[TypedAttribute, ...] = ()
