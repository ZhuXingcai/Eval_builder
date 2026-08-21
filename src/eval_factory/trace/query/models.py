from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractModel,
    Identifier,
    ObjectRef,
    SourceSpanRef,
)


class TraceQueryError(RuntimeError):
    pass


class QueryAuthorizationError(TraceQueryError):
    pass


class QueryBudgetExceededError(TraceQueryError):
    pass


class QueryValidationError(TraceQueryError):
    pass


class QueryKind(StrEnum):
    MANIFEST = "manifest"
    EVENTS = "events"
    EVENT = "event"
    SOURCE_SPAN = "source_span"
    TOOL_CALLS = "tool_calls"
    FILE_OBSERVATIONS = "file_observations"
    SEGMENTS = "segments"
    TEXT_SEARCH = "text_search"


class TrustedTracePrincipal(ContractModel):
    schema_version: Literal["eval-factory/trusted-trace-principal/v1"] = (
        "eval-factory/trusted-trace-principal/v1"
    )
    principal_id: Identifier
    consumer_stage: Identifier
    consumer_agent: Identifier
    allowed_purposes: frozenset[Identifier] = Field(min_length=1)
    allowed_trace_ir_version_ids: frozenset[Identifier] = Field(min_length=1)
    max_events: int = Field(ge=1)
    max_characters: int = Field(ge=0)
    audit: bool = False

    @field_validator("allowed_purposes", "allowed_trace_ir_version_ids", mode="before")
    @classmethod
    def freeze_sets(cls, value: object) -> frozenset[str]:
        if isinstance(value, frozenset):
            return value
        if isinstance(value, (list, tuple, set)):
            return frozenset(str(item) for item in value)
        raise TypeError("principal grants must be a collection")


class TraceQueryRequest(ContractModel):
    schema_version: Literal["eval-factory/trace-query-request/v1"] = "eval-factory/trace-query-request/v1"
    principal: TrustedTracePrincipal
    trace_ir_version_id: Identifier
    purpose: Identifier
    query_kind: QueryKind
    filters: dict[str, str | int | bool] = Field(default_factory=dict)
    limit: int = Field(ge=1)
    max_characters: int = Field(ge=0)
    allow_tainted: bool = False

    @field_validator("query_kind", mode="before")
    @classmethod
    def parse_kind(cls, value: object) -> QueryKind:
        if isinstance(value, QueryKind):
            return value
        if isinstance(value, str):
            return QueryKind(value)
        raise TypeError("query_kind must be a supported query kind")

    @field_validator("filters", mode="before")
    @classmethod
    def normalize_filters(cls, value: object) -> dict[str, str | int | bool]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("filters must be an object")
        normalized: dict[str, str | int | bool] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("filter keys must be strings")
            if isinstance(item, (str, int, bool)):
                normalized[key] = item
                continue
            raise TypeError("filter values must be scalar strings, integers, or booleans")
        return normalized

    @model_validator(mode="after")
    def reject_consumer_forgery(self) -> TraceQueryRequest:
        for forbidden in ("consumer_stage", "consumer_agent", "principal_id"):
            if forbidden in self.filters:
                raise ValueError("request filters cannot claim consumer identity")
        return self


class TraceEvidenceRequest(ContractModel):
    schema_version: Literal["eval-factory/trace-evidence-request/v1"] = (
        "eval-factory/trace-evidence-request/v1"
    )
    query: TraceQueryRequest


class TraceTextFragment(ContractModel):
    schema_version: Literal["eval-factory/trace-text-fragment/v1"] = "eval-factory/trace-text-fragment/v1"
    subject_ref: ObjectRef
    text: str


class TraceQueryResponse(ContractModel):
    schema_version: Literal["eval-factory/trace-query-response/v1"] = "eval-factory/trace-query-response/v1"
    query_id: Identifier
    trace_ir_version_id: Identifier
    principal_id: Identifier
    consumer_stage: Identifier
    consumer_agent: Identifier
    purpose: Identifier
    query_kind: QueryKind
    result_refs: tuple[ObjectRef, ...]
    source_span_refs: tuple[SourceSpanRef, ...] = ()
    text_fragments: tuple[TraceTextFragment, ...] = ()
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    truncated: bool
    audit: bool

    @field_validator("query_kind", mode="before")
    @classmethod
    def parse_kind(cls, value: object) -> QueryKind:
        if isinstance(value, QueryKind):
            return value
        if isinstance(value, str):
            return QueryKind(value)
        raise TypeError("query_kind must be a supported query kind")

    @model_validator(mode="after")
    def enforce_budget(self) -> TraceQueryResponse:
        if self.returned_characters > self.max_characters:
            raise ValueError("returned characters exceed query budget")
        return self


def query_id_for(request: TraceQueryRequest, refs: tuple[ObjectRef, ...]) -> str:
    payload = {
        "principal_id": request.principal.principal_id,
        "trace_ir_version_id": request.trace_ir_version_id,
        "purpose": request.purpose,
        "query_kind": request.query_kind.value,
        "filters": request.filters,
        "limit": request.limit,
        "max_characters": request.max_characters,
        "refs": [item.model_dump(mode="json", exclude_none=False) for item in refs],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"trace-query://sha256/{hashlib.sha256(encoded).hexdigest()}"
