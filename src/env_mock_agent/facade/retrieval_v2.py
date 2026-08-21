from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    Sha256,
)
from env_mock_agent.retrieval.registry import RetrievalRegistry


class PublicSourceRetrievalFacadeFailureKind(StrEnum):
    CAPABILITY = "CAPABILITY"
    POLICY = "POLICY"


class PublicSourceRetrievalFacadeError(RuntimeError):
    def __init__(
        self,
        kind: PublicSourceRetrievalFacadeFailureKind,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.code = code


class PublicSourceSearchRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/public-source-search-request/v2"] = (
        "env-mock-agent/public-source-search-request/v2"
    )
    search_request_id: Identifier
    query: str = Field(min_length=1, max_length=2000)
    provider_id: Identifier
    result_limit: int = Field(ge=1, le=100)
    allowed_schemes: tuple[str, ...] = Field(min_length=1)
    allowed_host_suffixes: tuple[str, ...] = Field(min_length=1)
    query_approval_ref: FacadeObjectRef
    retrieval_policy_ref: FacadeObjectRef
    idempotency_key: Identifier
    search_request_sha256: Sha256

    @field_validator("allowed_schemes")
    @classmethod
    def validate_schemes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_sorted_unique("allowed schemes", value)
        if any(item not in {"http", "https"} for item in value):
            raise ValueError("allowed schemes must contain only http or https")
        return value

    @field_validator("allowed_host_suffixes")
    @classmethod
    def validate_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_host_suffixes(value)
        return value

    @model_validator(mode="after")
    def validate_refs(self) -> PublicSourceSearchRequestV2:
        _require_ref(
            self.query_approval_ref,
            "public-search-query-approval",
            None,
            "query_approval_ref",
        )
        _require_ref(
            self.retrieval_policy_ref,
            "public-source-retrieval-policy",
            "v2",
            "retrieval_policy_ref",
        )
        return self


class PublicSourceSearchHitV2(FacadeModel):
    schema_version: Literal["env-mock-agent/public-source-search-hit/v2"] = (
        "env-mock-agent/public-source-search-hit/v2"
    )
    search_hit_id: Identifier
    source_uri: str = Field(min_length=3, max_length=2048)
    rank: int = Field(ge=1)
    provider_id: Identifier
    search_hit_sha256: Sha256

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        return _canonical_public_uri(value)


class PublicSourceSearchResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/public-source-search-result/v2"] = (
        "env-mock-agent/public-source-search-result/v2"
    )
    search_result_id: Identifier
    search_request_ref: FacadeObjectRef
    provider_id: Identifier
    hits: tuple[PublicSourceSearchHitV2, ...] = ()
    search_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> PublicSourceSearchResultV2:
        _require_ref(
            self.search_request_ref,
            "public-source-search-request",
            "v2",
            "search_request_ref",
        )
        if any(item.provider_id != self.provider_id for item in self.hits):
            raise ValueError("search hits must bind the result provider")
        keys = tuple(
            (
                item.rank,
                item.source_uri,
                item.search_hit_id,
            )
            for item in self.hits
        )
        if keys != tuple(sorted(keys)):
            raise ValueError("search hits must be sorted by rank, URI, and ID")
        _require_unique("search hit IDs", tuple(item.search_hit_id for item in self.hits))
        _require_unique("search hit URIs", tuple(item.source_uri for item in self.hits))
        return self


class PublicSourceFetchRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/public-source-fetch-request/v2"] = (
        "env-mock-agent/public-source-fetch-request/v2"
    )
    fetch_request_id: Identifier
    source_uri: str = Field(min_length=3, max_length=2048)
    provider_id: Identifier
    max_bytes: int = Field(ge=1)
    allowed_schemes: tuple[str, ...] = Field(min_length=1)
    allowed_host_suffixes: tuple[str, ...] = Field(min_length=1)
    source_approval_ref: FacadeObjectRef
    retrieval_policy_ref: FacadeObjectRef
    idempotency_key: Identifier
    fetch_request_sha256: Sha256

    @field_validator("allowed_schemes")
    @classmethod
    def validate_schemes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_sorted_unique("allowed schemes", value)
        if any(item not in {"http", "https"} for item in value):
            raise ValueError("allowed schemes must contain only http or https")
        return value

    @field_validator("allowed_host_suffixes")
    @classmethod
    def validate_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_host_suffixes(value)
        return value

    @model_validator(mode="after")
    def validate_request(self) -> PublicSourceFetchRequestV2:
        _require_ref(
            self.source_approval_ref,
            "public-source-approval",
            None,
            "source_approval_ref",
        )
        _require_ref(
            self.retrieval_policy_ref,
            "public-source-retrieval-policy",
            "v2",
            "retrieval_policy_ref",
        )
        return self


class PublicSourceFetchResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/public-source-fetch-result/v2"] = (
        "env-mock-agent/public-source-fetch-result/v2"
    )
    fetch_result_id: Identifier
    fetch_request_ref: FacadeObjectRef
    requested_source_uri: str = Field(min_length=3, max_length=2048)
    canonical_source_uri: str = Field(min_length=3, max_length=2048)
    retrieved_at: datetime
    status_code: int = Field(ge=100, le=599)
    media_type: str | None = Field(default=None, min_length=3, max_length=255)
    bytes_count: int = Field(ge=0)
    content_ref: FacadeObjectRef
    content_sha256: Sha256
    provider_id: Identifier
    response_metadata_sha256: Sha256
    fetch_result_sha256: Sha256

    @field_validator("requested_source_uri", "canonical_source_uri")
    @classmethod
    def validate_source_uri(cls, value: str) -> str:
        return _canonical_public_uri(value)

    @model_validator(mode="after")
    def validate_result(self) -> PublicSourceFetchResultV2:
        _require_ref(
            self.fetch_request_ref,
            "public-source-fetch-request",
            "v2",
            "fetch_request_ref",
        )
        _require_ref(
            self.content_ref,
            "public-source-content",
            "v1",
            "content_ref",
        )
        if self.content_ref.object_sha256 != self.content_sha256:
            raise ValueError("content_ref must bind content_sha256")
        return self


class PublicSourceRetrievalFacade(Protocol):
    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2: ...

    async def fetch(
        self,
        request: PublicSourceFetchRequestV2,
    ) -> PublicSourceFetchResultV2: ...


class RegistryPublicSourceRetrievalFacade:
    def __init__(
        self,
        registry: RetrievalRegistry,
        *,
        clock: Callable[[], datetime],
    ) -> None:
        self._registry = registry
        self._clock = clock

    async def search(
        self,
        request: PublicSourceSearchRequestV2,
    ) -> PublicSourceSearchResultV2:
        _validate_search_request_identity(request)
        try:
            provider = self._registry.search(request.provider_id)
        except KeyError as exc:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "SEARCH_PROVIDER_UNAVAILABLE",
                "approved search provider is unavailable",
            ) from exc
        try:
            legacy_hits = await provider.search(
                request.query,
                limit=request.result_limit,
            )
        except Exception as exc:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "SEARCH_FAILED",
                "approved search provider failed",
            ) from exc
        if len(legacy_hits) > request.result_limit:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.POLICY,
                "SEARCH_RESULT_LIMIT_EXCEEDED",
                "search provider exceeded the approved result limit",
            )

        values: list[PublicSourceSearchHitV2] = []
        for position, legacy in enumerate(legacy_hits, start=1):
            try:
                source_uri = _validate_uri_policy(
                    legacy.url,
                    request.allowed_schemes,
                    request.allowed_host_suffixes,
                )
            except PublicSourceRetrievalFacadeError:
                continue
            hit = PublicSourceSearchHitV2(
                search_hit_id="public-source-search-hit://pending",
                source_uri=source_uri,
                rank=legacy.rank if legacy.rank >= 1 else position,
                provider_id=request.provider_id,
                search_hit_sha256="0" * 64,
            )
            digest = public_source_search_hit_carried_sha256(hit)
            values.append(
                hit.model_copy(
                    update={
                        "search_hit_id": f"public-source-search-hit://sha256/{digest}",
                        "search_hit_sha256": digest,
                    }
                )
            )
        hits = tuple(
            sorted(
                values,
                key=lambda item: (
                    item.rank,
                    item.source_uri,
                    item.search_hit_id,
                ),
            )
        )
        request_ref = public_source_search_request_ref(request)
        result = PublicSourceSearchResultV2(
            search_result_id="public-source-search-result://pending",
            search_request_ref=request_ref,
            provider_id=request.provider_id,
            hits=hits,
            search_result_sha256="0" * 64,
        )
        digest = public_source_search_result_carried_sha256(result)
        return result.model_copy(
            update={
                "search_result_id": f"public-source-search-result://sha256/{digest}",
                "search_result_sha256": digest,
            }
        )

    async def fetch(
        self,
        request: PublicSourceFetchRequestV2,
    ) -> PublicSourceFetchResultV2:
        _validate_fetch_request_identity(request)
        requested_uri = _validate_uri_policy(
            request.source_uri,
            request.allowed_schemes,
            request.allowed_host_suffixes,
        )
        try:
            provider = self._registry.fetch(request.provider_id)
        except KeyError as exc:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "FETCH_PROVIDER_UNAVAILABLE",
                "approved fetch provider is unavailable",
            ) from exc
        try:
            legacy = await provider.fetch(requested_uri)
        except Exception as exc:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "FETCH_FAILED",
                "approved fetch provider failed",
            ) from exc
        canonical_uri = _validate_uri_policy(
            legacy.url,
            request.allowed_schemes,
            request.allowed_host_suffixes,
        )
        if not 200 <= legacy.status_code < 300:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "FETCH_STATUS_NOT_SUCCESSFUL",
                "fetch provider returned a non-success status",
            )
        if legacy.bytes_count > request.max_bytes:
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.POLICY,
                "FETCH_BYTE_LIMIT_EXCEEDED",
                "fetch provider exceeded the approved byte limit",
            )
        if not _is_sha256(legacy.sha256):
            raise PublicSourceRetrievalFacadeError(
                PublicSourceRetrievalFacadeFailureKind.CAPABILITY,
                "FETCH_CONTENT_HASH_INVALID",
                "fetch provider returned an invalid content hash",
            )
        content_ref = FacadeObjectRef(
            object_type="public-source-content",
            object_id=f"public-source-content://sha256/{legacy.sha256}",
            object_version="v1",
            object_sha256=legacy.sha256,
        )
        metadata_digest = _payload_sha256(
            {
                "status_code": legacy.status_code,
                "media_type": legacy.media_type,
                "bytes_count": legacy.bytes_count,
                "headers": {
                    key.casefold(): value
                    for key, value in sorted(legacy.headers.items())
                    if key.casefold()
                    in {
                        "content-length",
                        "content-type",
                        "etag",
                        "last-modified",
                    }
                },
            }
        )
        result = PublicSourceFetchResultV2(
            fetch_result_id="public-source-fetch-result://pending",
            fetch_request_ref=public_source_fetch_request_ref(request),
            requested_source_uri=requested_uri,
            canonical_source_uri=canonical_uri,
            retrieved_at=self._clock(),
            status_code=legacy.status_code,
            media_type=legacy.media_type,
            bytes_count=legacy.bytes_count,
            content_ref=content_ref,
            content_sha256=legacy.sha256,
            provider_id=request.provider_id,
            response_metadata_sha256=metadata_digest,
            fetch_result_sha256="0" * 64,
        )
        digest = public_source_fetch_result_carried_sha256(result)
        return result.model_copy(
            update={
                "fetch_result_id": f"public-source-fetch-result://sha256/{digest}",
                "fetch_result_sha256": digest,
            }
        )


def public_source_search_request_carried_sha256(
    request: PublicSourceSearchRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={"search_request_id", "search_request_sha256"},
            exclude_none=False,
        )
    )


def public_source_search_request_ref(
    request: PublicSourceSearchRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="public-source-search-request",
        object_id=request.search_request_id,
        object_version="v2",
        object_sha256=request.search_request_sha256,
    )


def public_source_search_hit_carried_sha256(
    hit: PublicSourceSearchHitV2,
) -> str:
    return _payload_sha256(
        hit.model_dump(
            mode="json",
            exclude={"search_hit_id", "search_hit_sha256"},
            exclude_none=False,
        )
    )


def public_source_search_result_carried_sha256(
    result: PublicSourceSearchResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={"search_result_id", "search_result_sha256"},
            exclude_none=False,
        )
    )


def public_source_search_result_ref(
    result: PublicSourceSearchResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="public-source-search-result",
        object_id=result.search_result_id,
        object_version="v2",
        object_sha256=result.search_result_sha256,
    )


def public_source_fetch_request_carried_sha256(
    request: PublicSourceFetchRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={"fetch_request_id", "fetch_request_sha256"},
            exclude_none=False,
        )
    )


def public_source_fetch_request_ref(
    request: PublicSourceFetchRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="public-source-fetch-request",
        object_id=request.fetch_request_id,
        object_version="v2",
        object_sha256=request.fetch_request_sha256,
    )


def public_source_fetch_result_carried_sha256(
    result: PublicSourceFetchResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={"fetch_result_id", "fetch_result_sha256"},
            exclude_none=False,
        )
    )


def public_source_fetch_result_ref(
    result: PublicSourceFetchResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="public-source-fetch-result",
        object_id=result.fetch_result_id,
        object_version="v2",
        object_sha256=result.fetch_result_sha256,
    )


def _validate_search_request_identity(
    request: PublicSourceSearchRequestV2,
) -> None:
    digest = public_source_search_request_carried_sha256(request)
    if (
        request.search_request_sha256 != digest
        or request.search_request_id != f"public-source-search-request://sha256/{digest}"
    ):
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "SEARCH_REQUEST_IDENTITY_INVALID",
            "search request identity is stale or invalid",
        )


def _validate_fetch_request_identity(
    request: PublicSourceFetchRequestV2,
) -> None:
    digest = public_source_fetch_request_carried_sha256(request)
    if (
        request.fetch_request_sha256 != digest
        or request.fetch_request_id != f"public-source-fetch-request://sha256/{digest}"
    ):
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "FETCH_REQUEST_IDENTITY_INVALID",
            "fetch request identity is stale or invalid",
        )


def _validate_uri_policy(
    value: str,
    allowed_schemes: tuple[str, ...],
    allowed_host_suffixes: tuple[str, ...],
) -> str:
    try:
        canonical = _canonical_public_uri(value)
    except ValueError as exc:
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "URI_NOT_PUBLIC",
            "source URI targets a private or local location",
        ) from exc
    parsed = urlsplit(canonical)
    if parsed.scheme not in allowed_schemes:
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "URI_SCHEME_NOT_APPROVED",
            "source URI scheme is not approved",
        )
    host = parsed.hostname or ""
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_host_suffixes):
        raise PublicSourceRetrievalFacadeError(
            PublicSourceRetrievalFacadeFailureKind.POLICY,
            "URI_HOST_NOT_APPROVED",
            "source URI host is not approved",
        )
    return canonical


def _canonical_public_uri(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("source URI must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URI cannot contain credentials")
    if parsed.fragment:
        raise ValueError("source URI cannot contain a fragment")
    host = (parsed.hostname or "").casefold().rstrip(".")
    if not host:
        raise ValueError("source URI requires a host")
    if _is_private_or_local_host(host):
        raise ValueError("source URI cannot target a private or local host")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            f"{host}{port}",
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _is_private_or_local_host(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _validate_host_suffixes(values: tuple[str, ...]) -> None:
    _validate_sorted_unique("allowed host suffixes", values)
    for value in values:
        if (
            value != value.casefold()
            or value.startswith(".")
            or value.startswith("*")
            or "/" in value
            or ":" in value
            or not value
            or _is_private_or_local_host(value)
        ):
            raise ValueError("allowed host suffixes must be normalized public DNS suffixes")


def _validate_sorted_unique(label: str, values: tuple[str, ...]) -> None:
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{label} must be sorted and unique")


def _require_ref(
    ref: FacadeObjectRef,
    object_type: str,
    object_version: str | None,
    field_name: str,
) -> None:
    if ref.object_type != object_type:
        raise ValueError(f"{field_name} must reference {object_type}")
    if object_version is not None and ref.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
