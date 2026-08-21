from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from eval_factory.ai_gateway.receipts import GatewayRecordStore
from eval_factory.contracts.ai_gateway_v2 import (
    RAGRequestV2,
    RAGResultV2,
    RAGSourcePolicyV2,
)
from eval_factory.contracts.core import ObjectRef


class RAGAuthorizationError(RuntimeError):
    pass


class RAGRetriever(Protocol):
    def retrieve(self, request: RAGRequestV2) -> tuple[RAGRetrievedDocument, ...]: ...


@dataclass(frozen=True, slots=True)
class RAGRetrievedDocument:
    content_ref: ObjectRef
    provenance_ref: ObjectRef
    source_class: str
    size_bytes: int
    character_count: int

    def __post_init__(self) -> None:
        if self.content_ref.object_type != "rag-content":
            raise ValueError("RAG document requires a private rag-content ref")
        if self.provenance_ref.object_type != "provenance-decision":
            raise ValueError("RAG document requires a provenance decision ref")
        if self.size_bytes < 0 or self.character_count < 0:
            raise ValueError("RAG document sizes must be non-negative")


class PurposeBoundRAGGateway:
    def __init__(
        self,
        *,
        policies: tuple[RAGSourcePolicyV2, ...],
        retriever: RAGRetriever,
        records: GatewayRecordStore,
    ) -> None:
        self._policies = {_ref_key(policy.to_ref()): policy for policy in policies}
        if len(self._policies) != len(policies):
            raise ValueError("duplicate RAG source policy authority")
        self._retriever = retriever
        self._records = records

    def retrieve(self, request: RAGRequestV2) -> RAGResultV2:
        replay = self._records.get_rag(request.to_ref())
        if replay is not None:
            return replay
        policy = self._policies.get(_ref_key(request.source_policy_ref))
        if policy is None:
            raise RAGAuthorizationError("unknown RAG source policy")
        self._authorize(request, policy)
        documents = tuple(
            sorted(self._retriever.retrieve(request), key=lambda item: _ref_key(item.content_ref))
        )
        approved: list[RAGRetrievedDocument] = []
        excluded: set[str] = set()
        total_bytes = 0
        total_characters = 0
        for document in documents:
            if (
                document.source_class not in policy.allowed_source_classes
                or document.source_class not in request.requested_source_classes
            ):
                excluded.add("SOURCE_CLASS_DENIED")
                continue
            if len(approved) >= request.max_results:
                excluded.add("RESULT_BUDGET_EXCEEDED")
                continue
            if total_bytes + document.size_bytes > request.max_bytes:
                excluded.add("BYTE_BUDGET_EXCEEDED")
                continue
            if total_characters + document.character_count > request.max_characters:
                excluded.add("CHARACTER_BUDGET_EXCEEDED")
                continue
            approved.append(document)
            total_bytes += document.size_bytes
            total_characters += document.character_count
        result = RAGResultV2.create(
            rag_result_id=f"rag-result://{request.rag_request_id.rsplit('://', 1)[-1]}",
            request_ref=request.to_ref(),
            approved_content_refs=tuple(item.content_ref for item in approved),
            provenance_refs=tuple(item.provenance_ref for item in approved),
            excluded_reason_codes=tuple(sorted(excluded)),
            result_count=len(approved),
            total_bytes=total_bytes,
            total_characters=total_characters,
            audit=request.audit,
        )
        return self._records.commit_rag(request, result)

    @staticmethod
    def _authorize(request: RAGRequestV2, policy: RAGSourcePolicyV2) -> None:
        if request.principal_id not in policy.allowed_principal_ids:
            raise RAGAuthorizationError("RAG principal is not authorized")
        if request.purpose not in policy.allowed_purposes:
            raise RAGAuthorizationError("RAG purpose is not authorized")
        if not set(request.requested_source_classes).issubset(policy.allowed_source_classes):
            raise RAGAuthorizationError("RAG source class is not authorized")
        if (
            request.max_results > policy.max_results
            or request.max_bytes > policy.max_bytes
            or request.max_characters > policy.max_characters
        ):
            raise RAGAuthorizationError("RAG request exceeds policy budget")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "PurposeBoundRAGGateway",
    "RAGAuthorizationError",
    "RAGRetrievedDocument",
    "RAGRetriever",
]
