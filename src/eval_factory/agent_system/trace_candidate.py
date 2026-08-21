from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.agent_system.specialists import GatewayIntentRewriteAgent
from eval_factory.contracts.agent_system_v2 import (
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, SourceSpanRef
from eval_factory.contracts.trace import ParseQuality, TraceEventType
from eval_factory.trace.normalization.models import (
    object_ref_for_event,
    object_ref_for_span,
)
from eval_factory.trace.storage.models import StoredTraceIndex


class TraceCandidatePreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TraceCandidatePreparation:
    decision: TraceCandidateDecisionV2
    extracted_prompt: ExtractedUserPromptV2 | None
    inferred_intent: InferredUserIntentV2 | None
    rewrite_candidate: TaskRewriteCandidateV2 | None
    route_refs: tuple[ObjectRef, ...]

    def validation_refs(self) -> tuple[ObjectRef, ...]:
        values = (
            self.decision.to_ref(),
            *((self.extracted_prompt.to_ref(),) if self.extracted_prompt is not None else ()),
            *((self.inferred_intent.to_ref(),) if self.inferred_intent is not None else ()),
            *((self.rewrite_candidate.to_ref(),) if self.rewrite_candidate is not None else ()),
            *self.route_refs,
        )
        return tuple(
            sorted(
                set(values),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        )


class TraceCandidatePreparationService:
    """Owns one source's prompt extraction, intent, rewrite, and decision."""

    def __init__(
        self,
        *,
        private_store: FactoryPrivateObjectStore,
        semantic_agent: GatewayIntentRewriteAgent,
    ) -> None:
        self.private_store = private_store
        self.semantic_agent = semantic_agent

    async def prepare(
        self,
        *,
        indexed: StoredTraceIndex,
        source_ref: ObjectRef,
        audit: ContractAudit,
    ) -> TraceCandidatePreparation:
        source_trace_id = indexed.manifest.source_trace_id
        if (
            source_ref.object_type != "trace-source"
            or source_ref.object_version != "v2"
            or source_ref.object_sha256 != indexed.manifest.raw_sha256
        ):
            raise TraceCandidatePreparationError(
                "trace candidate source authority differs from index",
            )
        user_event = next(
            (
                event
                for event in indexed.events
                if event.event_type is TraceEventType.USER_TEXT and event.content_ref is not None
            ),
            None,
        )
        if user_event is None or indexed.manifest.parse_quality == ParseQuality.UNPARSEABLE.value:
            return TraceCandidatePreparation(
                decision=TraceCandidateDecisionV2.create(
                    decision_id=(
                        f"trace-candidate-decision://{_decision_scope(source_trace_id)}/missing-user-prompt"
                    ),
                    source_trace_id=source_trace_id,
                    source_ref=source_ref,
                    disposition=(TraceCandidateDispositionV2.NON_CANDIDATE),
                    reason_codes=("USER_PROMPT_MISSING",),
                    cleaned_trace_ref=None,
                    audit=audit,
                ),
                extracted_prompt=None,
                inferred_intent=None,
                rewrite_candidate=None,
                route_refs=(),
            )
        prompt_text, extracted = self._extract_prompt(
            indexed=indexed,
            user_event_id=user_event.event_id,
            source_ref=source_ref,
            audit=audit,
        )
        intent_result = await self.semantic_agent.infer_intent(
            task_ref=_agent_task_ref("intent", extracted.object_id),
            extracted_prompt=extracted,
            prompt_text=prompt_text,
            audit=audit,
        )
        rewrite_result = await self.semantic_agent.rewrite(
            task_ref=_agent_task_ref("rewrite", extracted.object_id),
            extracted_prompt=extracted,
            intent=intent_result.intent,
            prompt_text=prompt_text,
            audit=audit,
        )
        return TraceCandidatePreparation(
            decision=TraceCandidateDecisionV2.create(
                decision_id=(f"trace-candidate-decision://{_decision_scope(source_trace_id)}/candidate"),
                source_trace_id=source_trace_id,
                source_ref=source_ref,
                disposition=TraceCandidateDispositionV2.CANDIDATE,
                reason_codes=(),
                cleaned_trace_ref=_cleaned_trace_ref(
                    indexed.trace_ir_version_id,
                    indexed.manifest.raw_sha256,
                ),
                audit=audit,
            ),
            extracted_prompt=extracted,
            inferred_intent=intent_result.intent,
            rewrite_candidate=rewrite_result.candidate,
            route_refs=(
                intent_result.route.to_ref(),
                rewrite_result.route.to_ref(),
            ),
        )

    def _extract_prompt(
        self,
        *,
        indexed: StoredTraceIndex,
        user_event_id: str,
        source_ref: ObjectRef,
        audit: ContractAudit,
    ) -> tuple[str, ExtractedUserPromptV2]:
        event = next(event for event in indexed.events if event.event_id == user_event_id)
        assert event.content_ref is not None
        blobs = {blob.content_ref.object_id: blob for blob in indexed.content_blobs}
        blob = blobs.get(event.content_ref.object_id)
        if blob is None:
            raise TraceCandidatePreparationError(
                "user prompt content blob is unresolved",
            )
        prompt_text = blob.canonical_bytes.decode("utf-8")
        private_content_ref = self.private_store.put_text(
            object_type="user-prompt-content",
            text=prompt_text,
        )
        event_ref = object_ref_for_event(event)
        segment = next(
            (
                value
                for value in indexed.interaction_segments
                if value.boundary_method == "user_turn" and event_ref in value.member_event_refs
            ),
            None,
        )
        if segment is None:
            raise TraceCandidatePreparationError(
                "user prompt interaction segment is unresolved",
            )
        span_by_id = {object_ref_for_span(span).object_id: span for span in indexed.source_spans}
        spans = tuple(
            span_by_id[reference.object_id]
            for reference in event.source_spans
            if reference.object_id in span_by_id
        )
        if not spans:
            raise TraceCandidatePreparationError(
                "user prompt has no source span",
            )
        if any(span.source_trace_id != indexed.manifest.source_trace_id for span in spans):
            raise TraceCandidatePreparationError(
                "user prompt source span crosses traces",
            )
        extracted = ExtractedUserPromptV2.create(
            extracted_prompt_id=(f"extracted-user-prompt://{_suffix(event.event_id)}"),
            trace_ref=_cleaned_trace_ref(
                indexed.trace_ir_version_id,
                source_ref.object_sha256,
            ),
            interaction_segment_ref=ObjectRef(
                object_type="interaction-segment",
                object_id=segment.segment_id,
                object_version="v2",
                object_sha256=segment.canonical_sha256(),
            ),
            content_ref=private_content_ref,
            source_spans=tuple(
                SourceSpanRef(
                    span_id=span.span_id,
                    source_trace_id=span.source_trace_id,
                    raw_sha256=span.raw_sha256,
                    approximate=span.approximate,
                )
                for span in spans
            ),
            context_segment_refs=(),
            audit=audit,
        )
        return prompt_text, extracted


def _cleaned_trace_ref(
    trace_ir_version_id: str,
    raw_sha256: str,
) -> ObjectRef:
    digest = hashlib.sha256(f"{trace_ir_version_id}|{raw_sha256}".encode()).hexdigest()
    return ObjectRef(
        object_type="trace-ir",
        object_id=trace_ir_version_id,
        object_version="v2",
        object_sha256=digest,
    )


def _agent_task_ref(kind: str, seed: str) -> ObjectRef:
    digest = hashlib.sha256(f"{kind}|{seed}".encode()).hexdigest()
    return ObjectRef(
        object_type="agent-task",
        object_id=f"agent-task://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _suffix(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def _decision_scope(source_trace_id: str) -> str:
    suffix = source_trace_id.rsplit("://", 1)[-1]
    return suffix.split("/", 1)[0]


__all__ = [
    "TraceCandidatePreparation",
    "TraceCandidatePreparationError",
    "TraceCandidatePreparationService",
]
