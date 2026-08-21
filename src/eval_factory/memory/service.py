from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol

from eval_factory.memory.models import (
    MemoryKindV1,
    MemoryRecallMatchV1,
    MemoryRecallQueryV1,
    MemoryRecallResultV1,
    MemoryRetrievalModeV1,
    StoredMemoryV1,
)
from eval_factory.memory.policy import MemoryCapabilityUnavailableError
from eval_factory.memory.store import AgentMemoryStore

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


class SemanticMemoryIndex(Protocol):
    def score(
        self,
        *,
        query_text: str,
        candidates: tuple[StoredMemoryV1, ...],
    ) -> dict[str, int]: ...


class AgentMemoryService:
    def __init__(
        self,
        *,
        store: AgentMemoryStore,
        semantic_index: SemanticMemoryIndex | None = None,
    ) -> None:
        self._store = store
        self._semantic_index = semantic_index

    def recall(self, query: MemoryRecallQueryV1) -> MemoryRecallResultV1:
        candidates = tuple(
            value
            for value in self._store.list_current(access=query.access)
            if _is_current(value, evaluated_at=query.evaluated_at)
            and _matches_kind(value, query.kinds)
            and set(query.required_tags).issubset(value.record.tags)
        )
        semantic_scores = self._semantic_scores(query=query, candidates=candidates)
        matches: list[MemoryRecallMatchV1] = []
        for candidate in candidates:
            lexical = _lexical_score(query.query_text, candidate.content)
            semantic = semantic_scores.get(candidate.record.memory_record_id)
            if query.mode is MemoryRetrievalModeV1.LEXICAL and lexical == 0:
                continue
            if query.mode is MemoryRetrievalModeV1.SEMANTIC_REQUIRED and semantic == 0:
                continue
            if query.mode is MemoryRetrievalModeV1.HYBRID_REQUIRED and lexical == 0 and semantic == 0:
                continue
            recency = _recency_score(
                evaluated_at=query.evaluated_at,
                created_at=candidate.record.created_at,
                half_life_seconds=query.recency_half_life_seconds,
            )
            importance = candidate.record.importance_basis_points
            total = _total_score(
                mode=query.mode,
                lexical=lexical,
                semantic=semantic,
                recency=recency,
                importance=importance,
            )
            matches.append(
                MemoryRecallMatchV1(
                    memory=candidate,
                    total_score_basis_points=total,
                    lexical_score_basis_points=lexical,
                    semantic_score_basis_points=(
                        semantic
                        if query.mode
                        in {
                            MemoryRetrievalModeV1.SEMANTIC_REQUIRED,
                            MemoryRetrievalModeV1.HYBRID_REQUIRED,
                        }
                        else None
                    ),
                    recency_score_basis_points=recency,
                    importance_score_basis_points=importance,
                )
            )
        selected = tuple(
            sorted(
                matches,
                key=lambda item: (
                    -item.total_score_basis_points,
                    item.memory.record.memory_id,
                    -item.memory.record.revision,
                ),
            )[: query.limit]
        )
        return MemoryRecallResultV1(
            query_sha256=query.canonical_sha256(),
            mode=query.mode,
            evaluated_at=query.evaluated_at,
            matches=selected,
            candidate_count=len(candidates),
        )

    def _semantic_scores(
        self,
        *,
        query: MemoryRecallQueryV1,
        candidates: tuple[StoredMemoryV1, ...],
    ) -> dict[str, int]:
        if query.mode is MemoryRetrievalModeV1.LEXICAL:
            return {}
        if self._semantic_index is None:
            raise MemoryCapabilityUnavailableError("required semantic memory index is unavailable")
        scores = self._semantic_index.score(
            query_text=query.query_text,
            candidates=candidates,
        )
        expected_ids = {item.record.memory_record_id for item in candidates}
        if not set(scores).issubset(expected_ids):
            raise MemoryCapabilityUnavailableError("semantic memory index returned an unauthorized candidate")
        if any(
            isinstance(score, bool) or not isinstance(score, int) or score < 0 or score > 10_000
            for score in scores.values()
        ):
            raise MemoryCapabilityUnavailableError("semantic memory index returned an invalid score")
        return {item_id: scores.get(item_id, 0) for item_id in expected_ids}


def _is_current(value: StoredMemoryV1, *, evaluated_at: datetime) -> bool:
    record = value.record
    return (
        record.created_at <= evaluated_at
        and record.valid_from <= evaluated_at
        and (record.valid_until is None or evaluated_at < record.valid_until)
        and (record.expires_at is None or evaluated_at < record.expires_at)
    )


def _matches_kind(
    value: StoredMemoryV1,
    kinds: tuple[MemoryKindV1, ...],
) -> bool:
    return not kinds or value.record.kind in kinds


def _lexical_score(query_text: str, content: str) -> int:
    query = query_text.casefold().strip()
    candidate = content.casefold()
    query_tokens = set(_TOKEN_PATTERN.findall(query))
    candidate_tokens = set(_TOKEN_PATTERN.findall(candidate))
    if not query_tokens:
        return 0
    overlap = len(query_tokens & candidate_tokens)
    token_score = overlap * 8_000 // len(query_tokens)
    phrase_score = 2_000 if query in candidate else 0
    return min(10_000, token_score + phrase_score)


def _recency_score(
    *,
    evaluated_at: datetime,
    created_at: datetime,
    half_life_seconds: int,
) -> int:
    age_seconds = max(0, int((evaluated_at - created_at).total_seconds()))
    periods = age_seconds // half_life_seconds
    remainder = age_seconds % half_life_seconds
    base = 10_000 // (1 << min(periods, 14))
    next_base = base // 2
    interpolated = base - ((base - next_base) * remainder // half_life_seconds)
    return max(0, min(10_000, interpolated))


def _total_score(
    *,
    mode: MemoryRetrievalModeV1,
    lexical: int,
    semantic: int | None,
    recency: int,
    importance: int,
) -> int:
    if mode is MemoryRetrievalModeV1.LEXICAL:
        return (lexical * 7_000 + recency * 1_500 + importance * 1_500) // 10_000
    if mode is MemoryRetrievalModeV1.SEMANTIC_REQUIRED:
        assert semantic is not None
        return (semantic * 7_000 + recency * 1_500 + importance * 1_500) // 10_000
    assert semantic is not None
    return (lexical * 3_500 + semantic * 3_500 + recency * 1_500 + importance * 1_500) // 10_000


__all__ = ["AgentMemoryService", "SemanticMemoryIndex"]
