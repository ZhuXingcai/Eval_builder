from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
    PromptLeakageFingerprintV2,
)

_TOKEN_PATTERN = re.compile(
    r"[\w]+(?:[._/-][\w]+)*",
    flags=re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class PromptLeakageFingerprintMatch:
    category: PromptLeakageCategoryV2
    span_start: int
    span_end: int
    fingerprint_ids: tuple[str, ...]


def normalize_prompt_leakage_tokens(text: str) -> tuple[str, ...]:
    return tuple(value for value, _, _ in normalize_prompt_leakage_tokens_with_spans(text))


def normalize_prompt_leakage_tokens_with_spans(
    text: str,
) -> tuple[tuple[str, int, int], ...]:
    values: list[tuple[str, int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        normalized = unicodedata.normalize(
            "NFKC",
            match.group(),
        ).casefold()
        if normalized:
            values.append((normalized, match.start(), match.end()))
    return tuple(values)


def digest_prompt_leakage_tokens(tokens: Iterable[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode()).hexdigest()


def match_prompt_leakage_fingerprints(
    text: str,
    fingerprints: tuple[PromptLeakageFingerprintV2, ...],
) -> tuple[PromptLeakageFingerprintMatch, ...]:
    prompt_tokens = normalize_prompt_leakage_tokens_with_spans(text)
    fingerprints_by_key: dict[
        tuple[int, str],
        list[PromptLeakageFingerprintV2],
    ] = {}
    for fingerprint in fingerprints:
        fingerprints_by_key.setdefault(
            (
                fingerprint.token_count,
                fingerprint.digest_sha256,
            ),
            [],
        ).append(fingerprint)
    matches: dict[
        tuple[PromptLeakageCategoryV2, int, int],
        set[str],
    ] = {}
    for token_count in sorted({key[0] for key in fingerprints_by_key}):
        if token_count > len(prompt_tokens):
            continue
        for start in range(0, len(prompt_tokens) - token_count + 1):
            window = prompt_tokens[start : start + token_count]
            digest = digest_prompt_leakage_tokens(item[0] for item in window)
            for fingerprint in fingerprints_by_key.get(
                (token_count, digest),
                (),
            ):
                key = (
                    fingerprint.category,
                    window[0][1],
                    window[-1][2],
                )
                matches.setdefault(key, set()).add(fingerprint.fingerprint_id)
    return tuple(
        PromptLeakageFingerprintMatch(
            category=key[0],
            span_start=key[1],
            span_end=key[2],
            fingerprint_ids=tuple(sorted(fingerprint_ids)),
        )
        for key, fingerprint_ids in sorted(
            matches.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1],
                item[0][2],
            ),
        )
    )
