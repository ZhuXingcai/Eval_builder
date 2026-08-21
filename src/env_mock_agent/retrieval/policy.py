from __future__ import annotations

from urllib.parse import urlparse

from env_mock_agent.schemas import SourceType, UsagePolicy


def classify_source(url: str) -> SourceType:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(".gov") or ".gov." in host:
        return SourceType.OFFICIAL
    if host.endswith(".edu") or ".edu." in host:
        return SourceType.AUTHORITATIVE
    return SourceType.SECONDARY


def default_usage_policy(source_type: SourceType) -> UsagePolicy:
    if source_type == SourceType.OFFICIAL:
        return UsagePolicy.FACTS_ONLY
    if source_type == SourceType.AUTHORITATIVE:
        return UsagePolicy.FACTS_ONLY
    return UsagePolicy.STRUCTURE_AND_STYLE_ONLY
