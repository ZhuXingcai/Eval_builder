from __future__ import annotations

from pathlib import Path

from env_mock_agent.retrieval.policy import classify_source, default_usage_policy
from env_mock_agent.schemas import FetchResult, SourceEvidence


def evidence_from_fetch(
    source_id: str,
    result: FetchResult,
    *,
    title: str = "",
    publisher: str = "",
    relevant_excerpt: str = "",
    supported_dependencies: list[str] | None = None,
    used_by: list[str] | None = None,
    usage_basis: str = "",
) -> SourceEvidence:
    source_type = classify_source(result.url)
    return SourceEvidence(
        source_id=source_id,
        url=result.url,
        local_path=str(Path(result.content_path).resolve()),
        title=title,
        publisher=publisher,
        source_type=source_type,
        usage_policy=default_usage_policy(source_type),
        sha256=result.sha256,
        media_type=result.media_type,
        relevant_excerpt=relevant_excerpt,
        supported_dependencies=supported_dependencies or [],
        used_by=used_by or [],
        usage_basis=usage_basis,
        metadata={
            "status_code": result.status_code,
            "bytes_count": result.bytes_count,
            "headers": result.headers,
        },
    )
