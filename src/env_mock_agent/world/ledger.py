from __future__ import annotations

from env_mock_agent.schemas import SourceEvidence, WorldLedger


def build_world_ledger(
    evidence: list[SourceEvidence],
    *,
    locked_facts: dict[str, object] | None = None,
) -> WorldLedger:
    entities: dict[str, dict[str, object]] = {}
    for source in evidence:
        entities[source.source_id] = {
            "title": source.title,
            "publisher": source.publisher,
            "source_type": source.source_type.value,
            "sha256": source.sha256,
            "supported_dependencies": source.supported_dependencies,
        }
    return WorldLedger(
        entities=entities,
        locked_facts=locked_facts or {},
    )


def validate_locked_facts(
    ledger: WorldLedger,
    candidate: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    for key, expected in ledger.locked_facts.items():
        if key in candidate and candidate[key] != expected:
            errors.append(f"locked fact mismatch for {key}: expected {expected!r}, got {candidate[key]!r}")
    return errors
