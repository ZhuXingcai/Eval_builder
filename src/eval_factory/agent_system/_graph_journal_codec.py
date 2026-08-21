from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from eval_factory.agent_system._graph_journal_types import (
    FactoryGraphJournalIntegrityError,
)
from eval_factory.contracts.core import ContractModel


def record_json(value: ContractModel) -> str:
    return value.canonical_json().decode()


def parse_record[ModelT: ContractModel](
    model_type: type[ModelT],
    raw: str,
    label: str,
) -> ModelT:
    try:
        value = model_type.model_validate_json(raw)
    except ValidationError as exc:
        raise FactoryGraphJournalIntegrityError(
            f"{label} record is malformed",
        ) from exc
    if record_json(value) != raw:
        raise FactoryGraphJournalIntegrityError(
            f"{label} record is not canonical",
        )
    return value


def request_sha256(*values: object) -> str:
    encoded = json.dumps(
        values,
        default=lambda value: (
            value.model_dump(mode="json", exclude_none=False)
            if isinstance(value, ContractModel)
            else str(value)
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["parse_record", "record_json", "request_sha256"]
