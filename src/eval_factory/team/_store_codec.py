from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ValidationError

from eval_factory.contracts.core_v2 import canonical_json_v2, canonical_value_v2
from eval_factory.team._store_types import TeamIntegrityError


def record_json(value: BaseModel) -> str:
    return canonical_json_v2(value).decode()


def parse_record[RecordT: BaseModel](
    model_type: type[RecordT],
    payload: str,
    label: str,
) -> RecordT:
    try:
        value = model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise TeamIntegrityError(f"{label} is invalid") from exc
    if record_json(value) != payload:
        raise TeamIntegrityError(f"{label} is not canonical")
    return value


def request_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["parse_record", "record_json", "request_sha256"]
