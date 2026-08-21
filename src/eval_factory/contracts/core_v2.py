from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from eval_factory.contracts.core import ContractModel, ObjectRef


def canonical_value_v2(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return canonical_value_v2(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, dict):
        return {
            str(key): canonical_value_v2(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_value_v2(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [canonical_value_v2(item) for item in value]
    if isinstance(value, datetime):
        rendered = value.isoformat()
        return rendered.removesuffix("+00:00") + "Z" if rendered.endswith("+00:00") else rendered
    if isinstance(value, Enum):
        return canonical_value_v2(value.value)
    return value


def canonical_json_v2(value: BaseModel) -> bytes:
    return json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def canonical_sha256_v2(value: BaseModel) -> str:
    return hashlib.sha256(canonical_json_v2(value)).hexdigest()


def require_matching_hash_v2(reference: ObjectRef, value: BaseModel) -> None:
    observed = canonical_sha256_v2(value)
    if observed != reference.object_sha256:
        raise ValueError(
            f"stale or mismatched v2 object reference: expected "
            f"{reference.object_sha256}, observed {observed}"
        )


class ContractModelV2(ContractModel):
    """Closed immutable contract with process-stable unordered-container hashing."""

    def canonical_json(self) -> bytes:
        return canonical_json_v2(self)

    def canonical_sha256(self) -> str:
        return canonical_sha256_v2(self)
