from __future__ import annotations

import hashlib
import json


def operation_key(
    run_id: str,
    operation: str,
    *,
    artifact_id: str = "",
    inputs: object = None,
) -> str:
    serialized = json.dumps(inputs, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    raw = f"{run_id}:{artifact_id}:{operation}:{digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
