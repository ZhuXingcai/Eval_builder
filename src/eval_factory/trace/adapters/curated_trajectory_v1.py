from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from eval_factory.trace.models import (
    TraceProbeDiagnostic,
    TraceProbeResultV2,
    TraceProbeStatus,
)

CURATED_TRAJECTORY_V1_REQUIRED_FIELDS = frozenset(
    {
        "messages",
        "system",
    }
)


def _diagnostic(
    code: str,
    message: str,
) -> TraceProbeDiagnostic:
    return TraceProbeDiagnostic(
        code=code,
        message=message,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _source_uri(path: Path) -> str:
    try:
        return path.expanduser().resolve(strict=False).as_uri()
    except ValueError:
        return f"file://{path.expanduser().absolute()}"


class CuratedTrajectoryV1Adapter:
    name: Literal["curated_trajectory_v1"] = "curated_trajectory_v1"
    version: Literal["1.0.0"] = "1.0.0"

    def probe(self, source: Path) -> TraceProbeResultV2:
        path = source.expanduser().absolute()
        source_uri = _source_uri(path)
        preflight = self._preflight(path, source_uri)
        if preflight is not None:
            return preflight
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
        except (OSError, PermissionError) as exc:
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-read-error",
                        f"cannot read logical trajectory member: {type(exc).__name__}",
                    ),
                ),
            )

        raw_sha256 = hashlib.sha256(raw).hexdigest()
        fields: tuple[str, ...] = ()
        diagnostics: tuple[TraceProbeDiagnostic, ...] = ()
        try:
            value = json.loads(
                raw,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            diagnostics = (
                _diagnostic(
                    "curated-trajectory-json-invalid",
                    f"logical trajectory JSON is invalid: {type(exc).__name__}",
                ),
            )
        else:
            fields, diagnostics = validate_curated_trajectory_v1(value)

        if _changed_during_probe(before, after, len(raw)):
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                raw_sha256=raw_sha256,
                size_bytes=len(raw),
                outer_fields=fields,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-source-changed",
                        "logical trajectory member changed while it was being probed",
                    ),
                ),
            )
        return self._result(
            source_uri=source_uri,
            status=(TraceProbeStatus.INVALID if diagnostics else TraceProbeStatus.SUPPORTED),
            raw_sha256=raw_sha256,
            size_bytes=len(raw),
            record_count=1,
            outer_fields=fields,
            diagnostics=diagnostics,
        )

    def _preflight(
        self,
        path: Path,
        source_uri: str,
    ) -> TraceProbeResultV2 | None:
        if not path.exists():
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-source-missing",
                        "logical trajectory member does not exist",
                    ),
                ),
            )
        if path.is_symlink():
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-symlink-unsupported",
                        "logical trajectory member must not be a symbolic link",
                    ),
                ),
            )
        if not path.is_file():
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-not-regular-file",
                        "logical trajectory member must be a regular file",
                    ),
                ),
            )
        if path.suffix.casefold() != ".json":
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "curated-trajectory-extension-mismatch",
                        "logical trajectory members must use the .json extension",
                    ),
                ),
            )
        return None

    def _result(
        self,
        *,
        source_uri: str,
        status: TraceProbeStatus,
        raw_sha256: str | None = None,
        size_bytes: int | None = None,
        record_count: int = 0,
        outer_fields: tuple[str, ...] = (),
        diagnostics: tuple[TraceProbeDiagnostic, ...] = (),
    ) -> TraceProbeResultV2:
        return TraceProbeResultV2(
            adapter_name=self.name,
            adapter_version=self.version,
            source_uri=source_uri,
            status=status,
            raw_sha256=raw_sha256,
            size_bytes=size_bytes,
            record_count=record_count,
            outer_fields=outer_fields,
            diagnostics=diagnostics,
        )


def validate_curated_trajectory_v1(
    value: object,
) -> tuple[tuple[str, ...], tuple[TraceProbeDiagnostic, ...]]:
    if not isinstance(value, Mapping):
        return (), (
            _diagnostic(
                "curated-trajectory-not-object",
                "logical trajectory must be a JSON object",
            ),
        )
    fields = tuple(sorted(str(key) for key in value))
    if set(fields) != CURATED_TRAJECTORY_V1_REQUIRED_FIELDS:
        return fields, (
            _diagnostic(
                "curated-trajectory-fields-mismatch",
                "logical trajectory fields differ from the v1 schema",
            ),
        )
    system = value.get("system")
    messages = value.get("messages")
    if not isinstance(system, str) or not system:
        return fields, (
            _diagnostic(
                "curated-trajectory-system-invalid",
                "logical trajectory system field must be a non-empty string",
            ),
        )
    if not isinstance(messages, list) or not messages:
        return fields, (
            _diagnostic(
                "curated-trajectory-messages-invalid",
                "logical trajectory messages field must be a non-empty array",
            ),
        )
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or set(message) != {"content", "role"}
            or message.get("role") not in {"assistant", "user"}
            or not isinstance(message.get("content"), list)
            or any(not isinstance(block, Mapping) for block in message["content"])
        ):
            return fields, (
                _diagnostic(
                    "curated-trajectory-message-invalid",
                    "logical trajectory contains an invalid message",
                ),
            )
    return fields, ()


def _changed_during_probe(
    before: os.stat_result,
    after: os.stat_result,
    observed_size: int,
) -> bool:
    return (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or after.st_size != observed_size
    )


__all__ = [
    "CURATED_TRAJECTORY_V1_REQUIRED_FIELDS",
    "CuratedTrajectoryV1Adapter",
    "validate_curated_trajectory_v1",
]
