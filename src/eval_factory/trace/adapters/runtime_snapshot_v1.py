from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval_factory.trace.models import (
    TraceProbeDiagnostic,
    TraceProbeResultV2,
    TraceProbeStatus,
)

RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS = frozenset(
    {
        "sid",
        "event_time",
        "api_type",
        "business",
        "real_model",
        "request_model",
        "request",
        "response",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _source_uri(path: Path) -> str:
    try:
        return path.expanduser().resolve(strict=False).as_uri()
    except ValueError:
        return f"file://{path.expanduser().absolute()}"


def _diagnostic(
    code: str,
    message: str,
) -> TraceProbeDiagnostic:
    return TraceProbeDiagnostic(
        code=code,
        message=message,
    )


class RuntimeSnapshotV1Adapter:
    name = "runtime_snapshot_v1"
    version = "1.0.0"

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
                        "runtime-snapshot-read-error",
                        f"cannot read trace source: {type(exc).__name__}",
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
                    "runtime-snapshot-outer-json-invalid",
                    f"outer JSON record is invalid: {type(exc).__name__}",
                ),
            )
        else:
            if not isinstance(value, dict):
                diagnostics = (
                    _diagnostic(
                        "runtime-snapshot-outer-not-object",
                        "runtime snapshot outer record must be an object",
                    ),
                )
            else:
                fields = tuple(sorted(str(key) for key in value))
                observed_fields = set(fields)
                if observed_fields != RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS:
                    diagnostics = (
                        _diagnostic(
                            "runtime-snapshot-envelope-fields-mismatch",
                            "runtime snapshot outer fields differ from the v1 schema",
                        ),
                    )
                else:
                    invalid_types = tuple(
                        sorted(
                            field
                            for field in RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS
                            if not isinstance(value.get(field), str)
                        )
                    )
                    if invalid_types:
                        diagnostics = (
                            _diagnostic(
                                "runtime-snapshot-envelope-field-type-invalid",
                                "runtime snapshot outer fields have invalid types",
                            ),
                        )

        if self._changed_during_probe(before, after, len(raw)):
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNREADABLE,
                raw_sha256=raw_sha256,
                size_bytes=len(raw),
                outer_fields=fields,
                diagnostics=(
                    _diagnostic(
                        "runtime-snapshot-source-changed",
                        "source metadata changed while it was being probed",
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
                        "runtime-snapshot-source-missing",
                        "trace source does not exist",
                    ),
                ),
            )
        if path.is_symlink():
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "runtime-snapshot-symlink-unsupported",
                        "trace source must be a regular file, not a symbolic link",
                    ),
                ),
            )
        if not path.is_file():
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "runtime-snapshot-not-regular-file",
                        "trace source must be a regular file",
                    ),
                ),
            )
        if path.suffix.casefold() != ".json":
            return self._result(
                source_uri=source_uri,
                status=TraceProbeStatus.UNSUPPORTED,
                diagnostics=(
                    _diagnostic(
                        "runtime-snapshot-extension-mismatch",
                        "runtime snapshot v1 sources must use the .json extension",
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

    @staticmethod
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
