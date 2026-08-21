from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import ExecutionTelemetryV2, FacadeObjectRef
from env_mock_agent.facade.execution_v2 import (
    AttachmentExecutionFailureCodeV2,
    AttachmentExecutionResultV2,
    AttachmentExecutionStatusV2,
    WorldLedgerFactLockV2,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def test_world_ledger_fact_lock_is_content_free_and_hash_bound() -> None:
    lock = WorldLedgerFactLockV2(
        fact_id="world-fact://organization",
        value_ref=_ref("world-fact-value", "organization"),
        value_sha256=HASH,
        source_refs=(_ref("source-evidence", "organization"),),
    )

    assert lock.value_ref.object_sha256 == lock.value_sha256
    serialized = lock.model_dump(mode="json")
    assert "value" not in serialized
    assert "key" not in serialized

    with pytest.raises(ValidationError, match="value"):
        WorldLedgerFactLockV2.model_validate(
            {
                **serialized,
                "value_sha256": OTHER_HASH,
            }
        )

    with pytest.raises(ValidationError, match="extra"):
        WorldLedgerFactLockV2.model_validate(
            {
                **serialized,
                "raw_value": "Example Co",
            }
        )


def test_attachment_execution_result_enforces_success_failure_matrix() -> None:
    request_ref = _ref("attachment-execution-request", "artifact")
    snapshot_ref = _ref("world-ledger-snapshot", "current")
    output_ref = _ref("attachment-output", "artifact")
    success = AttachmentExecutionResultV2(
        execution_result_id="attachment-execution-result://artifact/1",
        execution_request_ref=request_ref,
        artifact_id="artifact://input",
        attempt=1,
        status=AttachmentExecutionStatusV2.SUCCEEDED,
        world_ledger_snapshot_ref=snapshot_ref,
        selected_route_kind="PROVIDER",
        selected_route_id="text",
        worker_version="env_mock_agent.providers.text.TextProvider",
        output_ref=output_ref,
        output_sha256=HASH,
        retryable=False,
        failure_code=None,
        telemetry=ExecutionTelemetryV2.unavailable(observed_at=datetime(2026, 7, 28, tzinfo=UTC)),
        policy_version="artifact-execution/r5-06-v1",
        execution_result_sha256=HASH,
    )

    assert success.output_ref == output_ref
    assert success.failure_code is None

    failed_values = success.model_dump(mode="python")
    failed_values.update(
        status=AttachmentExecutionStatusV2.RETRYABLE_FAILURE.value,
        output_ref=None,
        output_sha256=None,
        worker_version=None,
        retryable=True,
        failure_code=AttachmentExecutionFailureCodeV2.RUNTIME_TIMEOUT.value,
    )
    failed = AttachmentExecutionResultV2.model_validate(failed_values)
    assert failed.retryable is True

    with pytest.raises(ValidationError, match="output"):
        AttachmentExecutionResultV2.model_validate(
            {
                **failed_values,
                "output_ref": output_ref,
                "output_sha256": HASH,
            }
        )

    with pytest.raises(ValidationError, match="failure"):
        AttachmentExecutionResultV2.model_validate(
            {
                **success.model_dump(mode="python"),
                "failure_code": AttachmentExecutionFailureCodeV2.OUTPUT_MISSING.value,
            }
        )


def test_execution_result_rejects_unknown_diagnostic_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        AttachmentExecutionResultV2.model_validate(
            {
                "execution_result_id": "attachment-execution-result://artifact/1",
                "execution_request_ref": _ref(
                    "attachment-execution-request",
                    "artifact",
                ).model_dump(mode="json"),
                "artifact_id": "artifact://input",
                "attempt": 1,
                "status": "TERMINAL_FAILURE",
                "world_ledger_snapshot_ref": _ref(
                    "world-ledger-snapshot",
                    "current",
                ).model_dump(mode="json"),
                "selected_route_kind": "PROVIDER",
                "selected_route_id": "text",
                "worker_version": None,
                "output_ref": None,
                "output_sha256": None,
                "retryable": False,
                "failure_code": "PROVIDER_EXECUTION_FAILED",
                "policy_version": "artifact-execution/r5-06-v1",
                "execution_result_sha256": HASH,
                "exception_message": "private provider detail",
                "created_at": datetime(2026, 7, 28, tzinfo=UTC).isoformat(),
            }
        )
