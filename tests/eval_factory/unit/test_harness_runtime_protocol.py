from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from env_mock_agent.facade.runtime_v2 import (
    RuntimeEventKindV2,
    RuntimeFailureCodeV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    GatewayReceiptV2,
    GatewayUsageV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.harness.runtime_protocol import (
    GatewayRuntimeEventProjector,
    GatewayRuntimeProtocolError,
)

NOW = datetime(2026, 8, 23, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="stage5-runtime-protocol-test",
        governing_versions=(
            VersionBinding(
                component="stage5-runtime-protocol",
                version="v1",
                sha256=_digest("stage5-runtime-protocol-v1"),
            ),
        ),
    )


def _closure(
    status: GatewayInvocationStatusV2,
    *,
    failure_code: str | None = None,
) -> tuple[GatewayReceiptV2, GatewayInvocationResultV2]:
    invocation_ref = _ref("gateway-invocation-request", "request")
    route_ref = _ref("model-route-decision", "route")
    receipt = GatewayReceiptV2.create(
        receipt_id="gateway-receipt://runtime-protocol",
        invocation_request_ref=invocation_ref,
        route_decision_ref=route_ref,
        status=status,
        response_body_ref=(
            _ref("model-response-content", "response")
            if status is GatewayInvocationStatusV2.SUCCEEDED
            else None
        ),
        usage=GatewayUsageV2(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
            charged_tokens=135,
            reported_cost_micro_usd=250,
        ),
        failure_code=failure_code,
        completed_at=NOW,
        audit=_audit(),
    )
    result = GatewayInvocationResultV2.create(
        invocation_result_id="gateway-invocation-result://runtime-protocol",
        invocation_request_ref=invocation_ref,
        route_decision_ref=route_ref,
        receipt_ref=receipt.to_ref(),
        status=status,
        output_ref=(
            _ref("requirement-proposal", "output") if status is GatewayInvocationStatusV2.SUCCEEDED else None
        ),
        failure_code=failure_code,
        audit=_audit(),
    )
    return receipt, result


def test_gateway_success_projects_common_runtime_event_family() -> None:
    receipt, result = _closure(GatewayInvocationStatusV2.SUCCEEDED)

    events = GatewayRuntimeEventProjector.project(
        run_id="run-stage5",
        work_id="requirement-stage5",
        runtime_id="gateway",
        runtime_version="v1",
        receipt=receipt,
        result=result,
        started_at=NOW,
    )

    assert [event.kind for event in events] == [
        RuntimeEventKindV2.RUN_STARTED,
        RuntimeEventKindV2.MODEL_REQUEST_STARTED,
        RuntimeEventKindV2.MODEL_RESPONSE_AVAILABLE,
        RuntimeEventKindV2.USAGE_REPORTED,
        RuntimeEventKindV2.RUN_COMPLETED,
    ]
    assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
    assert events[3].usage is not None
    assert events[3].usage.reported_cost.amount_microusd == 250
    assert events[-1].failure_code is None
    assert "model-response-content" not in events[-1].model_dump_json()


@pytest.mark.parametrize(
    ("gateway_code", "runtime_code"),
    [
        ("PROVIDER_AUTHENTICATION_FAILED", RuntimeFailureCodeV2.AUTHENTICATION_FAILED),
        ("MODEL_PROVIDER_RATE_LIMITED", RuntimeFailureCodeV2.RATE_LIMITED),
        ("MODEL_PROVIDER_OVERLOADED", RuntimeFailureCodeV2.OVERLOADED),
        ("MODEL_CONTEXT_LIMIT", RuntimeFailureCodeV2.CONTEXT_OVERFLOW),
        ("MODEL_PROVIDER_FAILED", RuntimeFailureCodeV2.PROVIDER_FAILED),
    ],
)
def test_gateway_failures_use_closed_runtime_codes(
    gateway_code: str,
    runtime_code: RuntimeFailureCodeV2,
) -> None:
    receipt, result = _closure(
        GatewayInvocationStatusV2.FAILED,
        failure_code=gateway_code,
    )

    events = GatewayRuntimeEventProjector.project(
        run_id="run-stage5",
        work_id="requirement-stage5",
        runtime_id="gateway",
        runtime_version="v1",
        receipt=receipt,
        result=result,
        started_at=NOW,
    )

    assert [event.kind for event in events] == [
        RuntimeEventKindV2.RUN_STARTED,
        RuntimeEventKindV2.MODEL_REQUEST_STARTED,
        RuntimeEventKindV2.USAGE_REPORTED,
        RuntimeEventKindV2.RUN_FAILED,
    ]
    assert events[-1].failure_code is runtime_code


def test_gateway_projector_rejects_mismatched_closure() -> None:
    receipt, result = _closure(GatewayInvocationStatusV2.SUCCEEDED)
    changed = result.model_copy(
        update={"receipt_ref": _ref("gateway-receipt", "other")},
    )

    with pytest.raises(GatewayRuntimeProtocolError, match="closure"):
        GatewayRuntimeEventProjector.project(
            run_id="run-stage5",
            work_id="requirement-stage5",
            runtime_id="gateway",
            runtime_version="v1",
            receipt=receipt,
            result=changed,
            started_at=NOW,
        )
