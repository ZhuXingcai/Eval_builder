from __future__ import annotations

from datetime import datetime

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.runtime_v2 import (
    RuntimeEventKindV2,
    RuntimeFailureCodeV2,
    UnifiedRuntimeEventV2,
    UnifiedRuntimeUsageV2,
)
from env_mock_agent.facade.telemetry_v2 import ReportedCostV2
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationResultV2,
    GatewayInvocationStatusV2,
    GatewayReceiptV2,
)
from eval_factory.contracts.core import ObjectRef


class GatewayRuntimeProtocolError(RuntimeError):
    pass


class GatewayRuntimeEventProjector:
    @staticmethod
    def project(
        *,
        run_id: str,
        work_id: str,
        runtime_id: str,
        runtime_version: str | None,
        receipt: GatewayReceiptV2,
        result: GatewayInvocationResultV2,
        started_at: datetime,
    ) -> tuple[UnifiedRuntimeEventV2, ...]:
        if (
            result.receipt_ref != receipt.to_ref()
            or result.invocation_request_ref != receipt.invocation_request_ref
            or result.route_decision_ref != receipt.route_decision_ref
            or result.status is not receipt.status
            or result.failure_code != receipt.failure_code
        ):
            raise GatewayRuntimeProtocolError(
                "Gateway receipt/result closure is inconsistent",
            )
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise GatewayRuntimeProtocolError(
                "Gateway runtime start timestamp must be timezone-aware",
            )
        if started_at > receipt.completed_at:
            raise GatewayRuntimeProtocolError(
                "Gateway runtime completion precedes its start",
            )

        events = [
            UnifiedRuntimeEventV2.create(
                run_id=run_id,
                work_id=work_id,
                runtime_id=runtime_id,
                runtime_version=runtime_version,
                sequence=1,
                kind=RuntimeEventKindV2.RUN_STARTED,
                occurred_at=started_at,
                source_refs=(
                    _facade_ref(receipt.invocation_request_ref),
                    _facade_ref(receipt.route_decision_ref),
                ),
            ),
            UnifiedRuntimeEventV2.create(
                run_id=run_id,
                work_id=work_id,
                runtime_id=runtime_id,
                runtime_version=runtime_version,
                sequence=2,
                kind=RuntimeEventKindV2.MODEL_REQUEST_STARTED,
                occurred_at=started_at,
                source_refs=(
                    _facade_ref(receipt.invocation_request_ref),
                    _facade_ref(receipt.route_decision_ref),
                ),
            ),
        ]
        sequence = 3
        if result.status is GatewayInvocationStatusV2.SUCCEEDED:
            assert result.output_ref is not None
            events.append(
                UnifiedRuntimeEventV2.create(
                    run_id=run_id,
                    work_id=work_id,
                    runtime_id=runtime_id,
                    runtime_version=runtime_version,
                    sequence=sequence,
                    kind=RuntimeEventKindV2.MODEL_RESPONSE_AVAILABLE,
                    occurred_at=receipt.completed_at,
                    source_refs=(
                        _facade_ref(result.to_ref()),
                        _facade_ref(result.output_ref),
                    ),
                )
            )
            sequence += 1
        events.append(
            UnifiedRuntimeEventV2.create(
                run_id=run_id,
                work_id=work_id,
                runtime_id=runtime_id,
                runtime_version=runtime_version,
                sequence=sequence,
                kind=RuntimeEventKindV2.USAGE_REPORTED,
                occurred_at=receipt.completed_at,
                source_refs=(
                    _facade_ref(receipt.to_ref()),
                    _facade_ref(result.to_ref()),
                ),
                usage=_usage(receipt),
            )
        )
        sequence += 1
        terminal_kind = (
            RuntimeEventKindV2.RUN_COMPLETED
            if result.status is GatewayInvocationStatusV2.SUCCEEDED
            else RuntimeEventKindV2.RUN_FAILED
        )
        events.append(
            UnifiedRuntimeEventV2.create(
                run_id=run_id,
                work_id=work_id,
                runtime_id=runtime_id,
                runtime_version=runtime_version,
                sequence=sequence,
                kind=terminal_kind,
                occurred_at=receipt.completed_at,
                source_refs=(
                    _facade_ref(receipt.to_ref()),
                    _facade_ref(result.to_ref()),
                ),
                failure_code=(
                    None
                    if terminal_kind is RuntimeEventKindV2.RUN_COMPLETED
                    else _failure_code(result.failure_code)
                ),
            )
        )
        return tuple(events)


def _usage(receipt: GatewayReceiptV2) -> UnifiedRuntimeUsageV2:
    usage = receipt.usage
    return UnifiedRuntimeUsageV2(
        model_requests=1,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        tool_calls=0,
        turns=1,
        duration_ms=None,
        reported_cost=(
            ReportedCostV2.reported(
                amount_microusd=usage.reported_cost_micro_usd,
            )
            if usage.reported_cost_micro_usd is not None
            else ReportedCostV2.unavailable()
        ),
    )


def _failure_code(value: str | None) -> RuntimeFailureCodeV2:
    return {
        "PROVIDER_AUTHENTICATION_FAILED": (RuntimeFailureCodeV2.AUTHENTICATION_FAILED),
        "MODEL_PROVIDER_RATE_LIMITED": RuntimeFailureCodeV2.RATE_LIMITED,
        "MODEL_PROVIDER_OVERLOADED": RuntimeFailureCodeV2.OVERLOADED,
        "MODEL_CONTEXT_LIMIT": RuntimeFailureCodeV2.CONTEXT_OVERFLOW,
        "MODEL_INVALID_REQUEST": RuntimeFailureCodeV2.INVALID_REQUEST,
        "MODEL_UNAVAILABLE": RuntimeFailureCodeV2.MODEL_UNAVAILABLE,
        "MODEL_PROVIDER_TIMEOUT": RuntimeFailureCodeV2.TIMEOUT,
        "MODEL_PROVIDER_FAILED": RuntimeFailureCodeV2.PROVIDER_FAILED,
    }.get(value or "", RuntimeFailureCodeV2.PROVIDER_FAILED)


def _facade_ref(value: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


__all__ = [
    "GatewayRuntimeEventProjector",
    "GatewayRuntimeProtocolError",
]
