from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.model_control_adapter import (
    RegistryModelControlFacade,
    _non_negative_int,
)
from env_mock_agent.facade.model_control_v2 import (
    ATTACHMENT_MODEL_CONTROL_POLICY_VERSION,
    ModelControlGrantV2,
    ModelControlReceiptOutcomeV2,
    ModelControlReceiptV2,
    ModelControlUsageSourceV2,
    ModelControlUsageV2,
    ModelDemandModeFacadeV2,
    ModelInvocationCancellationOutcomeV2,
    ModelInvocationCancellationRequestV2,
    ModelInvocationCancellationResultV2,
    ModelInvocationDescriptorV2,
    ModelProviderBackpressureKindV2,
    ModelProviderBackpressureV2,
    model_control_grant_ref,
    model_invocation_cancellation_request_ref,
    model_invocation_descriptor_ref,
    model_provider_backpressure_ref,
)
from env_mock_agent.models.base import ModelClientBackpressureError

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "current", *, version: str = "v2") -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-03/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _descriptor() -> ModelInvocationDescriptorV2:
    return ModelInvocationDescriptorV2.create(
        operation_ref=_ref("semantic-residual-request"),
        model_profile_ref=_ref("model-profile", version="v1"),
        mode=ModelDemandModeFacadeV2.DIRECT_REQUEST,
        request_allowance=1,
        input_token_allowance=1000,
        output_token_allowance=2000,
        cache_token_allowance=500,
        idempotency_key="model-invocation-r6-03",
    )


def _grant(descriptor: ModelInvocationDescriptorV2 | None = None) -> ModelControlGrantV2:
    descriptor = descriptor or _descriptor()
    return ModelControlGrantV2.create(
        reservation_ref=_ref("work-model-reservation"),
        invocation_handle_ref=_ref("model-invocation-handle", version="v1"),
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        model_profile_ref=descriptor.model_profile_ref,
        fencing_token=2,
        request_allowance=descriptor.request_allowance,
        token_allowance=descriptor.token_allowance,
        output_token_limit=descriptor.output_token_allowance,
    )


def _usage() -> ModelControlUsageV2:
    return ModelControlUsageV2(
        requests=1,
        input_tokens=100,
        output_tokens=200,
        cache_creation_input_tokens=10,
        cache_read_input_tokens=20,
        charged_tokens=330,
        source=ModelControlUsageSourceV2.REPORTED,
    )


def test_descriptor_and_grant_are_strict_operation_bound_contracts() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)

    assert descriptor.policy_version == ATTACHMENT_MODEL_CONTROL_POLICY_VERSION
    assert grant.descriptor_ref == model_invocation_descriptor_ref(descriptor)
    assert model_control_grant_ref(grant).object_sha256 == grant.model_control_grant_sha256
    grant.validate_descriptor(descriptor)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelControlGrantV2.model_validate(
            {
                **grant.model_dump(mode="python"),
                "api_key": "secret",
            }
        )


def test_grant_rejects_cross_descriptor_and_usage_overrun() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    other = descriptor.model_copy(
        update={
            "operation_ref": _ref("task-draft-authoring-request", "other"),
            "invocation_descriptor_id": "model-invocation-descriptor://pending",
            "invocation_descriptor_sha256": "0" * 64,
        }
    )
    other = ModelInvocationDescriptorV2.create(
        operation_ref=other.operation_ref,
        model_profile_ref=other.model_profile_ref,
        mode=other.mode,
        request_allowance=other.request_allowance,
        input_token_allowance=other.input_token_allowance,
        output_token_allowance=other.output_token_allowance,
        cache_token_allowance=other.cache_token_allowance,
        idempotency_key="model-invocation-other",
    )

    with pytest.raises(ValueError, match="descriptor"):
        grant.validate_descriptor(other)
    with pytest.raises(ValueError, match="token"):
        grant.validate_usage(
            ModelControlUsageV2(
                requests=1,
                input_tokens=1000,
                output_tokens=2000,
                cache_creation_input_tokens=500,
                cache_read_input_tokens=1,
                charged_tokens=3501,
                source=ModelControlUsageSourceV2.REPORTED,
            )
        )


def test_success_receipt_requires_result_and_complete_usage() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    result_ref = _ref("semantic-residual-result")
    receipt = ModelControlReceiptV2.create(
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        grant_ref=model_control_grant_ref(grant),
        outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
        result_ref=result_ref,
        usage=_usage(),
        backpressure_ref=None,
        failure_code=None,
        completed_at=NOW,
    )

    assert receipt.result_ref == result_ref
    with pytest.raises(ValidationError, match="successful"):
        ModelControlReceiptV2.create(
            descriptor_ref=model_invocation_descriptor_ref(descriptor),
            grant_ref=model_control_grant_ref(grant),
            outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
            result_ref=None,
            usage=_usage(),
            backpressure_ref=None,
            failure_code=None,
            completed_at=NOW,
        )


def test_backpressure_receipt_is_typed_and_content_free() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    usage = ModelControlUsageV2(
        requests=1,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelControlUsageSourceV2.REPORTED,
    )
    backpressure = ModelProviderBackpressureV2.create(
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        kind=ModelProviderBackpressureKindV2.RATE_LIMITED,
        retry_at=NOW + timedelta(seconds=30),
        observed_at=NOW,
    )
    receipt = ModelControlReceiptV2.create(
        descriptor_ref=model_invocation_descriptor_ref(descriptor),
        grant_ref=model_control_grant_ref(grant),
        outcome=ModelControlReceiptOutcomeV2.BACKPRESSURED,
        result_ref=None,
        usage=usage,
        backpressure_ref=model_provider_backpressure_ref(backpressure),
        failure_code=None,
        completed_at=NOW,
    )

    assert receipt.backpressure_ref == model_provider_backpressure_ref(backpressure)
    assert "retry_at" not in receipt.model_dump(mode="json")


def test_cancellation_result_binds_exact_request() -> None:
    grant = _grant()
    request = ModelInvocationCancellationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        invocation_handle_ref=grant.invocation_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )
    result = ModelInvocationCancellationResultV2.create(
        cancellation_request_ref=model_invocation_cancellation_request_ref(request),
        outcome=ModelInvocationCancellationOutcomeV2.CANCELLED,
        completed_at=NOW,
    )

    assert result.cancellation_request_ref == model_invocation_cancellation_request_ref(request)


class _StructuredClient:
    def __init__(self) -> None:
        self.calls = 0
        self.max_output_tokens: int | None = None

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        self.max_output_tokens = max_output_tokens
        return (
            {"ok": bool(system and prompt and schema)},
            {
                "input_tokens": 100,
                "output_tokens": 200,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            },
        )


class _BackpressureClient(_StructuredClient):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        raise ModelClientBackpressureError(
            kind="RATE_LIMITED",
            retry_after_seconds=30,
        )


class _OverrunClient(_StructuredClient):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        return (
            {"must_not_escape": True},
            {
                "input_tokens": 2000,
                "output_tokens": 2000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )


class _FailingClient(_StructuredClient):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        raise RuntimeError("raw provider failure must not cross the facade")


class _BlockingClient(_StructuredClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocking client should be cancelled")


@pytest.mark.asyncio
async def test_structured_adapter_enforces_grant_and_normalizes_usage() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    client = _StructuredClient()
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: client},
        clock=lambda: NOW,
    )
    result_ref = _ref("semantic-residual-result")

    output, receipt = await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=result_ref,
    )
    replay = await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=result_ref,
    )

    assert replay == (output, receipt)
    assert output == {"ok": True}
    assert receipt.outcome is ModelControlReceiptOutcomeV2.SUCCEEDED
    assert receipt.usage.charged_tokens == 330
    assert client.calls == 1
    assert client.max_output_tokens == descriptor.output_token_allowance
    with pytest.raises(ValueError, match="idempotency conflict"):
        await facade.generate_json_with_model_control(
            descriptor=descriptor,
            grant=grant,
            system="stable system",
            prompt="bounded prompt",
            schema={"type": "object"},
            result_ref=_ref("semantic-residual-result", "different"),
        )


@pytest.mark.asyncio
async def test_structured_adapter_blocks_missing_profile_before_model_call() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    facade = RegistryModelControlFacade({}, clock=lambda: NOW)

    output, receipt = await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=_ref("semantic-residual-result"),
    )

    assert output is None
    assert receipt.outcome is ModelControlReceiptOutcomeV2.BLOCKED_CAPABILITY
    assert receipt.failure_code == "MODEL_PROFILE_UNAVAILABLE"
    assert receipt.usage.charged_tokens == 0


@pytest.mark.asyncio
async def test_structured_adapter_rejects_opaque_envelope_before_model_call() -> None:
    baseline = _descriptor()
    descriptor = ModelInvocationDescriptorV2.create(
        operation_ref=baseline.operation_ref,
        model_profile_ref=baseline.model_profile_ref,
        mode=ModelDemandModeFacadeV2.OPAQUE_RUNTIME_ENVELOPE,
        request_allowance=2,
        input_token_allowance=baseline.input_token_allowance,
        output_token_allowance=baseline.output_token_allowance,
        cache_token_allowance=baseline.cache_token_allowance,
        idempotency_key="opaque-on-structured-client",
    )
    client = _StructuredClient()
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: client},
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="DIRECT_REQUEST"):
        await facade.generate_json_with_model_control(
            descriptor=descriptor,
            grant=_grant(descriptor),
            system="stable system",
            prompt="bounded prompt",
            schema={"type": "object"},
            result_ref=_ref("semantic-residual-result"),
        )

    assert client.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client", "expected_outcome", "expected_failure", "expected_tokens"),
    (
        (
            _OverrunClient(),
            ModelControlReceiptOutcomeV2.BLOCKED_POLICY,
            "MODEL_USAGE_OVERRUN",
            4000,
        ),
        (
            _FailingClient(),
            ModelControlReceiptOutcomeV2.FAILED,
            "MODEL_CALL_FAILED",
            3500,
        ),
    ),
)
async def test_structured_adapter_blocks_overrun_and_sanitizes_failures(
    client: _StructuredClient,
    expected_outcome: ModelControlReceiptOutcomeV2,
    expected_failure: str,
    expected_tokens: int,
) -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: client},
        clock=lambda: NOW,
    )

    output, receipt = await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=_ref("semantic-residual-result"),
    )

    assert output is None
    assert receipt.outcome is expected_outcome
    assert receipt.failure_code == expected_failure
    assert receipt.usage.charged_tokens == expected_tokens
    assert "raw provider failure" not in str(receipt.model_dump(mode="json"))


def test_model_usage_normalizer_accepts_integral_values_and_rejects_invalid_values() -> None:
    assert _non_negative_int("12") == 12
    assert _non_negative_int(3.0) == 3
    for invalid in (True, "not-an-int", 1.5, object(), -1):
        with pytest.raises(ValueError, match="model usage"):
            _non_negative_int(invalid)


@pytest.mark.asyncio
async def test_structured_adapter_converts_typed_backpressure_without_fallback() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    client = _BackpressureClient()
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: client},
        clock=lambda: NOW,
    )

    output, receipt = await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=_ref("semantic-residual-result"),
    )

    assert output is None
    assert receipt.outcome is ModelControlReceiptOutcomeV2.BACKPRESSURED
    assert receipt.backpressure_ref is not None
    assert receipt.failure_code is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_structured_adapter_cancels_only_exact_active_handle() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    client = _BlockingClient()
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: client},
        clock=lambda: NOW,
    )
    invocation = asyncio.create_task(
        facade.generate_json_with_model_control(
            descriptor=descriptor,
            grant=grant,
            system="stable system",
            prompt="bounded prompt",
            schema={"type": "object"},
            result_ref=_ref("semantic-residual-result"),
        )
    )
    await client.started.wait()
    request = ModelInvocationCancellationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        invocation_handle_ref=grant.invocation_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    result = await facade.cancel_model_invocation(request)

    assert result.outcome is ModelInvocationCancellationOutcomeV2.CANCELLED
    with pytest.raises(asyncio.CancelledError):
        await invocation


@pytest.mark.asyncio
async def test_structured_adapter_unknown_handle_fails_closed() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    facade = RegistryModelControlFacade({}, clock=lambda: NOW)
    request = ModelInvocationCancellationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        invocation_handle_ref=grant.invocation_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    result = await facade.cancel_model_invocation(request)

    assert result.outcome is ModelInvocationCancellationOutcomeV2.FAILED


@pytest.mark.asyncio
async def test_structured_adapter_finished_handle_cancellation_replays() -> None:
    descriptor = _descriptor()
    grant = _grant(descriptor)
    facade = RegistryModelControlFacade(
        {descriptor.model_profile_ref.object_id: _StructuredClient()},
        clock=lambda: NOW,
    )
    await facade.generate_json_with_model_control(
        descriptor=descriptor,
        grant=grant,
        system="stable system",
        prompt="bounded prompt",
        schema={"type": "object"},
        result_ref=_ref("semantic-residual-result"),
    )
    request = ModelInvocationCancellationRequestV2.create(
        reservation_ref=grant.reservation_ref,
        invocation_handle_ref=grant.invocation_handle_ref,
        fencing_token=grant.fencing_token,
        reason="CANCELLED",
        requested_at=NOW,
    )

    result = await facade.cancel_model_invocation(request)
    replay = await facade.cancel_model_invocation(request)

    assert replay == result
    assert result.outcome is ModelInvocationCancellationOutcomeV2.ALREADY_FINISHED
