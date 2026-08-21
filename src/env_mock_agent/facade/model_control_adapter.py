from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from env_mock_agent.facade.contracts import FacadeObjectRef
from env_mock_agent.facade.model_control_v2 import (
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
    validate_model_control_grant_identity,
    validate_model_invocation_cancellation_request_identity,
    validate_model_invocation_descriptor_identity,
)
from env_mock_agent.models.base import (
    ModelClientBackpressureError,
    StructuredModelClient,
)


class RegistryModelControlFacade:
    def __init__(
        self,
        clients: Mapping[str, StructuredModelClient],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._clients = dict(clients)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cache: dict[
            str,
            tuple[
                str,
                dict[str, object] | None,
                ModelControlReceiptV2,
            ],
        ] = {}
        self._active: dict[
            str,
            tuple[
                asyncio.Task[tuple[dict[str, object], dict[str, object]]],
                str,
                int,
            ],
        ] = {}
        self._stopped: set[str] = set()
        self._cancellations: dict[
            str,
            ModelInvocationCancellationResultV2,
        ] = {}

    async def generate_json_with_model_control(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        system: str,
        prompt: str,
        schema: dict[str, object],
        result_ref: FacadeObjectRef,
    ) -> tuple[dict[str, object] | None, ModelControlReceiptV2]:
        validate_model_invocation_descriptor_identity(descriptor)
        validate_model_control_grant_identity(grant)
        grant.validate_descriptor(descriptor)
        if descriptor.mode is not ModelDemandModeFacadeV2.DIRECT_REQUEST:
            raise ValueError("structured model control requires DIRECT_REQUEST mode")
        request_key = (
            f"{descriptor.invocation_descriptor_sha256}:"
            f"{grant.model_control_grant_sha256}:"
            f"{json.dumps(result_ref.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))}"
        )
        cached = self._cache.get(descriptor.idempotency_key)
        if cached is not None:
            if cached[0] != request_key:
                raise ValueError("model-controlled invocation idempotency conflict")
            return cached[1], cached[2]
        result: tuple[dict[str, object] | None, ModelControlReceiptV2]
        client = self._clients.get(descriptor.model_profile_ref.object_id)
        if client is None:
            result = (
                None,
                self._receipt(
                    descriptor=descriptor,
                    grant=grant,
                    result_ref=None,
                    outcome=ModelControlReceiptOutcomeV2.BLOCKED_CAPABILITY,
                    usage=_zero_usage(),
                    backpressure_ref=None,
                    failure_code="MODEL_PROFILE_UNAVAILABLE",
                ),
            )
            return self._cache_result(descriptor.idempotency_key, request_key, result)
        handle_id = grant.invocation_handle_ref.object_id
        task = asyncio.create_task(
            client.generate_json(
                system=system,
                prompt=prompt,
                schema=schema,
                max_output_tokens=grant.output_token_limit,
            )
        )
        self._active[handle_id] = (
            task,
            grant.reservation_ref.object_id,
            grant.fencing_token,
        )
        try:
            output, raw_usage = await task
            usage = _normalize_usage(raw_usage)
            try:
                grant.validate_usage(usage)
            except ValueError:
                result = (
                    None,
                    self._receipt(
                        descriptor=descriptor,
                        grant=grant,
                        result_ref=None,
                        outcome=ModelControlReceiptOutcomeV2.BLOCKED_POLICY,
                        usage=usage,
                        backpressure_ref=None,
                        failure_code="MODEL_USAGE_OVERRUN",
                    ),
                )
            else:
                result = (
                    output,
                    self._receipt(
                        descriptor=descriptor,
                        grant=grant,
                        result_ref=result_ref,
                        outcome=ModelControlReceiptOutcomeV2.SUCCEEDED,
                        usage=usage,
                        backpressure_ref=None,
                        failure_code=None,
                    ),
                )
        except ModelClientBackpressureError as exc:
            observed_at = self._clock()
            backpressure = ModelProviderBackpressureV2.create(
                descriptor_ref=model_invocation_descriptor_ref(descriptor),
                kind=ModelProviderBackpressureKindV2(exc.kind),
                retry_at=observed_at + timedelta(seconds=max(exc.retry_after_seconds, 1)),
                observed_at=observed_at,
            )
            result = (
                None,
                self._receipt(
                    descriptor=descriptor,
                    grant=grant,
                    result_ref=None,
                    outcome=ModelControlReceiptOutcomeV2.BACKPRESSURED,
                    usage=_zero_usage(),
                    backpressure_ref=model_provider_backpressure_ref(backpressure),
                    failure_code=None,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            result = (
                None,
                self._receipt(
                    descriptor=descriptor,
                    grant=grant,
                    result_ref=None,
                    outcome=ModelControlReceiptOutcomeV2.FAILED,
                    usage=ModelControlUsageV2.conservative(
                        requests=grant.request_allowance,
                        tokens=grant.token_allowance,
                    ),
                    backpressure_ref=None,
                    failure_code="MODEL_CALL_FAILED",
                ),
            )
        finally:
            self._active.pop(handle_id, None)
            self._stopped.add(handle_id)
        return self._cache_result(descriptor.idempotency_key, request_key, result)

    async def cancel_model_invocation(
        self,
        request: ModelInvocationCancellationRequestV2,
    ) -> ModelInvocationCancellationResultV2:
        validate_model_invocation_cancellation_request_identity(request)
        cached = self._cancellations.get(request.cancellation_request_id)
        if cached is not None:
            return cached
        handle_id = request.invocation_handle_ref.object_id
        active = self._active.get(handle_id)
        if active is None:
            outcome = (
                ModelInvocationCancellationOutcomeV2.ALREADY_FINISHED
                if handle_id in self._stopped
                else ModelInvocationCancellationOutcomeV2.FAILED
            )
        else:
            task, reservation_id, fence = active
            if reservation_id != request.reservation_ref.object_id or fence != request.fencing_token:
                outcome = ModelInvocationCancellationOutcomeV2.FAILED
            else:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                outcome = ModelInvocationCancellationOutcomeV2.CANCELLED
        result = ModelInvocationCancellationResultV2.create(
            cancellation_request_ref=model_invocation_cancellation_request_ref(request),
            outcome=outcome,
            completed_at=self._clock(),
        )
        self._cancellations[request.cancellation_request_id] = result
        return result

    def _receipt(
        self,
        *,
        descriptor: ModelInvocationDescriptorV2,
        grant: ModelControlGrantV2,
        result_ref: FacadeObjectRef | None,
        outcome: ModelControlReceiptOutcomeV2,
        usage: ModelControlUsageV2,
        backpressure_ref: FacadeObjectRef | None,
        failure_code: str | None,
    ) -> ModelControlReceiptV2:
        return ModelControlReceiptV2.create(
            descriptor_ref=model_invocation_descriptor_ref(descriptor),
            grant_ref=model_control_grant_ref(grant),
            outcome=outcome,
            result_ref=result_ref,
            usage=usage,
            backpressure_ref=backpressure_ref,
            failure_code=failure_code,
            completed_at=self._clock(),
        )

    def _cache_result(
        self,
        idempotency_key: str,
        request_key: str,
        result: tuple[dict[str, object] | None, ModelControlReceiptV2],
    ) -> tuple[dict[str, object] | None, ModelControlReceiptV2]:
        self._cache[idempotency_key] = (request_key, result[0], result[1])
        return result


def _normalize_usage(raw: dict[str, object]) -> ModelControlUsageV2:
    input_tokens = _non_negative_int(raw.get("input_tokens", raw.get("prompt_tokens", 0)))
    output_tokens = _non_negative_int(raw.get("output_tokens", raw.get("completion_tokens", 0)))
    cache_creation = _non_negative_int(raw.get("cache_creation_input_tokens", 0))
    cache_read = _non_negative_int(raw.get("cache_read_input_tokens", 0))
    charged_tokens = input_tokens + output_tokens + cache_creation + cache_read
    return ModelControlUsageV2(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        charged_tokens=charged_tokens,
        source=ModelControlUsageSourceV2.REPORTED,
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("model usage cannot be boolean")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError("model usage must be an integer") from exc
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    else:
        raise ValueError("model usage must be an integer")
    if parsed < 0:
        raise ValueError("model usage cannot be negative")
    return parsed


def _zero_usage() -> ModelControlUsageV2:
    return ModelControlUsageV2(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        charged_tokens=0,
        source=ModelControlUsageSourceV2.REPORTED,
    )
