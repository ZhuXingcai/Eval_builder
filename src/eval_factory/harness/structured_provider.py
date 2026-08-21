from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.ai_gateway.invocation import ProviderInvocationResult
from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationStatusV2,
    GatewayUsageV2,
    ModelCapabilityProfileV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.runtime_models import (
    RequirementIntakeV1,
    RequirementInterpretationProposalV1,
)


class StructuredJsonClient(Protocol):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]: ...


class RequirementStructuredProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RequirementStructuredProviderConfig:
    provider_id: str
    model_id: str
    system_template_ref: ObjectRef
    instruction_template_ref: ObjectRef
    output_schema_ref: ObjectRef
    max_output_tokens: int = 2_000

    def __post_init__(self) -> None:
        _require_private_ref(
            self.system_template_ref,
            "prompt-template-content",
            "system_template_ref",
        )
        _require_private_ref(
            self.instruction_template_ref,
            "prompt-template-content",
            "instruction_template_ref",
        )
        _require_private_ref(
            self.output_schema_ref,
            "json-schema",
            "output_schema_ref",
        )
        if self.max_output_tokens < 256 or self.max_output_tokens > 16_000:
            raise ValueError("structured provider output budget must be between 256 and 16000")


class RequirementStructuredGatewayProvider:
    def __init__(
        self,
        *,
        client: StructuredJsonClient,
        private_store: FactoryPrivateObjectStore,
        config: RequirementStructuredProviderConfig,
    ) -> None:
        self.client = client
        self.private_store = private_store
        self.config = config
        self._invocation_count = 0

    @property
    def invocation_count(self) -> int:
        return self._invocation_count

    async def invoke(
        self,
        request: GatewayInvocationRequestV2,
        *,
        model_profile: ModelCapabilityProfileV2,
    ) -> ProviderInvocationResult:
        if (
            model_profile.provider_id != self.config.provider_id
            or model_profile.model_id != self.config.model_id
        ):
            raise RequirementStructuredProviderError("selected model profile differs from provider config")
        if request.output_schema_ref != self.config.output_schema_ref:
            raise RequirementStructuredProviderError("invocation output schema differs from provider config")

        intake = self.private_store.get_model(
            request.prompt_rendering_ref,
            RequirementIntakeV1,
        )
        system = self.private_store.get_text(self.config.system_template_ref)
        instruction = self.private_store.get_text(self.config.instruction_template_ref)
        schema = _load_schema(
            self.private_store.get_bytes(self.config.output_schema_ref),
        )
        prompt = _render_prompt(instruction=instruction, intake=intake)

        self._invocation_count += 1
        raw_value, raw_usage = await self.client.generate_json(
            system=system,
            prompt=prompt,
            schema=schema,
            max_output_tokens=self.config.max_output_tokens,
        )
        response_payload = _canonical_json(raw_value)
        proposal = RequirementInterpretationProposalV1.model_validate_json(
            response_payload,
        )
        response_body_ref = self.private_store.put_bytes(
            object_type="model-response-content",
            payload=response_payload,
        )
        output_ref = self.private_store.put_model(
            object_type="requirement-proposal",
            value=proposal,
        )
        usage = _normalize_usage(raw_usage)
        return ProviderInvocationResult(
            status=GatewayInvocationStatusV2.SUCCEEDED,
            response_body_ref=response_body_ref,
            output_ref=output_ref,
            usage=usage,
            failure_code=None,
        )


def _render_prompt(
    *,
    instruction: str,
    intake: RequirementIntakeV1,
) -> str:
    input_value = {
        "artifact_envelope_refs": [value.model_dump(mode="json") for value in intake.artifact_envelope_refs],
        "user_text": intake.user_text,
    }
    rendered = json.dumps(
        input_value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{instruction}\n\nINPUT_JSON:\n{rendered}"


def _load_schema(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RequirementStructuredProviderError("private output schema is invalid") from exc
    if not isinstance(value, dict):
        raise RequirementStructuredProviderError("private output schema must be an object")
    return {str(key): item for key, item in value.items()}


def _normalize_usage(raw: dict[str, object]) -> GatewayUsageV2:
    input_tokens = _usage_int(raw, "input_tokens")
    output_tokens = _usage_int(raw, "output_tokens")
    cache_creation = _usage_int(raw, "cache_creation_input_tokens")
    cache_read = _usage_int(raw, "cache_read_input_tokens")
    reported_cost = raw.get("reported_cost_micro_usd")
    if reported_cost is not None and (
        isinstance(reported_cost, bool) or not isinstance(reported_cost, int) or reported_cost < 0
    ):
        raise RequirementStructuredProviderError("provider-reported cost is invalid")
    return GatewayUsageV2(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        charged_tokens=(input_tokens + output_tokens + cache_creation + cache_read),
        reported_cost_micro_usd=reported_cost,
    )


def _usage_int(raw: dict[str, object], key: str) -> int:
    value = raw.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RequirementStructuredProviderError(f"provider usage field {key} is invalid")
    return value


def _canonical_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _require_private_ref(
    reference: ObjectRef,
    object_type: str,
    label: str,
) -> None:
    if reference.object_type != object_type or reference.object_version != "v2":
        raise ValueError(f"{label} must reference private {object_type}/v2")


__all__ = [
    "RequirementStructuredGatewayProvider",
    "RequirementStructuredProviderConfig",
    "RequirementStructuredProviderError",
    "StructuredJsonClient",
]
