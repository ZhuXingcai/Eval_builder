from __future__ import annotations

import json

import anthropic

from env_mock_agent.models.base import ModelClientBackpressureError


class AnthropicStructuredClient:
    def __init__(self, model: str = "claude-opus-4-7") -> None:
        self.model = model
        self.client = anthropic.AsyncAnthropic(max_retries=0)

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=max_output_tokens,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "high",
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    },
                },
                cache_control={"type": "ephemeral"},
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = await stream.get_final_message()
        except anthropic.RateLimitError as exc:
            raise ModelClientBackpressureError(
                kind="RATE_LIMITED",
                retry_after_seconds=_retry_after_seconds(exc.response.headers.get("retry-after")),
            ) from None
        except anthropic.InternalServerError as exc:
            if exc.status_code != 529:
                raise
            raise ModelClientBackpressureError(
                kind="OVERLOADED",
                retry_after_seconds=_retry_after_seconds(exc.response.headers.get("retry-after")),
            ) from None
        text = next(block.text for block in message.content if block.type == "text")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("structured Claude output must be a JSON object")
        normalized = {str(key): item for key, item in value.items()}
        usage: dict[str, object] = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "cache_creation_input_tokens": message.usage.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": message.usage.cache_read_input_tokens or 0,
        }
        return normalized, usage


def _retry_after_seconds(value: str | None) -> int:
    if value is None:
        return 60
    try:
        parsed = int(value)
    except ValueError:
        return 60
    return min(max(parsed, 1), 86_400)
