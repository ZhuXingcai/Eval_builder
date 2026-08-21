from __future__ import annotations

import json
import os

import httpx

from env_mock_agent.models.base import ModelClientBackpressureError


class ArkStructuredClient:
    def __init__(
        self,
        model: str = "glm-5-2-260617",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]:
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError("ARK_API_KEY is required")
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            if response.status_code in {429, 529}:
                raise ModelClientBackpressureError(
                    kind=("RATE_LIMITED" if response.status_code == 429 else "OVERLOADED"),
                    retry_after_seconds=_retry_after_seconds(response.headers.get("retry-after")),
                )
            response.raise_for_status()
            result = response.json()
        content = result["choices"][0]["message"]["content"]
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("structured Ark output must be a JSON object")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return value, {str(key): item for key, item in usage.items()}


def _retry_after_seconds(value: str | None) -> int:
    if value is None:
        return 60
    try:
        parsed = int(value)
    except ValueError:
        return 60
    return min(max(parsed, 1), 86_400)
