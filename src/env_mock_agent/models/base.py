from __future__ import annotations

from typing import Literal, Protocol


class ModelClientBackpressureError(RuntimeError):
    def __init__(
        self,
        *,
        kind: Literal["RATE_LIMITED", "OVERLOADED"],
        retry_after_seconds: int,
    ) -> None:
        super().__init__(kind)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds


class StructuredModelClient(Protocol):
    async def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, object],
        max_output_tokens: int = 64_000,
    ) -> tuple[dict[str, object], dict[str, object]]: ...
