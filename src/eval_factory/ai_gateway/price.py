from __future__ import annotations

from eval_factory.contracts.ai_gateway_v2 import (
    ModelPriceScheduleV2,
    ModelRouteRequestV2,
)


def estimate_max_cost_micro_usd(
    request: ModelRouteRequestV2,
    price: ModelPriceScheduleV2,
) -> int:
    numerator = (
        request.input_token_budget * price.input_micro_usd_per_million_tokens
        + request.output_token_budget * price.output_micro_usd_per_million_tokens
    )
    return (numerator + 999_999) // 1_000_000


__all__ = ["estimate_max_cost_micro_usd"]
