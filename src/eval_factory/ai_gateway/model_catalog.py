from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from eval_factory.contracts.ai_gateway_v2 import (
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
)
from eval_factory.contracts.core import ObjectRef


class ModelCatalogError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRouteFacts:
    profile: ModelCapabilityProfileV2
    quality: ModelQualityBaselineV2 | None
    health: ModelHealthSnapshotV2 | None
    price: ModelPriceScheduleV2 | None


class ModelCatalog:
    def __init__(
        self,
        *,
        profiles: tuple[ModelCapabilityProfileV2, ...],
        quality_baselines: tuple[ModelQualityBaselineV2, ...],
        health_snapshots: tuple[ModelHealthSnapshotV2, ...],
        price_schedules: tuple[ModelPriceScheduleV2, ...],
    ) -> None:
        self._profiles = _index_records(
            profiles,
            lambda value: _ref_key(value.to_ref()),
            "model profile",
        )
        self._quality = _index_records(
            quality_baselines,
            lambda value: (_ref_key(value.model_profile_ref), value.task_kind),
            "quality baseline",
        )
        self._health = _index_records(
            health_snapshots,
            lambda value: _ref_key(value.model_profile_ref),
            "health snapshot",
        )
        self._prices = _index_records(
            price_schedules,
            lambda value: _ref_key(value.model_profile_ref),
            "price schedule",
        )

    @property
    def profiles(self) -> tuple[ModelCapabilityProfileV2, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))

    def get_profile(self, reference: ObjectRef) -> ModelCapabilityProfileV2:
        value = self._profiles.get(_ref_key(reference))
        if value is None:
            raise ModelCatalogError("unknown model capability profile")
        return value

    def facts(self, reference: ObjectRef, *, task_kind: str) -> ModelRouteFacts:
        key = _ref_key(reference)
        profile = self._profiles.get(key)
        if profile is None:
            raise ModelCatalogError("unknown model capability profile")
        return ModelRouteFacts(
            profile=profile,
            quality=self._quality.get((key, task_kind)),
            health=self._health.get(key),
            price=self._prices.get(key),
        )


def _index_records[KeyT, ValueT](
    values: tuple[ValueT, ...],
    key_for: Callable[[ValueT], KeyT],
    label: str,
) -> dict[KeyT, ValueT]:
    result: dict[KeyT, ValueT] = {}
    for value in values:
        key = key_for(value)
        if key in result:
            raise ModelCatalogError(f"duplicate {label} authority")
        result[key] = value
    return result


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = ["ModelCatalog", "ModelCatalogError", "ModelRouteFacts"]
