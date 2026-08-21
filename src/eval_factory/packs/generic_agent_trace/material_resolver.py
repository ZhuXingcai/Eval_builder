from __future__ import annotations

from typing import Protocol

from eval_factory.contracts.core import ObjectRef
from eval_factory.packs.generic_agent_trace.adapter_types import (
    CapabilityMaterialResolver,
    GenericCapabilityAdapterError,
)


class CapabilityMaterialSource(Protocol):
    def load(
        self,
        reference: ObjectRef,
        expected_type: type[object],
    ) -> object | None: ...

    def load_many(
        self,
        reference: ObjectRef,
        expected_type: type[object],
    ) -> tuple[object, ...] | None: ...


class CompositeCapabilityMaterialResolver(CapabilityMaterialResolver):
    """Rebuildable projection over canonical owner material sources."""

    def __init__(
        self,
        sources: tuple[CapabilityMaterialSource, ...] = (),
    ) -> None:
        self.sources = sources
        self._values: dict[ObjectRef, object] = {}
        self._collections: dict[ObjectRef, tuple[object, ...]] = {}

    def register(
        self,
        reference: ObjectRef,
        value: object,
    ) -> None:
        prior = self._values.get(reference)
        if prior is not None and prior != value:
            raise GenericCapabilityAdapterError(
                "capability material ref already binds another value",
            )
        _validate_value_reference(reference, value)
        self._values[reference] = value

    def register_many(
        self,
        reference: ObjectRef,
        values: tuple[object, ...],
    ) -> None:
        prior = self._collections.get(reference)
        if prior is not None and prior != values:
            raise GenericCapabilityAdapterError(
                "capability collection ref already binds another value",
            )
        self._collections[reference] = values

    def get[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> ValueT:
        value = self._values.get(reference)
        if value is None:
            for source in self.sources:
                value = source.load(
                    reference,
                    expected_type,
                )
                if value is not None:
                    break
        if value is None:
            raise GenericCapabilityAdapterError(
                "capability material authority is unavailable",
            )
        if not isinstance(value, expected_type):
            raise GenericCapabilityAdapterError(
                "capability material has the wrong owner type",
            )
        _validate_value_reference(reference, value)
        return value

    def get_many[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> tuple[ValueT, ...]:
        values = self._collections.get(reference)
        if values is None:
            for source in self.sources:
                values = source.load_many(
                    reference,
                    expected_type,
                )
                if values is not None:
                    break
        if values is None:
            raise GenericCapabilityAdapterError(
                "capability material collection is unavailable",
            )
        if any(not isinstance(value, expected_type) for value in values):
            raise GenericCapabilityAdapterError(
                "capability material collection has the wrong owner type",
            )
        return values  # type: ignore[return-value]


def _validate_value_reference(
    reference: ObjectRef,
    value: object,
) -> None:
    to_ref = getattr(value, "to_ref", None)
    if not callable(to_ref):
        return
    observed = to_ref()
    if isinstance(observed, ObjectRef) and observed != reference:
        raise GenericCapabilityAdapterError(
            "capability material differs from its owner ref",
        )


__all__ = [
    "CapabilityMaterialSource",
    "CompositeCapabilityMaterialResolver",
]
