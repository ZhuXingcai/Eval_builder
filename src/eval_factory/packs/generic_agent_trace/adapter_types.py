from __future__ import annotations

from typing import Protocol

from eval_factory.contracts.core import ObjectRef


class GenericCapabilityAdapterError(RuntimeError):
    """Structural adapter error; callers must not convert this to success."""


class CapabilityMaterialResolver(Protocol):
    def get[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> ValueT: ...

    def get_many[ValueT](
        self,
        reference: ObjectRef,
        expected_type: type[ValueT],
    ) -> tuple[ValueT, ...]: ...


__all__ = [
    "CapabilityMaterialResolver",
    "GenericCapabilityAdapterError",
]
