from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.graph_models import (
    HarnessGraphCheckpointV1,
)


class FactoryGraphJournalError(RuntimeError):
    pass


class FactoryGraphJournalConflictError(FactoryGraphJournalError):
    pass


class FactoryGraphJournalIntegrityError(FactoryGraphJournalError):
    pass


class FactoryGraphJournalNotFoundError(FactoryGraphJournalError):
    pass


class FactoryGraphJournalFaultPoint(StrEnum):
    AFTER_BINDING_RECORD = "after_binding_record"
    AFTER_BINDING_HEAD = "after_binding_head"
    AFTER_CHECKPOINT_RECORD = "after_checkpoint_record"
    AFTER_CHECKPOINT_HEAD = "after_checkpoint_head"
    AFTER_CHECKPOINT_OUTBOX = "after_checkpoint_outbox"
    AFTER_IDEMPOTENCY = "after_idempotency"
    AFTER_DELIVERY = "after_delivery"


class FactoryGraphJournalInjectedCrash(FactoryGraphJournalError):
    pass


@dataclass(frozen=True, slots=True)
class StaticFactoryGraphJournalFaultInjector:
    crash_points: frozenset[FactoryGraphJournalFaultPoint]

    def __call__(
        self,
        point: FactoryGraphJournalFaultPoint,
    ) -> None:
        if point in self.crash_points:
            raise FactoryGraphJournalInjectedCrash(point.value)


@dataclass(frozen=True, slots=True)
class FactoryGraphJournalHeadRebuild:
    binding_count: int
    checkpoint_count: int


@dataclass(frozen=True, slots=True)
class FactoryGraphCheckpointDelivery:
    checkpoint: HarnessGraphCheckpointV1
    session_ref: ObjectRef


__all__ = [
    "FactoryGraphCheckpointDelivery",
    "FactoryGraphJournalConflictError",
    "FactoryGraphJournalError",
    "FactoryGraphJournalFaultPoint",
    "FactoryGraphJournalHeadRebuild",
    "FactoryGraphJournalInjectedCrash",
    "FactoryGraphJournalIntegrityError",
    "FactoryGraphJournalNotFoundError",
    "StaticFactoryGraphJournalFaultInjector",
]
