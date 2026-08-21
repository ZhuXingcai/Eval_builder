"""Public TeamStore facade and stable runtime result types."""

from eval_factory.team._store_projection import TeamStoreProjection
from eval_factory.team._store_types import (
    TeamBlackboardError,
    TeamConcurrencyError,
    TeamConvergenceSource,
    TeamHeadRebuild,
    TeamIdempotencyConflictError,
    TeamIntegrityError,
    TeamMailboxError,
    TeamMessagePage,
    TeamNotFoundError,
    TeamProjectionSource,
    TeamSnapshot,
    TeamStaleLeaseError,
    TeamStoreError,
    TeamTaskCompletion,
    TeamTaskStateError,
    TeamTaskWork,
)


class TeamStore(TeamStoreProjection):
    """Dedicated SQLite facade for collaborative Team authority."""


__all__ = [
    "TeamBlackboardError",
    "TeamConcurrencyError",
    "TeamConvergenceSource",
    "TeamHeadRebuild",
    "TeamIdempotencyConflictError",
    "TeamIntegrityError",
    "TeamMailboxError",
    "TeamMessagePage",
    "TeamNotFoundError",
    "TeamProjectionSource",
    "TeamSnapshot",
    "TeamStaleLeaseError",
    "TeamStore",
    "TeamStoreError",
    "TeamTaskCompletion",
    "TeamTaskStateError",
    "TeamTaskWork",
]
