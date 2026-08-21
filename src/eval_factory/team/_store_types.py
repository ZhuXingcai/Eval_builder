from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.artifacts import ArtifactEnvelopeV1, ArtifactHeadV1
from eval_factory.harness.capability import CapabilityResultV1
from eval_factory.harness.interaction import ExecutionAuthorityV1
from eval_factory.team.models import (
    ArtifactSubscriptionV1,
    TeamConflictV1,
    TeamMemberV1,
    TeamMessageV1,
    TeamRosterV1,
    TeamTaskClaimV1,
    TeamTaskEventV1,
    TeamTaskGraphV1,
    TeamTaskLeaseV1,
    TeamTaskStatusV1,
    TeamTaskV1,
    TeamV1,
)


class TeamStoreError(RuntimeError):
    """Base error for Team authority failures."""


class TeamNotFoundError(TeamStoreError):
    """Raised when a Team-owned record does not exist."""


class TeamConcurrencyError(TeamStoreError):
    """Raised when a caller presents stale Team authority."""


class TeamIdempotencyConflictError(TeamStoreError):
    """Raised when one operation key is reused for a different request."""


class TeamIntegrityError(TeamStoreError):
    """Raised when immutable Team history or a current cache has drifted."""


class TeamTaskStateError(TeamStoreError):
    """Raised for an illegal task lifecycle transition."""


class TeamStaleLeaseError(TeamStoreError):
    """Raised when a lease holder, fence, version, or expiry is stale."""


class TeamMailboxError(TeamStoreError):
    """Raised when a Team message violates current mailbox authority."""


class TeamBlackboardError(TeamStoreError):
    """Raised when Blackboard ownership or lineage is invalid."""


@dataclass(frozen=True, slots=True)
class TeamSnapshot:
    team: TeamV1
    roster: TeamRosterV1
    graph: TeamTaskGraphV1
    authority: ExecutionAuthorityV1


@dataclass(frozen=True, slots=True)
class TeamTaskWork:
    task: TeamTaskV1
    status: TeamTaskStatusV1
    attempt: int
    event_version: int
    graph_revision: int
    claim: TeamTaskClaimV1 | None
    lease: TeamTaskLeaseV1 | None
    fencing_token: int | None
    effective_expires_at: datetime | None
    latest_event: TeamTaskEventV1 | None


@dataclass(frozen=True, slots=True)
class TeamTaskCompletion:
    result: CapabilityResultV1
    event: TeamTaskEventV1
    artifact_heads: tuple[ArtifactHeadV1, ...]
    snapshot: TeamSnapshot


@dataclass(frozen=True, slots=True)
class TeamMessagePage:
    messages: tuple[TeamMessageV1, ...]
    next_sequence: int | None


@dataclass(frozen=True, slots=True)
class TeamProjectionSource:
    snapshot: TeamSnapshot
    member: TeamMemberV1
    tasks: tuple[TeamTaskV1, ...]
    artifact_envelopes: tuple[ArtifactEnvelopeV1, ...]
    messages: tuple[TeamMessageV1, ...]
    subscriptions: tuple[ArtifactSubscriptionV1, ...]
    acceptance_check_refs: tuple[ObjectRef, ...]
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class TeamHeadRebuild:
    team_count: int
    task_count: int
    artifact_count: int
    subscription_count: int
    conflict_count: int
    projection_count: int
    checkpoint_count: int


@dataclass(frozen=True, slots=True)
class TeamConvergenceSource:
    snapshot: TeamSnapshot
    work: tuple[TeamTaskWork, ...]
    messages: tuple[TeamMessageV1, ...]
    conflicts: tuple[TeamConflictV1, ...]
    artifact_heads: tuple[ArtifactHeadV1, ...]


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
    "TeamStoreError",
    "TeamTaskCompletion",
    "TeamTaskStateError",
    "TeamTaskWork",
]
