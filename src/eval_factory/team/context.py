from __future__ import annotations

from eval_factory.contracts.core import ContractAudit
from eval_factory.harness.contracts import sorted_refs
from eval_factory.team.models import MemberContextProjectionV1
from eval_factory.team.store import TeamStore


class TeamContextProjector:
    """Builds one least-privilege member view from a TeamStore snapshot."""

    def __init__(self, store: TeamStore) -> None:
        self.store = store

    def project(
        self,
        team_id: str,
        member_id: str,
        *,
        audit: ContractAudit,
        persist: bool = True,
    ) -> MemberContextProjectionV1:
        source = self.store.projection_source(team_id, member_id)
        projection = MemberContextProjectionV1.create(
            projection_id=(
                f"{team_id}.{member_id}.{source.snapshot.team.team_version}.{source.source_fingerprint[:16]}"
            ),
            team_ref=source.snapshot.team.to_ref(),
            member_ref=source.member.to_ref(),
            authority_ref=source.snapshot.authority.to_ref(),
            task_refs=sorted_refs(value.to_ref() for value in source.tasks),
            artifact_envelope_refs=sorted_refs(value.to_ref() for value in source.artifact_envelopes),
            message_refs=sorted_refs(value.to_ref() for value in source.messages),
            subscription_refs=sorted_refs(value.to_ref() for value in source.subscriptions),
            acceptance_check_refs=source.acceptance_check_refs,
            audit=audit,
        )
        if persist:
            projection = self.store.persist_projection(
                projection,
                source_fingerprint=source.source_fingerprint,
            )
        return projection

    def validate_rebuild(
        self,
        team_id: str,
        member_id: str,
        *,
        audit: ContractAudit,
    ) -> MemberContextProjectionV1:
        del audit
        return self.store.validate_projection_rebuild(team_id, member_id)

    def rebuild(
        self,
        team_id: str,
        member_id: str,
        *,
        audit: ContractAudit,
    ) -> MemberContextProjectionV1:
        return self.project(team_id, member_id, audit=audit, persist=True)


__all__ = ["TeamContextProjector"]
