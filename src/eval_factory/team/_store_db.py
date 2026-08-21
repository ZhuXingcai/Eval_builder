from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from eval_factory.contracts.core import ObjectRef
from eval_factory.harness.interaction import ExecutionAuthorityV1
from eval_factory.team._store_codec import parse_record, record_json
from eval_factory.team._store_types import (
    TeamConcurrencyError,
    TeamIdempotencyConflictError,
    TeamIntegrityError,
    TeamNotFoundError,
    TeamSnapshot,
)
from eval_factory.team.models import (
    TeamMemberV1,
    TeamOutboxKindV1,
    TeamOutboxRecordV1,
    TeamRosterV1,
    TeamTaskGraphV1,
    TeamTaskV1,
    TeamV1,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS rosters (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, revision INTEGER NOT NULL,
  record_json TEXT NOT NULL, UNIQUE(team_id, revision));
CREATE TABLE IF NOT EXISTS graphs (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, revision INTEGER NOT NULL,
  record_json TEXT NOT NULL, UNIQUE(team_id, revision));
CREATE TABLE IF NOT EXISTS authorities (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, version INTEGER NOT NULL,
  record_json TEXT NOT NULL, UNIQUE(team_id, version));
CREATE TABLE IF NOT EXISTS teams (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, version INTEGER NOT NULL,
  roster_id TEXT NOT NULL REFERENCES rosters(object_id),
  graph_id TEXT NOT NULL REFERENCES graphs(object_id),
  authority_id TEXT NOT NULL REFERENCES authorities(object_id),
  record_json TEXT NOT NULL, UNIQUE(team_id, version));
CREATE TABLE IF NOT EXISTS team_heads (
  team_id TEXT PRIMARY KEY, team_id_ref TEXT NOT NULL REFERENCES teams(object_id),
  roster_id TEXT NOT NULL REFERENCES rosters(object_id),
  graph_id TEXT NOT NULL REFERENCES graphs(object_id),
  authority_id TEXT NOT NULL REFERENCES authorities(object_id));

CREATE TABLE IF NOT EXISTS task_claims (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  attempt INTEGER NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, task_id, attempt));
CREATE TABLE IF NOT EXISTS task_leases (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  attempt INTEGER NOT NULL, fence INTEGER NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, task_id, attempt), UNIQUE(team_id, task_id, fence));
CREATE TABLE IF NOT EXISTS task_events (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  version INTEGER NOT NULL, graph_revision INTEGER NOT NULL,
  kind TEXT NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, task_id, version));
CREATE TABLE IF NOT EXISTS task_heads (
  team_id TEXT NOT NULL, task_id TEXT NOT NULL, task_ref TEXT NOT NULL,
  event_id TEXT NOT NULL REFERENCES task_events(object_id),
  PRIMARY KEY(team_id, task_id));

CREATE TABLE IF NOT EXISTS capability_calls (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  record_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS capability_results (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  call_id TEXT NOT NULL REFERENCES capability_calls(object_id),
  record_json TEXT NOT NULL, UNIQUE(call_id));
CREATE TABLE IF NOT EXISTS completion_responses (
  result_id TEXT PRIMARY KEY REFERENCES capability_results(object_id),
  team_id TEXT NOT NULL, event_id TEXT NOT NULL UNIQUE REFERENCES task_events(object_id),
  team_object_id TEXT NOT NULL REFERENCES teams(object_id),
  artifact_head_ids_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS task_usage_events (
  result_id TEXT PRIMARY KEY REFERENCES capability_results(object_id),
  team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  model_requests INTEGER NOT NULL, model_tokens INTEGER NOT NULL,
  cost_micro_usd INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS envelopes (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, task_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL, revision INTEGER NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, artifact_id, revision));
CREATE TABLE IF NOT EXISTS artifact_heads (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, head_id TEXT NOT NULL,
  revision INTEGER NOT NULL, sequence INTEGER NOT NULL,
  envelope_id TEXT NOT NULL REFERENCES envelopes(object_id),
  record_json TEXT NOT NULL, UNIQUE(team_id, head_id, revision),
  UNIQUE(team_id, sequence));
CREATE TABLE IF NOT EXISTS artifact_current (
  team_id TEXT NOT NULL, head_id TEXT NOT NULL,
  object_id TEXT NOT NULL REFERENCES artifact_heads(object_id),
  PRIMARY KEY(team_id, head_id));

CREATE TABLE IF NOT EXISTS messages (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  sender_id TEXT NOT NULL, task_id TEXT, kind TEXT NOT NULL,
  record_json TEXT NOT NULL, UNIQUE(team_id, sequence));
CREATE TABLE IF NOT EXISTS subscriptions (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, subscription_id TEXT NOT NULL,
  revision INTEGER NOT NULL, cursor INTEGER NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, subscription_id, revision));
CREATE TABLE IF NOT EXISTS subscription_current (
  team_id TEXT NOT NULL, subscription_id TEXT NOT NULL,
  object_id TEXT NOT NULL REFERENCES subscriptions(object_id),
  PRIMARY KEY(team_id, subscription_id));
CREATE TABLE IF NOT EXISTS conflicts (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, conflict_id TEXT NOT NULL,
  revision INTEGER NOT NULL, status TEXT NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, conflict_id, revision));
CREATE TABLE IF NOT EXISTS conflict_current (
  team_id TEXT NOT NULL, conflict_id TEXT NOT NULL,
  object_id TEXT NOT NULL REFERENCES conflicts(object_id),
  PRIMARY KEY(team_id, conflict_id));

CREATE TABLE IF NOT EXISTS projections (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, member_id TEXT NOT NULL,
  revision INTEGER NOT NULL, fingerprint TEXT NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, member_id, revision));
CREATE TABLE IF NOT EXISTS projection_current (
  team_id TEXT NOT NULL, member_id TEXT NOT NULL,
  object_id TEXT NOT NULL REFERENCES projections(object_id),
  PRIMARY KEY(team_id, member_id));
CREATE TABLE IF NOT EXISTS checkpoints (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  team_version INTEGER NOT NULL, record_json TEXT NOT NULL,
  UNIQUE(team_id, sequence));
CREATE TABLE IF NOT EXISTS checkpoint_current (
  team_id TEXT PRIMARY KEY, object_id TEXT NOT NULL REFERENCES checkpoints(object_id));
CREATE TABLE IF NOT EXISTS convergence (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  outcome TEXT NOT NULL, record_json TEXT NOT NULL, UNIQUE(team_id, sequence));

CREATE TABLE IF NOT EXISTS outbox (
  object_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, sequence INTEGER NOT NULL,
  kind TEXT NOT NULL, record_json TEXT NOT NULL, UNIQUE(team_id, sequence));
CREATE TABLE IF NOT EXISTS outbox_delivery (
  outbox_id TEXT NOT NULL REFERENCES outbox(object_id), session_id TEXT NOT NULL,
  delivered_at TEXT NOT NULL, PRIMARY KEY(outbox_id, session_id));
CREATE TABLE IF NOT EXISTS idempotency (
  scope TEXT NOT NULL, operation_key TEXT NOT NULL, request_hash TEXT NOT NULL,
  response_type TEXT NOT NULL, response_id TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(scope, operation_key));
"""

RECORD_TABLES = frozenset(
    {
        "task_claims",
        "task_leases",
        "task_events",
        "capability_calls",
        "capability_results",
        "envelopes",
        "artifact_heads",
        "messages",
        "subscriptions",
        "conflicts",
        "projections",
        "checkpoints",
        "convergence",
        "outbox",
    },
)

PAIR_HEAD_SPECS = frozenset(
    {
        ("subscriptions", "subscription_current", "subscription_id", "revision"),
        ("conflicts", "conflict_current", "conflict_id", "revision"),
        ("projections", "projection_current", "member_id", "revision"),
    },
)


class TeamStoreDatabase:
    """Owns SQLite setup, canonical records, and generic transaction mechanics."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._fault = fault_injector or (lambda _: None)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _read(self) -> Generator[sqlite3.Connection]:
        connection = self._connection()
        try:
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.rollback()
            connection.close()

    @contextmanager
    def _write(self) -> Generator[sqlite3.Connection]:
        connection = self._connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _load[RecordT: BaseModel](
        connection: sqlite3.Connection,
        table: str,
        object_id: str,
        model: type[RecordT],
    ) -> RecordT:
        if table not in RECORD_TABLES:
            raise AssertionError("unapproved Team record table")
        row = connection.execute(
            f"SELECT record_json FROM {table} WHERE object_id = ?",
            (object_id,),
        ).fetchone()
        if row is None:
            raise TeamNotFoundError(f"{model.__name__} was not found")
        return parse_record(model, str(row["record_json"]), model.__name__)

    def _snapshot(
        self,
        connection: sqlite3.Connection,
        team_id: str,
    ) -> TeamSnapshot:
        row = connection.execute(
            """
            SELECT h.*, t.team_id stored_team_id,
                   t.version stored_team_version,
                   r.team_id roster_team_id,
                   r.revision roster_revision,
                   g.team_id graph_team_id,
                   g.revision graph_revision,
                   a.team_id authority_team_id,
                   a.version authority_version,
                   t.record_json team_json, r.record_json roster_json,
                   g.record_json graph_json, a.record_json authority_json
            FROM team_heads h
            JOIN teams t ON t.object_id = h.team_id_ref
            JOIN rosters r ON r.object_id = h.roster_id
            JOIN graphs g ON g.object_id = h.graph_id
            JOIN authorities a ON a.object_id = h.authority_id
            WHERE h.team_id = ?
            """,
            (team_id,),
        ).fetchone()
        if row is None:
            exists = connection.execute(
                "SELECT 1 FROM teams WHERE team_id = ? LIMIT 1",
                (team_id,),
            ).fetchone()
            if exists is not None:
                raise TeamIntegrityError("Team current head is missing")
            raise TeamNotFoundError("Team was not found")
        latest = connection.execute(
            """
            SELECT object_id FROM teams
            WHERE team_id = ? ORDER BY version DESC LIMIT 1
            """,
            (team_id,),
        ).fetchone()
        snapshot = TeamSnapshot(
            parse_record(TeamV1, str(row["team_json"]), "Team"),
            parse_record(TeamRosterV1, str(row["roster_json"]), "Team roster"),
            parse_record(TeamTaskGraphV1, str(row["graph_json"]), "Team graph"),
            parse_record(
                ExecutionAuthorityV1,
                str(row["authority_json"]),
                "execution authority",
            ),
        )
        if (
            latest is None
            or latest["object_id"] != row["team_id_ref"]
            or snapshot.team.object_id != row["team_id_ref"]
            or snapshot.team.team_id != row["stored_team_id"]
            or snapshot.team.team_version != row["stored_team_version"]
            or snapshot.roster.object_id != row["roster_id"]
            or snapshot.roster.team_id != row["roster_team_id"]
            or snapshot.roster.revision != row["roster_revision"]
            or snapshot.graph.object_id != row["graph_id"]
            or snapshot.graph.team_id != row["graph_team_id"]
            or snapshot.graph.revision != row["graph_revision"]
            or snapshot.authority.object_id != row["authority_id"]
            or snapshot.authority.team_id != row["authority_team_id"]
            or snapshot.authority.authority_version != row["authority_version"]
        ):
            raise TeamIntegrityError("Team current head drifted")
        self._validate_bundle(snapshot)
        return snapshot

    def _snapshot_at(
        self,
        connection: sqlite3.Connection,
        team_ref: ObjectRef,
    ) -> TeamSnapshot:
        row = connection.execute(
            """
            SELECT t.team_id, t.version team_version,
                   t.roster_id, t.graph_id, t.authority_id,
                   r.team_id roster_team_id,
                   r.revision roster_revision,
                   g.team_id graph_team_id,
                   g.revision graph_revision,
                   a.team_id authority_team_id,
                   a.version authority_version,
                   t.record_json team_json, r.record_json roster_json,
                   g.record_json graph_json, a.record_json authority_json
            FROM teams t
            JOIN rosters r ON r.object_id = t.roster_id
            JOIN graphs g ON g.object_id = t.graph_id
            JOIN authorities a ON a.object_id = t.authority_id
            WHERE t.object_id = ?
            """,
            (team_ref.object_id,),
        ).fetchone()
        if row is None:
            raise TeamNotFoundError("Team reference was not found")
        snapshot = TeamSnapshot(
            parse_record(TeamV1, str(row["team_json"]), "Team"),
            parse_record(TeamRosterV1, str(row["roster_json"]), "Team roster"),
            parse_record(TeamTaskGraphV1, str(row["graph_json"]), "Team graph"),
            parse_record(
                ExecutionAuthorityV1,
                str(row["authority_json"]),
                "execution authority",
            ),
        )
        if (
            snapshot.team.to_ref() != team_ref
            or snapshot.team.team_id != row["team_id"]
            or snapshot.team.team_version != row["team_version"]
            or snapshot.roster.object_id != row["roster_id"]
            or snapshot.roster.team_id != row["roster_team_id"]
            or snapshot.roster.revision != row["roster_revision"]
            or snapshot.graph.object_id != row["graph_id"]
            or snapshot.graph.team_id != row["graph_team_id"]
            or snapshot.graph.revision != row["graph_revision"]
            or snapshot.authority.object_id != row["authority_id"]
            or snapshot.authority.team_id != row["authority_team_id"]
            or snapshot.authority.authority_version != row["authority_version"]
        ):
            raise TeamIntegrityError("historical Team authority drifted")
        self._validate_bundle(snapshot)
        return snapshot

    @staticmethod
    def _validate_bundle(snapshot: TeamSnapshot) -> None:
        team, roster, graph, authority = (
            snapshot.team,
            snapshot.roster,
            snapshot.graph,
            snapshot.authority,
        )
        member_ids = tuple(member.member_id for member in roster.members)
        task_ids = {
            member_id: {task.task_id for task in graph.tasks if task.assigned_member_id == member_id}
            for member_id in member_ids
        }
        if (
            {roster.team_id, graph.team_id, authority.team_id} != {team.team_id}
            or team.roster_ref != roster.to_ref()
            or team.task_graph_ref != graph.to_ref()
            or team.authority_ref != authority.to_ref()
            or graph.roster_ref != roster.to_ref()
            or authority.roster_ref != roster.to_ref()
            or authority.task_graph_ref != graph.to_ref()
            or authority.team_incarnation_id != team.team_incarnation_id
            or graph.member_ids != member_ids
            or tuple(grant.member_id for grant in authority.grants) != member_ids
            or any(set(grant.task_ids) != task_ids[grant.member_id] for grant in authority.grants)
        ):
            raise TeamIntegrityError("Team authority bundle is inconsistent")
        coordinator = tuple(
            member for member in roster.members if member.member_id == team.coordinator_member_id
        )
        if len(coordinator) != 1 or not coordinator[0].is_coordinator:
            raise TeamIntegrityError("Team coordinator is stale")

    def _require_refs(
        self,
        connection: sqlite3.Connection,
        team_id: str,
        team_ref: ObjectRef,
        graph_ref: ObjectRef,
        authority_ref: ObjectRef,
    ) -> TeamSnapshot:
        snapshot = self._snapshot(connection, team_id)
        if (
            snapshot.team.to_ref() != team_ref
            or snapshot.graph.to_ref() != graph_ref
            or snapshot.authority.to_ref() != authority_ref
        ):
            raise TeamConcurrencyError("Team, graph, or authority is stale")
        return snapshot

    def _team_id(self, reference: ObjectRef) -> str:
        if reference.object_type != "agent-team":
            raise TeamIntegrityError("reference is not a Team")
        with self._read() as connection:
            row = connection.execute(
                "SELECT team_id, record_json FROM teams WHERE object_id = ?",
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise TeamNotFoundError("Team reference was not found")
            value = parse_record(TeamV1, str(row["record_json"]), "Team")
            if value.to_ref() != reference:
                raise TeamIntegrityError("Team reference drifted")
            return str(row["team_id"])

    @staticmethod
    def _task(snapshot: TeamSnapshot, task_id: str) -> TeamTaskV1:
        matches = tuple(task for task in snapshot.graph.tasks if task.task_id == task_id)
        if len(matches) != 1:
            raise TeamNotFoundError("Team task was not found")
        return matches[0]

    @staticmethod
    def _task_ref(
        snapshot: TeamSnapshot,
        reference: ObjectRef,
    ) -> TeamTaskV1:
        matches = tuple(task for task in snapshot.graph.tasks if task.to_ref() == reference)
        if len(matches) != 1:
            raise TeamConcurrencyError("Team task reference is stale")
        return matches[0]

    @staticmethod
    def _member_ref(
        snapshot: TeamSnapshot,
        reference: ObjectRef,
    ) -> TeamMemberV1:
        matches = tuple(member for member in snapshot.roster.members if member.to_ref() == reference)
        if len(matches) != 1:
            raise TeamConcurrencyError("Team member reference is stale")
        return matches[0]

    @staticmethod
    def _member(
        snapshot: TeamSnapshot,
        member_id: str,
    ) -> TeamMemberV1:
        matches = tuple(member for member in snapshot.roster.members if member.member_id == member_id)
        if len(matches) != 1:
            raise TeamNotFoundError("Team member was not found")
        return matches[0]

    @staticmethod
    def _replay(
        connection: sqlite3.Connection,
        scope: str,
        operation_key: str,
        request_hash: str,
    ) -> tuple[str, str] | None:
        row = connection.execute(
            """
            SELECT request_hash, response_type, response_id FROM idempotency
            WHERE scope = ? AND operation_key = ?
            """,
            (scope, operation_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise TeamIdempotencyConflictError(
                "operation key already binds another Team request",
            )
        return str(row["response_type"]), str(row["response_id"])

    def _remember(
        self,
        connection: sqlite3.Connection,
        scope: str,
        operation_key: str,
        request_hash: str,
        response_type: str,
        response_id: str,
    ) -> None:
        connection.execute(
            "INSERT INTO idempotency VALUES (?, ?, ?, ?, ?, ?)",
            (
                scope,
                operation_key,
                request_hash,
                response_type,
                response_id,
                self._clock().isoformat(),
            ),
        )

    def _outbox(
        self,
        connection: sqlite3.Connection,
        snapshot: TeamSnapshot,
        kind: TeamOutboxKindV1,
        record_ref: ObjectRef,
    ) -> TeamOutboxRecordV1:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM outbox WHERE team_id = ?",
            (snapshot.team.team_id,),
        ).fetchone()
        sequence = int(row[0]) + 1
        record = TeamOutboxRecordV1.create(
            outbox_key=f"{snapshot.team.team_id}.outbox-{sequence}",
            team_id=snapshot.team.team_id,
            sequence=sequence,
            event_kind=kind,
            aggregate_ref=snapshot.team.to_ref(),
            record_ref=record_ref,
            intended_member_session_refs=tuple(
                member.independent_session_ref for member in snapshot.roster.members
            ),
            occurred_at=self._clock(),
            audit=snapshot.team.audit,
        )
        connection.execute(
            "INSERT INTO outbox VALUES (?, ?, ?, ?, ?)",
            (
                record.object_id,
                record.team_id,
                record.sequence,
                record.event_kind.value,
                record_json(record),
            ),
        )
        return record

    @staticmethod
    def _validate_pair_head(
        connection: sqlite3.Connection,
        *,
        team_id: str,
        history_table: str,
        current_table: str,
        key_column: str,
        version_column: str,
    ) -> None:
        if (
            history_table,
            current_table,
            key_column,
            version_column,
        ) not in PAIR_HEAD_SPECS:
            raise AssertionError("unapproved Team current-head tables")
        expected = {
            str(row[key_column]): str(row["object_id"])
            for row in connection.execute(
                f"""
                SELECT h.{key_column}, h.object_id
                FROM {history_table} h
                JOIN (
                  SELECT {key_column}, MAX({version_column}) version
                  FROM {history_table}
                  WHERE team_id = ?
                  GROUP BY {key_column}
                ) latest
                  ON latest.{key_column} = h.{key_column}
                 AND latest.version = h.{version_column}
                WHERE h.team_id = ?
                """,
                (team_id, team_id),
            ).fetchall()
        }
        observed = {
            str(row[key_column]): str(row["object_id"])
            for row in connection.execute(
                f"""
                SELECT {key_column}, object_id
                FROM {current_table}
                WHERE team_id = ?
                """,
                (team_id,),
            ).fetchall()
        }
        if observed != expected:
            raise TeamIntegrityError(
                f"{current_table} differs from immutable history",
            )


__all__ = ["PAIR_HEAD_SPECS", "SCHEMA", "TeamStoreDatabase"]
