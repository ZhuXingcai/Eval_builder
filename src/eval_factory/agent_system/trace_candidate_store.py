from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.agent_system.trace_candidate import (
    TraceCandidatePreparation,
)
from eval_factory.contracts.agent_system_v2 import (
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.core import ContractAudit, ContractModel, ObjectRef
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique_refs,
    sorted_refs,
)


class FactoryTraceCandidateMaterialError(RuntimeError):
    pass


class FactoryTraceCandidateAuthorityV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/trace-candidate-authority/v1"] = (
        "generic-agent-trace/trace-candidate-authority/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "factory-trace-candidate-authority"

    run_id: str = Field(min_length=1, max_length=512)
    run_ref: ObjectRef
    compiled_plan_ref: ObjectRef
    trace_source_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    disposition: TraceCandidateDispositionV2
    decision_ref: ObjectRef
    extracted_prompt_ref: ObjectRef | None = None
    inferred_intent_ref: ObjectRef | None = None
    rewrite_candidate_ref: ObjectRef | None = None
    route_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10,
    )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        for reference, object_type, label in (
            (self.run_ref, "factory-run", "run_ref"),
            (
                self.compiled_plan_ref,
                "compiled-dataset-build-plan",
                "compiled_plan_ref",
            ),
            (self.trace_source_ref, "trace-source", "trace_source_ref"),
            (
                self.trace_envelope_ref,
                "trace-envelope",
                "trace_envelope_ref",
            ),
            (
                self.decision_ref,
                "trace-candidate-decision",
                "decision_ref",
            ),
        ):
            require_ref(
                reference,
                object_type,
                label,
                object_version="v2",
            )
        optional = (
            (
                self.extracted_prompt_ref,
                "extracted-user-prompt",
                "extracted_prompt_ref",
            ),
            (
                self.inferred_intent_ref,
                "inferred-user-intent",
                "inferred_intent_ref",
            ),
            (
                self.rewrite_candidate_ref,
                "task-rewrite-candidate",
                "rewrite_candidate_ref",
            ),
        )
        for optional_reference, object_type, label in optional:
            if optional_reference is not None:
                require_ref(
                    optional_reference,
                    object_type,
                    label,
                    object_version="v2",
                )
        require_sorted_unique_refs(self.route_refs, "route_refs")
        if any(
            reference.object_type != "model-route-decision" or reference.object_version != "v2"
            for reference in self.route_refs
        ):
            raise ValueError("route_refs contain an invalid authority")
        candidate_values = (
            self.extracted_prompt_ref,
            self.inferred_intent_ref,
            self.rewrite_candidate_ref,
        )
        if self.disposition is TraceCandidateDispositionV2.CANDIDATE:
            if any(value is None for value in candidate_values):
                raise ValueError(
                    "candidate authority requires complete semantic material",
                )
        elif any(value is not None for value in candidate_values):
            raise ValueError(
                "non-candidate authority cannot retain semantic material",
            )
        return self


@dataclass(frozen=True, slots=True)
class FactoryTraceCandidateMaterial:
    authority: FactoryTraceCandidateAuthorityV1
    preparation: TraceCandidatePreparation


class FactoryTraceCandidateMaterialStore:
    """Persists per-source candidate material before dataset fan-in."""

    def __init__(self, store: FactoryControlStore) -> None:
        self.store = store
        self._initialize()

    def commit(
        self,
        *,
        run_id: str,
        trace_source_ref: ObjectRef,
        trace_envelope_ref: ObjectRef,
        preparation: TraceCandidatePreparation,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> FactoryTraceCandidateMaterial:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = self.store._load_current_run(connection, run_id)
            if run.compiled_plan_ref is None:
                raise FactoryTraceCandidateMaterialError(
                    "trace candidate requires current compiled plan",
                )
            authority = self._authority(
                run_id=run_id,
                run_ref=run.to_ref(),
                compiled_plan_ref=run.compiled_plan_ref,
                trace_source_ref=trace_source_ref,
                trace_envelope_ref=trace_envelope_ref,
                preparation=preparation,
                audit=audit,
            )
            scope = f"commit-trace-candidate:{run_id}:{trace_source_ref.object_id}"
            replay_id = self.store._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=authority.object_sha256,
                response_type=authority.OBJECT_TYPE,
            )
            if replay_id is not None:
                replay = self._load_authority(
                    connection,
                    replay_id,
                )
                material = self._material(connection, replay)
                connection.rollback()
                return material
            current = self._current(
                connection,
                run_id=run_id,
                trace_source_ref=trace_source_ref,
                required=False,
            )
            if current is not None:
                if current != authority:
                    raise FactoryControlConflictError(
                        "trace candidate current authority conflicts",
                    )
                self.store._insert_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=authority.object_sha256,
                    response_type=authority.OBJECT_TYPE,
                    response_id=authority.object_id,
                )
                connection.commit()
                return self._material_for_authority(authority, preparation)
            for value in self._values(preparation):
                self._insert_material(connection, value)
            connection.execute(
                """
                INSERT INTO factory_trace_candidate_authorities (
                    object_id, run_id, trace_source_object_id,
                    compiled_plan_object_id, object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    authority.object_id,
                    run_id,
                    trace_source_ref.object_id,
                    run.compiled_plan_ref.object_id,
                    authority.object_sha256,
                    authority.canonical_json().decode(),
                ),
            )
            connection.execute(
                """
                INSERT INTO factory_trace_candidate_current_heads
                    (run_id, trace_source_object_id, authority_object_id)
                VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    trace_source_ref.object_id,
                    authority.object_id,
                ),
            )
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=run.run_version,
                event_type="factory-trace-candidate-committed",
                object_id=authority.object_id,
                created_at=audit.created_at.isoformat(),
            )
            self.store._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=authority.object_sha256,
                response_type=authority.OBJECT_TYPE,
                response_id=authority.object_id,
            )
            connection.commit()
            return self._material_for_authority(
                authority,
                preparation,
            )
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError(
                "trace candidate authority already exists",
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(
        self,
        *,
        run_id: str,
        trace_source_ref: ObjectRef,
    ) -> FactoryTraceCandidateMaterial:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            authority = self._current(
                connection,
                run_id=run_id,
                trace_source_ref=trace_source_ref,
                required=True,
            )
            assert authority is not None
            run = self.store._load_current_run(connection, run_id)
            bound_run = self.store._load_run_by_object_id(
                connection,
                authority.run_ref.object_id,
            )
            if bound_run.to_ref() != authority.run_ref:
                raise FactoryControlIntegrityError(
                    "trace candidate run authority drifted",
                )
            if (
                bound_run.run_id != run.run_id
                or bound_run.compiled_plan_ref != authority.compiled_plan_ref
                or run.compiled_plan_ref != authority.compiled_plan_ref
            ):
                raise FactoryControlIntegrityError(
                    "trace candidate authority is stale",
                )
            value = self._material(connection, authority)
            connection.rollback()
            return value
        finally:
            connection.close()

    def load(
        self,
        reference: ObjectRef,
        expected_type: type[object],
    ) -> object | None:
        if not issubclass(expected_type, ContractModel):
            return None
        connection = self.store._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM factory_trace_candidate_materials
                WHERE object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                return None
            return self._parse_material(
                row,
                reference,
                expected_type,
            )
        finally:
            connection.close()

    def load_many(
        self,
        reference: ObjectRef,
        expected_type: type[object],
    ) -> tuple[object, ...] | None:
        del reference, expected_type
        return None

    @staticmethod
    def _authority(
        *,
        run_id: str,
        run_ref: ObjectRef,
        compiled_plan_ref: ObjectRef,
        trace_source_ref: ObjectRef,
        trace_envelope_ref: ObjectRef,
        preparation: TraceCandidatePreparation,
        audit: ContractAudit,
    ) -> FactoryTraceCandidateAuthorityV1:
        return FactoryTraceCandidateAuthorityV1.create(
            run_id=run_id,
            run_ref=run_ref,
            compiled_plan_ref=compiled_plan_ref,
            trace_source_ref=trace_source_ref,
            trace_envelope_ref=trace_envelope_ref,
            disposition=preparation.decision.disposition,
            decision_ref=preparation.decision.to_ref(),
            extracted_prompt_ref=(
                preparation.extracted_prompt.to_ref() if preparation.extracted_prompt is not None else None
            ),
            inferred_intent_ref=(
                preparation.inferred_intent.to_ref() if preparation.inferred_intent is not None else None
            ),
            rewrite_candidate_ref=(
                preparation.rewrite_candidate.to_ref() if preparation.rewrite_candidate is not None else None
            ),
            route_refs=sorted_refs(preparation.route_refs),
            audit=audit,
        )

    @staticmethod
    def _values(
        preparation: TraceCandidatePreparation,
    ) -> tuple[ContractModel, ...]:
        return tuple(
            value
            for value in (
                preparation.decision,
                preparation.extracted_prompt,
                preparation.inferred_intent,
                preparation.rewrite_candidate,
            )
            if value is not None
        )

    @staticmethod
    def _material_for_authority(
        authority: FactoryTraceCandidateAuthorityV1,
        preparation: TraceCandidatePreparation,
    ) -> FactoryTraceCandidateMaterial:
        if authority.decision_ref != preparation.decision.to_ref() or authority.route_refs != sorted_refs(
            preparation.route_refs
        ):
            raise FactoryControlIntegrityError(
                "trace candidate preparation differs from authority",
            )
        return FactoryTraceCandidateMaterial(
            authority=authority,
            preparation=preparation,
        )

    def _material(
        self,
        connection: sqlite3.Connection,
        authority: FactoryTraceCandidateAuthorityV1,
    ) -> FactoryTraceCandidateMaterial:
        decision = self._get_material(
            connection,
            authority.decision_ref,
            TraceCandidateDecisionV2,
        )
        extracted = self._optional_material(
            connection,
            authority.extracted_prompt_ref,
            ExtractedUserPromptV2,
        )
        intent = self._optional_material(
            connection,
            authority.inferred_intent_ref,
            InferredUserIntentV2,
        )
        rewrite = self._optional_material(
            connection,
            authority.rewrite_candidate_ref,
            TaskRewriteCandidateV2,
        )
        return self._material_for_authority(
            authority,
            TraceCandidatePreparation(
                decision=decision,
                extracted_prompt=extracted,
                inferred_intent=intent,
                rewrite_candidate=rewrite,
                route_refs=authority.route_refs,
            ),
        )

    def _current(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        trace_source_ref: ObjectRef,
        required: bool,
    ) -> FactoryTraceCandidateAuthorityV1 | None:
        row = connection.execute(
            """
            SELECT authority_object_id
            FROM factory_trace_candidate_current_heads
            WHERE run_id = ? AND trace_source_object_id = ?
            """,
            (run_id, trace_source_ref.object_id),
        ).fetchone()
        if row is None:
            immutable = connection.execute(
                """
                SELECT 1 FROM factory_trace_candidate_authorities
                WHERE run_id = ? AND trace_source_object_id = ?
                LIMIT 1
                """,
                (run_id, trace_source_ref.object_id),
            ).fetchone()
            if immutable is not None:
                raise FactoryControlIntegrityError(
                    "trace candidate current head is missing",
                )
            if required:
                raise FactoryControlNotFoundError(
                    "trace candidate authority was not found",
                )
            return None
        authority = self._load_authority(
            connection,
            str(row["authority_object_id"]),
        )
        if authority.trace_source_ref != trace_source_ref:
            raise FactoryControlIntegrityError(
                "trace candidate source authority drifted",
            )
        return authority

    @staticmethod
    def _load_authority(
        connection: sqlite3.Connection,
        object_id: str,
    ) -> FactoryTraceCandidateAuthorityV1:
        row = connection.execute(
            """
            SELECT * FROM factory_trace_candidate_authorities
            WHERE object_id = ?
            """,
            (object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError(
                "trace candidate authority is missing",
            )
        try:
            value = FactoryTraceCandidateAuthorityV1.model_validate_json(
                str(row["record_json"]),
            )
        except Exception as exc:
            raise FactoryControlIntegrityError(
                "trace candidate authority schema is invalid",
            ) from exc
        if (
            value.object_id != object_id
            or value.object_sha256 != row["object_sha256"]
            or value.run_id != row["run_id"]
            or value.trace_source_ref.object_id != row["trace_source_object_id"]
            or value.compiled_plan_ref.object_id != row["compiled_plan_object_id"]
            or value.canonical_json().decode() != row["record_json"]
        ):
            raise FactoryControlIntegrityError(
                "trace candidate authority drifted",
            )
        return value

    @staticmethod
    def _insert_material(
        connection: sqlite3.Connection,
        value: ContractModel,
    ) -> None:
        reference = value.to_ref()  # type: ignore[attr-defined]
        payload = value.canonical_json().decode()
        row = connection.execute(
            """
            SELECT * FROM factory_trace_candidate_materials
            WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if row is not None:
            if (
                row["object_type"] != reference.object_type
                or row["object_version"] != reference.object_version
                or row["object_sha256"] != reference.object_sha256
                or row["record_json"] != payload
            ):
                raise FactoryControlIntegrityError(
                    "trace candidate material identity conflicts",
                )
            return
        connection.execute(
            """
            INSERT INTO factory_trace_candidate_materials (
                object_id, object_type, object_version,
                object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                reference.object_id,
                reference.object_type,
                reference.object_version,
                reference.object_sha256,
                payload,
            ),
        )

    @staticmethod
    def _get_material[ValueT: ContractModel](
        connection: sqlite3.Connection,
        reference: ObjectRef,
        model_type: type[ValueT],
    ) -> ValueT:
        row = connection.execute(
            """
            SELECT * FROM factory_trace_candidate_materials
            WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError(
                "trace candidate material is missing",
            )
        return FactoryTraceCandidateMaterialStore._parse_material(
            row,
            reference,
            model_type,
        )

    @staticmethod
    def _optional_material[ValueT: ContractModel](
        connection: sqlite3.Connection,
        reference: ObjectRef | None,
        model_type: type[ValueT],
    ) -> ValueT | None:
        if reference is None:
            return None
        return FactoryTraceCandidateMaterialStore._get_material(
            connection,
            reference,
            model_type,
        )

    @staticmethod
    def _parse_material[ValueT: ContractModel](
        row: sqlite3.Row,
        reference: ObjectRef,
        model_type: type[ValueT],
    ) -> ValueT:
        try:
            value = model_type.model_validate_json(
                str(row["record_json"]),
            )
        except Exception as exc:
            raise FactoryControlIntegrityError(
                "trace candidate material schema is invalid",
            ) from exc
        if (
            row["object_type"] != reference.object_type
            or row["object_version"] != reference.object_version
            or row["object_sha256"] != reference.object_sha256
            or value.to_ref() != reference  # type: ignore[attr-defined]
            or value.canonical_json().decode() != row["record_json"]
        ):
            raise FactoryControlIntegrityError(
                "trace candidate material drifted",
            )
        return value

    def _initialize(self) -> None:
        connection = self.store._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS
                factory_trace_candidate_materials (
                    object_id TEXT PRIMARY KEY,
                    object_type TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS
                factory_trace_candidate_authorities (
                    object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    trace_source_object_id TEXT NOT NULL,
                    compiled_plan_object_id TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(run_id, trace_source_object_id)
                );
                CREATE TABLE IF NOT EXISTS
                factory_trace_candidate_current_heads (
                    run_id TEXT NOT NULL,
                    trace_source_object_id TEXT NOT NULL,
                    authority_object_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(run_id, trace_source_object_id)
                );
                """
            )
            connection.commit()
        finally:
            connection.close()


__all__ = [
    "FactoryTraceCandidateAuthorityV1",
    "FactoryTraceCandidateMaterial",
    "FactoryTraceCandidateMaterialError",
    "FactoryTraceCandidateMaterialStore",
]
