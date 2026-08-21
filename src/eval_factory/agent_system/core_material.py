from __future__ import annotations

import sqlite3
from typing import TypeVar

from eval_factory.agent_system.core_vertical import (
    CoreVerticalExecution,
)
from eval_factory.agent_system.store import (
    FactoryControlConflictError,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    CoreVerticalResultV2,
    ExtractedUserPromptV2,
    InferredUserIntentV2,
    TaskRewriteCandidateV2,
    TraceCandidateDecisionV2,
)
from eval_factory.contracts.core import ContractModel, ObjectRef

ModelT = TypeVar("ModelT", bound=ContractModel)


class FactoryCoreMaterialError(RuntimeError):
    pass


class FactoryCoreMaterialStore:
    def __init__(
        self,
        store: FactoryControlStore,
    ) -> None:
        self.store = store
        self._initialize()

    def commit(
        self,
        *,
        run_id: str,
        compiled_plan_ref: ObjectRef,
        execution: CoreVerticalExecution,
        idempotency_key: str,
    ) -> CoreVerticalExecution:
        self._validate_execution(execution)
        request_sha256 = execution.result.object_sha256
        scope = f"commit-core-execution:{run_id}"
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self.store._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="core-vertical-result",
            )
            if replay_id is not None:
                replay = self._load(
                    connection,
                    run_id=run_id,
                    result_object_id=replay_id,
                )
                connection.rollback()
                return replay
            run = self.store._load_current_run(
                connection,
                run_id,
            )
            plan_head = connection.execute(
                """
                SELECT compiled_plan_object_id
                FROM plan_current_heads
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                plan_head is None
                or plan_head["compiled_plan_object_id"] != compiled_plan_ref.object_id
                or run.compiled_plan_ref != compiled_plan_ref
            ):
                raise FactoryCoreMaterialError("core execution does not bind current compiled plan")
            request_row = connection.execute(
                """
                SELECT * FROM factory_dataset_run_requests
                WHERE dataset_run_id = ?
                """,
                (run_id,),
            ).fetchone()
            planning_head = connection.execute(
                """
                SELECT authority_object_id
                FROM factory_dataset_planning_current_heads
                WHERE dataset_run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if request_row is None or planning_head is None:
                raise FactoryCoreMaterialError("core execution lacks request or planning authority")
            request = self.store._parse_dataset_request_row(
                connection,
                request_row,
            )
            planning = self.store._load_planning_authority_by_object_id(
                connection,
                str(planning_head["authority_object_id"]),
            )
            if (
                execution.result.requirement_spec_ref != request.requirement_spec_ref
                or execution.result.manifest_ref != request.manifest_ref
                or planning.plan_ref != run.current_plan_ref
            ):
                raise FactoryCoreMaterialError("core execution source authority is stale")
            for value in self._values(execution):
                self._insert_material(connection, value)
            connection.execute(
                """
                INSERT INTO factory_core_executions (
                    result_object_id, run_id,
                    compiled_plan_object_id,
                    planning_authority_object_id,
                    manifest_object_id,
                    requirement_object_id
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.result.object_id,
                    run_id,
                    compiled_plan_ref.object_id,
                    planning.object_id,
                    execution.result.manifest_ref.object_id,
                    execution.result.requirement_spec_ref.object_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO factory_core_current_heads (
                    run_id, compiled_plan_object_id,
                    result_object_id
                ) VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    compiled_plan_object_id =
                        excluded.compiled_plan_object_id,
                    result_object_id = excluded.result_object_id
                """,
                (
                    run_id,
                    compiled_plan_ref.object_id,
                    execution.result.object_id,
                ),
            )
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=run.run_version,
                event_type="factory-core-execution-committed",
                object_id=execution.result.object_id,
                created_at=execution.result.audit.created_at.isoformat(),
            )
            self.store._insert_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="core-vertical-result",
                response_id=execution.result.object_id,
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise FactoryControlConflictError("factory core execution authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get(run_id)

    def get(
        self,
        run_id: str,
    ) -> CoreVerticalExecution:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT * FROM factory_core_current_heads
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if head is None:
                immutable = connection.execute(
                    """
                    SELECT 1 FROM factory_core_executions
                    WHERE run_id = ?
                    LIMIT 1
                    """,
                    (run_id,),
                ).fetchone()
                if immutable is not None:
                    raise FactoryControlIntegrityError("factory core current head is missing")
                raise FactoryControlNotFoundError("factory core execution was not found")
            value = self._load(
                connection,
                run_id=run_id,
                result_object_id=str(head["result_object_id"]),
            )
            run = self.store._load_current_run(
                connection,
                run_id,
            )
            if (
                run.compiled_plan_ref is None
                or head["compiled_plan_object_id"] != run.compiled_plan_ref.object_id
            ):
                raise FactoryControlIntegrityError("factory core head is stale for current plan")
            connection.rollback()
            return value
        finally:
            connection.close()

    def rebuild_current_heads(self) -> int:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM factory_core_executions
                ORDER BY run_id, compiled_plan_object_id
                """
            ).fetchall()
            connection.execute("DELETE FROM factory_core_current_heads")
            count = 0
            for row in rows:
                plan_head = connection.execute(
                    """
                    SELECT compiled_plan_object_id
                    FROM plan_current_heads
                    WHERE run_id = ?
                    """,
                    (row["run_id"],),
                ).fetchone()
                if (
                    plan_head is None
                    or plan_head["compiled_plan_object_id"] != row["compiled_plan_object_id"]
                ):
                    continue
                self._load(
                    connection,
                    run_id=str(row["run_id"]),
                    result_object_id=str(row["result_object_id"]),
                )
                connection.execute(
                    """
                    INSERT INTO factory_core_current_heads (
                        run_id, compiled_plan_object_id,
                        result_object_id
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        row["run_id"],
                        row["compiled_plan_object_id"],
                        row["result_object_id"],
                    ),
                )
                count += 1
            connection.commit()
            return count
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _load(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        result_object_id: str,
    ) -> CoreVerticalExecution:
        row = connection.execute(
            """
            SELECT * FROM factory_core_executions
            WHERE result_object_id = ?
            """,
            (result_object_id,),
        ).fetchone()
        if row is None or row["run_id"] != run_id:
            raise FactoryControlIntegrityError("factory core execution record is missing")
        result = self._get_material(
            connection,
            _material_ref(
                connection,
                result_object_id,
            ),
            CoreVerticalResultV2,
        )
        if (
            row["manifest_object_id"] != result.manifest_ref.object_id
            or row["requirement_object_id"] != result.requirement_spec_ref.object_id
        ):
            raise FactoryControlIntegrityError("factory core execution columns drifted")
        decision_refs = (
            *result.candidate_decision_refs,
            *result.non_candidate_decision_refs,
            *result.blocked_decision_refs,
        )
        execution = CoreVerticalExecution(
            result=result,
            decisions=tuple(
                sorted(
                    (
                        self._get_material(
                            connection,
                            ref,
                            TraceCandidateDecisionV2,
                        )
                        for ref in decision_refs
                    ),
                    key=lambda value: value.source_trace_id,
                )
            ),
            extracted_prompts=tuple(
                sorted(
                    (
                        self._get_material(
                            connection,
                            ref,
                            ExtractedUserPromptV2,
                        )
                        for ref in result.extracted_prompt_refs
                    ),
                    key=lambda value: value.object_id,
                )
            ),
            inferred_intents=tuple(
                sorted(
                    (
                        self._get_material(
                            connection,
                            ref,
                            InferredUserIntentV2,
                        )
                        for ref in result.inferred_intent_refs
                    ),
                    key=lambda value: value.object_id,
                )
            ),
            rewrite_candidates=tuple(
                sorted(
                    (
                        self._get_material(
                            connection,
                            ref,
                            TaskRewriteCandidateV2,
                        )
                        for ref in result.rewrite_candidate_refs
                    ),
                    key=lambda value: value.object_id,
                )
            ),
        )
        self._validate_execution(execution)
        return execution

    @staticmethod
    def _values(
        execution: CoreVerticalExecution,
    ) -> tuple[ContractModel, ...]:
        return (
            *execution.decisions,
            *execution.extracted_prompts,
            *execution.inferred_intents,
            *execution.rewrite_candidates,
            execution.result,
        )

    @staticmethod
    def _validate_execution(
        execution: CoreVerticalExecution,
    ) -> None:
        result = execution.result
        decision_refs = {value.to_ref() for value in execution.decisions}
        expected_decisions = set(
            (
                *result.candidate_decision_refs,
                *result.non_candidate_decision_refs,
                *result.blocked_decision_refs,
            )
        )
        if (
            decision_refs != expected_decisions
            or {value.to_ref() for value in execution.extracted_prompts} != set(result.extracted_prompt_refs)
            or {value.to_ref() for value in execution.inferred_intents} != set(result.inferred_intent_refs)
            or {value.to_ref() for value in execution.rewrite_candidates}
            != set(result.rewrite_candidate_refs)
        ):
            raise FactoryCoreMaterialError("core execution material inventory is not exact")

    @staticmethod
    def _insert_material(
        connection: sqlite3.Connection,
        value: ContractModel,
    ) -> None:
        reference = value.to_ref()  # type: ignore[attr-defined]
        payload = value.canonical_json().decode()
        existing = connection.execute(
            """
            SELECT * FROM factory_core_materials
            WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if existing is not None:
            if (
                existing["object_type"] != reference.object_type
                or existing["object_version"] != reference.object_version
                or existing["object_sha256"] != reference.object_sha256
                or existing["record_json"] != payload
            ):
                raise FactoryControlIntegrityError("factory core material identity conflicts")
            return
        connection.execute(
            """
            INSERT INTO factory_core_materials (
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
    def _get_material(
        connection: sqlite3.Connection,
        reference: ObjectRef,
        model_type: type[ModelT],
    ) -> ModelT:
        row = connection.execute(
            """
            SELECT * FROM factory_core_materials
            WHERE object_id = ?
            """,
            (reference.object_id,),
        ).fetchone()
        if row is None:
            raise FactoryControlIntegrityError("factory core material is missing")
        try:
            value = model_type.model_validate_json(str(row["record_json"]))
        except Exception as exc:
            raise FactoryControlIntegrityError("factory core material schema is invalid") from exc
        if (
            row["object_type"] != reference.object_type
            or row["object_version"] != reference.object_version
            or row["object_sha256"] != reference.object_sha256
            or value.to_ref() != reference  # type: ignore[attr-defined]
            or value.canonical_json().decode() != row["record_json"]
        ):
            raise FactoryControlIntegrityError("factory core material drifted")
        return value

    def _initialize(self) -> None:
        connection = self.store._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS factory_core_materials (
                    object_id TEXT PRIMARY KEY,
                    object_type TEXT NOT NULL,
                    object_version TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS factory_core_executions (
                    result_object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    compiled_plan_object_id TEXT NOT NULL,
                    planning_authority_object_id TEXT NOT NULL,
                    manifest_object_id TEXT NOT NULL,
                    requirement_object_id TEXT NOT NULL,
                    UNIQUE(run_id, compiled_plan_object_id)
                );

                CREATE TABLE IF NOT EXISTS factory_core_current_heads (
                    run_id TEXT PRIMARY KEY,
                    compiled_plan_object_id TEXT NOT NULL,
                    result_object_id TEXT NOT NULL UNIQUE
                );
                """
            )
            connection.commit()
        finally:
            connection.close()


def _material_ref(
    connection: sqlite3.Connection,
    object_id: str,
) -> ObjectRef:
    row = connection.execute(
        """
        SELECT object_type, object_version, object_sha256
        FROM factory_core_materials
        WHERE object_id = ?
        """,
        (object_id,),
    ).fetchone()
    if row is None:
        raise FactoryControlIntegrityError("factory core material ref is missing")
    return ObjectRef(
        object_type=str(row["object_type"]),
        object_id=object_id,
        object_version=str(row["object_version"]),
        object_sha256=str(row["object_sha256"]),
    )


__all__ = [
    "FactoryCoreMaterialError",
    "FactoryCoreMaterialStore",
]
