from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from eval_factory.contracts.ai_gateway_v2 import (
    GatewayInvocationRequestV2,
    GatewayInvocationResultV2,
    GatewayReceiptV2,
    ModelRouteDecisionV2,
    ModelRouteRequestV2,
    RAGRequestV2,
    RAGResultV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2


class GatewayRecordError(RuntimeError):
    pass


class GatewayRecordConflictError(GatewayRecordError):
    pass


class GatewayRecordIntegrityError(GatewayRecordError):
    pass


class GatewayRecordStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_route_records (
                    request_object_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    agent_task_object_id TEXT NOT NULL,
                    route_version INTEGER NOT NULL,
                    decision_object_id TEXT NOT NULL UNIQUE,
                    decision_sha256 TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    UNIQUE(agent_task_object_id, route_version)
                );

                CREATE TABLE IF NOT EXISTS gateway_invocation_records (
                    request_object_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    route_decision_object_id TEXT NOT NULL,
                    receipt_object_id TEXT NOT NULL UNIQUE,
                    result_object_id TEXT NOT NULL UNIQUE,
                    request_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_result_records (
                    request_object_id TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    result_object_id TEXT NOT NULL UNIQUE,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    def commit_route(
        self,
        request: ModelRouteRequestV2,
        decision: ModelRouteDecisionV2,
    ) -> ModelRouteDecisionV2:
        if decision.request_ref != request.to_ref():
            raise GatewayRecordConflictError("route decision does not bind request")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM model_route_records WHERE request_object_id = ?",
                (request.object_id,),
            ).fetchone()
            if row is not None:
                stored_request = _parse(
                    ModelRouteRequestV2,
                    str(row["request_json"]),
                    "route request",
                )
                stored_decision = _parse(
                    ModelRouteDecisionV2,
                    str(row["decision_json"]),
                    "route decision",
                )
                if stored_request != request or stored_decision != decision:
                    raise GatewayRecordConflictError("route request already has different authority")
                connection.rollback()
                return stored_decision
            connection.execute(
                """
                INSERT INTO model_route_records (
                    request_object_id, request_sha256, agent_task_object_id,
                    route_version, decision_object_id, decision_sha256,
                    request_json, decision_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.object_id,
                    request.object_sha256,
                    request.agent_task_ref.object_id,
                    request.route_version,
                    decision.object_id,
                    decision.object_sha256,
                    _json(request),
                    _json(decision),
                ),
            )
            connection.commit()
            return decision
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise GatewayRecordConflictError("route authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_route(self, reference: ObjectRef) -> ModelRouteDecisionV2:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM model_route_records WHERE decision_object_id = ?",
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise GatewayRecordIntegrityError("route decision authority is missing")
            value = _parse(
                ModelRouteDecisionV2,
                str(row["decision_json"]),
                "route decision",
            )
            if value.to_ref() != reference or row["decision_sha256"] != value.object_sha256:
                raise GatewayRecordIntegrityError("route decision materialized identity drifted")
            return value
        finally:
            connection.close()

    def commit_invocation(
        self,
        request: GatewayInvocationRequestV2,
        receipt: GatewayReceiptV2,
        result: GatewayInvocationResultV2,
    ) -> GatewayInvocationResultV2:
        if (
            receipt.invocation_request_ref != request.to_ref()
            or result.invocation_request_ref != request.to_ref()
            or result.receipt_ref != receipt.to_ref()
            or receipt.route_decision_ref != request.route_decision_ref
            or result.route_decision_ref != request.route_decision_ref
        ):
            raise GatewayRecordConflictError("invocation closure refs are inconsistent")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM gateway_invocation_records
                WHERE request_object_id = ?
                """,
                (request.object_id,),
            ).fetchone()
            if row is not None:
                stored = self._parse_invocation_row(row)
                if stored != (request, receipt, result):
                    raise GatewayRecordConflictError("invocation request already has different authority")
                connection.rollback()
                return stored[2]
            connection.execute(
                """
                INSERT INTO gateway_invocation_records (
                    request_object_id, request_sha256,
                    route_decision_object_id, receipt_object_id,
                    result_object_id, request_json, receipt_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.object_id,
                    request.object_sha256,
                    request.route_decision_ref.object_id,
                    receipt.object_id,
                    result.object_id,
                    _json(request),
                    _json(receipt),
                    _json(result),
                ),
            )
            connection.commit()
            return result
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise GatewayRecordConflictError("invocation authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_invocation(
        self,
        request_ref: ObjectRef,
    ) -> tuple[GatewayInvocationRequestV2, GatewayReceiptV2, GatewayInvocationResultV2] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM gateway_invocation_records
                WHERE request_object_id = ?
                """,
                (request_ref.object_id,),
            ).fetchone()
            if row is None:
                return None
            values = self._parse_invocation_row(row)
            if values[0].to_ref() != request_ref:
                raise GatewayRecordIntegrityError("invocation request ref drifted")
            return values
        finally:
            connection.close()

    def get_receipt(self, reference: ObjectRef) -> GatewayReceiptV2:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM gateway_invocation_records
                WHERE receipt_object_id = ?
                """,
                (reference.object_id,),
            ).fetchone()
            if row is None:
                raise GatewayRecordIntegrityError("gateway receipt authority is missing")
            receipt = _parse(GatewayReceiptV2, str(row["receipt_json"]), "gateway receipt")
            if receipt.to_ref() != reference:
                raise GatewayRecordIntegrityError("gateway receipt identity drifted")
            return receipt
        finally:
            connection.close()

    def commit_rag(
        self,
        request: RAGRequestV2,
        result: RAGResultV2,
    ) -> RAGResultV2:
        if result.request_ref != request.to_ref():
            raise GatewayRecordConflictError("RAG result does not bind request")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM rag_result_records WHERE request_object_id = ?",
                (request.object_id,),
            ).fetchone()
            if row is not None:
                stored_request = _parse(
                    RAGRequestV2,
                    str(row["request_json"]),
                    "RAG request",
                )
                stored_result = _parse(
                    RAGResultV2,
                    str(row["result_json"]),
                    "RAG result",
                )
                if stored_request != request or stored_result != result:
                    raise GatewayRecordConflictError("RAG request already has different authority")
                connection.rollback()
                return stored_result
            connection.execute(
                """
                INSERT INTO rag_result_records (
                    request_object_id, request_sha256, result_object_id,
                    request_json, result_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request.object_id,
                    request.object_sha256,
                    result.object_id,
                    _json(request),
                    _json(result),
                ),
            )
            connection.commit()
            return result
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise GatewayRecordConflictError("RAG authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_rag(self, request_ref: ObjectRef) -> RAGResultV2 | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM rag_result_records WHERE request_object_id = ?",
                (request_ref.object_id,),
            ).fetchone()
            if row is None:
                return None
            request = _parse(RAGRequestV2, str(row["request_json"]), "RAG request")
            result = _parse(RAGResultV2, str(row["result_json"]), "RAG result")
            if request.to_ref() != request_ref or result.request_ref != request_ref:
                raise GatewayRecordIntegrityError("RAG record identity drifted")
            return result
        finally:
            connection.close()

    @staticmethod
    def _parse_invocation_row(
        row: sqlite3.Row,
    ) -> tuple[GatewayInvocationRequestV2, GatewayReceiptV2, GatewayInvocationResultV2]:
        request = _parse(
            GatewayInvocationRequestV2,
            str(row["request_json"]),
            "gateway invocation request",
        )
        receipt = _parse(
            GatewayReceiptV2,
            str(row["receipt_json"]),
            "gateway receipt",
        )
        result = _parse(
            GatewayInvocationResultV2,
            str(row["result_json"]),
            "gateway invocation result",
        )
        if (
            row["request_sha256"] != request.object_sha256
            or row["route_decision_object_id"] != request.route_decision_ref.object_id
            or row["receipt_object_id"] != receipt.object_id
            or row["result_object_id"] != result.object_id
        ):
            raise GatewayRecordIntegrityError("gateway invocation materialized columns drifted")
        return request, receipt, result


def _json(value: ContractModelV2) -> str:
    return value.canonical_json().decode()


def _parse[RecordT: ContractModelV2](
    model_type: type[RecordT],
    payload: str,
    label: str,
) -> RecordT:
    try:
        value = model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise GatewayRecordIntegrityError(f"{label} is invalid") from exc
    if _json(value) != payload:
        raise GatewayRecordIntegrityError(f"{label} is not canonical")
    return value


__all__ = [
    "GatewayRecordConflictError",
    "GatewayRecordError",
    "GatewayRecordIntegrityError",
    "GatewayRecordStore",
]
