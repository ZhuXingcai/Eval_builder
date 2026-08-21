from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.contracts.core import ContractModel, ObjectRef


class GradingDesignMaterialError(RuntimeError):
    pass


class GradingDesignMaterialStore:
    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.private_store = FactoryPrivateObjectStore(
            self.root / "cas",
        )
        self.database = self.root / "grading-materials.sqlite3"
        self._initialize()

    def put_model(
        self,
        *,
        behavior_ref: ObjectRef,
        value: ContractModel,
    ) -> ObjectRef:
        payload = value.canonical_json()
        content_ref = self.private_store.put_bytes(
            object_type="grading-private-material",
            payload=payload,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM grading_materials
                WHERE behavior_object_id = ?
                """,
                (behavior_ref.object_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["behavior_object_type"] != behavior_ref.object_type
                    or existing["behavior_object_version"] != behavior_ref.object_version
                    or existing["behavior_object_sha256"] != behavior_ref.object_sha256
                    or existing["content_object_id"] != content_ref.object_id
                    or existing["content_sha256"] != content_ref.object_sha256
                ):
                    raise GradingDesignMaterialError(
                        "grading behavior ref is already bound to different material"
                    )
                connection.rollback()
                return content_ref
            connection.execute(
                """
                INSERT INTO grading_materials (
                    behavior_object_id, behavior_object_type,
                    behavior_object_version, behavior_object_sha256,
                    content_object_id, content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    behavior_ref.object_id,
                    behavior_ref.object_type,
                    behavior_ref.object_version,
                    behavior_ref.object_sha256,
                    content_ref.object_id,
                    content_ref.object_sha256,
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return content_ref

    def get_model[ModelT: ContractModel](
        self,
        behavior_ref: ObjectRef,
        model_type: type[ModelT],
    ) -> ModelT:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                """
                SELECT * FROM grading_materials
                WHERE behavior_object_id = ?
                """,
                (behavior_ref.object_id,),
            ).fetchone()
            if row is None:
                raise GradingDesignMaterialError("grading private material is missing")
            if (
                row["behavior_object_type"] != behavior_ref.object_type
                or row["behavior_object_version"] != behavior_ref.object_version
                or row["behavior_object_sha256"] != behavior_ref.object_sha256
            ):
                raise GradingDesignMaterialError("grading behavior ref binding drifted")
            content_ref = ObjectRef(
                object_type="grading-private-material",
                object_id=str(row["content_object_id"]),
                object_version="v2",
                object_sha256=str(row["content_sha256"]),
            )
            connection.rollback()
        finally:
            connection.close()
        payload = self.private_store.get_bytes(content_ref)
        try:
            value = model_type.model_validate_json(payload)
        except ValidationError as exc:
            raise GradingDesignMaterialError("grading private material has the wrong schema") from exc
        if value.canonical_json() != payload:
            raise GradingDesignMaterialError("grading private material is not canonical")
        return value

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS grading_materials (
                    behavior_object_id TEXT PRIMARY KEY,
                    behavior_object_type TEXT NOT NULL,
                    behavior_object_version TEXT NOT NULL,
                    behavior_object_sha256 TEXT NOT NULL,
                    content_object_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


__all__ = [
    "GradingDesignMaterialError",
    "GradingDesignMaterialStore",
]
