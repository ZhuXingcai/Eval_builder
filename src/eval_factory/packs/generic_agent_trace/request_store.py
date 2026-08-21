from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, Field, model_validator

from eval_factory.agent_system.private_store import FactoryPrivateObjectStore
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.harness import CapabilityRuntimeRegistry
from eval_factory.harness.contracts import (
    HarnessObjectV1,
    require_ref,
    require_sorted_unique_refs,
    sorted_refs,
)
from eval_factory.packs.generic_agent_trace.factory_workflow import (
    GenericAgentCapabilityRequestSource,
    GenericAgentFactoryWorkflowError,
    GenericAgentTeamTaskSpec,
)
from eval_factory.team import TeamSnapshot, TeamTaskWork


class GenericAgentCapabilityRequestStoreError(RuntimeError):
    pass


class GenericAgentCapabilityRequestConflictError(
    GenericAgentCapabilityRequestStoreError,
):
    pass


class GenericAgentCapabilityRequestIntegrityError(
    GenericAgentCapabilityRequestStoreError,
):
    pass


class GenericAgentCapabilityRequestNotFoundError(
    GenericAgentCapabilityRequestStoreError,
):
    pass


class GenericAgentCapabilityRequestBindingV1(HarnessObjectV1):
    schema_version: Literal["generic-agent-trace/capability-request-binding/v1"] = (
        "generic-agent-trace/capability-request-binding/v1"
    )
    OBJECT_TYPE: ClassVar[str] = "generic-capability-request-binding"

    task_ref: ObjectRef
    capability_definition_ref: ObjectRef
    request_ref: ObjectRef
    request_material_ref: ObjectRef
    source_authority_refs: tuple[ObjectRef, ...] = Field(
        default=(),
        max_length=10_000,
    )
    binding_version: int = Field(ge=1, le=1_000_000)
    predecessor_binding_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        require_ref(self.task_ref, "team-task", "task_ref")
        require_ref(
            self.capability_definition_ref,
            "harness-capability-definition",
            "capability_definition_ref",
        )
        require_ref(
            self.request_material_ref,
            "generic-agent-capability-request-body",
            "request_material_ref",
            object_version="v2",
        )
        require_sorted_unique_refs(
            self.source_authority_refs,
            "source_authority_refs",
        )
        if self.binding_version == 1:
            if self.predecessor_binding_ref is not None:
                raise ValueError("first request binding has no predecessor")
        elif self.predecessor_binding_ref is None:
            raise ValueError("request binding successor requires predecessor")
        else:
            require_ref(
                self.predecessor_binding_ref,
                self.OBJECT_TYPE,
                "predecessor_binding_ref",
            )
        return self


class GenericAgentCapabilityRequestStore:
    """Durable content-free binding from Team tasks to private requests."""

    def __init__(
        self,
        path: Path,
        *,
        private_store: FactoryPrivateObjectStore,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.private_store = private_store
        self._fault = fault_injector or (lambda _: None)
        self._initialize()

    def commit(
        self,
        *,
        task_ref: ObjectRef,
        capability_definition_ref: ObjectRef,
        request: ContractModelV2,
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> GenericAgentCapabilityRequestBindingV1:
        request_ref = _model_ref(request)
        ordered_sources = sorted_refs(source_authority_refs)
        if ordered_sources != source_authority_refs:
            raise ValueError("request source authority refs must be sorted")
        material_ref = self.private_store.put_model(
            object_type="generic-agent-capability-request-body",
            value=request,
        )
        request_sha256 = _command_sha256(
            task_ref=task_ref,
            capability_definition_ref=capability_definition_ref,
            request_ref=request_ref,
            material_ref=material_ref,
            source_authority_refs=ordered_sources,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay(
                connection,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay is not None:
                connection.rollback()
                return replay
            current = self._current(
                connection,
                task_ref,
                required=False,
            )
            if current is not None and (
                current.capability_definition_ref == capability_definition_ref
                and current.request_ref == request_ref
                and current.request_material_ref == material_ref
                and current.source_authority_refs == ordered_sources
            ):
                self._remember(
                    connection,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_id=current.object_id,
                )
                connection.commit()
                return current
            binding = GenericAgentCapabilityRequestBindingV1.create(
                task_ref=task_ref,
                capability_definition_ref=capability_definition_ref,
                request_ref=request_ref,
                request_material_ref=material_ref,
                source_authority_refs=ordered_sources,
                binding_version=(1 if current is None else current.binding_version + 1),
                predecessor_binding_ref=(None if current is None else current.to_ref()),
                audit=audit,
            )
            connection.execute(
                """
                INSERT INTO capability_request_bindings (
                    object_id, task_object_id, binding_version,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    binding.object_id,
                    task_ref.object_id,
                    binding.binding_version,
                    binding.object_sha256,
                    binding.canonical_json().decode(),
                ),
            )
            self._fault("after_binding")
            if current is None:
                connection.execute(
                    "INSERT INTO capability_request_heads VALUES (?, ?, ?)",
                    (
                        task_ref.object_id,
                        binding.object_id,
                        binding.binding_version,
                    ),
                )
            else:
                changed = connection.execute(
                    """
                    UPDATE capability_request_heads
                    SET binding_object_id = ?, binding_version = ?
                    WHERE task_object_id = ? AND binding_version = ?
                    """,
                    (
                        binding.object_id,
                        binding.binding_version,
                        task_ref.object_id,
                        current.binding_version,
                    ),
                ).rowcount
                if changed != 1:
                    raise GenericAgentCapabilityRequestConflictError(
                        "capability request head changed concurrently",
                    )
            self._fault("after_head")
            connection.execute(
                """
                INSERT INTO capability_request_outbox (
                    binding_object_id, task_object_id, delivered
                ) VALUES (?, ?, 0)
                """,
                (binding.object_id, task_ref.object_id),
            )
            self._fault("after_outbox")
            self._remember(
                connection,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_id=binding.object_id,
            )
            self._fault("after_idempotency")
            connection.commit()
            return binding
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise GenericAgentCapabilityRequestConflictError(
                "capability request authority already exists",
            ) from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(
        self,
        task_ref: ObjectRef,
    ) -> GenericAgentCapabilityRequestBindingV1:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            binding = self._current(
                connection,
                task_ref,
                required=True,
            )
            assert binding is not None
            connection.rollback()
            return binding
        finally:
            connection.close()

    def load_request[RequestT: ContractModelV2](
        self,
        binding: GenericAgentCapabilityRequestBindingV1,
        request_model: type[RequestT],
    ) -> RequestT:
        current = self.get(binding.task_ref)
        if current != binding:
            raise GenericAgentCapabilityRequestIntegrityError(
                "capability request binding is stale",
            )
        request = self.private_store.get_model(
            binding.request_material_ref,
            request_model,
        )
        if _model_ref(request) != binding.request_ref:
            raise GenericAgentCapabilityRequestIntegrityError(
                "private capability request differs from binding authority",
            )
        return request

    def rebuild_current_heads(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT task_object_id, record_json
                FROM capability_request_bindings
                ORDER BY task_object_id, binding_version
                """
            ).fetchall()
            current: dict[
                str,
                GenericAgentCapabilityRequestBindingV1,
            ] = {}
            for row in rows:
                binding = _parse_binding(str(row["record_json"]))
                task_id = str(row["task_object_id"])
                prior = current.get(task_id)
                if (
                    prior is None
                    and (binding.binding_version != 1 or binding.predecessor_binding_ref is not None)
                ) or (
                    prior is not None
                    and (
                        binding.binding_version != prior.binding_version + 1
                        or binding.predecessor_binding_ref != prior.to_ref()
                    )
                ):
                    raise GenericAgentCapabilityRequestIntegrityError(
                        "capability request binding chain is not contiguous",
                    )
                current[task_id] = binding
            connection.execute("DELETE FROM capability_request_heads")
            for task_id, binding in sorted(current.items()):
                connection.execute(
                    "INSERT INTO capability_request_heads VALUES (?, ?, ?)",
                    (task_id, binding.object_id, binding.binding_version),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return tuple(sorted(current))

    def _current(
        self,
        connection: sqlite3.Connection,
        task_ref: ObjectRef,
        *,
        required: bool,
    ) -> GenericAgentCapabilityRequestBindingV1 | None:
        require_ref(task_ref, "team-task", "task_ref")
        row = connection.execute(
            """
            SELECT h.binding_version head_version, b.*
            FROM capability_request_heads h
            JOIN capability_request_bindings b
              ON b.object_id = h.binding_object_id
            WHERE h.task_object_id = ?
            """,
            (task_ref.object_id,),
        ).fetchone()
        if row is None:
            if required:
                raise GenericAgentCapabilityRequestNotFoundError(
                    "capability request binding was not found",
                )
            return None
        binding = _parse_binding(str(row["record_json"]))
        if (
            binding.task_ref != task_ref
            or row["task_object_id"] != task_ref.object_id
            or row["object_id"] != binding.object_id
            or row["object_sha256"] != binding.object_sha256
            or row["binding_version"] != binding.binding_version
            or row["head_version"] != binding.binding_version
        ):
            raise GenericAgentCapabilityRequestIntegrityError(
                "capability request head drifted",
            )
        return binding

    def _replay(
        self,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        request_sha256: str,
    ) -> GenericAgentCapabilityRequestBindingV1 | None:
        row = connection.execute(
            """
            SELECT request_sha256, response_id
            FROM capability_request_idempotency
            WHERE scope = 'commit-request' AND idempotency_key = ?
            """,
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha256:
            raise GenericAgentCapabilityRequestConflictError(
                "capability request idempotency key binds another command",
            )
        stored = connection.execute(
            "SELECT record_json FROM capability_request_bindings WHERE object_id = ?",
            (row["response_id"],),
        ).fetchone()
        if stored is None:
            raise GenericAgentCapabilityRequestIntegrityError(
                "capability request replay response is missing",
            )
        return _parse_binding(str(stored["record_json"]))

    @staticmethod
    def _remember(
        connection: sqlite3.Connection,
        *,
        idempotency_key: str,
        request_sha256: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO capability_request_idempotency (
                scope, idempotency_key, request_sha256, response_id
            ) VALUES ('commit-request', ?, ?, ?)
            """,
            (idempotency_key, request_sha256, response_id),
        )

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
                CREATE TABLE IF NOT EXISTS capability_request_bindings (
                    object_id TEXT PRIMARY KEY,
                    task_object_id TEXT NOT NULL,
                    binding_version INTEGER NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(task_object_id, binding_version)
                );
                CREATE TABLE IF NOT EXISTS capability_request_heads (
                    task_object_id TEXT PRIMARY KEY,
                    binding_object_id TEXT NOT NULL UNIQUE,
                    binding_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capability_request_outbox (
                    binding_object_id TEXT PRIMARY KEY,
                    task_object_id TEXT NOT NULL,
                    delivered INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capability_request_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );
                """
            )
            connection.commit()
        finally:
            connection.close()


class StoredGenericAgentCapabilityRequestSource(
    GenericAgentCapabilityRequestSource,
):
    def __init__(
        self,
        *,
        store: GenericAgentCapabilityRequestStore,
        registry: CapabilityRuntimeRegistry,
    ) -> None:
        self.store = store
        self.registry = registry

    def build(
        self,
        *,
        snapshot: TeamSnapshot,
        work: TeamTaskWork,
        spec: GenericAgentTeamTaskSpec,
    ) -> BaseModel:
        graph_tasks = {value.task_id: value for value in snapshot.graph.tasks}
        if graph_tasks.get(work.task.task_id) != work.task:
            raise GenericAgentFactoryWorkflowError(
                "request source task differs from Team authority",
            )
        binding = self.store.get(work.task.to_ref())
        definition = next(
            (
                value
                for value in self.registry.registration.capability_definitions
                if value.to_ref() == work.task.capability_definition_ref
            ),
            None,
        )
        if (
            definition is None
            or definition.capability_id != spec.capability_id
            or binding.capability_definition_ref != definition.to_ref()
        ):
            raise GenericAgentFactoryWorkflowError(
                "stored request capability differs from Team task",
            )
        provider_binding = next(
            value
            for value in self.registry.registration.provider_bindings
            if value.capability_definition_ref == definition.to_ref()
        )
        provider = next(
            value
            for value in self.registry.providers
            if value.provider_binding_ref == provider_binding.to_ref()
        )
        request = self.store.load_request(
            binding,
            provider.request_model,  # type: ignore[type-var]
        )
        if not isinstance(request, BaseModel):
            raise GenericAgentFactoryWorkflowError(
                "stored capability request is not a Pydantic model",
            )
        return request


def _model_ref(value: BaseModel) -> ObjectRef:
    to_ref = getattr(value, "to_ref", None)
    if not callable(to_ref):
        raise ValueError("capability request lacks immutable reference authority")
    reference = to_ref()
    if not isinstance(reference, ObjectRef):
        raise ValueError("capability request reference authority is invalid")
    return reference


def _command_sha256(
    *,
    task_ref: ObjectRef,
    capability_definition_ref: ObjectRef,
    request_ref: ObjectRef,
    material_ref: ObjectRef,
    source_authority_refs: tuple[ObjectRef, ...],
) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(
                {
                    "task_ref": task_ref,
                    "capability_definition_ref": capability_definition_ref,
                    "request_ref": request_ref,
                    "material_ref": material_ref,
                    "source_authority_refs": source_authority_refs,
                }
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _parse_binding(
    payload: str,
) -> GenericAgentCapabilityRequestBindingV1:
    try:
        binding = GenericAgentCapabilityRequestBindingV1.model_validate_json(
            payload,
        )
    except Exception as exc:
        raise GenericAgentCapabilityRequestIntegrityError(
            "capability request binding is invalid",
        ) from exc
    if binding.canonical_json().decode() != payload:
        raise GenericAgentCapabilityRequestIntegrityError(
            "capability request binding is not canonical",
        )
    return binding


__all__ = [
    "GenericAgentCapabilityRequestBindingV1",
    "GenericAgentCapabilityRequestConflictError",
    "GenericAgentCapabilityRequestIntegrityError",
    "GenericAgentCapabilityRequestNotFoundError",
    "GenericAgentCapabilityRequestStore",
    "GenericAgentCapabilityRequestStoreError",
    "StoredGenericAgentCapabilityRequestSource",
]
