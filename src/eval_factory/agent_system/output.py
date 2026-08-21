from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.agent_system.core_vertical import CoreVerticalExecution
from eval_factory.agent_system.store import (
    FactoryControlNotFoundError,
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    DatasetDeliveryManifestV2,
    EvaluationRequirementSpecV2,
    FactoryCompletionOutcomeV2,
    FactoryRunCompletionV2,
    FactoryRunV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2


class CoreOutputError(RuntimeError):
    pass


class CoreOutputConflictError(CoreOutputError):
    pass


class CoreOutputIntegrityError(CoreOutputError):
    pass


class CoreOutputFileV1(ContractModelV2):
    schema_version: Literal["eval-factory/core-output-file/v1"] = "eval-factory/core-output-file/v1"
    relative_path: str = Field(min_length=1, max_length=512)
    size_bytes: int = Field(ge=0, le=1_000_000_000)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("core output path must be safe and relative")
        return self


class CoreOutputInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/core-output-inventory/v1"] = "eval-factory/core-output-inventory/v1"
    inventory_id: str
    run_id: str
    source_result_ref: ObjectRef
    completion_ref: ObjectRef
    delivery_manifest_ref: ObjectRef
    files: tuple[CoreOutputFileV1, ...] = Field(min_length=1, max_length=64)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        paths = tuple(value.relative_path for value in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("core output files must be sorted and unique")
        observed_bundle = _hash_json(
            [
                {
                    "content_sha256": value.content_sha256,
                    "relative_path": value.relative_path,
                    "size_bytes": value.size_bytes,
                }
                for value in self.files
            ]
        )
        if self.bundle_sha256 != observed_bundle:
            raise ValueError("core output bundle identity is stale")
        observed_inventory = _inventory_sha256(self)
        if (
            self.inventory_sha256 != observed_inventory
            or self.inventory_id != f"core-output-inventory://sha256/{observed_inventory}"
        ):
            raise ValueError("core output inventory identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="core-output-inventory",
            object_id=self.inventory_id,
            object_version="v1",
            object_sha256=self.inventory_sha256,
        )


class CoreOutputViewV1(ContractModelV2):
    schema_version: Literal["eval-factory/core-output-view/v1"] = "eval-factory/core-output-view/v1"
    run_id: str
    completion: FactoryRunCompletionV2
    delivery_manifest: DatasetDeliveryManifestV2
    inventory: CoreOutputInventoryV1


@dataclass(frozen=True, slots=True)
class CoreOutputWrite:
    view: CoreOutputViewV1
    bundle_path: Path
    reused: bool


class CoreOutputAssembler:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        root: Path,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.root = root.expanduser().resolve()
        if self.root.is_symlink() or (self.root.exists() and not self.root.is_dir()):
            raise CoreOutputError("core output root must be a real directory")
        self._cas = self.root / "cas/sha256"
        self._staging = self.root / ".staging"
        self._cas.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._fault_injector = fault_injector
        self._initialize()

    def assemble(
        self,
        *,
        run_id: str,
        requirement: EvaluationRequirementSpecV2,
        execution: CoreVerticalExecution,
        audit: ContractAudit,
        idempotency_key: str,
    ) -> CoreOutputWrite:
        replay = self._replay_if_existing(
            run_id=run_id,
            requirement=requirement,
            execution=execution,
            idempotency_key=idempotency_key,
        )
        if replay is not None:
            return replay
        run = self.store.get_run(run_id)
        _, compiled = self.store.get_plan(run_id)
        if (
            run.requirement_spec_ref != requirement.to_ref()
            or execution.result.requirement_spec_ref != requirement.to_ref()
        ):
            raise CoreOutputConflictError("core output inputs do not bind the current requirement")
        completion = self._completion(
            run=run,
            compiled_plan_ref=compiled.to_ref(),
            execution=execution,
            audit=audit,
        )
        payloads, manifest = self._payloads(
            run=run,
            requirement=requirement,
            execution=execution,
            completion=completion,
            audit=audit,
        )
        payloads["manifest.json"] = manifest.canonical_json() + b"\n"
        payloads["completion.json"] = completion.canonical_json() + b"\n"
        inventory = _inventory(
            run_id=run_id,
            source_result_ref=execution.result.to_ref(),
            completion_ref=completion.to_ref(),
            delivery_manifest_ref=manifest.to_ref(),
            payloads=payloads,
        )
        bundle_path, reused = self._write_bundle(payloads, inventory)
        self._fault("after_bundle")
        request_sha256 = _request_sha256(
            run_id,
            requirement.to_ref(),
            execution.result.to_ref(),
            completion.to_ref(),
            manifest.to_ref(),
            inventory.to_ref(),
        )
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_inventory_id = self._idempotent_inventory(
                connection,
                run_id=run_id,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_inventory_id is not None:
                connection.rollback()
                view = self.get(run_id)
                self._verify_bundle(bundle_path, view.inventory)
                return CoreOutputWrite(
                    view=view,
                    bundle_path=bundle_path,
                    reused=True,
                )
            current = self.store._load_current_run(connection, run_id)
            if current != run:
                raise CoreOutputConflictError("Factory run changed during output assembly")
            self._insert_completion(connection, run_id, completion)
            self._fault("after_completion")
            self._insert_manifest(connection, run_id, manifest)
            self._fault("after_manifest")
            self._insert_inventory(connection, inventory)
            self._fault("after_inventory")
            connection.execute(
                """
                INSERT INTO core_output_current_heads (
                    run_id, inventory_id, manifest_object_id
                ) VALUES (?, ?, ?)
                """,
                (run_id, inventory.inventory_id, manifest.object_id),
            )
            self._fault("after_output_head")
            result_refs = _sorted_refs(
                (
                    *current.result_refs,
                    completion.to_ref(),
                    manifest.to_ref(),
                    inventory.to_ref(),
                )
            )
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=current.status,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=current.current_plan_ref,
                compiled_plan_ref=current.compiled_plan_ref,
                active_task_refs=current.active_task_refs,
                result_refs=result_refs,
                pending_review_ref=current.pending_review_ref,
                planner_assessment_ref=current.planner_assessment_ref,
                completion_ref=current.completion_ref,
                delivery_manifest_ref=current.delivery_manifest_ref,
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self.store._insert_run(connection, successor)
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    run_id,
                    current.run_version,
                ),
            ).rowcount
            if changed != 1:
                raise CoreOutputConflictError("Factory run head changed during output commit")
            self._fault("after_run_head")
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run_id,
                aggregate_version=successor.run_version,
                event_type="core-output-assembled",
                object_id=inventory.inventory_id,
                created_at=audit.created_at.isoformat(),
            )
            self._fault("after_outbox")
            connection.execute(
                """
                INSERT INTO factory_control_idempotency (
                    scope, idempotency_key, request_sha256,
                    response_type, response_id
                ) VALUES (?, ?, ?, 'core-output-inventory', ?)
                """,
                (
                    f"assemble-core-output:{run_id}",
                    idempotency_key,
                    request_sha256,
                    inventory.inventory_id,
                ),
            )
            self._fault("after_idempotency")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CoreOutputConflictError("core output authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return CoreOutputWrite(
            view=self.get(run_id),
            bundle_path=bundle_path,
            reused=reused,
        )

    def get(self, run_id: str) -> CoreOutputViewV1:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                "SELECT * FROM core_output_current_heads WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if head is None:
                raise FactoryControlNotFoundError(f"core output not found: {run_id}")
            inventory_row = connection.execute(
                "SELECT * FROM core_output_inventories WHERE inventory_id = ?",
                (head["inventory_id"],),
            ).fetchone()
            if inventory_row is None:
                raise CoreOutputIntegrityError("core output head points to missing inventory")
            inventory = CoreOutputInventoryV1.model_validate_json(str(inventory_row["record_json"]))
            completion_row = connection.execute(
                "SELECT * FROM factory_run_completions WHERE object_id = ?",
                (inventory.completion_ref.object_id,),
            ).fetchone()
            manifest_row = connection.execute(
                "SELECT * FROM dataset_delivery_manifests WHERE object_id = ?",
                (inventory.delivery_manifest_ref.object_id,),
            ).fetchone()
            if completion_row is None or manifest_row is None:
                raise CoreOutputIntegrityError("core output authority is incomplete")
            completion = FactoryRunCompletionV2.model_validate_json(str(completion_row["record_json"]))
            manifest = DatasetDeliveryManifestV2.model_validate_json(str(manifest_row["record_json"]))
            if (
                head["manifest_object_id"] != manifest.object_id
                or inventory_row["run_id"] != run_id
                or inventory_row["bundle_sha256"] != inventory.bundle_sha256
                or inventory_row["object_sha256"] != inventory.inventory_sha256
                or completion.to_ref() != inventory.completion_ref
                or manifest.to_ref() != inventory.delivery_manifest_ref
            ):
                raise CoreOutputIntegrityError("core output materialized columns drifted")
            connection.rollback()
        finally:
            connection.close()
        self._verify_bundle(self._bundle_path(inventory.bundle_sha256), inventory)
        return CoreOutputViewV1(
            run_id=run_id,
            completion=completion,
            delivery_manifest=manifest,
            inventory=inventory,
        )

    def _completion(
        self,
        *,
        run: FactoryRunV2,
        compiled_plan_ref: ObjectRef,
        execution: CoreVerticalExecution,
        audit: ContractAudit,
    ) -> FactoryRunCompletionV2:
        reason_codes = ["CORE_VERTICAL_ONLY"]
        if execution.result.blocked_decision_refs:
            reason_codes.append("BLOCKED_TRACES_PRESENT")
        return FactoryRunCompletionV2.create(
            completion_id=f"factory-run-completion://{_slug(run.run_id)}/core",
            run_ref=run.to_ref(),
            compiled_plan_ref=compiled_plan_ref,
            outcome=FactoryCompletionOutcomeV2.INCOMPLETE,
            completed_task_refs=(),
            incomplete_task_refs=(_stable_ref("agent-task", f"{run.run_id}:deferred-attachment-evaluation"),),
            blocked_task_refs=(),
            validator_result_refs=(execution.result.to_ref(),),
            reason_codes=tuple(sorted(reason_codes)),
            audit=audit,
        )

    def _payloads(
        self,
        *,
        run: FactoryRunV2,
        requirement: EvaluationRequirementSpecV2,
        execution: CoreVerticalExecution,
        completion: FactoryRunCompletionV2,
        audit: ContractAudit,
    ) -> tuple[dict[str, bytes], DatasetDeliveryManifestV2]:
        requirement_bytes = _json_bytes(
            {
                "assumptions": requirement.assumptions,
                "constraints": requirement.constraints,
                "goals": requirement.goals,
                "open_questions": requirement.open_questions,
                "requirement_ref": requirement.to_ref(),
            }
        )
        candidate_bytes = _jsonl(execution.decisions)
        prompt_bytes = _jsonl(execution.extracted_prompts)
        intent_bytes = _jsonl(execution.inferred_intents)
        rewrite_bytes = _jsonl(execution.rewrite_candidates)
        unresolved_bytes = _json_bytes(
            {
                "blocked_decision_refs": (execution.result.blocked_decision_refs),
                "non_candidate_decision_refs": (execution.result.non_candidate_decision_refs),
                "reason_codes": completion.reason_codes,
            }
        )
        audit_bytes = _json_bytes(
            {
                "audit": audit,
                "completion_ref": completion.to_ref(),
                "core_result_ref": execution.result.to_ref(),
                "manifest_ref": execution.result.manifest_ref,
                "route_decision_refs": execution.result.route_decision_refs,
            }
        )
        requirement_ref = _content_ref(
            "core-output-requirement-summary",
            requirement_bytes,
        )
        candidates_ref = _content_ref(
            "core-output-trace-candidate-table",
            candidate_bytes,
        )
        unresolved_ref = _content_ref(
            "core-output-unresolved-report",
            unresolved_bytes,
        )
        audit_ref = _content_ref("core-output-audit", audit_bytes)
        manifest = DatasetDeliveryManifestV2.create(
            delivery_manifest_id=f"dataset-delivery-manifest://{_slug(run.run_id)}/core",
            run_ref=run.to_ref(),
            completion_ref=completion.to_ref(),
            requirement_summary_ref=requirement_ref,
            trace_candidate_table_ref=candidates_ref,
            extracted_prompt_refs=execution.result.extracted_prompt_refs,
            inferred_intent_refs=execution.result.inferred_intent_refs,
            rewrite_candidate_refs=execution.result.rewrite_candidate_refs,
            unresolved_report_ref=unresolved_ref,
            audit_ref=audit_ref,
            item_count=len(execution.rewrite_candidates),
            audit=audit,
        )
        readme = (
            "# Eval Dataset Factory Core Output\n\n"
            f"- Run: `{run.run_id}`\n"
            f"- Sources: `{execution.result.source_count}`\n"
            f"- Rewrite candidates: `{len(execution.rewrite_candidates)}`\n"
            "- Scope: core vertical candidate output only\n"
            "- Deferred: attachments, criteria, rubrics, grading, production release\n"
        ).encode()
        return (
            {
                "README.md": readme,
                "audit-refs.json": audit_bytes,
                "extracted-user-prompts.jsonl": prompt_bytes,
                "inferred-user-intents.jsonl": intent_bytes,
                "requirement-summary.json": requirement_bytes,
                "task-rewrite-candidates.jsonl": rewrite_bytes,
                "trace-candidates.jsonl": candidate_bytes,
                "unresolved-items.json": unresolved_bytes,
            },
            manifest,
        )

    def _write_bundle(
        self,
        payloads: dict[str, bytes],
        inventory: CoreOutputInventoryV1,
    ) -> tuple[Path, bool]:
        final = self._bundle_path(inventory.bundle_sha256)
        if final.exists():
            self._verify_bundle(final, inventory)
            return final, True
        temporary = Path(tempfile.mkdtemp(prefix="core-output-", dir=self._staging))
        renamed = False
        try:
            self._fault("after_temporary_directory")
            for relative_path, payload in sorted(payloads.items()):
                _write_file(temporary, relative_path, payload)
            self._fault("after_files")
            self._verify_bundle(temporary, inventory)
            _fsync_tree(temporary)
            self._fault("before_rename")
            final.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.rename(temporary, final)
            except OSError:
                if not final.exists():
                    raise
                self._verify_bundle(final, inventory)
                shutil.rmtree(temporary)
                return final, True
            renamed = True
            _fsync_directory(final.parent)
            self._fault("after_rename")
            return final, False
        finally:
            if not renamed and temporary.exists():
                shutil.rmtree(temporary)

    def _verify_bundle(
        self,
        path: Path,
        inventory: CoreOutputInventoryV1,
    ) -> None:
        if path.is_symlink() or not path.is_dir():
            raise CoreOutputIntegrityError("core output bundle is missing or unsafe")
        files: list[CoreOutputFileV1] = []
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise CoreOutputIntegrityError("core output bundle contains a symlink")
            mode = child.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise CoreOutputIntegrityError("core output bundle contains a special file")
            relative = child.relative_to(path).as_posix()
            payload = child.read_bytes()
            files.append(
                CoreOutputFileV1(
                    relative_path=relative,
                    size_bytes=len(payload),
                    content_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
        if tuple(files) != inventory.files:
            raise CoreOutputIntegrityError("core output bundle inventory differs")

    def _bundle_path(self, digest: str) -> Path:
        return self._cas / digest[:2] / digest

    def _initialize(self) -> None:
        connection = self.store._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS core_output_inventories (
                    inventory_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    manifest_object_id TEXT NOT NULL UNIQUE,
                    bundle_sha256 TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS core_output_current_heads (
                    run_id TEXT PRIMARY KEY,
                    inventory_id TEXT NOT NULL UNIQUE,
                    manifest_object_id TEXT NOT NULL UNIQUE
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _insert_completion(
        connection: sqlite3.Connection,
        run_id: str,
        value: FactoryRunCompletionV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO factory_run_completions (
                completion_id, run_id, outcome, object_id,
                object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                value.completion_id,
                run_id,
                value.outcome.value,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_manifest(
        connection: sqlite3.Connection,
        run_id: str,
        value: DatasetDeliveryManifestV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO dataset_delivery_manifests (
                delivery_manifest_id, run_id, object_id,
                object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                value.delivery_manifest_id,
                run_id,
                value.object_id,
                value.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_inventory(
        connection: sqlite3.Connection,
        value: CoreOutputInventoryV1,
    ) -> None:
        connection.execute(
            """
            INSERT INTO core_output_inventories (
                inventory_id, run_id, manifest_object_id,
                bundle_sha256, object_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                value.inventory_id,
                value.run_id,
                value.delivery_manifest_ref.object_id,
                value.bundle_sha256,
                value.inventory_sha256,
                value.model_dump_json(),
            ),
        )

    def _idempotent_inventory(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        idempotency_key: str,
        request_sha256: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT request_sha256, response_type, response_id
            FROM factory_control_idempotency
            WHERE scope = ? AND idempotency_key = ?
            """,
            (f"assemble-core-output:{run_id}", idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if row["request_sha256"] != request_sha256:
            raise CoreOutputConflictError("core output idempotency request changed")
        if row["response_type"] != "core-output-inventory":
            raise CoreOutputIntegrityError("core output idempotency response type drifted")
        inventory = connection.execute(
            "SELECT 1 FROM core_output_inventories WHERE inventory_id = ?",
            (row["response_id"],),
        ).fetchone()
        if inventory is None:
            raise CoreOutputIntegrityError("core output idempotency inventory is missing")
        return str(row["response_id"])

    def _replay_if_existing(
        self,
        *,
        run_id: str,
        requirement: EvaluationRequirementSpecV2,
        execution: CoreVerticalExecution,
        idempotency_key: str,
    ) -> CoreOutputWrite | None:
        connection = self.store._connect()
        try:
            row = connection.execute(
                """
                SELECT response_type, response_id
                FROM factory_control_idempotency
                WHERE scope = ? AND idempotency_key = ?
                """,
                (f"assemble-core-output:{run_id}", idempotency_key),
            ).fetchone()
            current_head = connection.execute(
                "SELECT 1 FROM core_output_current_heads WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            if current_head is not None:
                raise CoreOutputConflictError("Factory run already has core output authority")
            return None
        if row["response_type"] != "core-output-inventory":
            raise CoreOutputIntegrityError("core output replay response type drifted")
        run = self.store.get_run(run_id)
        view = self.get(run_id)
        if (
            run.requirement_spec_ref != requirement.to_ref()
            or view.inventory.source_result_ref != execution.result.to_ref()
            or view.inventory.inventory_id != row["response_id"]
        ):
            raise CoreOutputConflictError("core output idempotency input changed")
        path = self._bundle_path(view.inventory.bundle_sha256)
        self._verify_bundle(path, view.inventory)
        return CoreOutputWrite(view=view, bundle_path=path, reused=True)

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)


def _inventory(
    *,
    run_id: str,
    source_result_ref: ObjectRef,
    completion_ref: ObjectRef,
    delivery_manifest_ref: ObjectRef,
    payloads: dict[str, bytes],
) -> CoreOutputInventoryV1:
    files = tuple(
        CoreOutputFileV1(
            relative_path=path,
            size_bytes=len(payload),
            content_sha256=hashlib.sha256(payload).hexdigest(),
        )
        for path, payload in sorted(payloads.items())
    )
    bundle_sha256 = _hash_json(
        [
            {
                "content_sha256": value.content_sha256,
                "relative_path": value.relative_path,
                "size_bytes": value.size_bytes,
            }
            for value in files
        ]
    )
    values = {
        "run_id": run_id,
        "source_result_ref": source_result_ref,
        "completion_ref": completion_ref,
        "delivery_manifest_ref": delivery_manifest_ref,
        "files": files,
        "bundle_sha256": bundle_sha256,
    }
    digest = _hash_json(values)
    return CoreOutputInventoryV1(
        **values,  # type: ignore[arg-type]
        inventory_id=f"core-output-inventory://sha256/{digest}",
        inventory_sha256=digest,
    )


def _inventory_sha256(value: CoreOutputInventoryV1) -> str:
    return _hash_json(
        value.model_dump(
            mode="python",
            exclude={
                "schema_version",
                "inventory_id",
                "inventory_sha256",
            },
        )
    )


def _jsonl(values: tuple[ContractModelV2, ...]) -> bytes:
    return b"".join(value.canonical_json() + b"\n" for value in values)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _content_ref(object_type: str, payload: bytes) -> ObjectRef:
    digest = hashlib.sha256(payload).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v1",
        object_sha256=digest,
    )


def _stable_ref(object_type: str, seed: str) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _request_sha256(*values: object) -> str:
    return _hash_json(values)


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _write_file(root: Path, relative_path: str, payload: bytes) -> None:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise CoreOutputError("core output path is unsafe")
    target = root.joinpath(*relative.parts)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _slug(value: str) -> str:
    return value.rsplit("://", 1)[-1].replace("/", "-")


__all__ = [
    "CoreOutputAssembler",
    "CoreOutputConflictError",
    "CoreOutputError",
    "CoreOutputFileV1",
    "CoreOutputIntegrityError",
    "CoreOutputInventoryV1",
    "CoreOutputViewV1",
    "CoreOutputWrite",
]
