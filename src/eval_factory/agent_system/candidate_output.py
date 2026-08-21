from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionResult,
)
from eval_factory.agent_system.store import (
    FactoryControlStore,
)
from eval_factory.contracts.agent_system_v2 import (
    FactoryCompletionOutcomeV2,
    FactoryRunCompletionV2,
    FactoryRunStatusV2,
    FactoryRunV2,
    PlanKindV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    CandidateDatasetDeliveryManifestV2,
    CompiledDatasetDeliveryPlanV2,
    DatasetDeliveryPlanV2,
    FactoryDatasetAggregateResultV2,
    FactoryItemStageOutcomeV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
    provenance_manifest_v2_ref,
)
from eval_factory.contracts.release_projection_v2 import (
    evaluation_item_v2_ref,
)


class CandidateOutputError(RuntimeError):
    pass


class CandidateOutputConflictError(CandidateOutputError):
    pass


class CandidateOutputIntegrityError(CandidateOutputError):
    pass


class CandidateOutputFileV1(ContractModelV2):
    schema_version: Literal["eval-factory/candidate-output-file/v1"] = "eval-factory/candidate-output-file/v1"

    relative_path: str = Field(min_length=1, max_length=1_024)
    size_bytes: int = Field(ge=0, le=10_000_000_000)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_path(self) -> CandidateOutputFileV1:
        _normalized_relative_path(self.relative_path)
        return self


class CandidateDatasetInventoryV1(ContractModelV2):
    schema_version: Literal["eval-factory/candidate-dataset-inventory/v1"] = (
        "eval-factory/candidate-dataset-inventory/v1"
    )

    inventory_id: str
    dataset_run_id: str
    aggregate_result_ref: ObjectRef
    delivery_plan_ref: ObjectRef
    files: tuple[CandidateOutputFileV1, ...] = Field(
        min_length=1,
        max_length=10_000_000,
    )
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> CandidateDatasetInventoryV1:
        paths = tuple(value.relative_path for value in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("candidate output files must be sorted and unique")
        observed_bundle = _hash_json(
            [
                {
                    "relative_path": value.relative_path,
                    "size_bytes": value.size_bytes,
                    "content_sha256": value.content_sha256,
                }
                for value in self.files
            ]
        )
        if self.bundle_sha256 != observed_bundle:
            raise ValueError("candidate bundle identity is stale")
        observed = _inventory_sha256(self)
        if (
            self.inventory_sha256 != observed
            or self.inventory_id != f"candidate-dataset-inventory://sha256/{observed}"
        ):
            raise ValueError("candidate inventory identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="candidate-dataset-inventory",
            object_id=self.inventory_id,
            object_version="v2",
            object_sha256=self.inventory_sha256,
        )


@dataclass(frozen=True, slots=True)
class AuthorizedCandidateExport:
    item_id: str
    files: tuple[tuple[str, bytes], ...]


def authorized_candidate_export_ref(
    value: AuthorizedCandidateExport,
) -> ObjectRef:
    digest = hashlib.sha256()
    digest.update(value.item_id.encode())
    for relative_path, payload in value.files:
        digest.update(b"\0")
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    value_sha256 = digest.hexdigest()
    return ObjectRef(
        object_type="authorized-candidate-export",
        object_id=(f"authorized-candidate-export://sha256/{value_sha256}"),
        object_version="v2",
        object_sha256=value_sha256,
    )


@dataclass(frozen=True, slots=True)
class CandidateOutputWrite:
    completion: FactoryRunCompletionV2
    manifest: CandidateDatasetDeliveryManifestV2
    inventory: CandidateDatasetInventoryV1
    bundle_path: Path
    reused: bool


class CandidateDatasetOutputAssembler:
    def __init__(
        self,
        *,
        store: FactoryControlStore,
        root: Path,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        expanded_root = root.expanduser()
        if expanded_root.is_symlink():
            raise CandidateOutputError("candidate output root must be a real directory")
        self.root = expanded_root.resolve()
        if self.root.exists() and not self.root.is_dir():
            raise CandidateOutputError("candidate output root must be a real directory")
        self._cas = self.root / "cas" / "sha256"
        self._staging = self.root / ".staging"
        self._cas.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._staging.mkdir(
            mode=0o700,
            parents=True,
            exist_ok=True,
        )
        self._fault_injector = fault_injector
        self._initialize()

    def assemble(
        self,
        *,
        dataset_run_id: str,
        aggregate: FactoryDatasetAggregateResultV2,
        plan: DatasetDeliveryPlanV2,
        compiled_plan: CompiledDatasetDeliveryPlanV2,
        parent_compiled_plan_ref: ObjectRef,
        candidates: tuple[
            FactoryCandidateProjectionResult,
            ...,
        ],
        exports: tuple[AuthorizedCandidateExport, ...],
        audit: ContractAudit,
        idempotency_key: str,
    ) -> CandidateOutputWrite:
        payloads, provenance_refs = self._payloads(
            aggregate=aggregate,
            candidates=candidates,
            exports=exports,
        )
        request_sha256 = _assembly_request_sha256(
            dataset_run_id=dataset_run_id,
            aggregate=aggregate,
            plan=plan,
            compiled_plan=compiled_plan,
            parent_compiled_plan_ref=parent_compiled_plan_ref,
            payloads=payloads,
        )
        replay = self._replay(
            dataset_run_id=dataset_run_id,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        if replay is not None:
            return replay
        run = self.store.get_run(dataset_run_id)
        stored_plan = self.store.get_domain_plan(
            dataset_run_id,
            PlanKindV2.FINAL_DELIVERY,
        )
        if (
            self.store.get_dataset_aggregate(dataset_run_id) != aggregate
            or stored_plan.plan_ref != plan.to_ref()
            or stored_plan.compiled_plan_ref != compiled_plan.to_ref()
            or compiled_plan.source_plan_ref != plan.to_ref()
            or compiled_plan.policy_ref != run.policy_ref
            or parent_compiled_plan_ref.object_type != "compiled-dataset-build-plan"
            or parent_compiled_plan_ref.object_version != "v2"
        ):
            raise CandidateOutputConflictError("candidate output authority is stale")
        candidate_item_refs = tuple(
            sorted(
                (evaluation_item_v2_ref(value.projection.evaluation_item) for value in candidates),
                key=_ref_key,
            )
        )
        candidate_projection_refs = tuple(
            sorted(
                (value.projection.to_ref() for value in candidates),
                key=_ref_key,
            )
        )
        if aggregate.batch_quality_ref is None:
            raise CandidateOutputConflictError("candidate output requires batch quality")
        candidate_stage_head_refs = []
        for value in candidates:
            stage_head = self.store.get_item_stage_head(
                value.source.item.item_id,
                FactoryItemStageV2.RELEASE_CANDIDATE,
            )
            required_dependencies = {
                aggregate.batch_quality_ref,
                item_quality_compilation_result_ref(value.source.item_quality),
                value.criteria_result_ref,
                value.grading_result_ref,
            }
            if (
                stage_head.result_ref != value.projection.to_ref()
                or stage_head.outcome is not FactoryItemStageOutcomeV2.SUCCEEDED
                or set(stage_head.dependency_result_refs) != required_dependencies
            ):
                raise CandidateOutputConflictError("candidate stage authority is stale")
            candidate_stage_head_refs.append(stage_head.to_ref())
        ordered_stage_head_refs = tuple(
            sorted(
                candidate_stage_head_refs,
                key=_ref_key,
            )
        )
        if (
            candidate_item_refs != plan.candidate_item_refs
            or candidate_projection_refs != plan.candidate_projection_refs
            or ordered_stage_head_refs != plan.candidate_stage_head_refs
        ):
            raise CandidateOutputConflictError("candidate output inventory differs from plan")
        batch_quality_ref = aggregate.batch_quality_ref
        completion = FactoryRunCompletionV2.create(
            completion_id=(f"factory-run-completion://candidate/{aggregate.object_sha256}"),
            run_ref=run.to_ref(),
            compiled_plan_ref=parent_compiled_plan_ref,
            outcome=FactoryCompletionOutcomeV2.COMPLETE,
            completed_task_refs=tuple(
                sorted(
                    plan.candidate_projection_refs,
                    key=_ref_key,
                )
            ),
            incomplete_task_refs=(),
            blocked_task_refs=(),
            validator_result_refs=(
                batch_quality_ref,
                compiled_plan.to_ref(),
            ),
            reason_codes=(),
            audit=audit,
        )
        payloads["completion.json"] = completion.canonical_json() + b"\n"
        inventory = _inventory(
            dataset_run_id=dataset_run_id,
            aggregate_result_ref=aggregate.to_ref(),
            delivery_plan_ref=plan.to_ref(),
            payloads=payloads,
        )
        manifest = CandidateDatasetDeliveryManifestV2.create(
            dataset_run_ref=run.to_ref(),
            aggregate_result_ref=aggregate.to_ref(),
            batch_quality_ref=batch_quality_ref,
            candidate_item_refs=candidate_item_refs,
            candidate_projection_refs=candidate_projection_refs,
            candidate_stage_head_refs=ordered_stage_head_refs,
            rejected_binding_refs=(aggregate.rejected_binding_refs),
            blocked_binding_refs=aggregate.blocked_binding_refs,
            inventory_ref=inventory.to_ref(),
            provenance_manifest_refs=provenance_refs,
            audit=audit,
        )
        payloads["dataset-manifest.json"] = manifest.canonical_json() + b"\n"
        _require_reviewed_limits(payloads, plan)
        bundle_path, reused = self._write_bundle(
            payloads,
            inventory,
            manifest,
        )
        self._fault("after_bundle")
        self._commit(
            run=run,
            aggregate=aggregate,
            plan=plan,
            completion=completion,
            manifest=manifest,
            inventory=inventory,
            audit=audit,
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        return CandidateOutputWrite(
            completion=completion,
            manifest=manifest,
            inventory=inventory,
            bundle_path=bundle_path,
            reused=reused,
        )

    def _payloads(
        self,
        *,
        aggregate: FactoryDatasetAggregateResultV2,
        candidates: tuple[
            FactoryCandidateProjectionResult,
            ...,
        ],
        exports: tuple[AuthorizedCandidateExport, ...],
    ) -> tuple[dict[str, bytes], tuple[ObjectRef, ...]]:
        exports_by_item = {value.item_id: value.files for value in exports}
        if len(exports_by_item) != len(exports):
            raise CandidateOutputConflictError("candidate exports contain duplicate items")
        payloads: dict[str, bytes] = {
            "README.md": (
                "# Candidate Evaluation Dataset\n\n"
                f"- Candidates: {len(candidates)}\n"
                f"- Rejected: {aggregate.rejected_count}\n"
                f"- Blocked: {aggregate.blocked_count}\n"
                "- Release: production blocked\n"
            ).encode(),
            "candidate-items.jsonl": _jsonl(tuple(value.projection.evaluation_item for value in candidates)),
            "candidate-projections.jsonl": _jsonl(tuple(value.projection for value in candidates)),
            "blocked-bindings.jsonl": _jsonl(aggregate.blocked_binding_refs),
            "rejected-bindings.jsonl": _jsonl(aggregate.rejected_binding_refs),
        }
        provenance_refs: list[ObjectRef] = []
        for value in candidates:
            source = value.source
            quality = source.item_quality
            package = quality.final_package_manifest
            provenance = quality.provenance_manifest
            environment = quality.environment_spec
            if (
                package is None
                or provenance is None
                or environment is None
                or package.entry_refs != provenance.package_inventory_entry_refs
            ):
                raise CandidateOutputConflictError("candidate package/provenance is incomplete")
            item_slug = hashlib.sha256(source.item.item_id.encode()).hexdigest()
            prefix = f"items/{item_slug}"
            payloads[f"{prefix}/evaluation-item.json"] = (
                value.projection.evaluation_item.canonical_json() + b"\n"
            )
            payloads[f"{prefix}/provenance-manifest.json"] = provenance.canonical_json() + b"\n"
            payloads[f"{prefix}/environment-spec.json"] = environment.canonical_json() + b"\n"
            provenance_refs.append(provenance_manifest_v2_ref(provenance))
            provided = dict(exports_by_item.get(source.item.item_id, ()))
            if len(provided) != len(exports_by_item.get(source.item.item_id, ())):
                raise CandidateOutputConflictError("candidate export paths are duplicated")
            expected = {
                entry.inventory_member.normalized_path: (entry.inventory_member.content_sha256)
                for entry in package.entries
                if entry.inventory_member.container_ref is None
                and entry.inventory_member.member_type == "FILE"
            }
            if set(provided) != set(expected):
                raise CandidateOutputConflictError("candidate export inventory is not exact")
            for relative_path, payload in provided.items():
                safe = _safe_relative_path(relative_path)
                if hashlib.sha256(payload).hexdigest() != expected[relative_path]:
                    raise CandidateOutputConflictError("candidate export hash differs from final package")
                payloads[f"{prefix}/workspace/{safe}"] = payload
        if set(exports_by_item) - {value.source.item.item_id for value in candidates}:
            raise CandidateOutputConflictError("candidate exports contain unknown items")
        return (
            payloads,
            tuple(sorted(provenance_refs, key=_ref_key)),
        )

    def _commit(
        self,
        *,
        run: FactoryRunV2,
        aggregate: FactoryDatasetAggregateResultV2,
        plan: DatasetDeliveryPlanV2,
        completion: FactoryRunCompletionV2,
        manifest: CandidateDatasetDeliveryManifestV2,
        inventory: CandidateDatasetInventoryV1,
        audit: ContractAudit,
        idempotency_key: str,
        request_sha256: str,
    ) -> None:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self.store._check_idempotency(
                connection,
                scope=f"assemble-candidate-output:{run.run_id}",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="candidate-dataset-inventory",
            )
            if replay is not None:
                connection.rollback()
                return
            current = self.store._load_current_run(
                connection,
                run.run_id,
            )
            if current != run:
                raise CandidateOutputConflictError("Factory run changed during candidate output")
            connection.execute(
                """
                INSERT INTO candidate_output_inventories (
                    inventory_id, run_id, bundle_sha256,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    inventory.inventory_id,
                    run.run_id,
                    inventory.bundle_sha256,
                    inventory.inventory_sha256,
                    inventory.canonical_json().decode(),
                ),
            )
            self._fault("after_inventory")
            connection.execute(
                """
                INSERT INTO candidate_delivery_manifests (
                    manifest_object_id, run_id,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    manifest.object_id,
                    run.run_id,
                    manifest.object_sha256,
                    manifest.canonical_json().decode(),
                ),
            )
            self._fault("after_manifest")
            connection.execute(
                """
                INSERT INTO candidate_run_completions (
                    completion_object_id, run_id,
                    object_sha256, record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    completion.object_id,
                    run.run_id,
                    completion.object_sha256,
                    completion.canonical_json().decode(),
                ),
            )
            self._fault("after_completion")
            connection.execute(
                """
                INSERT INTO candidate_output_current_heads (
                    run_id, inventory_id, manifest_object_id,
                    completion_object_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    inventory.inventory_id,
                    manifest.object_id,
                    completion.object_id,
                ),
            )
            self._fault("after_output_head")
            successor = FactoryRunV2.create(
                run_id=current.run_id,
                run_version=current.run_version + 1,
                status=FactoryRunStatusV2.COMPLETED,
                policy_ref=current.policy_ref,
                requirement_spec_ref=current.requirement_spec_ref,
                current_plan_ref=current.current_plan_ref,
                compiled_plan_ref=current.compiled_plan_ref,
                active_task_refs=(),
                result_refs=tuple(
                    sorted(
                        {
                            *current.result_refs,
                            aggregate.to_ref(),
                            completion.to_ref(),
                            manifest.to_ref(),
                            inventory.to_ref(),
                        },
                        key=_ref_key,
                    )
                ),
                pending_review_ref=None,
                planner_assessment_ref=(current.planner_assessment_ref),
                completion_ref=completion.to_ref(),
                delivery_manifest_ref=manifest.to_ref(),
                transition_count=current.transition_count + 1,
                model_requests_used=current.model_requests_used,
                model_tokens_used=current.model_tokens_used,
                cost_micro_usd_used=current.cost_micro_usd_used,
                audit=audit,
            )
            self.store._insert_run(connection, successor)
            self._fault("after_run")
            changed = connection.execute(
                """
                UPDATE factory_run_current_heads
                SET run_version = ?, run_object_id = ?
                WHERE run_id = ? AND run_version = ?
                """,
                (
                    successor.run_version,
                    successor.object_id,
                    run.run_id,
                    current.run_version,
                ),
            ).rowcount
            if changed != 1:
                raise CandidateOutputConflictError("Factory run head changed during candidate commit")
            self._fault("after_run_head")
            self.store._insert_outbox(
                connection,
                aggregate_type="FACTORY_RUN",
                aggregate_id=run.run_id,
                aggregate_version=successor.run_version,
                event_type="candidate-dataset-delivered",
                object_id=manifest.object_id,
                created_at=audit.created_at.isoformat(),
            )
            self._fault("after_outbox")
            self.store._insert_idempotency(
                connection,
                scope=f"assemble-candidate-output:{run.run_id}",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="candidate-dataset-inventory",
                response_id=inventory.inventory_id,
            )
            self._fault("after_idempotency")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise CandidateOutputConflictError("candidate output authority already exists") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get(self, run_id: str) -> CandidateOutputWrite:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN")
            head = connection.execute(
                """
                SELECT * FROM candidate_output_current_heads
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if head is None:
                raise CandidateOutputIntegrityError("candidate output was not found")
            inventory_row = connection.execute(
                """
                SELECT * FROM candidate_output_inventories
                WHERE inventory_id = ?
                """,
                (head["inventory_id"],),
            ).fetchone()
            manifest_row = connection.execute(
                """
                SELECT * FROM candidate_delivery_manifests
                WHERE manifest_object_id = ?
                """,
                (head["manifest_object_id"],),
            ).fetchone()
            completion_row = connection.execute(
                """
                SELECT * FROM candidate_run_completions
                WHERE completion_object_id = ?
                """,
                (head["completion_object_id"],),
            ).fetchone()
            if inventory_row is None or manifest_row is None or completion_row is None:
                raise CandidateOutputIntegrityError("candidate output authority is incomplete")
            inventory = CandidateDatasetInventoryV1.model_validate_json(str(inventory_row["record_json"]))
            manifest = CandidateDatasetDeliveryManifestV2.model_validate_json(
                str(manifest_row["record_json"])
            )
            completion = FactoryRunCompletionV2.model_validate_json(str(completion_row["record_json"]))
            if (
                inventory_row["run_id"] != run_id
                or manifest_row["run_id"] != run_id
                or completion_row["run_id"] != run_id
                or inventory_row["inventory_id"] != inventory.inventory_id
                or inventory_row["bundle_sha256"] != inventory.bundle_sha256
                or inventory_row["object_sha256"] != inventory.inventory_sha256
                or manifest_row["manifest_object_id"] != manifest.object_id
                or manifest_row["object_sha256"] != manifest.object_sha256
                or completion_row["completion_object_id"] != completion.object_id
                or completion_row["object_sha256"] != completion.object_sha256
                or manifest.inventory_ref != inventory.to_ref()
                or completion.to_ref() != self.store.get_run(run_id).completion_ref
            ):
                raise CandidateOutputIntegrityError("candidate output authority drifted")
            connection.rollback()
        finally:
            connection.close()
        bundle = self._bundle_path(inventory.bundle_sha256)
        self._verify_bundle(bundle, inventory, manifest)
        return CandidateOutputWrite(
            completion=completion,
            manifest=manifest,
            inventory=inventory,
            bundle_path=bundle,
            reused=True,
        )

    def rebuild_current_heads(self) -> tuple[str, ...]:
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            inventory_rows = connection.execute(
                """
                SELECT run_id, inventory_id
                FROM candidate_output_inventories
                ORDER BY run_id
                """
            ).fetchall()
            manifest_rows = connection.execute(
                """
                SELECT run_id, manifest_object_id
                FROM candidate_delivery_manifests
                ORDER BY run_id
                """
            ).fetchall()
            completion_rows = connection.execute(
                """
                SELECT run_id, completion_object_id
                FROM candidate_run_completions
                ORDER BY run_id
                """
            ).fetchall()
            inventories = {str(row["run_id"]): str(row["inventory_id"]) for row in inventory_rows}
            manifests = {str(row["run_id"]): str(row["manifest_object_id"]) for row in manifest_rows}
            completions = {str(row["run_id"]): str(row["completion_object_id"]) for row in completion_rows}
            if set(inventories) != set(manifests) or set(inventories) != set(completions):
                raise CandidateOutputIntegrityError("candidate output immutable authority is incomplete")
            connection.execute("DELETE FROM candidate_output_current_heads")
            for run_id in sorted(inventories):
                connection.execute(
                    """
                    INSERT INTO candidate_output_current_heads (
                        run_id, inventory_id, manifest_object_id,
                        completion_object_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        inventories[run_id],
                        manifests[run_id],
                        completions[run_id],
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        for run_id in sorted(inventories):
            self.get(run_id)
        return tuple(sorted(inventories))

    def _replay(
        self,
        *,
        dataset_run_id: str,
        idempotency_key: str,
        request_sha256: str,
    ) -> CandidateOutputWrite | None:
        connection = self.store._connect()
        try:
            row = connection.execute(
                """
                SELECT request_sha256, response_type
                FROM factory_control_idempotency
                WHERE scope = ? AND idempotency_key = ?
                """,
                (
                    f"assemble-candidate-output:{dataset_run_id}",
                    idempotency_key,
                ),
            ).fetchone()
            if row is None:
                head = connection.execute(
                    """
                    SELECT 1
                    FROM candidate_output_current_heads
                    WHERE run_id = ?
                    """,
                    (dataset_run_id,),
                ).fetchone()
                if head is not None:
                    raise CandidateOutputConflictError("candidate output authority already exists")
                return None
            if (
                row["request_sha256"] != request_sha256
                or row["response_type"] != "candidate-dataset-inventory"
            ):
                raise CandidateOutputConflictError("candidate output replay type drifted")
        finally:
            connection.close()
        return self.get(dataset_run_id)

    def _initialize(self) -> None:
        connection = self.store._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidate_output_inventories (
                    inventory_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    bundle_sha256 TEXT NOT NULL,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_delivery_manifests (
                    manifest_object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_run_completions (
                    completion_object_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    object_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_output_current_heads (
                    run_id TEXT PRIMARY KEY,
                    inventory_id TEXT NOT NULL UNIQUE,
                    manifest_object_id TEXT NOT NULL UNIQUE,
                    completion_object_id TEXT NOT NULL UNIQUE
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _write_bundle(
        self,
        payloads: dict[str, bytes],
        inventory: CandidateDatasetInventoryV1,
        manifest: CandidateDatasetDeliveryManifestV2,
    ) -> tuple[Path, bool]:
        final = self._bundle_path(inventory.bundle_sha256)
        if final.exists():
            self._verify_bundle(final, inventory, manifest)
            return final, True
        temporary = Path(
            tempfile.mkdtemp(
                prefix="candidate-output-",
                dir=self._staging,
            )
        )
        renamed = False
        try:
            for relative_path, payload in sorted(payloads.items()):
                _write_file(
                    temporary,
                    relative_path,
                    payload,
                )
            self._verify_bundle(
                temporary,
                inventory,
                manifest,
            )
            _fsync_tree(temporary)
            self._fault("before_rename")
            final.parent.mkdir(
                mode=0o700,
                parents=True,
                exist_ok=True,
            )
            try:
                os.rename(temporary, final)
            except OSError:
                if not final.exists():
                    raise
                self._verify_bundle(final, inventory, manifest)
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
        inventory: CandidateDatasetInventoryV1,
        manifest: CandidateDatasetDeliveryManifestV2 | None = None,
    ) -> None:
        if path.is_symlink() or not path.is_dir():
            raise CandidateOutputIntegrityError("candidate output bundle is unsafe")
        observed: list[CandidateOutputFileV1] = []
        manifest_payload: bytes | None = None
        for child in sorted(path.rglob("*")):
            if child.is_symlink():
                raise CandidateOutputIntegrityError("candidate output contains a symlink")
            if child.is_dir():
                continue
            if not child.is_file():
                raise CandidateOutputIntegrityError("candidate output contains a special file")
            payload = child.read_bytes()
            relative_path = child.relative_to(path).as_posix()
            if relative_path == "dataset-manifest.json":
                manifest_payload = payload
                continue
            observed.append(
                CandidateOutputFileV1(
                    relative_path=relative_path,
                    size_bytes=len(payload),
                    content_sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
        if tuple(observed) != inventory.files:
            raise CandidateOutputIntegrityError("candidate output inventory drifted")
        if manifest_payload is None:
            raise CandidateOutputIntegrityError("candidate output manifest is missing")
        if manifest is not None and manifest_payload != manifest.canonical_json() + b"\n":
            raise CandidateOutputIntegrityError("candidate output manifest drifted")

    def _bundle_path(self, digest: str) -> Path:
        return self._cas / digest[:2] / digest

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)


def _inventory(
    *,
    dataset_run_id: str,
    aggregate_result_ref: ObjectRef,
    delivery_plan_ref: ObjectRef,
    payloads: dict[str, bytes],
) -> CandidateDatasetInventoryV1:
    files = tuple(
        CandidateOutputFileV1(
            relative_path=relative_path,
            size_bytes=len(payload),
            content_sha256=hashlib.sha256(payload).hexdigest(),
        )
        for relative_path, payload in sorted(payloads.items())
    )
    bundle_sha256 = _hash_json(
        [
            {
                "relative_path": value.relative_path,
                "size_bytes": value.size_bytes,
                "content_sha256": value.content_sha256,
            }
            for value in files
        ]
    )
    provisional = CandidateDatasetInventoryV1.model_construct(
        inventory_id="candidate-dataset-inventory://pending",
        dataset_run_id=dataset_run_id,
        aggregate_result_ref=aggregate_result_ref,
        delivery_plan_ref=delivery_plan_ref,
        files=files,
        bundle_sha256=bundle_sha256,
        inventory_sha256="0" * 64,
    )
    digest = _inventory_sha256(provisional)
    return CandidateDatasetInventoryV1(
        inventory_id=(f"candidate-dataset-inventory://sha256/{digest}"),
        dataset_run_id=dataset_run_id,
        aggregate_result_ref=aggregate_result_ref,
        delivery_plan_ref=delivery_plan_ref,
        files=files,
        bundle_sha256=bundle_sha256,
        inventory_sha256=digest,
    )


def _inventory_sha256(
    value: CandidateDatasetInventoryV1,
) -> str:
    return _hash_json(
        {
            "dataset_run_id": value.dataset_run_id,
            "aggregate_result_ref": value.aggregate_result_ref,
            "delivery_plan_ref": value.delivery_plan_ref,
            "files": value.files,
            "bundle_sha256": value.bundle_sha256,
        }
    )


def _write_file(
    root: Path,
    relative_path: str,
    payload: bytes,
) -> None:
    safe = _safe_relative_path(relative_path)
    destination = root / safe
    destination.parent.mkdir(
        mode=0o700,
        parents=True,
        exist_ok=True,
    )
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_relative_path(value: str) -> str:
    try:
        return _normalized_relative_path(value)
    except ValueError as exc:
        raise CandidateOutputConflictError("candidate export path is unsafe") from exc


def _normalized_relative_path(value: str) -> str:
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in raw_parts)
        or not path.parts
    ):
        raise ValueError("candidate output path must be safe and relative")
    return path.as_posix()


def _assembly_request_sha256(
    *,
    dataset_run_id: str,
    aggregate: FactoryDatasetAggregateResultV2,
    plan: DatasetDeliveryPlanV2,
    compiled_plan: CompiledDatasetDeliveryPlanV2,
    parent_compiled_plan_ref: ObjectRef,
    payloads: dict[str, bytes],
) -> str:
    return _hash_json(
        {
            "dataset_run_id": dataset_run_id,
            "aggregate_ref": aggregate.to_ref(),
            "plan_ref": plan.to_ref(),
            "compiled_plan_ref": compiled_plan.to_ref(),
            "parent_compiled_plan_ref": parent_compiled_plan_ref,
            "payloads": tuple(
                {
                    "relative_path": relative_path,
                    "size_bytes": len(payload),
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                }
                for relative_path, payload in sorted(payloads.items())
            ),
        }
    )


def _require_reviewed_limits(
    payloads: dict[str, bytes],
    plan: DatasetDeliveryPlanV2,
) -> None:
    if (
        len(payloads) > plan.max_files
        or sum(len(value) for value in payloads.values()) > plan.max_total_bytes
    ):
        raise CandidateOutputConflictError("candidate output exceeds reviewed limits")


def _fsync_tree(root: Path) -> None:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _jsonl(values: tuple[object, ...]) -> bytes:
    return b"".join(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
        for value in values
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


def _ref_key(
    value: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "AuthorizedCandidateExport",
    "CandidateDatasetInventoryV1",
    "CandidateDatasetOutputAssembler",
    "CandidateOutputConflictError",
    "CandidateOutputError",
    "CandidateOutputFileV1",
    "CandidateOutputIntegrityError",
    "CandidateOutputWrite",
    "authorized_candidate_export_ref",
]
