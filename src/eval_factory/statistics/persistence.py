from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import LabelSpecV2
from eval_factory.contracts.statistics_v2 import (
    IndependentLabelPartitionManifestV2,
    IndependentLabelTestSetAccessPolicyV2,
    IndependentLabelTestSetAccessReceiptV2,
    IndependentLabelTestSetFreezeRecordV2,
    IndependentLabelTestSetFreezeResultV2,
    IndependentLabelTestSetManifestV2,
    IndependentLabelTestSetPolicyV2,
    LabelTestSetAccessOutcomeV2,
    LabelTestSetAccessReasonV2,
)
from eval_factory.statistics.freeze import (
    IndependentLabelTestSetCompiler,
    IndependentLabelTestSetFreezeCompilation,
)
from eval_factory.statistics.material_store import (
    IndependentLabelMaterialError,
    IndependentLabelMaterialStore,
)
from eval_factory.statistics.models import (
    IndependentLabelCandidatePoolV1,
    IndependentLabelPartitionMaterialV1,
    IndependentLabelTestSetAccessRequestV1,
    IndependentLabelTestSetMaterialV1,
    TrustedIndependentLabelTestSetPrincipalV1,
)


class IndependentLabelTestSetPersistenceError(RuntimeError):
    pass


class IndependentLabelTestSetConflictError(IndependentLabelTestSetPersistenceError):
    pass


class IndependentLabelTestSetIntegrityError(IndependentLabelTestSetPersistenceError):
    pass


class IndependentLabelTestSetAuthorizationError(IndependentLabelTestSetPersistenceError):
    def __init__(
        self,
        message: str,
        *,
        receipt: IndependentLabelTestSetAccessReceiptV2,
    ) -> None:
        self.receipt = receipt
        super().__init__(message)


class IndependentLabelTestSetPersistenceInjectedCrash(IndependentLabelTestSetPersistenceError):
    pass


class IndependentLabelTestSetPersistenceFaultInjector(Protocol):
    def maybe_raise(self, point: str) -> None: ...


@dataclass(frozen=True, slots=True)
class StaticIndependentLabelTestSetPersistenceFaultInjector:
    crash_points: frozenset[str] = frozenset()

    def maybe_raise(self, point: str) -> None:
        if point in self.crash_points:
            raise IndependentLabelTestSetPersistenceInjectedCrash(
                f"injected independent label persistence crash at {point}"
            )


@dataclass(frozen=True, slots=True)
class IndependentLabelHeadRebuild:
    head_count: int


@dataclass(frozen=True, slots=True)
class IndependentLabelAccessResult:
    material: IndependentLabelTestSetMaterialV1
    receipt: IndependentLabelTestSetAccessReceiptV2


class IndependentLabelTestSetPersistenceService:
    def __init__(
        self,
        path: Path,
        *,
        material_store: IndependentLabelMaterialStore,
        compiler: IndependentLabelTestSetCompiler | None = None,
        fault_injector: IndependentLabelTestSetPersistenceFaultInjector | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.material_store = material_store
        self.compiler = compiler or IndependentLabelTestSetCompiler()
        self.fault_injector = fault_injector
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS label_test_set_policies (
                    policy_id TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS label_test_set_access_policies (
                    access_policy_id TEXT PRIMARY KEY,
                    access_policy_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS label_test_set_partition_manifests (
                    partition_manifest_id TEXT PRIMARY KEY,
                    partition_kind TEXT NOT NULL,
                    private_material_id TEXT NOT NULL,
                    private_material_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS independent_label_test_set_manifests (
                    dataset_id TEXT PRIMARY KEY,
                    dataset_series_id TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL,
                    private_material_id TEXT NOT NULL,
                    private_material_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(dataset_series_id, dataset_version)
                );

                CREATE TABLE IF NOT EXISTS label_test_set_freeze_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    freeze_id TEXT NOT NULL UNIQUE,
                    dataset_series_id TEXT NOT NULL,
                    requested_dataset_version TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    dataset_manifest_id TEXT,
                    supersedes_freeze_id TEXT,
                    freeze_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    UNIQUE(dataset_series_id, requested_dataset_version)
                );

                CREATE TABLE IF NOT EXISTS label_test_set_freeze_results (
                    result_id TEXT PRIMARY KEY,
                    freeze_id TEXT NOT NULL UNIQUE,
                    dataset_series_id TEXT NOT NULL,
                    requested_dataset_version TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    dataset_manifest_id TEXT,
                    result_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS label_test_set_current_heads (
                    dataset_series_id TEXT PRIMARY KEY,
                    freeze_id TEXT NOT NULL UNIQUE,
                    dataset_id TEXT NOT NULL UNIQUE,
                    freeze_sequence INTEGER NOT NULL UNIQUE,
                    result_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS label_test_set_access_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS label_test_set_idempotency (
                    scope TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    response_type TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    PRIMARY KEY(scope, idempotency_key)
                );
                """
            )

    def freeze(
        self,
        *,
        idempotency_key: str,
        dataset_series_id: str,
        dataset_version: str,
        label_specs: tuple[LabelSpecV2, ...],
        policy: IndependentLabelTestSetPolicyV2,
        access_policy: IndependentLabelTestSetAccessPolicyV2,
        candidate_pool: IndependentLabelCandidatePoolV1,
        train_partition_manifest: IndependentLabelPartitionManifestV2,
        train_partition_material: IndependentLabelPartitionMaterialV1,
        development_partition_manifest: IndependentLabelPartitionManifestV2,
        development_partition_material: IndependentLabelPartitionMaterialV1,
        supersedes_dataset_ref: ObjectRef | None,
        supersedes_freeze_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetFreezeResultV2:
        request_sha256 = _freeze_request_sha256(
            dataset_series_id=dataset_series_id,
            dataset_version=dataset_version,
            label_specs=label_specs,
            policy=policy,
            access_policy=access_policy,
            candidate_pool=candidate_pool,
            train_partition_manifest=train_partition_manifest,
            train_partition_material=train_partition_material,
            development_partition_manifest=development_partition_manifest,
            development_partition_material=development_partition_material,
            supersedes_dataset_ref=supersedes_dataset_ref,
            supersedes_freeze_ref=supersedes_freeze_ref,
        )
        replay_id = self._preflight_idempotency(
            scope="freeze",
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
        )
        if replay_id is not None:
            return self.get_result(_result_ref(replay_id))

        compilation = self.compiler.compile(
            dataset_series_id=dataset_series_id,
            dataset_version=dataset_version,
            label_specs=label_specs,
            policy=policy,
            access_policy=access_policy,
            candidate_pool=candidate_pool,
            train_partition_manifest=train_partition_manifest,
            train_partition_material=train_partition_material,
            development_partition_manifest=development_partition_manifest,
            development_partition_material=development_partition_material,
            supersedes_dataset_ref=supersedes_dataset_ref,
            supersedes_freeze_ref=supersedes_freeze_ref,
            audit=audit,
        )
        self.material_store.put_candidate_pool(candidate_pool)
        self.material_store.put_partition(train_partition_material)
        self.material_store.put_partition(development_partition_material)
        if compilation.selected_material is not None:
            self.material_store.put_test_set(compilation.selected_material)
        self._fault("after_material")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope="freeze",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                connection.rollback()
                return self.get_result(_result_ref(replay_id))
            existing_version = connection.execute(
                """
                SELECT result_id
                FROM label_test_set_freeze_results
                WHERE dataset_series_id = ? AND requested_dataset_version = ?
                """,
                (dataset_series_id, dataset_version),
            ).fetchone()
            if existing_version is not None:
                raise IndependentLabelTestSetConflictError("dataset version already has a freeze authority")
            self._validate_supersession(
                connection,
                dataset_series_id=dataset_series_id,
                supersedes_dataset_ref=supersedes_dataset_ref,
                supersedes_freeze_ref=supersedes_freeze_ref,
            )
            current = self.compiler.compile(
                dataset_series_id=dataset_series_id,
                dataset_version=dataset_version,
                label_specs=label_specs,
                policy=policy,
                access_policy=access_policy,
                candidate_pool=candidate_pool,
                train_partition_manifest=train_partition_manifest,
                train_partition_material=train_partition_material,
                development_partition_manifest=development_partition_manifest,
                development_partition_material=development_partition_material,
                supersedes_dataset_ref=supersedes_dataset_ref,
                supersedes_freeze_ref=supersedes_freeze_ref,
                audit=audit,
            )
            if current.result.to_ref() != compilation.result.to_ref() or _optional_material_ref(
                current
            ) != _optional_material_ref(compilation):
                raise IndependentLabelTestSetConflictError("freeze sources changed before transaction")
            self._insert_policy(connection, policy)
            self._fault("after_policy")
            self._insert_access_policy(connection, access_policy)
            self._fault("after_access_policy")
            self._insert_partition(connection, train_partition_manifest)
            self._insert_partition(connection, development_partition_manifest)
            self._fault("after_partitions")
            if current.result.dataset_manifest is not None:
                self._insert_manifest(connection, current.result.dataset_manifest)
                self._fault("after_manifest")
            freeze_sequence = self._insert_freeze_record(connection, current.result)
            self._fault("after_freeze_record")
            self._insert_result(connection, current.result)
            self._fault("after_result")
            if current.result.dataset_manifest is not None:
                connection.execute(
                    """
                    INSERT INTO label_test_set_current_heads (
                        dataset_series_id, freeze_id, dataset_id,
                        freeze_sequence, result_id
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(dataset_series_id) DO UPDATE SET
                        freeze_id = excluded.freeze_id,
                        dataset_id = excluded.dataset_id,
                        freeze_sequence = excluded.freeze_sequence,
                        result_id = excluded.result_id
                    """,
                    (
                        dataset_series_id,
                        current.result.freeze_record.freeze_id,
                        current.result.dataset_manifest.dataset_id,
                        freeze_sequence,
                        current.result.result_id,
                    ),
                )
                self._fault("after_head")
            self._record_idempotency(
                connection,
                scope="freeze",
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="independent-label-test-set-freeze-result",
                response_id=current.result.result_id,
            )
            self._fault("after_idempotency")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            raise IndependentLabelTestSetConflictError("freeze authority conflicted") from exc
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return self.get_result(current.result.to_ref())

    def get_result(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetFreezeResultV2:
        _require_ref(
            ref,
            "independent-label-test-set-freeze-result",
            "v2",
            "result_ref",
        )
        with self._connection() as connection:
            result = self._load_result(connection, ref.object_id)
            self._validate_series_history(
                connection,
                result.freeze_record.dataset_series_id,
            )
        if result.to_ref() != ref:
            raise IndependentLabelTestSetIntegrityError("freeze result differs from requested reference")
        return result

    def get_current(
        self,
        dataset_series_id: str,
    ) -> IndependentLabelTestSetFreezeResultV2:
        with self._connection() as connection:
            self._validate_series_history(connection, dataset_series_id)
            head = connection.execute(
                """
                SELECT freeze_id, dataset_id, freeze_sequence, result_id
                FROM label_test_set_current_heads
                WHERE dataset_series_id = ?
                """,
                (dataset_series_id,),
            ).fetchone()
            latest_frozen = connection.execute(
                """
                SELECT sequence, freeze_id
                FROM label_test_set_freeze_records
                WHERE dataset_series_id = ? AND outcome = 'FROZEN'
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (dataset_series_id,),
            ).fetchone()
            if head is None:
                if latest_frozen is not None:
                    raise IndependentLabelTestSetIntegrityError("frozen dataset current head is missing")
                raise IndependentLabelTestSetConflictError("frozen dataset authority is unavailable")
            if (
                latest_frozen is None
                or int(head["freeze_sequence"]) != int(latest_frozen["sequence"])
                or str(head["freeze_id"]) != str(latest_frozen["freeze_id"])
            ):
                raise IndependentLabelTestSetIntegrityError("frozen dataset current head is stale")
            result = self._load_result(connection, str(head["result_id"]))
            manifest = result.dataset_manifest
            if (
                manifest is None
                or result.freeze_record.freeze_id != str(head["freeze_id"])
                or manifest.dataset_id != str(head["dataset_id"])
                or manifest.dataset_series_id != dataset_series_id
            ):
                raise IndependentLabelTestSetIntegrityError(
                    "frozen dataset current head differs from immutable result"
                )
            return result

    def list_results(
        self,
        *,
        limit: int,
    ) -> tuple[IndependentLabelTestSetFreezeResultV2, ...]:
        if limit < 1 or limit > 1_000:
            raise IndependentLabelTestSetConflictError("freeze result page limit is invalid")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT r.result_id
                FROM label_test_set_freeze_results AS r
                JOIN label_test_set_freeze_records AS f
                  ON f.freeze_id = r.freeze_id
                ORDER BY f.sequence
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            results = tuple(self._load_result(connection, str(row["result_id"])) for row in rows)
            for dataset_series_id in sorted({result.freeze_record.dataset_series_id for result in results}):
                self._validate_series_history(connection, dataset_series_id)
            return results

    def rebuild_current_heads(self) -> IndependentLabelHeadRebuild:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT r.result_id, f.sequence
                FROM label_test_set_freeze_results AS r
                JOIN label_test_set_freeze_records AS f
                  ON f.freeze_id = r.freeze_id
                WHERE f.outcome = 'FROZEN'
                ORDER BY f.sequence
                """
            ).fetchall()
            series_rows = connection.execute(
                """
                SELECT DISTINCT dataset_series_id
                FROM label_test_set_freeze_records
                ORDER BY dataset_series_id
                """
            ).fetchall()
            for row in series_rows:
                self._validate_series_history(
                    connection,
                    str(row["dataset_series_id"]),
                )
            latest: dict[str, tuple[IndependentLabelTestSetFreezeResultV2, int]] = {}
            for row in rows:
                result = self._load_result(connection, str(row["result_id"]))
                latest[result.freeze_record.dataset_series_id] = (
                    result,
                    int(row["sequence"]),
                )
            connection.execute("DELETE FROM label_test_set_current_heads")
            for dataset_series_id, (result, sequence) in sorted(latest.items()):
                manifest = result.dataset_manifest
                if manifest is None:
                    raise IndependentLabelTestSetIntegrityError("FROZEN result is missing dataset manifest")
                connection.execute(
                    """
                    INSERT INTO label_test_set_current_heads (
                        dataset_series_id, freeze_id, dataset_id,
                        freeze_sequence, result_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_series_id,
                        result.freeze_record.freeze_id,
                        manifest.dataset_id,
                        sequence,
                        result.result_id,
                    ),
                )
            connection.commit()
            return IndependentLabelHeadRebuild(head_count=len(latest))
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def authorize_and_get_material(
        self,
        *,
        principal: TrustedIndependentLabelTestSetPrincipalV1,
        request: IndependentLabelTestSetAccessRequestV1,
        audit: ContractAudit,
    ) -> IndependentLabelAccessResult:
        request_sha256 = _access_request_sha256(principal, request)
        replay_id = self._preflight_idempotency(
            scope="access",
            idempotency_key=request.idempotency_key,
            request_sha256=request_sha256,
        )
        if replay_id is not None:
            return self._replay_access(replay_id)

        result = self._result_for_dataset(request.dataset_manifest_ref)
        manifest = result.dataset_manifest
        if manifest is None:
            raise IndependentLabelTestSetIntegrityError("dataset access resolved a pending freeze")
        access_policy = self._load_access_policy(manifest.access_policy_ref)
        reason = _authorization_reason(
            principal=principal,
            request=request,
            manifest=manifest,
            access_policy=access_policy,
        )
        material: IndependentLabelTestSetMaterialV1 | None = None
        if reason is LabelTestSetAccessReasonV2.AUTHORIZED:
            try:
                material = self.material_store.get_test_set(manifest.private_material_ref)
            except IndependentLabelMaterialError:
                reason = LabelTestSetAccessReasonV2.MATERIAL_INTEGRITY_FAILED
            else:
                if len(material.members) > request.max_members:
                    reason = LabelTestSetAccessReasonV2.MEMBER_LIMIT_EXCEEDED
                    material = None
        granted = reason is LabelTestSetAccessReasonV2.AUTHORIZED
        receipt = IndependentLabelTestSetAccessReceiptV2.create(
            dataset_manifest_ref=manifest.to_ref(),
            principal_ref=principal.principal_ref,
            purpose=request.purpose,
            access_policy_ref=access_policy.to_ref(),
            outcome=(LabelTestSetAccessOutcomeV2.GRANTED if granted else LabelTestSetAccessOutcomeV2.DENIED),
            reason=reason,
            returned_member_count=0 if material is None else len(material.members),
            material_verified=material is not None,
            audit=audit,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay_id = self._check_idempotency(
                connection,
                scope="access",
                idempotency_key=request.idempotency_key,
                request_sha256=request_sha256,
            )
            if replay_id is not None:
                connection.rollback()
                return self._replay_access(replay_id)
            self._insert_access_receipt(connection, receipt)
            self._record_idempotency(
                connection,
                scope="access",
                idempotency_key=request.idempotency_key,
                request_sha256=request_sha256,
                response_type="independent-label-test-set-access-receipt",
                response_id=receipt.receipt_id,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if material is None:
            raise IndependentLabelTestSetAuthorizationError(
                _authorization_message(reason),
                receipt=receipt,
            )
        return IndependentLabelAccessResult(material=material, receipt=receipt)

    def _replay_access(self, receipt_id: str) -> IndependentLabelAccessResult:
        receipt = self._load_access_receipt(receipt_id)
        if receipt.outcome is LabelTestSetAccessOutcomeV2.DENIED:
            raise IndependentLabelTestSetAuthorizationError(
                _authorization_message(receipt.reason),
                receipt=receipt,
            )
        result = self._result_for_dataset(receipt.dataset_manifest_ref)
        manifest = result.dataset_manifest
        if manifest is None:
            raise IndependentLabelTestSetIntegrityError("granted access receipt points to pending result")
        material = self.material_store.get_test_set(manifest.private_material_ref)
        if len(material.members) != receipt.returned_member_count:
            raise IndependentLabelTestSetIntegrityError(
                "access receipt member count differs from private material"
            )
        return IndependentLabelAccessResult(material=material, receipt=receipt)

    def _validate_supersession(
        self,
        connection: sqlite3.Connection,
        *,
        dataset_series_id: str,
        supersedes_dataset_ref: ObjectRef | None,
        supersedes_freeze_ref: ObjectRef | None,
    ) -> None:
        latest = connection.execute(
            """
            SELECT freeze_id, record_json
            FROM label_test_set_freeze_records
            WHERE dataset_series_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (dataset_series_id,),
        ).fetchone()
        if latest is None:
            if supersedes_freeze_ref is not None:
                raise IndependentLabelTestSetConflictError("first freeze cannot supersede another freeze")
        else:
            latest_record = _parse_freeze_record(str(latest["record_json"]))
            if supersedes_freeze_ref != latest_record.to_ref():
                raise IndependentLabelTestSetConflictError(
                    "freeze supersession does not match latest attempt"
                )
        head = connection.execute(
            """
            SELECT result_id
            FROM label_test_set_current_heads
            WHERE dataset_series_id = ?
            """,
            (dataset_series_id,),
        ).fetchone()
        if head is None:
            if supersedes_dataset_ref is not None:
                raise IndependentLabelTestSetConflictError("freeze cannot supersede a missing dataset")
        else:
            current = self._load_result(connection, str(head["result_id"]))
            manifest = current.dataset_manifest
            if manifest is None or supersedes_dataset_ref != manifest.to_ref():
                raise IndependentLabelTestSetConflictError(
                    "dataset supersession does not match current authority"
                )

    def _validate_series_history(
        self,
        connection: sqlite3.Connection,
        dataset_series_id: str,
    ) -> None:
        rows = connection.execute(
            """
            SELECT r.result_id
            FROM label_test_set_freeze_results AS r
            JOIN label_test_set_freeze_records AS f
              ON f.freeze_id = r.freeze_id
            WHERE f.dataset_series_id = ?
            ORDER BY f.sequence
            """,
            (dataset_series_id,),
        ).fetchall()
        previous_freeze_ref: ObjectRef | None = None
        current_dataset_ref: ObjectRef | None = None
        for row in rows:
            result = self._load_result(connection, str(row["result_id"]))
            record = result.freeze_record
            if record.supersedes_freeze_ref != previous_freeze_ref:
                raise IndependentLabelTestSetIntegrityError("freeze supersession chain is broken")
            if result.dataset_manifest is not None:
                if result.dataset_manifest.supersedes_dataset_ref != current_dataset_ref:
                    raise IndependentLabelTestSetIntegrityError("dataset supersession chain is broken")
                current_dataset_ref = result.dataset_manifest.to_ref()
            previous_freeze_ref = record.to_ref()

    def _load_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> IndependentLabelTestSetFreezeResultV2:
        row = connection.execute(
            """
            SELECT result_id, freeze_id, dataset_series_id,
                   requested_dataset_version, outcome,
                   dataset_manifest_id, result_sha256, record_json
            FROM label_test_set_freeze_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("freeze result was not found")
        try:
            result = IndependentLabelTestSetFreezeResultV2.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelTestSetIntegrityError("freeze result JSON is invalid") from exc
        if result.canonical_json().decode() != str(row["record_json"]):
            raise IndependentLabelTestSetIntegrityError("freeze result JSON is not canonical")
        manifest_id = None if result.dataset_manifest is None else result.dataset_manifest.dataset_id
        if (
            result.result_id != str(row["result_id"])
            or result.result_sha256 != str(row["result_sha256"])
            or result.freeze_record.freeze_id != str(row["freeze_id"])
            or result.freeze_record.dataset_series_id != str(row["dataset_series_id"])
            or result.freeze_record.requested_dataset_version != str(row["requested_dataset_version"])
            or result.freeze_record.outcome.value != str(row["outcome"])
            or manifest_id != row["dataset_manifest_id"]
        ):
            raise IndependentLabelTestSetIntegrityError("freeze result materialized columns drifted")
        freeze = self._load_freeze_record(
            connection,
            result.freeze_record.freeze_id,
        )
        if freeze != result.freeze_record:
            raise IndependentLabelTestSetIntegrityError("freeze result differs from immutable freeze record")
        self._load_policy(connection, result.policy_ref)
        self._load_access_policy_from_connection(connection, result.access_policy_ref)
        for partition_ref in result.partition_refs:
            self._load_partition(connection, partition_ref)
        if result.dataset_manifest is not None:
            manifest = self._load_manifest(
                connection,
                result.dataset_manifest.to_ref(),
            )
            if manifest != result.dataset_manifest:
                raise IndependentLabelTestSetIntegrityError(
                    "freeze result differs from immutable dataset manifest"
                )
            self.material_store.verify_test_set(manifest.private_material_ref)
        self.material_store.verify_candidate_pool(result.freeze_record.candidate_pool_ref)
        for partition_ref in result.partition_refs:
            partition = self._load_partition(connection, partition_ref)
            self.material_store.verify_partition(partition.private_material_ref)
        return result

    def _load_freeze_record(
        self,
        connection: sqlite3.Connection,
        freeze_id: str,
    ) -> IndependentLabelTestSetFreezeRecordV2:
        row = connection.execute(
            """
            SELECT freeze_id, dataset_series_id, requested_dataset_version,
                   outcome, dataset_manifest_id, supersedes_freeze_id,
                   freeze_sha256, record_json
            FROM label_test_set_freeze_records
            WHERE freeze_id = ?
            """,
            (freeze_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("freeze record was not found")
        record = _parse_freeze_record(str(row["record_json"]))
        if record.canonical_json().decode() != str(row["record_json"]):
            raise IndependentLabelTestSetIntegrityError("freeze record JSON is not canonical")
        manifest_id = None if record.dataset_manifest_ref is None else record.dataset_manifest_ref.object_id
        supersedes_id = (
            None if record.supersedes_freeze_ref is None else record.supersedes_freeze_ref.object_id
        )
        if (
            record.freeze_id != str(row["freeze_id"])
            or record.dataset_series_id != str(row["dataset_series_id"])
            or record.requested_dataset_version != str(row["requested_dataset_version"])
            or record.outcome.value != str(row["outcome"])
            or manifest_id != row["dataset_manifest_id"]
            or supersedes_id != row["supersedes_freeze_id"]
            or record.freeze_sha256 != str(row["freeze_sha256"])
        ):
            raise IndependentLabelTestSetIntegrityError("freeze record materialized columns drifted")
        return record

    def _load_policy(
        self,
        connection: sqlite3.Connection,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetPolicyV2:
        row = connection.execute(
            """
            SELECT policy_id, policy_sha256, record_json
            FROM label_test_set_policies
            WHERE policy_id = ?
            """,
            (ref.object_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("freeze policy is missing")
        try:
            value = IndependentLabelTestSetPolicyV2.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelTestSetIntegrityError("freeze policy JSON is invalid") from exc
        if (
            value.to_ref() != ref
            or value.policy_id != str(row["policy_id"])
            or value.policy_sha256 != str(row["policy_sha256"])
            or value.canonical_json().decode() != str(row["record_json"])
        ):
            raise IndependentLabelTestSetIntegrityError("freeze policy materialized columns drifted")
        return value

    def _load_access_policy(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetAccessPolicyV2:
        with self._connection() as connection:
            return self._load_access_policy_from_connection(connection, ref)

    def _load_access_policy_from_connection(
        self,
        connection: sqlite3.Connection,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetAccessPolicyV2:
        row = connection.execute(
            """
            SELECT access_policy_id, access_policy_sha256, record_json
            FROM label_test_set_access_policies
            WHERE access_policy_id = ?
            """,
            (ref.object_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("access policy is missing")
        try:
            value = IndependentLabelTestSetAccessPolicyV2.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelTestSetIntegrityError("access policy JSON is invalid") from exc
        if (
            value.to_ref() != ref
            or value.access_policy_id != str(row["access_policy_id"])
            or value.access_policy_sha256 != str(row["access_policy_sha256"])
            or value.canonical_json().decode() != str(row["record_json"])
        ):
            raise IndependentLabelTestSetIntegrityError("access policy materialized columns drifted")
        return value

    def _load_partition(
        self,
        connection: sqlite3.Connection,
        ref: ObjectRef,
    ) -> IndependentLabelPartitionManifestV2:
        row = connection.execute(
            """
            SELECT partition_manifest_id, partition_kind,
                   private_material_id, private_material_sha256, record_json
            FROM label_test_set_partition_manifests
            WHERE partition_manifest_id = ?
            """,
            (ref.object_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("partition manifest is missing")
        try:
            value = IndependentLabelPartitionManifestV2.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelTestSetIntegrityError("partition manifest JSON is invalid") from exc
        if (
            value.to_ref() != ref
            or value.partition_manifest_id != str(row["partition_manifest_id"])
            or value.partition_kind.value != str(row["partition_kind"])
            or value.private_material_ref.object_id != str(row["private_material_id"])
            or value.private_material_ref.object_sha256 != str(row["private_material_sha256"])
            or value.canonical_json().decode() != str(row["record_json"])
        ):
            raise IndependentLabelTestSetIntegrityError("partition manifest materialized columns drifted")
        return value

    def _load_manifest(
        self,
        connection: sqlite3.Connection,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetManifestV2:
        row = connection.execute(
            """
            SELECT dataset_id, dataset_series_id, dataset_version,
                   dataset_sha256, private_material_id,
                   private_material_sha256, record_json
            FROM independent_label_test_set_manifests
            WHERE dataset_id = ?
            """,
            (ref.object_id,),
        ).fetchone()
        if row is None:
            raise IndependentLabelTestSetIntegrityError("dataset manifest is missing")
        try:
            value = IndependentLabelTestSetManifestV2.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise IndependentLabelTestSetIntegrityError("dataset manifest JSON is invalid") from exc
        if (
            value.to_ref() != ref
            or value.dataset_id != str(row["dataset_id"])
            or value.dataset_series_id != str(row["dataset_series_id"])
            or value.dataset_version != str(row["dataset_version"])
            or value.dataset_sha256 != str(row["dataset_sha256"])
            or value.private_material_ref.object_id != str(row["private_material_id"])
            or value.private_material_ref.object_sha256 != str(row["private_material_sha256"])
            or value.canonical_json().decode() != str(row["record_json"])
        ):
            raise IndependentLabelTestSetIntegrityError("dataset manifest materialized columns drifted")
        return value

    def _result_for_dataset(
        self,
        ref: ObjectRef,
    ) -> IndependentLabelTestSetFreezeResultV2:
        _require_ref(
            ref,
            "independent-label-test-set-manifest",
            "v2",
            "dataset_manifest_ref",
        )
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT result_id
                FROM label_test_set_freeze_results
                WHERE dataset_manifest_id = ?
                """,
                (ref.object_id,),
            ).fetchone()
            if row is None:
                raise IndependentLabelTestSetIntegrityError("dataset manifest has no freeze result")
            result = self._load_result(connection, str(row["result_id"]))
            if result.dataset_manifest is None or result.dataset_manifest.to_ref() != ref:
                raise IndependentLabelTestSetIntegrityError("dataset manifest differs from freeze result")
            return result

    def _load_access_receipt(
        self,
        receipt_id: str,
    ) -> IndependentLabelTestSetAccessReceiptV2:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT receipt_id, dataset_id, principal_id, purpose,
                       outcome, reason, receipt_sha256, record_json
                FROM label_test_set_access_receipts
                WHERE receipt_id = ?
                """,
                (receipt_id,),
            ).fetchone()
            if row is None:
                raise IndependentLabelTestSetIntegrityError("access receipt was not found")
            try:
                receipt = IndependentLabelTestSetAccessReceiptV2.model_validate_json(str(row["record_json"]))
            except (ValidationError, ValueError) as exc:
                raise IndependentLabelTestSetIntegrityError("access receipt JSON is invalid") from exc
            if (
                receipt.receipt_id != str(row["receipt_id"])
                or receipt.dataset_manifest_ref.object_id != str(row["dataset_id"])
                or receipt.principal_ref.object_id != str(row["principal_id"])
                or receipt.purpose.value != str(row["purpose"])
                or receipt.outcome.value != str(row["outcome"])
                or receipt.reason.value != str(row["reason"])
                or receipt.receipt_sha256 != str(row["receipt_sha256"])
                or receipt.canonical_json().decode() != str(row["record_json"])
            ):
                raise IndependentLabelTestSetIntegrityError("access receipt materialized columns drifted")
            return receipt

    @staticmethod
    def _insert_policy(
        connection: sqlite3.Connection,
        value: IndependentLabelTestSetPolicyV2,
    ) -> None:
        _insert_exact(
            connection,
            table="label_test_set_policies",
            key_column="policy_id",
            key=value.policy_id,
            columns=("policy_sha256", "record_json"),
            values=(value.policy_sha256, value.canonical_json().decode()),
        )

    @staticmethod
    def _insert_access_policy(
        connection: sqlite3.Connection,
        value: IndependentLabelTestSetAccessPolicyV2,
    ) -> None:
        _insert_exact(
            connection,
            table="label_test_set_access_policies",
            key_column="access_policy_id",
            key=value.access_policy_id,
            columns=("access_policy_sha256", "record_json"),
            values=(
                value.access_policy_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_partition(
        connection: sqlite3.Connection,
        value: IndependentLabelPartitionManifestV2,
    ) -> None:
        _insert_exact(
            connection,
            table="label_test_set_partition_manifests",
            key_column="partition_manifest_id",
            key=value.partition_manifest_id,
            columns=(
                "partition_kind",
                "private_material_id",
                "private_material_sha256",
                "record_json",
            ),
            values=(
                value.partition_kind.value,
                value.private_material_ref.object_id,
                value.private_material_ref.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_manifest(
        connection: sqlite3.Connection,
        value: IndependentLabelTestSetManifestV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO independent_label_test_set_manifests (
                dataset_id, dataset_series_id, dataset_version,
                dataset_sha256, private_material_id,
                private_material_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.dataset_id,
                value.dataset_series_id,
                value.dataset_version,
                value.dataset_sha256,
                value.private_material_ref.object_id,
                value.private_material_ref.object_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_freeze_record(
        connection: sqlite3.Connection,
        result: IndependentLabelTestSetFreezeResultV2,
    ) -> int:
        value = result.freeze_record
        cursor = connection.execute(
            """
            INSERT INTO label_test_set_freeze_records (
                freeze_id, dataset_series_id, requested_dataset_version,
                outcome, dataset_manifest_id, supersedes_freeze_id,
                freeze_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.freeze_id,
                value.dataset_series_id,
                value.requested_dataset_version,
                value.outcome.value,
                (None if value.dataset_manifest_ref is None else value.dataset_manifest_ref.object_id),
                (None if value.supersedes_freeze_ref is None else value.supersedes_freeze_ref.object_id),
                value.freeze_sha256,
                value.canonical_json().decode(),
            ),
        )
        if cursor.lastrowid is None:
            raise IndependentLabelTestSetIntegrityError("freeze record sequence was not allocated")
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_result(
        connection: sqlite3.Connection,
        value: IndependentLabelTestSetFreezeResultV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO label_test_set_freeze_results (
                result_id, freeze_id, dataset_series_id,
                requested_dataset_version, outcome,
                dataset_manifest_id, result_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.result_id,
                value.freeze_record.freeze_id,
                value.freeze_record.dataset_series_id,
                value.freeze_record.requested_dataset_version,
                value.freeze_record.outcome.value,
                (None if value.dataset_manifest is None else value.dataset_manifest.dataset_id),
                value.result_sha256,
                value.canonical_json().decode(),
            ),
        )

    @staticmethod
    def _insert_access_receipt(
        connection: sqlite3.Connection,
        value: IndependentLabelTestSetAccessReceiptV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO label_test_set_access_receipts (
                receipt_id, dataset_id, principal_id, purpose,
                outcome, reason, receipt_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.receipt_id,
                value.dataset_manifest_ref.object_id,
                value.principal_ref.object_id,
                value.purpose.value,
                value.outcome.value,
                value.reason.value,
                value.receipt_sha256,
                value.canonical_json().decode(),
            ),
        )

    def _preflight_idempotency(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
    ) -> str | None:
        with self._connection() as connection:
            return self._check_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )

    @staticmethod
    def _check_idempotency(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
    ) -> str | None:
        row = connection.execute(
            """
            SELECT request_sha256, response_id
            FROM label_test_set_idempotency
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        if str(row["request_sha256"]) != request_sha256:
            raise IndependentLabelTestSetConflictError("idempotency key was reused for a different request")
        return str(row["response_id"])

    @staticmethod
    def _record_idempotency(
        connection: sqlite3.Connection,
        *,
        scope: str,
        idempotency_key: str,
        request_sha256: str,
        response_type: str,
        response_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO label_test_set_idempotency (
                scope, idempotency_key, request_sha256,
                response_type, response_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                scope,
                idempotency_key,
                request_sha256,
                response_type,
                response_id,
            ),
        )

    def _fault(self, point: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector.maybe_raise(point)


def _insert_exact(
    connection: sqlite3.Connection,
    *,
    table: str,
    key_column: str,
    key: str,
    columns: tuple[str, ...],
    values: tuple[str, ...],
) -> None:
    selected = ", ".join(columns)
    row = connection.execute(
        f"SELECT {selected} FROM {table} WHERE {key_column} = ?",
        (key,),
    ).fetchone()
    if row is not None:
        if tuple(str(row[column]) for column in columns) != values:
            raise IndependentLabelTestSetIntegrityError(f"immutable {table} row conflicts")
        return
    placeholders = ", ".join("?" for _ in range(len(columns) + 1))
    names = ", ".join((key_column, *columns))
    connection.execute(
        f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        (key, *values),
    )


def _parse_freeze_record(value: str) -> IndependentLabelTestSetFreezeRecordV2:
    try:
        return IndependentLabelTestSetFreezeRecordV2.model_validate_json(value)
    except (ValidationError, ValueError) as exc:
        raise IndependentLabelTestSetIntegrityError("freeze record JSON is invalid") from exc


def _authorization_reason(
    *,
    principal: TrustedIndependentLabelTestSetPrincipalV1,
    request: IndependentLabelTestSetAccessRequestV1,
    manifest: IndependentLabelTestSetManifestV2,
    access_policy: IndependentLabelTestSetAccessPolicyV2,
) -> LabelTestSetAccessReasonV2:
    if request.access_policy_ref != manifest.access_policy_ref:
        return LabelTestSetAccessReasonV2.ACCESS_POLICY_STALE
    grant = next(
        (item for item in access_policy.principal_grants if item.principal_ref == principal.principal_ref),
        None,
    )
    if grant is None:
        return LabelTestSetAccessReasonV2.PRINCIPAL_NOT_AUTHORIZED
    if request.purpose not in principal.allowed_purposes or request.purpose not in grant.purposes:
        return LabelTestSetAccessReasonV2.PURPOSE_NOT_AUTHORIZED
    if request.dataset_manifest_ref not in principal.allowed_dataset_refs:
        return LabelTestSetAccessReasonV2.DATASET_NOT_AUTHORIZED
    if (
        request.max_members > principal.max_members
        or request.max_members > access_policy.max_members_per_read
        or manifest.selected_member_count > request.max_members
    ):
        return LabelTestSetAccessReasonV2.MEMBER_LIMIT_EXCEEDED
    return LabelTestSetAccessReasonV2.AUTHORIZED


def _authorization_message(reason: LabelTestSetAccessReasonV2) -> str:
    if reason is LabelTestSetAccessReasonV2.PRINCIPAL_NOT_AUTHORIZED:
        return "statistical principal is not authorized"
    if reason is LabelTestSetAccessReasonV2.PURPOSE_NOT_AUTHORIZED:
        return "statistical purpose is not authorized"
    if reason is LabelTestSetAccessReasonV2.DATASET_NOT_AUTHORIZED:
        return "statistical dataset is not authorized"
    if reason is LabelTestSetAccessReasonV2.ACCESS_POLICY_STALE:
        return "statistical access policy is stale"
    if reason is LabelTestSetAccessReasonV2.MEMBER_LIMIT_EXCEEDED:
        return "statistical member limit exceeded"
    if reason is LabelTestSetAccessReasonV2.MATERIAL_INTEGRITY_FAILED:
        return "statistical private material integrity failed"
    return "statistical access denied"


def _freeze_request_sha256(
    *,
    dataset_series_id: str,
    dataset_version: str,
    label_specs: tuple[LabelSpecV2, ...],
    policy: IndependentLabelTestSetPolicyV2,
    access_policy: IndependentLabelTestSetAccessPolicyV2,
    candidate_pool: IndependentLabelCandidatePoolV1,
    train_partition_manifest: IndependentLabelPartitionManifestV2,
    train_partition_material: IndependentLabelPartitionMaterialV1,
    development_partition_manifest: IndependentLabelPartitionManifestV2,
    development_partition_material: IndependentLabelPartitionMaterialV1,
    supersedes_dataset_ref: ObjectRef | None,
    supersedes_freeze_ref: ObjectRef | None,
) -> str:
    return _payload_sha256(
        {
            "dataset_series_id": dataset_series_id,
            "dataset_version": dataset_version,
            "label_specs": [
                {
                    "id": value.label_spec_id,
                    "version": value.label_version,
                    "claimed_sha256": value.label_spec_sha256,
                    "observed_sha256": _payload_sha256(
                        value.model_dump(
                            mode="python",
                            exclude={
                                "label_spec_id",
                                "label_spec_sha256",
                                "audit",
                            },
                            exclude_none=False,
                        )
                    ),
                }
                for value in sorted(label_specs, key=lambda item: item.label_spec_id)
            ],
            "policy_ref": _ref_payload(policy.to_ref()),
            "access_policy_ref": _ref_payload(access_policy.to_ref()),
            "candidate_pool_ref": _ref_payload(candidate_pool.to_ref()),
            "train_partition_ref": _ref_payload(train_partition_manifest.to_ref()),
            "train_material_ref": _ref_payload(train_partition_material.to_ref()),
            "development_partition_ref": _ref_payload(development_partition_manifest.to_ref()),
            "development_material_ref": _ref_payload(development_partition_material.to_ref()),
            "supersedes_dataset_ref": (
                None if supersedes_dataset_ref is None else _ref_payload(supersedes_dataset_ref)
            ),
            "supersedes_freeze_ref": (
                None if supersedes_freeze_ref is None else _ref_payload(supersedes_freeze_ref)
            ),
        }
    )


def _access_request_sha256(
    principal: TrustedIndependentLabelTestSetPrincipalV1,
    request: IndependentLabelTestSetAccessRequestV1,
) -> str:
    return _payload_sha256(
        {
            "principal": {
                "principal_ref": _ref_payload(principal.principal_ref),
                "allowed_purposes": sorted(value.value for value in principal.allowed_purposes),
                "allowed_dataset_refs": [
                    _ref_payload(value)
                    for value in sorted(
                        principal.allowed_dataset_refs,
                        key=lambda value: (
                            value.object_type,
                            value.object_id,
                            value.object_version,
                            value.object_sha256,
                        ),
                    )
                ],
                "max_members": principal.max_members,
                "audit_case_ref": (
                    None if principal.audit_case_ref is None else _ref_payload(principal.audit_case_ref)
                ),
            },
            "request": request.model_dump(mode="python", exclude_none=False),
        }
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_payload(value: ObjectRef) -> dict[str, str]:
    return {
        "object_type": value.object_type,
        "object_id": value.object_id,
        "object_version": value.object_version,
        "object_sha256": value.object_sha256,
    }


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise IndependentLabelTestSetIntegrityError(
            f"{field_name} must reference {object_type} {object_version}"
        )


def _result_ref(result_id: str) -> ObjectRef:
    digest = result_id.rsplit("/", 1)[-1]
    return ObjectRef(
        object_type="independent-label-test-set-freeze-result",
        object_id=result_id,
        object_version="v2",
        object_sha256=digest,
    )


def _optional_material_ref(
    value: IndependentLabelTestSetFreezeCompilation,
) -> ObjectRef | None:
    return None if value.selected_material is None else value.selected_material.to_ref()


__all__ = [
    "IndependentLabelAccessResult",
    "IndependentLabelHeadRebuild",
    "IndependentLabelTestSetAuthorizationError",
    "IndependentLabelTestSetConflictError",
    "IndependentLabelTestSetIntegrityError",
    "IndependentLabelTestSetPersistenceError",
    "IndependentLabelTestSetPersistenceFaultInjector",
    "IndependentLabelTestSetPersistenceInjectedCrash",
    "IndependentLabelTestSetPersistenceService",
    "StaticIndependentLabelTestSetPersistenceFaultInjector",
]
