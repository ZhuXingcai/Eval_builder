from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from env_mock_agent.facade import LHWorkspaceExportFacade
from eval_factory.approval.persistence import UserDecisionPersistenceService
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPhaseV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_ref,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
)
from eval_factory.contracts.release_publication_v2 import (
    LHExportReceiptV2,
    LHReleaseItemManifestV2,
    NonProductionRegistryEntryV2,
    NonProductionReleaseManifestV2,
    PublishedItemProjectionV2,
    ReleasePublicationPolicyV2,
    ReleasePublicationResultV2,
    validate_lh_export_receipt_v2_identity,
    validate_lh_release_item_manifest_v2_identity,
    validate_nonproduction_registry_entry_v2_identity,
    validate_nonproduction_release_manifest_v2_identity,
    validate_published_item_projection_v2_identity,
    validate_release_publication_policy_v2_identity,
    validate_release_publication_result_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.dataset.bundle_store import (
    NonProductionReleaseBundleStore,
    ReleaseBundleIntegrityError,
    ReleaseBundleWriteResult,
)
from eval_factory.dataset.export import (
    ReleaseBundleFacts,
    ReleasePublicationCompiler,
    ReleasePublicationItemSource,
    ReleasePublicationManifestCompilation,
)
from eval_factory.dataset.persistence import ReleaseProjectionPersistenceService
from eval_factory.orchestration.job_store import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import ItemRecord, JobRecord


class ReleasePublicationConflictError(ValueError):
    pass


class ReleasePublicationIntegrityError(ImmutableResultError):
    pass


class ReleasePublicationPersistenceService:
    def __init__(
        self,
        store: JobStore,
        *,
        bundle_store: NonProductionReleaseBundleStore,
        workspace_facade: LHWorkspaceExportFacade,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self.bundle_store = bundle_store
        self.workspace_facade = workspace_facade
        self.compiler = ReleasePublicationCompiler()
        self.release_projection = ReleaseProjectionPersistenceService(store)
        self._fault_injector = fault_injector

    def publish_job(
        self,
        *,
        item_sources: tuple[ReleasePublicationItemSource, ...],
        rejected_item_ids: tuple[str, ...],
        policy: ReleasePublicationPolicyV2,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> ReleasePublicationResultV2:
        if not item_sources:
            raise ReleasePublicationConflictError("publication requires at least one approved Item")
        job_ids = {source.approved_result.release_subject.job_id for source in item_sources}
        if len(job_ids) != 1:
            raise ReleasePublicationConflictError("publication sources must belong to one Job")
        job_id = next(iter(job_ids))
        scope = f"publish-release-job:{job_id}"
        with self.store._connect() as connection:
            preliminary = self._compile_current_manifest(
                connection,
                item_sources=item_sources,
                rejected_item_ids=rejected_item_ids,
                policy=policy,
                base_contract_manifest_ref=base_contract_manifest_ref,
                overlay_contract_manifest_ref=overlay_contract_manifest_ref,
                release_profile_decision_ref=release_profile_decision_ref,
                audit=audit,
            )
            request_sha256 = _publication_request_sha256(
                preliminary,
                policy=policy,
            )
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_replay(
                    connection,
                    prior=prior,
                    manifest=preliminary,
                    policy=policy,
                )

        bundles = self.bundle_store.build(
            preliminary,
            facade=self.workspace_facade,
        )

        with self.store._transaction() as connection:
            current = self._compile_current_manifest(
                connection,
                item_sources=item_sources,
                rejected_item_ids=rejected_item_ids,
                policy=policy,
                base_contract_manifest_ref=base_contract_manifest_ref,
                overlay_contract_manifest_ref=overlay_contract_manifest_ref,
                release_profile_decision_ref=release_profile_decision_ref,
                audit=audit,
            )
            if current != preliminary:
                raise ReleasePublicationConflictError(
                    "publication authority changed while bundles were built"
                )
            current_request_sha256 = _publication_request_sha256(
                current,
                policy=policy,
            )
            if current_request_sha256 != request_sha256:
                raise ReleasePublicationConflictError("publication request changed while bundles were built")
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_replay(
                    connection,
                    prior=prior,
                    manifest=current,
                    policy=policy,
                )
            if (
                connection.execute(
                    """
                SELECT manifest_id
                FROM nonproduction_release_manifests
                WHERE job_id = ?
                """,
                    (job_id,),
                ).fetchone()
                is not None
            ):
                raise ReleasePublicationConflictError("the Job already has a non-production publication")
            ordered_bundles = self._validate_bundles(
                current,
                bundles,
            )
            published_at = self.store._clock()
            result = self.compiler.compile_publication(
                manifest=current,
                workspace_results=tuple(value.workspace_export_result for value in ordered_bundles),
                bundle_facts=tuple(value.facts for value in ordered_bundles),
                policy=policy,
                published_at=published_at,
                audit=audit,
            ).result
            self._persist_publication(
                connection,
                result=result,
                policy=policy,
            )
            self._fault("after_publication_rows")
            projected_items = tuple(
                self._project_released_item(
                    connection,
                    item_id=projection.item_id,
                )
                for projection in result.item_projections
            )
            self._fault("after_item_statuses")
            for projected, projection in zip(
                projected_items,
                result.item_projections,
                strict=True,
            ):
                self.store._append_outbox(
                    connection,
                    aggregate_type="ITEM",
                    aggregate_id=projected.item_id,
                    aggregate_version=projected.row_version,
                    event_type="item-published-nonproduction",
                    attributes=_attributes(
                        channel=result.registry_entry.channel.value,
                        projection_id=projection.projection_id,
                        registry_entry_id=(result.registry_entry.registry_entry_id),
                        result_id=result.result_id,
                    ),
                )
            job = self.store._get_record(
                connection,
                "jobs",
                "job_id",
                job_id,
                JobRecord,
            )
            self.store._append_outbox(
                connection,
                aggregate_type="JOB",
                aggregate_id=job_id,
                aggregate_version=job.row_version,
                event_type="release-published-nonproduction",
                attributes=_attributes(
                    channel=result.registry_entry.channel.value,
                    item_count=len(result.item_projections),
                    registry=result.registry_entry.registry,
                    registry_entry_id=result.registry_entry.registry_entry_id,
                    result_id=result.result_id,
                ),
            )
            self._fault("after_outbox")
            self.store._record_idempotency(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                response_type="RELEASE_PUBLICATION",
                response_id=result.result_id,
                created_at=published_at,
            )
            self._fault("after_idempotency")
            return result

    def get_publication(
        self,
        result_id: str,
    ) -> ReleasePublicationResultV2:
        with self.store._connect() as connection:
            result = self._load_result(connection, result_id)
            self._assert_current_publication(connection, result)
            return result

    def get_registry_entry(
        self,
        registry_entry_id: str,
    ) -> NonProductionRegistryEntryV2:
        with self.store._connect() as connection:
            row = connection.execute(
                """
                SELECT result_id
                FROM release_publication_results
                WHERE registry_entry_id = ?
                """,
                (registry_entry_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"non-production registry entry not found: {registry_entry_id}")
            result = self._load_result(
                connection,
                str(row["result_id"]),
            )
            if result.registry_entry.registry_entry_id != registry_entry_id:
                raise ReleasePublicationIntegrityError("registry entry differs from publication result")
            self._assert_current_publication(connection, result)
            return result.registry_entry

    def get_current_item_publication(
        self,
        item_id: str,
    ) -> PublishedItemProjectionV2:
        with self.store._connect() as connection:
            projection, result = self._load_current_projection(
                connection,
                item_id,
            )
            self._verify_result_bundles(result)
            return projection

    def list_registry(
        self,
        *,
        channel: ReleaseChannel,
        registry: str,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[NonProductionRegistryEntryV2, ...]:
        if channel not in {
            ReleaseChannel.CANARY,
            ReleaseChannel.INTERNAL_REVIEW,
        }:
            raise ReleasePublicationConflictError("R7 registry reads are non-production only")
        if offset < 0 or not 1 <= limit <= 500:
            raise ReleasePublicationConflictError("registry pagination is out of bounds")
        with self.store._connect() as connection:
            rows = connection.execute(
                """
                SELECT registry_entry_id
                FROM nonproduction_registry_entries
                WHERE channel = ? AND registry = ?
                ORDER BY registry_entry_id
                LIMIT ? OFFSET ?
                """,
                (channel.value, registry, limit, offset),
            ).fetchall()
            return tuple(self.get_registry_entry(str(row["registry_entry_id"])) for row in rows)

    def resolve_bundle(
        self,
        *,
        registry_entry_id: str,
        item_id: str,
    ) -> Path:
        with self.store._connect() as connection:
            row = connection.execute(
                """
                SELECT result_id
                FROM release_publication_results
                WHERE registry_entry_id = ?
                """,
                (registry_entry_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"non-production registry entry not found: {registry_entry_id}")
            result = self._load_result(
                connection,
                str(row["result_id"]),
            )
            self._assert_current_publication(connection, result)
            try:
                index = result.registry_entry.item_ids.index(item_id)
            except ValueError as exc:
                raise RecordNotFoundError(f"published Item not found in registry entry: {item_id}") from exc
            receipt = result.export_receipts[index]
            return self.bundle_store.verify_facts(
                result.registry_entry.channel,
                _receipt_facts(receipt),
                receipt.workspace_export_result,
            )

    def rebuild_job_publication(
        self,
        job_id: str,
    ) -> ReleasePublicationResultV2:
        with self.store._transaction() as connection:
            row = connection.execute(
                """
                SELECT result_id
                FROM release_publication_results
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"release publication not found for Job: {job_id}")
            result = self._load_result(
                connection,
                str(row["result_id"]),
            )
            connection.execute(
                """
                DELETE FROM item_publication_current_projections
                WHERE job_id = ?
                """,
                (job_id,),
            )
            for projection in result.item_projections:
                self._insert_current_projection(
                    connection,
                    result_id=result.result_id,
                    projection=projection,
                )
                item = self._load_item(connection, projection.item_id)
                if item.status is ItemStatus.APPROVED:
                    self._rewrite_item(
                        connection,
                        item,
                        ItemStatus.RELEASED,
                    )
                elif item.status is not ItemStatus.RELEASED:
                    raise ReleasePublicationIntegrityError(
                        "publication rebuild cannot reconcile the Item state"
                    )
            self._assert_current_publication(connection, result)
            return result

    def _compile_current_manifest(
        self,
        connection: sqlite3.Connection,
        *,
        item_sources: tuple[ReleasePublicationItemSource, ...],
        rejected_item_ids: tuple[str, ...],
        policy: ReleasePublicationPolicyV2,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ReleasePublicationManifestCompilation:
        sources: list[ReleasePublicationItemSource] = []
        for source in item_sources:
            approved = self.release_projection._load_current_result_required(
                connection,
                source.approved_result.release_subject.item_id,
            )
            if release_projection_result_v2_ref(approved) != release_projection_result_v2_ref(
                source.approved_result
            ):
                raise ReleasePublicationIntegrityError(
                    "publication source differs from current R7-08 authority"
                )
            decision = source.query_packaging_decision
            if decision is not None:
                stored = UserDecisionPersistenceService(self.store)._load_commit(
                    connection,
                    decision.commit_id,
                )
                if stored != decision:
                    raise ReleasePublicationIntegrityError("query packaging decision differs from storage")
                decision = stored
            sources.append(
                replace(
                    source,
                    approved_result=approved,
                    query_packaging_decision=decision,
                )
            )
        ordered = tuple(
            sorted(
                sources,
                key=lambda value: value.approved_result.release_subject.item_id,
            )
        )
        if not ordered:
            raise ReleasePublicationConflictError("publication requires approved Item sources")
        job_id = ordered[0].approved_result.release_subject.job_id
        self._validate_job_partition(
            connection,
            job_id=job_id,
            approved_item_ids=tuple(value.approved_result.release_subject.item_id for value in ordered),
            rejected_item_ids=rejected_item_ids,
        )
        return self.compiler.compile_manifest(
            job_id=job_id,
            item_sources=ordered,
            rejected_item_ids=rejected_item_ids,
            policy=policy,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            release_profile_decision_ref=release_profile_decision_ref,
            audit=audit,
        )

    def _validate_job_partition(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        approved_item_ids: tuple[str, ...],
        rejected_item_ids: tuple[str, ...],
    ) -> None:
        rows = connection.execute(
            """
            SELECT item_id, status
            FROM items
            WHERE job_id = ?
            ORDER BY item_id
            """,
            (job_id,),
        ).fetchall()
        observed = {str(row["item_id"]): ItemStatus(str(row["status"])) for row in rows}
        approved = tuple(sorted(set(approved_item_ids)))
        rejected = tuple(sorted(set(rejected_item_ids)))
        if (
            tuple(sorted(observed)) != tuple(sorted((*approved, *rejected)))
            or any(
                observed.get(item_id) not in {ItemStatus.APPROVED, ItemStatus.RELEASED}
                for item_id in approved
            )
            or any(observed.get(item_id) is not ItemStatus.REJECTED for item_id in rejected)
        ):
            raise ReleasePublicationConflictError(
                "publication must cover the exact terminal Job Item partition"
            )

    def _validate_bundles(
        self,
        manifest: ReleasePublicationManifestCompilation,
        bundles: tuple[ReleaseBundleWriteResult, ...],
    ) -> tuple[ReleaseBundleWriteResult, ...]:
        by_item = {value.item_id: value for value in bundles}
        if len(by_item) != len(bundles):
            raise ReleasePublicationIntegrityError("publication bundles contain duplicate Items")
        try:
            ordered = tuple(by_item[item.item_manifest.item_id] for item in manifest.items)
        except KeyError as exc:
            raise ReleasePublicationIntegrityError("publication bundle coverage is incomplete") from exc
        if set(by_item) != {item.item_manifest.item_id for item in manifest.items}:
            raise ReleasePublicationIntegrityError("publication bundle coverage is not exact")
        for item, bundle in zip(
            manifest.items,
            ordered,
            strict=True,
        ):
            if bundle.channel is not manifest.release_manifest.channel:
                raise ReleasePublicationIntegrityError("publication bundle channel differs from manifest")
            self.bundle_store.verify_item(
                manifest.release_manifest,
                item,
                bundle.facts,
            )
        return ordered

    def _persist_publication(
        self,
        connection: sqlite3.Connection,
        *,
        result: ReleasePublicationResultV2,
        policy: ReleasePublicationPolicyV2,
    ) -> None:
        approved_sources = self._load_approved_sources(
            connection,
            result,
        )
        self._validate_approved_source_bindings(
            result,
            approved_sources,
        )
        self._persist_policy(connection, policy)
        self._fault("after_policy")
        for item_manifest in result.release_manifest.item_manifests:
            self._persist_item_manifest(
                connection,
                item_manifest,
                policy_id=policy.policy_id,
            )
        self._fault("after_item_manifests")
        self._persist_manifest(connection, result.release_manifest)
        self._fault("after_manifest")
        for receipt in result.export_receipts:
            self._persist_receipt(connection, receipt)
        self._fault("after_receipts")
        for source, decision in zip(
            approved_sources,
            result.published_decisions,
            strict=True,
        ):
            self._persist_decision(
                connection,
                source=source,
                decision=decision,
            )
        self._fault("after_decisions")
        for source, item in zip(
            approved_sources,
            result.published_items,
            strict=True,
        ):
            self._persist_evaluation_item(
                connection,
                source=source,
                item=item,
            )
        self._fault("after_evaluation_items")
        self._persist_registry(connection, result.registry_entry)
        self._fault("after_registry")
        for projection in result.item_projections:
            self._persist_projection(connection, projection)
        self._fault("after_projections")
        self._persist_result(connection, result)
        self._fault("after_result")
        for projection in result.item_projections:
            self._insert_current_projection(
                connection,
                result_id=result.result_id,
                projection=projection,
            )
        self._fault("after_current_projections")

    def _persist_policy(
        self,
        connection: sqlite3.Connection,
        policy: ReleasePublicationPolicyV2,
    ) -> None:
        validate_release_publication_policy_v2_identity(policy)
        row = connection.execute(
            """
            SELECT record_json
            FROM release_publication_policies
            WHERE policy_id = ?
            """,
            (policy.policy_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO release_publication_policies (
                    policy_id, policy_sha256, record_json
                ) VALUES (?, ?, ?)
                """,
                (
                    policy.policy_id,
                    policy.policy_sha256,
                    self.store._record_json(policy),
                ),
            )
            return
        if self._load_policy(connection, policy.policy_id) != policy:
            raise ReleasePublicationIntegrityError("stored publication policy differs from input")

    def _persist_item_manifest(
        self,
        connection: sqlite3.Connection,
        value: LHReleaseItemManifestV2,
        *,
        policy_id: str,
    ) -> None:
        validate_lh_release_item_manifest_v2_identity(value)
        row = connection.execute(
            """
            SELECT item_manifest_id
            FROM lh_release_manifest_items
            WHERE item_manifest_id = ?
            """,
            (value.item_manifest_id,),
        ).fetchone()
        if row is not None:
            if (
                self._load_item_manifest(
                    connection,
                    value.item_manifest_id,
                )
                != value
            ):
                raise ReleasePublicationIntegrityError("stored release Item manifest differs from input")
            return
        connection.execute(
            """
            INSERT INTO lh_release_manifest_items (
                item_manifest_id, job_id, item_id, approved_result_id,
                policy_id, item_manifest_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.item_manifest_id,
                value.job_id,
                value.item_id,
                value.approved_release_result_ref.object_id,
                policy_id,
                value.item_manifest_sha256,
                self.store._record_json(value),
            ),
        )

    def _persist_manifest(
        self,
        connection: sqlite3.Connection,
        value: NonProductionReleaseManifestV2,
    ) -> None:
        validate_nonproduction_release_manifest_v2_identity(value)
        connection.execute(
            """
            INSERT INTO nonproduction_release_manifests (
                manifest_id, job_id, channel, registry, policy_id,
                manifest_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.manifest_id,
                value.job_id,
                value.channel.value,
                value.registry,
                value.policy_ref.object_id,
                value.manifest_sha256,
                self.store._record_json(value),
            ),
        )

    def _persist_receipt(
        self,
        connection: sqlite3.Connection,
        value: LHExportReceiptV2,
    ) -> None:
        validate_lh_export_receipt_v2_identity(value)
        connection.execute(
            """
            INSERT INTO lh_export_receipts (
                receipt_id, manifest_id, item_manifest_id,
                workspace_export_result_id, bundle_sha256,
                receipt_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.receipt_id,
                value.release_manifest_ref.object_id,
                value.item_manifest_ref.object_id,
                value.workspace_export_result_ref.object_id,
                value.bundle_sha256,
                value.receipt_sha256,
                self.store._record_json(value),
            ),
        )

    def _persist_decision(
        self,
        connection: sqlite3.Connection,
        *,
        source: ReleaseProjectionResultV2,
        decision: ReleaseDecisionV2,
    ) -> None:
        if decision.action is not ReleaseActionV2.PUBLISH or decision.state is not ReleaseStateV2.RELEASED:
            raise ReleasePublicationIntegrityError("publication persistence requires PUBLISH decisions")
        connection.execute(
            """
            INSERT INTO release_decisions_v2 (
                release_decision_id, job_id, item_id, release_subject_id,
                chain_id, previous_decision_id, action, state,
                decision_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.release_decision_id,
                source.release_subject.job_id,
                decision.item_id,
                source.release_subject.release_subject_id,
                decision.chain_id,
                decision.previous_decision_ref.object_id
                if decision.previous_decision_ref is not None
                else None,
                decision.action.value,
                decision.state.value,
                decision.decision_sha256,
                self.store._record_json(decision),
            ),
        )

    def _persist_evaluation_item(
        self,
        connection: sqlite3.Connection,
        *,
        source: ReleaseProjectionResultV2,
        item: EvaluationItemV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO evaluation_items_v2 (
                evaluation_item_id, job_id, item_id, release_decision_id,
                item_version, item_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.evaluation_item_id,
                source.release_subject.job_id,
                source.release_subject.item_id,
                item.release_decision_ref.object_id,
                item.item_version,
                item.item_sha256,
                self.store._record_json(item),
            ),
        )

    def _persist_registry(
        self,
        connection: sqlite3.Connection,
        value: NonProductionRegistryEntryV2,
    ) -> None:
        validate_nonproduction_registry_entry_v2_identity(value)
        connection.execute(
            """
            INSERT INTO nonproduction_registry_entries (
                registry_entry_id, manifest_id, job_id, channel, registry,
                published_at, entry_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.registry_entry_id,
                value.release_manifest_ref.object_id,
                value.job_id,
                value.channel.value,
                value.registry,
                value.published_at,
                value.entry_sha256,
                self.store._record_json(value),
            ),
        )

    def _persist_projection(
        self,
        connection: sqlite3.Connection,
        value: PublishedItemProjectionV2,
    ) -> None:
        validate_published_item_projection_v2_identity(value)
        connection.execute(
            """
            INSERT INTO item_publication_projection_events (
                projection_id, job_id, item_id, approved_result_id,
                manifest_id, receipt_id, registry_entry_id,
                publish_decision_id, published_evaluation_item_id,
                item_status, projection_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.projection_id,
                value.job_id,
                value.item_id,
                value.approved_result_ref.object_id,
                value.release_manifest_ref.object_id,
                value.export_receipt_ref.object_id,
                value.registry_entry_ref.object_id,
                value.publish_decision_ref.object_id,
                value.published_evaluation_item_ref.object_id,
                value.item_status.value,
                value.projection_sha256,
                self.store._record_json(value),
            ),
        )

    def _persist_result(
        self,
        connection: sqlite3.Connection,
        value: ReleasePublicationResultV2,
    ) -> None:
        validate_release_publication_result_v2_identity(value)
        connection.execute(
            """
            INSERT INTO release_publication_results (
                result_id, job_id, manifest_id, registry_entry_id,
                policy_id, result_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                value.result_id,
                value.release_manifest.job_id,
                value.release_manifest.manifest_id,
                value.registry_entry.registry_entry_id,
                value.policy_ref.object_id,
                value.result_sha256,
                self.store._record_json(value),
            ),
        )

    def _insert_current_projection(
        self,
        connection: sqlite3.Connection,
        *,
        result_id: str,
        projection: PublishedItemProjectionV2,
    ) -> None:
        connection.execute(
            """
            INSERT INTO item_publication_current_projections (
                item_id, job_id, projection_id, result_id,
                item_status, record_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                projection.item_id,
                projection.job_id,
                projection.projection_id,
                result_id,
                projection.item_status.value,
                self.store._record_json(projection),
            ),
        )

    def _project_released_item(
        self,
        connection: sqlite3.Connection,
        *,
        item_id: str,
    ) -> ItemRecord:
        item = self._load_item(connection, item_id)
        if item.status is not ItemStatus.APPROVED:
            raise ConcurrencyConflictError(f"publication Item is not APPROVED: {item_id}")
        return self._rewrite_item(
            connection,
            item,
            ItemStatus.RELEASED,
        )

    def _rewrite_item(
        self,
        connection: sqlite3.Connection,
        item: ItemRecord,
        target: ItemStatus,
    ) -> ItemRecord:
        updated = ItemRecord.model_validate(
            {
                **item.model_dump(mode="python"),
                "status": target,
                "row_version": item.row_version + 1,
                "updated_at": self.store._clock(),
            }
        )
        self.store._update_record(
            connection,
            "items",
            "item_id",
            item.item_id,
            updated.status,
            updated.row_version,
            updated,
            item.row_version,
        )
        return updated

    def _load_item(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> ItemRecord:
        row = connection.execute(
            """
            SELECT item_id, job_id, status, row_version,
                   idempotency_key, record_json
            FROM items
            WHERE item_id = ?
            """,
            (item_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ItemRecord not found: {item_id}")
        return self.store._load_item_row(row)

    def _load_replay(
        self,
        connection: sqlite3.Connection,
        *,
        prior: tuple[str, str],
        manifest: ReleasePublicationManifestCompilation,
        policy: ReleasePublicationPolicyV2,
    ) -> ReleasePublicationResultV2:
        response_type, response_id = prior
        if response_type != "RELEASE_PUBLICATION":
            raise IdempotencyConflictError("release publication replay response type is corrupt")
        stored = self._load_result(connection, response_id)
        if (
            stored.release_manifest.to_ref() != manifest.release_manifest.to_ref()
            or stored.policy_ref != policy.to_ref()
        ):
            raise ReleasePublicationIntegrityError("publication replay differs from current authority")
        self._assert_current_publication(connection, stored)
        return stored

    def _load_policy(
        self,
        connection: sqlite3.Connection,
        policy_id: str,
    ) -> ReleasePublicationPolicyV2:
        row = connection.execute(
            """
            SELECT policy_id, policy_sha256, record_json
            FROM release_publication_policies
            WHERE policy_id = ?
            """,
            (policy_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing its policy")
        try:
            value = ReleasePublicationPolicyV2.model_validate_json(str(row["record_json"]))
            validate_release_publication_policy_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored publication policy is malformed") from exc
        if str(row["policy_id"]) != value.policy_id or str(row["policy_sha256"]) != value.policy_sha256:
            raise ReleasePublicationIntegrityError("stored publication policy columns are inconsistent")
        return value

    def _load_item_manifest(
        self,
        connection: sqlite3.Connection,
        item_manifest_id: str,
    ) -> LHReleaseItemManifestV2:
        row = connection.execute(
            """
            SELECT item_manifest_id, job_id, item_id, approved_result_id,
                   item_manifest_sha256, record_json
            FROM lh_release_manifest_items
            WHERE item_manifest_id = ?
            """,
            (item_manifest_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing an Item manifest")
        try:
            value = LHReleaseItemManifestV2.model_validate_json(str(row["record_json"]))
            validate_lh_release_item_manifest_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored release Item manifest is malformed") from exc
        if (
            str(row["item_manifest_id"]) != value.item_manifest_id
            or str(row["job_id"]) != value.job_id
            or str(row["item_id"]) != value.item_id
            or str(row["approved_result_id"]) != value.approved_release_result_ref.object_id
            or str(row["item_manifest_sha256"]) != value.item_manifest_sha256
        ):
            raise ReleasePublicationIntegrityError("stored release Item manifest columns are inconsistent")
        return value

    def _load_manifest(
        self,
        connection: sqlite3.Connection,
        manifest_id: str,
    ) -> NonProductionReleaseManifestV2:
        row = connection.execute(
            """
            SELECT manifest_id, job_id, channel, registry, policy_id,
                   manifest_sha256, record_json
            FROM nonproduction_release_manifests
            WHERE manifest_id = ?
            """,
            (manifest_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing its release manifest")
        try:
            value = NonProductionReleaseManifestV2.model_validate_json(str(row["record_json"]))
            validate_nonproduction_release_manifest_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored release manifest is malformed") from exc
        if (
            str(row["manifest_id"]) != value.manifest_id
            or str(row["job_id"]) != value.job_id
            or str(row["channel"]) != value.channel.value
            or str(row["registry"]) != value.registry
            or str(row["policy_id"]) != value.policy_ref.object_id
            or str(row["manifest_sha256"]) != value.manifest_sha256
        ):
            raise ReleasePublicationIntegrityError("stored release manifest columns are inconsistent")
        children = tuple(
            self._load_item_manifest(
                connection,
                ref.object_id,
            )
            for ref in value.item_manifest_refs
        )
        if children != value.item_manifests:
            raise ReleasePublicationIntegrityError("release manifest differs from immutable Item rows")
        return value

    def _load_receipt(
        self,
        connection: sqlite3.Connection,
        receipt_id: str,
    ) -> LHExportReceiptV2:
        row = connection.execute(
            """
            SELECT receipt_id, manifest_id, item_manifest_id,
                   workspace_export_result_id, bundle_sha256,
                   receipt_sha256, record_json
            FROM lh_export_receipts
            WHERE receipt_id = ?
            """,
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing an export receipt")
        try:
            value = LHExportReceiptV2.model_validate_json(str(row["record_json"]))
            validate_lh_export_receipt_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored export receipt is malformed") from exc
        if (
            str(row["receipt_id"]) != value.receipt_id
            or str(row["manifest_id"]) != value.release_manifest_ref.object_id
            or str(row["item_manifest_id"]) != value.item_manifest_ref.object_id
            or str(row["workspace_export_result_id"]) != value.workspace_export_result_ref.object_id
            or str(row["bundle_sha256"]) != value.bundle_sha256
            or str(row["receipt_sha256"]) != value.receipt_sha256
        ):
            raise ReleasePublicationIntegrityError("stored export receipt columns are inconsistent")
        return value

    def _load_registry(
        self,
        connection: sqlite3.Connection,
        registry_entry_id: str,
    ) -> NonProductionRegistryEntryV2:
        row = connection.execute(
            """
            SELECT registry_entry_id, manifest_id, job_id, channel,
                   registry, published_at, entry_sha256, record_json
            FROM nonproduction_registry_entries
            WHERE registry_entry_id = ?
            """,
            (registry_entry_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing its registry entry")
        try:
            value = NonProductionRegistryEntryV2.model_validate_json(str(row["record_json"]))
            validate_nonproduction_registry_entry_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored registry entry is malformed") from exc
        if (
            str(row["registry_entry_id"]) != value.registry_entry_id
            or str(row["manifest_id"]) != value.release_manifest_ref.object_id
            or str(row["job_id"]) != value.job_id
            or str(row["channel"]) != value.channel.value
            or str(row["registry"]) != value.registry
            or str(row["published_at"]) != value.published_at
            or str(row["entry_sha256"]) != value.entry_sha256
        ):
            raise ReleasePublicationIntegrityError("stored registry entry columns are inconsistent")
        return value

    def _load_projection(
        self,
        connection: sqlite3.Connection,
        projection_id: str,
    ) -> PublishedItemProjectionV2:
        row = connection.execute(
            """
            SELECT projection_id, job_id, item_id, approved_result_id,
                   manifest_id, receipt_id, registry_entry_id,
                   publish_decision_id, published_evaluation_item_id,
                   item_status, projection_sha256, record_json
            FROM item_publication_projection_events
            WHERE projection_id = ?
            """,
            (projection_id,),
        ).fetchone()
        if row is None:
            raise ReleasePublicationIntegrityError("publication is missing an Item projection")
        try:
            value = PublishedItemProjectionV2.model_validate_json(str(row["record_json"]))
            validate_published_item_projection_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored Item publication projection is malformed") from exc
        if (
            str(row["projection_id"]) != value.projection_id
            or str(row["job_id"]) != value.job_id
            or str(row["item_id"]) != value.item_id
            or str(row["approved_result_id"]) != value.approved_result_ref.object_id
            or str(row["manifest_id"]) != value.release_manifest_ref.object_id
            or str(row["receipt_id"]) != value.export_receipt_ref.object_id
            or str(row["registry_entry_id"]) != value.registry_entry_ref.object_id
            or str(row["publish_decision_id"]) != value.publish_decision_ref.object_id
            or str(row["published_evaluation_item_id"]) != value.published_evaluation_item_ref.object_id
            or str(row["item_status"]) != value.item_status.value
            or str(row["projection_sha256"]) != value.projection_sha256
        ):
            raise ReleasePublicationIntegrityError(
                "stored Item publication projection columns are inconsistent"
            )
        return value

    def _load_result(
        self,
        connection: sqlite3.Connection,
        result_id: str,
    ) -> ReleasePublicationResultV2:
        row = connection.execute(
            """
            SELECT result_id, job_id, manifest_id, registry_entry_id,
                   policy_id, result_sha256, record_json
            FROM release_publication_results
            WHERE result_id = ?
            """,
            (result_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"ReleasePublicationResultV2 not found: {result_id}")
        try:
            value = ReleasePublicationResultV2.model_validate_json(str(row["record_json"]))
            validate_release_publication_result_v2_identity(value)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored release publication result is malformed") from exc
        if (
            str(row["result_id"]) != value.result_id
            or str(row["job_id"]) != value.release_manifest.job_id
            or str(row["manifest_id"]) != value.release_manifest.manifest_id
            or str(row["registry_entry_id"]) != value.registry_entry.registry_entry_id
            or str(row["policy_id"]) != value.policy_ref.object_id
            or str(row["result_sha256"]) != value.result_sha256
        ):
            raise ReleasePublicationIntegrityError(
                "stored release publication result columns are inconsistent"
            )
        approved_sources = self._load_approved_sources(
            connection,
            value,
        )
        if (
            self._load_policy(connection, value.policy_ref.object_id).to_ref() != value.policy_ref
            or self._load_manifest(
                connection,
                value.release_manifest.manifest_id,
            )
            != value.release_manifest
            or self._load_registry(
                connection,
                value.registry_entry.registry_entry_id,
            )
            != value.registry_entry
            or tuple(self._load_receipt(connection, receipt.receipt_id) for receipt in value.export_receipts)
            != value.export_receipts
            or tuple(
                self._load_projection(connection, projection.projection_id)
                for projection in value.item_projections
            )
            != value.item_projections
            or tuple(
                self.release_projection._load_decision(
                    connection,
                    decision.release_decision_id,
                )
                for decision in value.published_decisions
            )
            != value.published_decisions
            or tuple(
                self.release_projection._load_evaluation_item(
                    connection,
                    item.evaluation_item_id,
                )
                for item in value.published_items
            )
            != value.published_items
        ):
            raise ReleasePublicationIntegrityError("publication result differs from immutable child rows")
        self._validate_approved_source_bindings(
            value,
            approved_sources,
        )
        self._verify_result_bundles(value)
        return value

    def _load_current_projection(
        self,
        connection: sqlite3.Connection,
        item_id: str,
    ) -> tuple[PublishedItemProjectionV2, ReleasePublicationResultV2]:
        row = connection.execute(
            """
            SELECT item_id, job_id, projection_id, result_id,
                   item_status, record_json
            FROM item_publication_current_projections
            WHERE item_id = ?
            """,
            (item_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"current Item publication not found: {item_id}")
        try:
            projection = PublishedItemProjectionV2.model_validate_json(str(row["record_json"]))
            validate_published_item_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationIntegrityError("stored current Item publication is malformed") from exc
        if (
            str(row["item_id"]) != projection.item_id
            or str(row["job_id"]) != projection.job_id
            or str(row["projection_id"]) != projection.projection_id
            or str(row["item_status"]) != projection.item_status.value
        ):
            raise ReleasePublicationIntegrityError("stored current Item publication columns are inconsistent")
        result = self._load_result(
            connection,
            str(row["result_id"]),
        )
        if projection not in result.item_projections:
            raise ReleasePublicationIntegrityError("current Item publication differs from immutable result")
        item = self._load_item(connection, item_id)
        if item.status is not ItemStatus.RELEASED:
            raise ReleasePublicationIntegrityError("current publication Item is not RELEASED")
        return projection, result

    def _assert_current_publication(
        self,
        connection: sqlite3.Connection,
        result: ReleasePublicationResultV2,
    ) -> None:
        for expected in result.item_projections:
            try:
                observed, current_result = self._load_current_projection(
                    connection,
                    expected.item_id,
                )
            except RecordNotFoundError as exc:
                raise ReleasePublicationIntegrityError(
                    "publication result is missing its current Item projection"
                ) from exc
            if observed != expected or current_result.result_id != result.result_id:
                raise ReleasePublicationIntegrityError("publication result is not the current Item authority")

    def _verify_result_bundles(
        self,
        result: ReleasePublicationResultV2,
    ) -> None:
        try:
            for receipt in result.export_receipts:
                self.bundle_store.verify_facts(
                    result.registry_entry.channel,
                    _receipt_facts(receipt),
                    receipt.workspace_export_result,
                )
        except ReleaseBundleIntegrityError as exc:
            raise ReleasePublicationIntegrityError("published release bundle is corrupt") from exc

    def _load_approved_sources(
        self,
        connection: sqlite3.Connection,
        result: ReleasePublicationResultV2,
    ) -> tuple[ReleaseProjectionResultV2, ...]:
        sources = tuple(
            self.release_projection._load_result(
                connection,
                ref.object_id,
            )
            for ref in result.approved_source_result_refs
        )
        if (
            tuple(release_projection_result_v2_ref(source) for source in sources)
            != result.approved_source_result_refs
        ):
            raise ReleasePublicationIntegrityError("publication approved source refs are stale")
        return sources

    @staticmethod
    def _validate_approved_source_bindings(
        result: ReleasePublicationResultV2,
        sources: tuple[ReleaseProjectionResultV2, ...],
    ) -> None:
        for item_manifest, decision, item, projection, source in zip(
            result.release_manifest.item_manifests,
            result.published_decisions,
            result.published_items,
            result.item_projections,
            sources,
            strict=True,
        ):
            try:
                expected_version = str(int(source.release_decision.item_version) + 1)
            except ValueError as exc:
                raise ReleasePublicationIntegrityError(
                    "approved release decision item version is invalid"
                ) from exc
            if (
                source.phase is not ReleaseProjectionPhaseV2.TERMINAL
                or source.release_decision.action is not ReleaseActionV2.APPROVE
                or source.release_decision.state is not ReleaseStateV2.APPROVED
                or item_manifest.approved_release_result_ref != release_projection_result_v2_ref(source)
                or item_manifest.approved_evaluation_item_ref
                != evaluation_item_v2_ref(source.evaluation_item)
                or item_manifest.approved_release_decision_ref
                != release_decision_v2_ref(source.release_decision)
                or item_manifest.release_subject_ref != source.release_subject_ref
                or item_manifest.query_spec_ref != source.query_spec_ref
                or item_manifest.rubric_set_ref != source.release_subject.components.rubric_set_ref
                or item_manifest.environment_spec_ref
                != source.release_subject.components.environment_spec_ref
                or item_manifest.provenance_manifest_ref
                != source.release_subject.components.provenance_manifest_ref
                or item_manifest.quality_report_ref != source.release_subject.components.quality_report_ref
                or item_manifest.final_package_manifest_ref != source.release_subject.package_manifest_ref
                or item_manifest.package_sha256 != source.release_subject.package_sha256
            ):
                raise ReleasePublicationIntegrityError(
                    "publication Item manifest differs from approved source"
                )
            if (
                decision.previous_decision_ref != source.release_decision_ref
                or decision.chain_id != source.release_decision.chain_id
                or decision.item_id != source.release_decision.item_id
                or decision.item_version != expected_version
                or decision.components != source.release_decision.components
                or decision.release_subject_sha256 != source.release_decision.release_subject_sha256
                or decision.package_manifest_ref != source.release_decision.package_manifest_ref
                or decision.package_sha256 != source.release_decision.package_sha256
                or decision.batch_quality_report_ref != source.release_decision.batch_quality_report_ref
                or decision.automated_quality_passed != source.release_decision.automated_quality_passed
                or decision.open_p0_count != source.release_decision.open_p0_count
                or decision.open_p1_count != source.release_decision.open_p1_count
                or decision.unresolved_non_waivable_count
                != source.release_decision.unresolved_non_waivable_count
                or decision.user_approval_policy_ref != source.release_decision.user_approval_policy_ref
                or decision.required_checkpoints != source.release_decision.required_checkpoints
                or decision.checkpoint_decisions != source.release_decision.checkpoint_decisions
                or decision.channel is not source.release_decision.channel
                or decision.registry != source.release_decision.registry
                or decision.export_profile != source.release_decision.export_profile
                or decision.export_profile_version != source.release_decision.export_profile_version
                or decision.production_attestation_ref is not None
            ):
                raise ReleasePublicationIntegrityError("PUBLISH decision differs from approved source")
            if (
                item.item_version != decision.item_version
                or item.source_trace_refs != source.evaluation_item.source_trace_refs
                or item.label_decision_refs != source.evaluation_item.label_decision_refs
                or item.task_draft_ref != source.evaluation_item.task_draft_ref
                or item.attachment_reconstruction_result_ref
                != source.evaluation_item.attachment_reconstruction_result_ref
                or item.components != source.evaluation_item.components
                or item.user_approval_policy_ref != source.evaluation_item.user_approval_policy_ref
                or item.user_decision_record_refs != source.evaluation_item.user_decision_record_refs
                or item.release_decision_ref != release_decision_v2_ref(decision)
            ):
                raise ReleasePublicationIntegrityError(
                    "published EvaluationItem differs from approved source"
                )
            if (
                projection.approved_result_ref != release_projection_result_v2_ref(source)
                or projection.approved_projection_ref != source.item_projection_ref
            ):
                raise ReleasePublicationIntegrityError("published projection differs from approved source")

    def _fault(self, point: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point)


def _publication_request_sha256(
    manifest: ReleasePublicationManifestCompilation,
    *,
    policy: ReleasePublicationPolicyV2,
) -> str:
    return _request_sha256(
        {
            "release_manifest_ref": manifest.release_manifest.to_ref(),
            "policy_ref": policy.to_ref(),
            "approved_source_result_refs": [
                release_projection_result_v2_ref(item.source.approved_result) for item in manifest.items
            ],
            "rejected_item_ids": manifest.rejected_item_ids,
        }
    )


def _receipt_facts(value: LHExportReceiptV2) -> ReleaseBundleFacts:
    return ReleaseBundleFacts(
        bundle_sha256=value.bundle_sha256,
        bundle_file_count=value.bundle_file_count,
        bundle_total_bytes=value.bundle_total_bytes,
    )


__all__ = [
    "ReleasePublicationConflictError",
    "ReleasePublicationIntegrityError",
    "ReleasePublicationPersistenceService",
]
