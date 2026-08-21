from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError, model_validator

from env_mock_agent.facade import LHWorkspaceExportFacade
from eval_factory.contracts.approval_decision_v2 import (
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.production_release_v2 import (
    ProductionLHExportReceiptV2,
    ProductionPublishedItemProjectionV2,
    ProductionReleasePolicyV2,
    ProductionReleaseResultV2,
    validate_production_published_item_projection_v2_identity,
)
from eval_factory.contracts.quality_v2 import (
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionResultV2,
    release_projection_result_v2_ref,
)
from eval_factory.contracts.release_publication_v2 import (
    PublishedItemProjectionV2,
    ReleasePublicationResultV2,
    validate_published_item_projection_v2_identity,
    validate_release_publication_result_v2_identity,
)
from eval_factory.contracts.task_v2 import rubric_set_ref
from eval_factory.dataset.export import (
    ReleaseBundleFacts,
    ReleasePublicationItemSource,
    validate_lh_release_source,
)
from eval_factory.dataset.persistence import (
    ReleaseProjectionPersistenceService,
)
from eval_factory.dataset.production_bundle_store import (
    ProductionReleaseBundleIntegrityError,
    ProductionReleaseBundleStore,
    ProductionReleaseBundleWriteResult,
)
from eval_factory.dataset.production_release import (
    ProductionReleaseAttestationGate,
    ProductionReleaseCompiler,
    ProductionReleaseConflictError,
    ProductionReleaseIntegrityError,
    ProductionReleaseManifestCompilation,
    ProductionReleasePredecessor,
)
from eval_factory.dataset.production_release_store import (
    ProductionRegistryStore,
)
from eval_factory.orchestration.job_store import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    JobStore,
    RecordNotFoundError,
    _attributes,
    _request_sha256,
)
from eval_factory.orchestration.models import ItemRecord, JobRecord


class ProductionReleaseAcceptanceV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-release-acceptance/private-v1"] = (
        "eval-factory/production-release-acceptance/private-v1"
    )
    acceptance_id: str
    job_id: str
    result_ref: ObjectRef
    registry_entry_ref: ObjectRef
    policy_ref: ObjectRef
    attestation_authority_ref: ObjectRef
    published_at: datetime
    acceptance_sha256: Sha256

    @model_validator(mode="after")
    def validate_acceptance(self) -> Self:
        expected = _acceptance_sha256(
            job_id=self.job_id,
            result_ref=self.result_ref,
            registry_entry_ref=self.registry_entry_ref,
            policy_ref=self.policy_ref,
            attestation_authority_ref=self.attestation_authority_ref,
            published_at=self.published_at,
        )
        if (
            self.result_ref.object_type != "production-release-result"
            or self.registry_entry_ref.object_type != "production-registry-entry"
            or self.policy_ref.object_type != "production-release-policy"
            or self.attestation_authority_ref.object_type != "production-attestation-authority"
            or self.acceptance_sha256 != expected
            or self.acceptance_id != f"production-release-acceptance://sha256/{expected}"
        ):
            raise ValueError("production release acceptance identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        result: ProductionReleaseResultV2,
    ) -> ProductionReleaseAcceptanceV1:
        assert result.registry_entry_ref is not None
        assert result.attestation_authority_ref is not None
        assert result.registry_entry is not None
        digest = _acceptance_sha256(
            job_id=job_id,
            result_ref=result.to_ref(),
            registry_entry_ref=result.registry_entry_ref,
            policy_ref=result.policy_ref,
            attestation_authority_ref=result.attestation_authority_ref,
            published_at=result.registry_entry.published_at,
        )
        return cls(
            acceptance_id=(f"production-release-acceptance://sha256/{digest}"),
            job_id=job_id,
            result_ref=result.to_ref(),
            registry_entry_ref=result.registry_entry_ref,
            policy_ref=result.policy_ref,
            attestation_authority_ref=result.attestation_authority_ref,
            published_at=result.registry_entry.published_at,
            acceptance_sha256=digest,
        )


class ProductionReleasePersistenceService:
    def __init__(
        self,
        store: JobStore,
        *,
        bundle_store: ProductionReleaseBundleStore,
        registry_store: ProductionRegistryStore,
        workspace_facade: LHWorkspaceExportFacade,
        attestation_gate: ProductionReleaseAttestationGate,
    ) -> None:
        _require_disjoint_roots(
            (
                bundle_store.root,
                registry_store.root,
                attestation_gate.accepted_store.material_store.root,
                attestation_gate.accepted_store.report_store.root,
            )
        )
        self.store = store
        self.bundle_store = bundle_store
        self.registry_store = registry_store
        self.workspace_facade = workspace_facade
        self.attestation_gate = attestation_gate
        self.compiler = ProductionReleaseCompiler()
        self.release_projection = ReleaseProjectionPersistenceService(store)

    def publish_job(
        self,
        *,
        item_sources: tuple[ReleasePublicationItemSource, ...],
        policy: ProductionReleasePolicyV2,
        release_profile_payload: bytes,
        registry: str,
        attestation_acceptance_key: str,
        previous_attestation_acceptance_key: str | None,
        idempotency_key: str,
        audit: ContractAudit,
    ) -> ProductionReleaseResultV2:
        if not item_sources:
            raise ProductionReleaseConflictError("production publication requires approved Items")
        item_sources = tuple(validate_lh_release_source(value) for value in item_sources)
        job_ids = {value.approved_result.release_subject.job_id for value in item_sources}
        if len(job_ids) != 1:
            raise ProductionReleaseConflictError("production publication sources cross Jobs")
        job_id = next(iter(job_ids))
        scope = f"publish-production-job:{job_id}"
        request_sha256 = _production_request_sha256(
            item_sources=item_sources,
            policy=policy,
            release_profile_payload=release_profile_payload,
            registry=registry,
            attestation_acceptance_key=attestation_acceptance_key,
            previous_attestation_acceptance_key=(previous_attestation_acceptance_key),
        )
        with self.store._read_snapshot() as connection:
            prior = self.store._idempotent_response(
                connection,
                scope=scope,
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
            )
            if prior is not None:
                return self._load_replay(connection, prior)
        with self.attestation_gate.hold_current(
            acceptance_key=attestation_acceptance_key,
            previous_acceptance_key=previous_attestation_acceptance_key,
            policy=policy,
            clock=self.store._clock,
            audit=audit,
        ) as admission:
            if admission.authority is None:
                return ProductionReleaseResultV2.create_blocked(
                    policy_ref=policy.to_ref(),
                    evidence_class=policy.evidence_class,
                    reason_codes=(admission.reason_code,),
                    repository_evidence_ref=None,
                    audit=audit,
                )
            observed_at = admission.authority.verified_at
            with self.store._read_snapshot() as connection:
                if (
                    connection.execute(
                        """
                        SELECT acceptance_id
                        FROM production_release_acceptances
                        WHERE job_id = ?
                        """,
                        (job_id,),
                    ).fetchone()
                    is not None
                ):
                    raise ProductionReleaseConflictError("the Job already has production authority")
                preliminary_sources, predecessors = self._current_sources(
                    connection,
                    item_sources,
                )
            preliminary = self.compiler.compile_manifest(
                job_id=job_id,
                item_sources=preliminary_sources,
                predecessors=predecessors,
                attestation=admission.authority,
                release_profile_payload=release_profile_payload,
                registry=registry,
                policy=policy,
                audit=audit,
            )
            bundles = self.bundle_store.build(
                preliminary,
                facade=self.workspace_facade,
            )
            ordered = _order_bundles(preliminary, bundles)
            expected = self.compiler.compile_publication(
                manifest=preliminary,
                workspace_results=tuple(value.workspace_export_result for value in ordered),
                bundle_facts=tuple(value.facts for value in ordered),
                policy=policy,
                published_at=observed_at,
                audit=audit,
            ).result
            self.registry_store.put_closure(policy=policy, result=expected)
            with self.store._transaction() as connection:
                prior = self.store._idempotent_response(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                )
                if prior is not None:
                    return self._load_replay(connection, prior)
                if (
                    connection.execute(
                        """
                        SELECT acceptance_id
                        FROM production_release_acceptances
                        WHERE job_id = ?
                        """,
                        (job_id,),
                    ).fetchone()
                    is not None
                ):
                    raise ProductionReleaseConflictError("the Job already has production authority")
                current_sources, current_predecessors = self._current_sources(
                    connection,
                    item_sources,
                )
                current_manifest = self.compiler.compile_manifest(
                    job_id=job_id,
                    item_sources=current_sources,
                    predecessors=current_predecessors,
                    attestation=admission.authority,
                    release_profile_payload=release_profile_payload,
                    registry=registry,
                    policy=policy,
                    audit=audit,
                )
                current = self.compiler.compile_publication(
                    manifest=current_manifest,
                    workspace_results=tuple(value.workspace_export_result for value in ordered),
                    bundle_facts=tuple(value.facts for value in ordered),
                    policy=policy,
                    published_at=observed_at,
                    audit=audit,
                ).result
                if (
                    current_manifest != preliminary
                    or current != expected
                    or self.registry_store.get_result(current.to_ref()) != current
                ):
                    raise ProductionReleaseConflictError("production authority changed before commit")
                acceptance = ProductionReleaseAcceptanceV1.create(
                    job_id=job_id,
                    result=current,
                )
                self._persist_decisions_and_items(
                    connection,
                    result=current,
                    sources=current_sources,
                )
                self._persist_acceptance(
                    connection,
                    acceptance=acceptance,
                )
                self._persist_current(
                    connection,
                    acceptance=acceptance,
                    result=current,
                )
                projected = self._project_items(
                    connection,
                    result=current,
                    updated_at=observed_at,
                )
                for item, projection in zip(
                    projected,
                    current.item_projections,
                    strict=True,
                ):
                    self.store._append_outbox(
                        connection,
                        aggregate_type="ITEM",
                        aggregate_id=item.item_id,
                        aggregate_version=item.row_version,
                        event_type="item-published-production",
                        attributes=_attributes(
                            acceptance_id=acceptance.acceptance_id,
                            projection_id=projection.projection_id,
                            result_id=current.result_id,
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
                    event_type="release-published-production",
                    attributes=_attributes(
                        acceptance_id=acceptance.acceptance_id,
                        item_count=len(current.item_projections),
                        registry=registry,
                        result_id=current.result_id,
                    ),
                )
                self.store._record_idempotency(
                    connection,
                    scope=scope,
                    idempotency_key=idempotency_key,
                    request_sha256=request_sha256,
                    response_type="PRODUCTION_RELEASE",
                    response_id=acceptance.acceptance_id,
                    created_at=observed_at,
                )
                return current

    def get_publication(
        self,
        acceptance_id: str,
    ) -> ProductionReleaseResultV2:
        with self.store._read_snapshot() as connection:
            acceptance = self._load_acceptance(connection, acceptance_id)
            return self._verify_acceptance(connection, acceptance)

    def get_current_item_publication(
        self,
        item_id: str,
    ) -> ProductionPublishedItemProjectionV2:
        with self.store._read_snapshot() as connection:
            row = connection.execute(
                """
                SELECT acceptance_id
                FROM production_item_publication_current
                WHERE item_id = ?
                """,
                (item_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"current production publication not found: {item_id}")
            acceptance = self._load_acceptance(
                connection,
                str(row["acceptance_id"]),
            )
            result = self._verify_acceptance(connection, acceptance)
            matches = tuple(value for value in result.item_projections if value.item_id == item_id)
            if len(matches) != 1:
                raise ProductionReleaseIntegrityError("production result does not contain one current Item")
            return matches[0]

    def resolve_bundle(
        self,
        *,
        acceptance_id: str,
        item_id: str,
    ) -> Path:
        result = self.get_publication(acceptance_id)
        matches = tuple(
            (receipt, projection)
            for receipt, projection in zip(
                result.export_receipts,
                result.item_projections,
                strict=True,
            )
            if projection.item_id == item_id
        )
        if len(matches) != 1:
            raise RecordNotFoundError(f"production bundle not found: {item_id}")
        receipt, _projection = matches[0]
        return self.bundle_store.verify_facts(
            _receipt_facts(receipt),
            receipt.workspace_export_result,
        )

    def rebuild_job_publication(
        self,
        job_id: str,
    ) -> ProductionReleaseResultV2:
        with self.store._transaction() as connection:
            row = connection.execute(
                """
                SELECT acceptance_id
                FROM production_release_acceptances
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"production release not found for Job: {job_id}")
            acceptance = self._load_acceptance(
                connection,
                str(row["acceptance_id"]),
            )
            result = self._verify_immutable_acceptance(connection, acceptance)
            connection.execute(
                """
                DELETE FROM production_item_publication_current
                WHERE job_id = ?
                """,
                (job_id,),
            )
            for projection in result.item_projections:
                connection.execute(
                    """
                    DELETE FROM production_item_publication_current
                    WHERE item_id = ?
                    """,
                    (projection.item_id,),
                )
            self._persist_current(
                connection,
                acceptance=acceptance,
                result=result,
            )
            self._project_items(
                connection,
                result=result,
                updated_at=acceptance.published_at,
            )
            self._verify_current_authority(
                connection,
                acceptance=acceptance,
                result=result,
            )
            return result

    def _current_sources(
        self,
        connection: sqlite3.Connection,
        item_sources: tuple[ReleasePublicationItemSource, ...],
    ) -> tuple[
        tuple[ReleasePublicationItemSource, ...],
        tuple[ProductionReleasePredecessor, ...],
    ]:
        current_sources: list[ReleasePublicationItemSource] = []
        predecessors: list[ProductionReleasePredecessor] = []
        for source in item_sources:
            current = self.release_projection._load_current_result_required(
                connection,
                source.approved_result.release_subject.item_id,
            )
            if release_projection_result_v2_ref(current) != (
                release_projection_result_v2_ref(source.approved_result)
            ):
                raise ProductionReleaseIntegrityError("production source differs from current R7 authority")
            current_sources.append(replace(source, approved_result=current))
            predecessors.append(self._resolve_predecessor(connection, current))
        ordered = sorted(
            zip(current_sources, predecessors, strict=True),
            key=lambda pair: pair[0].approved_result.release_subject.item_id,
        )
        return (
            tuple(value[0] for value in ordered),
            tuple(value[1] for value in ordered),
        )

    def _resolve_predecessor(
        self,
        connection: sqlite3.Connection,
        approved: ReleaseProjectionResultV2,
    ) -> ProductionReleasePredecessor:
        row = connection.execute(
            """
            SELECT projection_id, result_id, record_json
            FROM item_publication_current_projections
            WHERE item_id = ?
            """,
            (approved.release_subject.item_id,),
        ).fetchone()
        if row is None:
            return ProductionReleasePredecessor.direct(approved)
        try:
            projection = PublishedItemProjectionV2.model_validate_json(str(row["record_json"]))
            validate_published_item_projection_v2_identity(projection)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("current non-production projection is invalid") from exc
        result_row = connection.execute(
            """
            SELECT record_json
            FROM release_publication_results
            WHERE result_id = ?
            """,
            (str(row["result_id"]),),
        ).fetchone()
        if result_row is None:
            raise ProductionReleaseIntegrityError("current non-production result is missing")
        try:
            publication = ReleasePublicationResultV2.model_validate_json(str(result_row["record_json"]))
            validate_release_publication_result_v2_identity(publication)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("current non-production result is invalid") from exc
        if (
            projection not in publication.item_projections
            or projection.approved_result_ref != release_projection_result_v2_ref(approved)
        ):
            raise ProductionReleaseIntegrityError("non-production predecessor differs from approved source")
        index = publication.item_projections.index(projection)
        return ProductionReleasePredecessor(
            decision=publication.published_decisions[index],
            evaluation_item=publication.published_items[index],
            source_nonproduction_publication_ref=publication.to_ref(),
        )

    def _persist_decisions_and_items(
        self,
        connection: sqlite3.Connection,
        *,
        result: ProductionReleaseResultV2,
        sources: tuple[ReleasePublicationItemSource, ...],
    ) -> None:
        for source, decision, item in zip(
            sources,
            result.published_decisions,
            result.published_items,
            strict=True,
        ):
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
                    source.approved_result.release_subject.job_id,
                    decision.item_id,
                    source.approved_result.release_subject.release_subject_id,
                    decision.chain_id,
                    decision.previous_decision_ref.object_id if decision.previous_decision_ref else None,
                    decision.action.value,
                    decision.state.value,
                    decision.decision_sha256,
                    self.store._record_json(decision),
                ),
            )
            connection.execute(
                """
                INSERT INTO evaluation_items_v2 (
                    evaluation_item_id, job_id, item_id, release_decision_id,
                    item_version, item_sha256, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.evaluation_item_id,
                    source.approved_result.release_subject.job_id,
                    source.approved_result.release_subject.item_id,
                    item.release_decision_ref.object_id,
                    item.item_version,
                    item.item_sha256,
                    self.store._record_json(item),
                ),
            )

    def _persist_acceptance(
        self,
        connection: sqlite3.Connection,
        *,
        acceptance: ProductionReleaseAcceptanceV1,
    ) -> None:
        connection.execute(
            """
            INSERT INTO production_release_acceptances (
                acceptance_id, job_id, result_id, result_sha256,
                registry_entry_id, policy_id, attestation_authority_id,
                published_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acceptance.acceptance_id,
                acceptance.job_id,
                acceptance.result_ref.object_id,
                acceptance.result_ref.object_sha256,
                acceptance.registry_entry_ref.object_id,
                acceptance.policy_ref.object_id,
                acceptance.attestation_authority_ref.object_id,
                acceptance.published_at.isoformat(),
                self.store._record_json(acceptance),
            ),
        )

    def _persist_current(
        self,
        connection: sqlite3.Connection,
        *,
        acceptance: ProductionReleaseAcceptanceV1,
        result: ProductionReleaseResultV2,
    ) -> None:
        for projection in result.item_projections:
            connection.execute(
                """
                INSERT INTO production_item_publication_current (
                    item_id, job_id, projection_id, result_id,
                    acceptance_id, item_status, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    projection.item_id,
                    projection.job_id,
                    projection.projection_id,
                    result.result_id,
                    acceptance.acceptance_id,
                    projection.item_status.value,
                    self.store._record_json(projection),
                ),
            )

    def _project_items(
        self,
        connection: sqlite3.Connection,
        *,
        result: ProductionReleaseResultV2,
        updated_at: datetime,
    ) -> tuple[ItemRecord, ...]:
        projected: list[ItemRecord] = []
        for projection in result.item_projections:
            item = self.release_projection._load_item(
                connection,
                projection.item_id,
            )
            if item.status is ItemStatus.RELEASED:
                projected.append(item)
                continue
            if item.status is not ItemStatus.APPROVED:
                raise ConcurrencyConflictError("production publication Item is not APPROVED")
            updated = ItemRecord.model_validate(
                {
                    **item.model_dump(mode="python"),
                    "status": ItemStatus.RELEASED,
                    "row_version": item.row_version + 1,
                    "updated_at": updated_at,
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
            projected.append(updated)
        return tuple(projected)

    def _load_replay(
        self,
        connection: sqlite3.Connection,
        prior: tuple[str, str],
    ) -> ProductionReleaseResultV2:
        response_type, response_id = prior
        if response_type != "PRODUCTION_RELEASE":
            raise IdempotencyConflictError("production release replay response type is corrupt")
        acceptance = self._load_acceptance(connection, response_id)
        return self._verify_acceptance(connection, acceptance)

    def _load_acceptance(
        self,
        connection: sqlite3.Connection,
        acceptance_id: str,
    ) -> ProductionReleaseAcceptanceV1:
        row = connection.execute(
            """
            SELECT acceptance_id, job_id, result_id, result_sha256,
                   registry_entry_id, policy_id, attestation_authority_id,
                   published_at, record_json
            FROM production_release_acceptances
            WHERE acceptance_id = ?
            """,
            (acceptance_id,),
        ).fetchone()
        if row is None:
            raise ProductionReleaseIntegrityError("production release acceptance is missing")
        try:
            value = ProductionReleaseAcceptanceV1.model_validate_json(str(row["record_json"]))
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseIntegrityError("production release acceptance is invalid") from exc
        if (
            str(row["acceptance_id"]) != value.acceptance_id
            or str(row["job_id"]) != value.job_id
            or str(row["result_id"]) != value.result_ref.object_id
            or str(row["result_sha256"]) != value.result_ref.object_sha256
            or str(row["registry_entry_id"]) != value.registry_entry_ref.object_id
            or str(row["policy_id"]) != value.policy_ref.object_id
            or str(row["attestation_authority_id"]) != value.attestation_authority_ref.object_id
            or str(row["published_at"]) != value.published_at.isoformat()
        ):
            raise ProductionReleaseIntegrityError("production release acceptance columns are inconsistent")
        return value

    def _verify_acceptance(
        self,
        connection: sqlite3.Connection,
        acceptance: ProductionReleaseAcceptanceV1,
    ) -> ProductionReleaseResultV2:
        result = self._verify_immutable_acceptance(connection, acceptance)
        self._verify_current_authority(
            connection,
            acceptance=acceptance,
            result=result,
        )
        return result

    def _verify_immutable_acceptance(
        self,
        connection: sqlite3.Connection,
        acceptance: ProductionReleaseAcceptanceV1,
    ) -> ProductionReleaseResultV2:
        result = self.registry_store.get_result(acceptance.result_ref)
        registry_entry = result.registry_entry
        release_manifest = result.release_manifest
        if registry_entry is None or release_manifest is None:
            raise ProductionReleaseIntegrityError("production registry closure is incomplete")
        if (
            result.registry_entry_ref != acceptance.registry_entry_ref
            or result.policy_ref != acceptance.policy_ref
            or result.attestation_authority_ref != acceptance.attestation_authority_ref
            or release_manifest.job_id != acceptance.job_id
            or registry_entry.job_id != acceptance.job_id
            or registry_entry.published_at != acceptance.published_at
        ):
            raise ProductionReleaseIntegrityError("production registry closure differs from acceptance")
        for decision, item, _projection in zip(
            result.published_decisions,
            result.published_items,
            result.item_projections,
            strict=True,
        ):
            stored_decision = self.release_projection._load_decision(
                connection,
                decision.release_decision_id,
            )
            stored_item = self.release_projection._load_evaluation_item(
                connection,
                item.evaluation_item_id,
            )
            if stored_decision != decision or stored_item != item:
                raise ProductionReleaseIntegrityError("production immutable authority differs from result")
        try:
            for receipt, projection, registry_item_id in zip(
                result.export_receipts,
                result.item_projections,
                registry_entry.item_ids,
                strict=True,
            ):
                if registry_item_id != projection.item_id:
                    raise ProductionReleaseIntegrityError(
                        "production registry Item order differs from result"
                    )
                self.bundle_store.verify_facts(
                    _receipt_facts(receipt),
                    receipt.workspace_export_result,
                )
        except (ValueError, ProductionReleaseBundleIntegrityError) as exc:
            raise ProductionReleaseIntegrityError("production bundle closure is corrupt") from exc
        return result

    def _verify_current_authority(
        self,
        connection: sqlite3.Connection,
        *,
        acceptance: ProductionReleaseAcceptanceV1,
        result: ProductionReleaseResultV2,
    ) -> None:
        for projection in result.item_projections:
            row = connection.execute(
                """
                SELECT job_id, projection_id, result_id, acceptance_id,
                       item_status, record_json
                FROM production_item_publication_current
                WHERE item_id = ?
                """,
                (projection.item_id,),
            ).fetchone()
            if row is None:
                raise ProductionReleaseIntegrityError("production current projection is missing")
            try:
                stored_projection = ProductionPublishedItemProjectionV2.model_validate_json(
                    str(row["record_json"])
                )
                validate_production_published_item_projection_v2_identity(stored_projection)
            except (ValidationError, ValueError) as exc:
                raise ProductionReleaseIntegrityError("production current projection is invalid") from exc
            if (
                stored_projection != projection
                or str(row["job_id"]) != projection.job_id
                or str(row["projection_id"]) != projection.projection_id
                or str(row["result_id"]) != result.result_id
                or str(row["acceptance_id"]) != acceptance.acceptance_id
                or str(row["item_status"]) != projection.item_status.value
            ):
                raise ProductionReleaseIntegrityError("production current authority differs from result")
            item_record = self.release_projection._load_item(
                connection,
                projection.item_id,
            )
            if item_record.status is not ItemStatus.RELEASED:
                raise ProductionReleaseIntegrityError("production Item status is not RELEASED")


def _production_request_sha256(
    *,
    item_sources: tuple[ReleasePublicationItemSource, ...],
    policy: ProductionReleasePolicyV2,
    release_profile_payload: bytes,
    registry: str,
    attestation_acceptance_key: str,
    previous_attestation_acceptance_key: str | None,
) -> str:
    sources = tuple(
        sorted(
            (
                {
                    "item_id": value.approved_result.release_subject.item_id,
                    "approved_result_ref": release_projection_result_v2_ref(value.approved_result),
                    "rubric_set_ref": rubric_set_ref(value.rubric_set),
                    "item_quality_ref": item_quality_compilation_result_ref(value.item_quality),
                    "query_packaging_decision_ref": (
                        user_decision_commit_result_v2_ref(value.query_packaging_decision)
                        if value.query_packaging_decision is not None
                        else None
                    ),
                }
                for value in item_sources
            ),
            key=lambda value: str(value["item_id"]),
        )
    )
    return _request_sha256(
        {
            "approved_sources": sources,
            "policy_ref": policy.to_ref(),
            "release_profile_sha256": hashlib.sha256(release_profile_payload).hexdigest(),
            "registry": registry,
            "attestation_acceptance_key": attestation_acceptance_key,
            "previous_attestation_acceptance_key": (previous_attestation_acceptance_key),
        }
    )


def _acceptance_sha256(
    *,
    job_id: str,
    result_ref: ObjectRef,
    registry_entry_ref: ObjectRef,
    policy_ref: ObjectRef,
    attestation_authority_ref: ObjectRef,
    published_at: datetime,
) -> str:
    return _request_sha256(
        {
            "job_id": job_id,
            "result_ref": result_ref,
            "registry_entry_ref": registry_entry_ref,
            "policy_ref": policy_ref,
            "attestation_authority_ref": attestation_authority_ref,
            "published_at": published_at,
        }
    )


def _order_bundles(
    manifest: ProductionReleaseManifestCompilation,
    bundles: tuple[ProductionReleaseBundleWriteResult, ...],
) -> tuple[ProductionReleaseBundleWriteResult, ...]:
    by_item = {value.item_id: value for value in bundles}
    if len(by_item) != len(bundles):
        raise ProductionReleaseIntegrityError("production bundles contain duplicate Items")
    try:
        ordered = tuple(by_item[value.item_manifest.item_id] for value in manifest.items)
    except KeyError as exc:
        raise ProductionReleaseIntegrityError("production bundle coverage is incomplete") from exc
    return ordered


def _receipt_facts(value: ProductionLHExportReceiptV2) -> ReleaseBundleFacts:
    return ReleaseBundleFacts(
        bundle_sha256=value.bundle_sha256,
        bundle_file_count=value.bundle_file_count,
        bundle_total_bytes=value.bundle_total_bytes,
    )


def _require_disjoint_roots(values: tuple[Path, ...]) -> None:
    resolved = tuple(value.resolve() for value in values)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ProductionReleaseIntegrityError("production release roots must be pairwise disjoint")


__all__ = [
    "ProductionReleaseAcceptanceV1",
    "ProductionReleasePersistenceService",
]
