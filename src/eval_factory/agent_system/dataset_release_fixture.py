from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from env_mock_agent.facade import (
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_carried_sha256,
    attachment_duplicate_fingerprint_request_ref,
    attachment_duplicate_fingerprint_result_carried_sha256,
)
from eval_factory.agent_system.attachment_quality_material import (
    FactoryAttachmentQualityMaterialStore,
)
from eval_factory.agent_system.batch_quality_material import (
    FactoryBatchQualityMaterialStore,
)
from eval_factory.agent_system.batch_quality_runtime import (
    FactoryBatchQualityContext,
    FactoryBatchQualityRuntime,
)
from eval_factory.agent_system.candidate_output import (
    AuthorizedCandidateExport,
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.candidate_projection_material import (
    FactoryCandidateProjectionMaterialStore,
)
from eval_factory.agent_system.candidate_projection_runtime import (
    FactoryCandidateProjectionRuntime,
)
from eval_factory.agent_system.delivery_runtime import (
    FactoryDeliveryRuntime,
    FactoryDeliveryRuntimeConfig,
)
from eval_factory.agent_system.job_store_bridge import (
    FactoryJobStoreAuthority,
)
from eval_factory.agent_system.job_store_witness_bridge import (
    FactoryJobStoreWitnessBridge,
)
from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.release_runtime import (
    FactoryDatasetReleaseContext,
    FactoryDatasetReleaseRuntime,
)
from eval_factory.agent_system.release_source_builder import (
    FactoryReleaseSourceBuilder,
)
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_policy_ref,
)
from eval_factory.contracts.batch_quality_v2 import BatchQualityPolicyV2
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.cross_item_safety_v2 import (
    CrossItemSafetyPolicyV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryItemStageV2,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
)
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPolicyV2,
)
from eval_factory.orchestration.job_store import JobStore


class FactoryDatasetReleaseRuntimeConfigV1(ContractModelV2):
    schema_version: Literal["eval-factory/factory-dataset-release-runtime-config/v1"] = (
        "eval-factory/factory-dataset-release-runtime-config/v1"
    )
    claim_scope: Literal["DEVELOPMENT_FIXTURE_ONLY"] = "DEVELOPMENT_FIXTURE_ONLY"
    duplicate_policy: DuplicateDetectionPolicyV2
    cross_item_policy: CrossItemSafetyPolicyV2
    batch_policy: BatchQualityPolicyV2
    approval_policy: UserApprovalPolicy
    release_policy: ReleaseProjectionPolicyV2
    not_required_checkpoints: tuple[
        ApprovalCheckpoint,
        ...,
    ] = ()
    output_target_ref: ObjectRef
    max_files: int = Field(ge=1, le=10_000_000)
    max_total_bytes: int = Field(
        ge=1,
        le=10_000_000_000,
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if (
            self.output_target_ref.object_type != "candidate-output-target"
            or self.output_target_ref.object_version != "v2"
        ):
            raise ValueError("release output target must be candidate-output-target/v2")
        order = {value: index for index, value in enumerate(ApprovalCheckpoint)}
        if self.not_required_checkpoints != tuple(
            sorted(
                set(self.not_required_checkpoints),
                key=lambda value: order[value],
            )
        ):
            raise ValueError("not-required checkpoints must be canonical")
        return self


@dataclass(frozen=True, slots=True)
class FactoryFixtureReleaseComponents:
    batch_runtime: FactoryBatchQualityRuntime
    batch_context_factory: FactoryFixtureBatchContextFactory
    release_context_factory: FactoryFixtureReleaseContextFactory
    witness_bridge: FactoryJobStoreWitnessBridge


class FactoryFixtureBatchContextFactory:
    def __init__(
        self,
        *,
        job_store: JobStore,
        config: FactoryDatasetReleaseRuntimeConfigV1,
    ) -> None:
        self.job_store = job_store
        self.config = config
        self._contexts: dict[str, FactoryBatchQualityContext] = {}

    def build(
        self,
        *,
        job_authority: FactoryJobStoreAuthority,
    ) -> FactoryBatchQualityContext:
        graph = self.job_store.get_job_work_graph(job_authority.job_spec.job_id)
        if graph != job_authority.graph:
            raise ValueError("batch context requires current JobStore graph")
        current = self._contexts.get(graph.resolved_job_work_graph_id)
        if current is not None:
            return current
        context = FactoryBatchQualityContext(
            resolved_job_work_graph=graph,
            duplicate_policy=self.config.duplicate_policy,
            duplicate_facade=_FixtureDuplicateFingerprintFacade(),
            cross_item_policy=self.config.cross_item_policy,
            cross_item_facade=_FixtureCrossItemSafetyFacade(),
            batch_policy=self.config.batch_policy,
        )
        self._contexts[graph.resolved_job_work_graph_id] = context
        return context


class FactoryFixtureReleaseContextFactory:
    def __init__(
        self,
        *,
        job_store: JobStore,
        config: FactoryDatasetReleaseRuntimeConfigV1,
        store: FactoryControlStore,
        private_store: FactoryPrivateObjectStore,
        plan_reviews: PlanReviewService,
        output_root: Path,
        requested_by: str,
    ) -> None:
        self.job_store = job_store
        self.config = config
        self.store = store
        self.quality_materials = FactoryAttachmentQualityMaterialStore(
            private_store,
        )
        self._contexts: dict[
            tuple[str, tuple[str, ...]],
            FactoryDatasetReleaseContext,
        ] = {}
        self._runtime = FactoryDatasetReleaseRuntime(
            candidates=FactoryCandidateProjectionRuntime(
                store=store,
                job_store=job_store,
                materials=FactoryCandidateProjectionMaterialStore(private_store),
            ),
            delivery=FactoryDeliveryRuntime(
                store=store,
                plan_reviews=plan_reviews,
                assembler=CandidateDatasetOutputAssembler(
                    store=store,
                    root=output_root,
                ),
                config=FactoryDeliveryRuntimeConfig(
                    output_target_ref=config.output_target_ref,
                    max_files=config.max_files,
                    max_total_bytes=config.max_total_bytes,
                ),
                requested_by=requested_by,
            ),
        )

    def build(
        self,
        *,
        job_authority: FactoryJobStoreAuthority,
        item_ids: tuple[str, ...],
    ) -> FactoryDatasetReleaseContext:
        graph = self.job_store.get_job_work_graph(job_authority.job_spec.job_id)
        candidate_item_ids = tuple(sorted(set(item_ids)))
        if (
            graph != job_authority.graph
            or user_approval_policy_ref(self.config.approval_policy)
            != job_authority.job_spec.approval_policy_ref
            or len(candidate_item_ids) != len(item_ids)
            or not set(candidate_item_ids).issubset(graph.item_ids)
        ):
            raise ValueError("release context requires current JobStore policy and graph")
        context_key = (
            graph.resolved_job_work_graph_id,
            candidate_item_ids,
        )
        current = self._contexts.get(context_key)
        if current is not None:
            return current
        context = FactoryDatasetReleaseContext(
            source_builder=FactoryReleaseSourceBuilder(
                store=self.store,
                job_store=self.job_store,
            ),
            runtime=self._runtime,
            approval_policy=self.config.approval_policy,
            release_policy=self.config.release_policy,
            exports=tuple(
                self._export(
                    item_id,
                )
                for item_id in candidate_item_ids
            ),
            not_required_checkpoints=(self.config.not_required_checkpoints),
        )
        self._contexts[context_key] = context
        return context

    def _export(
        self,
        item_id: str,
    ) -> AuthorizedCandidateExport:
        quality_head = self.store.get_item_stage_head(
            item_id,
            FactoryItemStageV2.ITEM_QUALITY,
        )
        quality = self.quality_materials.get(
            self.store.get_item_stage_material_ref(
                quality_head.to_ref(),
            )
        ).finalization.item_quality
        package = quality.final_package_manifest
        environment = quality.environment_spec
        if package is None or environment is None:
            raise ValueError("release fixture requires final package authority")
        artifacts = {value.output_ref: value for value in environment.artifacts}
        files = []
        for entry in package.entries:
            member = entry.inventory_member
            if member.container_ref is not None or member.member_type != "FILE":
                continue
            artifact = artifacts.get(entry.output_ref)
            if artifact is None:
                raise ValueError("release fixture package artifact is unavailable")
            payload = fixture_attachment_export_payload(
                artifact.artifact_id,
            )
            if (
                len(payload) != member.size_bytes
                or hashlib.sha256(payload).hexdigest() != member.content_sha256
            ):
                raise ValueError("release fixture payload differs from package authority")
            files.append(
                (
                    member.normalized_path,
                    payload,
                )
            )
        return AuthorizedCandidateExport(
            item_id=item_id,
            files=tuple(sorted(files)),
        )


def build_fixture_release_components(
    *,
    config: FactoryDatasetReleaseRuntimeConfigV1,
    job_store: JobStore,
    store: FactoryControlStore,
    private_store: FactoryPrivateObjectStore,
    plan_reviews: PlanReviewService,
    output_root: Path,
    requested_by: str,
) -> FactoryFixtureReleaseComponents:
    return FactoryFixtureReleaseComponents(
        batch_runtime=FactoryBatchQualityRuntime(
            store=store,
            materials=FactoryBatchQualityMaterialStore(private_store),
        ),
        batch_context_factory=FactoryFixtureBatchContextFactory(
            job_store=job_store,
            config=config,
        ),
        release_context_factory=FactoryFixtureReleaseContextFactory(
            job_store=job_store,
            config=config,
            store=store,
            private_store=private_store,
            plan_reviews=plan_reviews,
            output_root=output_root,
            requested_by=requested_by,
        ),
        witness_bridge=FactoryJobStoreWitnessBridge(
            job_store=job_store,
        ),
    )


def fixture_attachment_export_payload(
    artifact_id: str,
) -> bytes:
    return hashlib.sha256(
        artifact_id.encode(),
    ).digest()


class _FixtureDuplicateFingerprintFacade:
    async def fingerprint(
        self,
        request: AttachmentDuplicateFingerprintRequestV2,
    ) -> AttachmentDuplicateFingerprintResultV2:
        fingerprint = hashlib.sha256(request.content_sha256.encode()).hexdigest()
        value = AttachmentDuplicateFingerprintResultV2(
            fingerprint_result_id=("attachment-duplicate-fingerprint-result://pending"),
            fingerprint_request_ref=(attachment_duplicate_fingerprint_request_ref(request)),
            environment_artifact_ref=(request.environment_artifact_ref),
            output_ref=request.output_ref,
            exact_content_sha256=request.content_sha256,
            outcome=(AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED),
            similarity_fingerprint=fingerprint,
            token_count=12,
            shingle_count=10,
            failure_code=None,
            extractor_version=("attachment-duplicate-extractor/r7-01-v1"),
            normalization_version=request.normalization_version,
            policy_version=request.policy_version,
            result_sha256="0" * 64,
        )
        digest = attachment_duplicate_fingerprint_result_carried_sha256(value)
        return value.model_copy(
            update={
                "fingerprint_result_id": (f"attachment-duplicate-fingerprint-result://sha256/{digest}"),
                "result_sha256": digest,
            }
        )


class _FixtureCrossItemSafetyFacade:
    async def scan(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> AttachmentCrossItemSafetyScanResultV2:
        value = AttachmentCrossItemSafetyScanResultV2(
            scan_result_id=("attachment-cross-item-safety-scan-result://pending"),
            scan_request_ref=(attachment_cross_item_safety_scan_request_ref(request)),
            environment_artifact_ref=(request.environment_artifact_ref),
            output_ref=request.output_ref,
            output_sha256=request.content_sha256,
            status=AttachmentCrossItemSafetyScanStatusV2.PASSED,
            matched_fingerprint_ids=(),
            scanned_member_count=1,
            scan_complete=True,
            failure_code=None,
            extractor_version=("attachment-cross-item-safety-extractor/r7-02-v1"),
            normalization_version=request.normalization_version,
            policy_version=request.policy_version,
            result_sha256="0" * 64,
        )
        digest = attachment_cross_item_safety_scan_result_carried_sha256(value)
        return value.model_copy(
            update={
                "scan_result_id": (f"attachment-cross-item-safety-scan-result://sha256/{digest}"),
                "result_sha256": digest,
            }
        )


__all__ = [
    "FactoryDatasetReleaseRuntimeConfigV1",
    "FactoryFixtureBatchContextFactory",
    "FactoryFixtureReleaseComponents",
    "FactoryFixtureReleaseContextFactory",
    "build_fixture_release_components",
    "fixture_attachment_export_payload",
]
