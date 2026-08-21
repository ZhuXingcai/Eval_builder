from __future__ import annotations

import hashlib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import ValidationError, model_validator

from env_mock_agent.facade import (
    FacadeObjectRef,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
    lh_workspace_export_request_ref,
    validate_lh_workspace_export_result_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.production_attestation_v2 import (
    ProductionAttestationEvidenceClassV2,
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationStateV2,
)
from eval_factory.contracts.production_release_v2 import (
    ProductionAttestationAuthorityV2,
    ProductionLHExportReceiptV2,
    ProductionPublishedItemProjectionV2,
    ProductionRegistryEntryV2,
    ProductionReleaseEvidenceClassV2,
    ProductionReleaseItemManifestV2,
    ProductionReleaseManifestV2,
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseReasonCodeV2,
    ProductionReleaseResultV2,
    validate_production_attestation_authority_v2_identity,
    validate_production_release_policy_v2_identity,
)
from eval_factory.contracts.quality_v2 import (
    environment_spec_v2_ref,
    final_package_manifest_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionResultV2,
    evaluation_item_v2_carried_sha256,
    evaluation_item_v2_ref,
    release_decision_v2_carried_sha256,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
    validate_evaluation_item_v2_identity,
    validate_release_decision_v2_identity,
    validate_release_projection_result_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task_v2 import rubric_set_ref
from eval_factory.contracts.validation_v2 import (
    CandidatePackageInventoryEntryV2,
)
from eval_factory.dataset.export import (
    ReleaseBundleFacts,
    ReleasePublicationItemSource,
    render_lh_query_yaml,
    render_lh_rubrics_json,
    validate_lh_release_source,
)
from eval_factory.readiness.production_attestation import (
    ProductionAttestationEvaluator,
)
from eval_factory.readiness.production_attestation_store import (
    ProductionAttestationAcceptedRunStore,
    ProductionAttestationStoreCodec,
)

R8_08_ACCEPTED_REPOSITORY_RESULT_SHA256: Literal[
    "af5025277bf0404ea1253299538292a68406fc7bab08e2bdcb9990d94e611418"
] = "af5025277bf0404ea1253299538292a68406fc7bab08e2bdcb9990d94e611418"
R0_11_RELEASE_PROFILE_DECISION_SHA256: Literal[
    "6b4c0f86d76757c2ea7a6331022e2d6926cfb79f1424dfc664929ff6357d5ace"
] = "6b4c0f86d76757c2ea7a6331022e2d6926cfb79f1424dfc664929ff6357d5ace"
R7_09_NONPRODUCTION_GOLD_SHA256: Literal[
    "9a554ce55063b603051ce6948f0abb8aec5638af1773c1253afa49f5be0631f7"
] = "9a554ce55063b603051ce6948f0abb8aec5638af1773c1253afa49f5be0631f7"


class ProductionReleaseError(RuntimeError):
    pass


class ProductionReleasePolicyError(ProductionReleaseError):
    pass


class ProductionReleaseAuthorizationError(ProductionReleaseError):
    pass


class ProductionReleaseIntegrityError(ProductionReleaseError):
    pass


class ProductionReleaseConflictError(ProductionReleaseError):
    pass


class RepositoryPendingProductionReleaseEvidenceV1(ContractModelV2):
    schema_version: Literal["eval-factory/repository-pending-production-release-evidence/v1"] = (
        "eval-factory/repository-pending-production-release-evidence/v1"
    )
    evidence_class: Literal["REPOSITORY_PENDING_ONLY"]
    attestation_result_sha256: Literal["af5025277bf0404ea1253299538292a68406fc7bab08e2bdcb9990d94e611418"]
    attestation_outcome: Literal["ATTESTATION_PENDING"]
    attestation_evidence_class: Literal["REPOSITORY_PENDING_ONLY"]
    passed_gate_count: Literal[0]
    failed_gate_count: Literal[0]
    pending_gate_count: Literal[5]
    attestation_projection_present: Literal[False]
    frozen_attestation_present: Literal[False]
    satisfies_sc_010_through_sc_015: Literal[False]
    authorizes_production_release: Literal[False]
    release_profile_decision_sha256: Literal[
        "6b4c0f86d76757c2ea7a6331022e2d6926cfb79f1424dfc664929ff6357d5ace"
    ]
    lh_profile: Literal["LH"]
    lh_profile_version: Literal["v1"]
    lh_production_enabled: Literal[False]
    lh_production_gate: Literal["R8_PRODUCTION_READINESS_ATTESTATION"]
    generic_production_enabled: Literal[False]
    r7_nonproduction_gold_sha256: Literal["9a554ce55063b603051ce6948f0abb8aec5638af1773c1253afa49f5be0631f7"]
    r7_claim_scope: Literal["NONPRODUCTION_ONLY"]
    expected_outcome: Literal["PRODUCTION_RELEASE_BLOCKED"]
    expected_reason: Literal["ATTESTATION_PENDING"]

    @model_validator(mode="after")
    def validate_evidence(self) -> RepositoryPendingProductionReleaseEvidenceV1:
        if self.attestation_result_sha256 != R8_08_ACCEPTED_REPOSITORY_RESULT_SHA256:
            raise ValueError("repository attestation result is not accepted")
        if self.release_profile_decision_sha256 != R0_11_RELEASE_PROFILE_DECISION_SHA256:
            raise ValueError("repository release-profile decision is not accepted")
        if self.r7_nonproduction_gold_sha256 != R7_09_NONPRODUCTION_GOLD_SHA256:
            raise ValueError("repository R7 publication evidence is not accepted")
        return self


class _ReleaseProfileEntryV1(ContractModelV2):
    profile: Literal["LH", "GENERIC"]
    profile_version: Literal["v1"]
    decision: Literal["REQUIRED", "DEFERRED_APPROVAL_REQUIRED"]
    enabled_channels: tuple[str, ...]
    production_enabled: Literal[False]
    production_gate: Literal[
        "R8_PRODUCTION_READINESS_ATTESTATION",
        "NEW_APPROVED_RELEASE_PROFILE_DECISION",
    ]
    registry_isolation_required: Literal[True]
    reason: str


class _ReleaseProfileDecisionV1(ContractModelV2):
    schema_version: Literal["eval-factory-release-profile-decision/v1"]
    decision_id: Literal["release-profile-decision://r0-11/v1"]
    status: Literal["APPROVED"]
    approved_spec: str
    contract_manifest_sha256: str
    profiles: tuple[_ReleaseProfileEntryV1, _ReleaseProfileEntryV1]
    approved_at: str

    @model_validator(mode="after")
    def validate_profiles(self) -> _ReleaseProfileDecisionV1:
        by_name = {value.profile: value for value in self.profiles}
        if set(by_name) != {"LH", "GENERIC"}:
            raise ValueError("release-profile decision inventory is not exact")
        lh = by_name["LH"]
        generic = by_name["GENERIC"]
        if (
            lh.decision != "REQUIRED"
            or lh.enabled_channels != ("CANARY", "INTERNAL_REVIEW")
            or lh.production_gate != "R8_PRODUCTION_READINESS_ATTESTATION"
            or generic.decision != "DEFERRED_APPROVAL_REQUIRED"
            or generic.enabled_channels != ()
            or generic.production_gate != "NEW_APPROVED_RELEASE_PROFILE_DECISION"
        ):
            raise ValueError("release-profile decision semantics are not approved")
        return self


@dataclass(frozen=True, slots=True)
class ProductionReleasePredecessor:
    decision: ReleaseDecisionV2
    evaluation_item: EvaluationItemV2
    source_nonproduction_publication_ref: ObjectRef | None = None

    @classmethod
    def direct(
        cls,
        approved: ReleaseProjectionResultV2,
    ) -> ProductionReleasePredecessor:
        return cls(
            decision=approved.release_decision,
            evaluation_item=approved.evaluation_item,
        )


@dataclass(frozen=True, slots=True)
class ProductionReleaseAttestationAdmission:
    authority: ProductionAttestationAuthorityV2 | None
    reason_code: ProductionReleaseReasonCodeV2


@dataclass(frozen=True, slots=True)
class ProductionReleaseItemCompilation:
    source: ReleasePublicationItemSource
    predecessor: ProductionReleasePredecessor
    item_manifest: ProductionReleaseItemManifestV2
    query_yaml_bytes: bytes
    rubrics_json_bytes: bytes
    workspace_request: LHWorkspaceExportRequestV2


@dataclass(frozen=True, slots=True)
class ProductionReleaseManifestCompilation:
    release_manifest: ProductionReleaseManifestV2
    attestation: ProductionAttestationAuthorityV2
    items: tuple[ProductionReleaseItemCompilation, ...]


@dataclass(frozen=True, slots=True)
class ProductionReleaseCompilation:
    policy: ProductionReleasePolicyV2
    result: ProductionReleaseResultV2
    repository_evidence: RepositoryPendingProductionReleaseEvidenceV1 | None


class ProductionReleaseCompiler:
    def compile_repository_blocked(
        self,
        *,
        payload: bytes,
        policy: ProductionReleasePolicyV2,
    ) -> ProductionReleaseCompilation:
        current_policy = _validate_policy(policy)
        if (
            current_policy.evidence_class is not ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY
            or current_policy.repository_evidence_ref is None
            or current_policy.repository_attestation_result_ref is None
        ):
            raise ProductionReleasePolicyError("repository compilation requires repository-pending policy")
        digest = hashlib.sha256(payload).hexdigest()
        if current_policy.repository_evidence_ref.object_sha256 != digest:
            raise ProductionReleasePolicyError(
                "repository production-release evidence digest differs from policy"
            )
        try:
            evidence = RepositoryPendingProductionReleaseEvidenceV1.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleasePolicyError(
                "repository production-release evidence violates its contract"
            ) from exc
        if (
            current_policy.repository_attestation_result_ref.object_sha256
            != evidence.attestation_result_sha256
            or current_policy.release_profile_decision_sha256 != evidence.release_profile_decision_sha256
        ):
            raise ProductionReleasePolicyError("repository evidence differs from policy authority")
        result = ProductionReleaseResultV2.create_blocked(
            policy_ref=current_policy.to_ref(),
            evidence_class=(ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY),
            reason_codes=(ProductionReleaseReasonCodeV2.ATTESTATION_PENDING,),
            repository_evidence_ref=current_policy.repository_evidence_ref,
            audit=current_policy.audit,
        )
        if (
            result.outcome is not ProductionReleaseOutcomeV2.PRODUCTION_RELEASE_BLOCKED
            or result.satisfies_sc_015
        ):
            raise ProductionReleaseIntegrityError("repository production-release result overclaims authority")
        return ProductionReleaseCompilation(
            policy=current_policy,
            result=result,
            repository_evidence=evidence,
        )

    def compile_manifest(
        self,
        *,
        job_id: str,
        item_sources: tuple[ReleasePublicationItemSource, ...],
        predecessors: tuple[ProductionReleasePredecessor, ...],
        attestation: ProductionAttestationAuthorityV2,
        release_profile_payload: bytes,
        registry: str,
        policy: ProductionReleasePolicyV2,
        audit: ContractAudit,
    ) -> ProductionReleaseManifestCompilation:
        current_policy = _validate_policy(policy)
        if current_policy.evidence_class is not ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED:
            raise ProductionReleasePolicyError(
                "live production compilation requires production-verified policy"
            )
        try:
            current_attestation = ProductionAttestationAuthorityV2.model_validate_json(
                attestation.canonical_json()
            )
            validate_production_attestation_authority_v2_identity(current_attestation)
        except (ValidationError, ValueError) as exc:
            raise ProductionReleaseAuthorizationError("production attestation authority is stale") from exc
        if registry not in current_policy.allowed_production_registry_ids or (
            registry in current_policy.forbidden_nonproduction_registry_ids
        ):
            raise ProductionReleasePolicyError("production registry is not approved by policy")
        _validate_release_profile(
            release_profile_payload,
            current_policy,
        )
        if (
            not item_sources
            or len(item_sources) > current_policy.max_items
            or len(item_sources) != len(predecessors)
        ):
            raise ProductionReleasePolicyError("production release source coverage is invalid")
        prepared = tuple(
            sorted(
                (
                    (
                        validate_lh_release_source(source),
                        _validate_predecessor(source.approved_result, predecessor),
                    )
                    for source, predecessor in zip(
                        item_sources,
                        predecessors,
                        strict=True,
                    )
                ),
                key=lambda pair: pair[0].approved_result.release_subject.item_id,
            )
        )
        item_ids = tuple(source.approved_result.release_subject.item_id for source, _ in prepared)
        if len(set(item_ids)) != len(item_ids):
            raise ProductionReleasePolicyError("production release Item sources must be unique")
        if any(source.approved_result.release_subject.job_id != job_id for source, _ in prepared):
            raise ProductionReleasePolicyError("production release sources cross Jobs")
        item_compilations: list[ProductionReleaseItemCompilation] = []
        for source, predecessor in prepared:
            result = source.approved_result
            quality = source.item_quality
            package = quality.final_package_manifest
            environment = quality.environment_spec
            provenance = quality.provenance_manifest
            assert package is not None
            assert environment is not None
            assert provenance is not None
            query_bytes = render_lh_query_yaml(
                job_id=job_id,
                result=result,
                environment=environment,
            )
            rubric_bytes = render_lh_rubrics_json(source.rubric_set)
            members = tuple(_facade_member(value) for value in package.entries)
            if (
                len(members) > current_policy.max_workspace_members_per_item
                or sum(value.size_bytes for value in members) > current_policy.max_workspace_bytes_per_item
            ):
                raise ProductionReleasePolicyError("production release workspace budget exceeded")
            item_manifest = ProductionReleaseItemManifestV2.create(
                job_id=job_id,
                item_id=result.release_subject.item_id,
                approved_release_result_ref=release_projection_result_v2_ref(result),
                approved_projection_ref=result.item_projection_ref,
                release_subject_ref=result.release_subject_ref,
                predecessor_decision_ref=release_decision_v2_ref(predecessor.decision),
                predecessor_evaluation_item_ref=evaluation_item_v2_ref(predecessor.evaluation_item),
                source_nonproduction_publication_ref=(predecessor.source_nonproduction_publication_ref),
                query_spec_ref=result.query_spec_ref,
                rubric_set_ref=rubric_set_ref(source.rubric_set),
                environment_spec_ref=environment_spec_v2_ref(environment),
                provenance_manifest_ref=provenance_manifest_v2_ref(provenance),
                quality_report_ref=quality_report_v2_ref(quality.quality_report),
                final_package_manifest_ref=final_package_manifest_ref(package),
                package_sha256=package.package_sha256,
                query_yaml_sha256=hashlib.sha256(query_bytes).hexdigest(),
                rubrics_json_sha256=hashlib.sha256(rubric_bytes).hexdigest(),
                expected_workspace_members=members,
                attestation_authority_ref=current_attestation.to_ref(),
                registry=registry,
                policy_ref=current_policy.to_ref(),
                audit=audit,
            )
            workspace_request = LHWorkspaceExportRequestV2.create(
                item_id=result.release_subject.item_id,
                final_package_manifest_ref=_facade_ref(final_package_manifest_ref(package)),
                package_sha256=package.package_sha256,
                output_refs=tuple(_facade_ref(value) for value in package.output_refs),
                members=members,
                max_member_count=(current_policy.max_workspace_members_per_item),
                max_total_bytes=(current_policy.max_workspace_bytes_per_item),
            )
            item_compilations.append(
                ProductionReleaseItemCompilation(
                    source=source,
                    predecessor=predecessor,
                    item_manifest=item_manifest,
                    query_yaml_bytes=query_bytes,
                    rubrics_json_bytes=rubric_bytes,
                    workspace_request=workspace_request,
                )
            )
        release_manifest = ProductionReleaseManifestV2.create(
            job_id=job_id,
            item_manifests=tuple(value.item_manifest for value in item_compilations),
            base_contract_manifest_ref=current_policy.base_contract_manifest_ref,
            overlay_contract_manifest_ref=(current_policy.overlay_contract_manifest_ref),
            release_profile_decision_ref=(current_policy.release_profile_decision_ref),
            attestation_authority_ref=current_attestation.to_ref(),
            registry=registry,
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        if len(item_compilations) * 16 + 5 > current_policy.max_manifest_refs:
            raise ProductionReleasePolicyError("production release manifest ref budget exceeded")
        return ProductionReleaseManifestCompilation(
            release_manifest=release_manifest,
            attestation=current_attestation,
            items=tuple(item_compilations),
        )

    def compile_publication(
        self,
        *,
        manifest: ProductionReleaseManifestCompilation,
        workspace_results: tuple[LHWorkspaceExportResultV2, ...],
        bundle_facts: tuple[ReleaseBundleFacts, ...],
        policy: ProductionReleasePolicyV2,
        published_at: datetime,
        audit: ContractAudit,
    ) -> ProductionReleaseCompilation:
        current_policy = _validate_policy(policy)
        release_manifest = manifest.release_manifest
        if (
            release_manifest.policy_ref != current_policy.to_ref()
            or release_manifest.attestation_authority_ref != manifest.attestation.to_ref()
            or len(workspace_results) != len(manifest.items)
            or len(bundle_facts) != len(manifest.items)
        ):
            raise ProductionReleasePolicyError("production publication coverage or policy is stale")
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ProductionReleasePolicyError("production publication time must be timezone-aware")
        try:
            for item, workspace in zip(
                manifest.items,
                workspace_results,
                strict=True,
            ):
                validate_lh_workspace_export_result_identity(workspace)
                if (
                    workspace.request_ref != lh_workspace_export_request_ref(item.workspace_request)
                    or workspace.observed_members != item.workspace_request.members
                    or workspace.copied_output_refs != item.workspace_request.output_refs
                ):
                    raise ValueError("workspace export differs from request")
        except ValueError as exc:
            raise ProductionReleasePolicyError("production workspace export is stale") from exc
        receipts = tuple(
            ProductionLHExportReceiptV2.create(
                release_manifest_ref=release_manifest.to_ref(),
                item_manifest_ref=item.item_manifest.to_ref(),
                workspace_export_result=workspace,
                query_yaml_sha256=item.item_manifest.query_yaml_sha256,
                rubrics_json_sha256=item.item_manifest.rubrics_json_sha256,
                bundle_sha256=facts.bundle_sha256,
                bundle_file_count=facts.bundle_file_count,
                bundle_total_bytes=facts.bundle_total_bytes,
                policy_ref=current_policy.to_ref(),
                audit=audit,
            )
            for item, workspace, facts in zip(
                manifest.items,
                workspace_results,
                bundle_facts,
                strict=True,
            )
        )
        authority_ref = release_manifest.attestation_authority_ref
        decisions = tuple(
            _production_decision(
                item,
                registry=release_manifest.registry,
                frozen_attestation_ref=(manifest.attestation.frozen_attestation_ref),
                published_at=published_at,
                audit=audit,
            )
            for item in manifest.items
        )
        published_items = tuple(
            _production_item(item, decision, audit=audit)
            for item, decision in zip(manifest.items, decisions, strict=True)
        )
        registry_entry = ProductionRegistryEntryV2.create(
            release_manifest_ref=release_manifest.to_ref(),
            job_id=release_manifest.job_id,
            registry=release_manifest.registry,
            item_ids=release_manifest.item_ids,
            item_manifest_refs=release_manifest.item_manifest_refs,
            export_receipt_refs=tuple(value.to_ref() for value in receipts),
            published_release_decision_refs=tuple(release_decision_v2_ref(value) for value in decisions),
            published_evaluation_item_refs=tuple(evaluation_item_v2_ref(value) for value in published_items),
            bundle_sha256s=tuple(value.bundle_sha256 for value in bundle_facts),
            attestation_authority_ref=authority_ref,
            published_at=published_at,
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        projections = tuple(
            ProductionPublishedItemProjectionV2.create(
                job_id=release_manifest.job_id,
                item_id=item.item_manifest.item_id,
                chain_id=decision.chain_id,
                approved_result_ref=(item.item_manifest.approved_release_result_ref),
                approved_projection_ref=(item.item_manifest.approved_projection_ref),
                source_nonproduction_publication_ref=(item.predecessor.source_nonproduction_publication_ref),
                release_manifest_ref=release_manifest.to_ref(),
                export_receipt_ref=receipt.to_ref(),
                registry_entry_ref=registry_entry.to_ref(),
                publish_decision_ref=release_decision_v2_ref(decision),
                published_evaluation_item_ref=evaluation_item_v2_ref(published_item),
                attestation_authority_ref=authority_ref,
                audit=audit,
            )
            for item, receipt, decision, published_item in zip(
                manifest.items,
                receipts,
                decisions,
                published_items,
                strict=True,
            )
        )
        result = ProductionReleaseResultV2.create_published(
            policy_ref=current_policy.to_ref(),
            attestation_authority=manifest.attestation,
            release_manifest=release_manifest,
            export_receipts=receipts,
            published_decisions=decisions,
            published_items=published_items,
            item_projections=projections,
            registry_entry=registry_entry,
            approved_source_result_refs=tuple(
                value.item_manifest.approved_release_result_ref for value in manifest.items
            ),
            audit=audit,
        )
        return ProductionReleaseCompilation(
            policy=current_policy,
            result=result,
            repository_evidence=None,
        )


class ProductionReleaseAttestationGate:
    def __init__(
        self,
        accepted_store: ProductionAttestationAcceptedRunStore,
    ) -> None:
        self.accepted_store = accepted_store

    @contextmanager
    def hold_current(
        self,
        *,
        acceptance_key: str,
        previous_acceptance_key: str | None,
        policy: ProductionReleasePolicyV2,
        observed_at: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
        audit: ContractAudit,
    ) -> Generator[ProductionReleaseAttestationAdmission, None, None]:
        if (observed_at is None) == (clock is None):
            raise ProductionReleasePolicyError("attestation gate requires exactly one trusted clock source")
        current_policy = _validate_policy(policy)
        if current_policy.evidence_class is not ProductionReleaseEvidenceClassV2.PRODUCTION_VERIFIED:
            yield ProductionReleaseAttestationAdmission(
                authority=None,
                reason_code=ProductionReleaseReasonCodeV2.MECHANISM_ONLY,
            )
            return
        result = self.accepted_store.get_accepted_result(
            acceptance_key,
            previous_acceptance_key=previous_acceptance_key,
        )
        if (
            result.evidence_class is not ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED
            or result.outcome is not ProductionReadinessAttestationOutcomeV2.ISSUED
            or result.projection is None
        ):
            reason = (
                ProductionReleaseReasonCodeV2.MECHANISM_ONLY
                if result.evidence_class is ProductionAttestationEvidenceClassV2.MECHANISM_VALIDATION_ONLY
                else ProductionReleaseReasonCodeV2.ATTESTATION_PENDING
                if result.outcome is ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING
                else ProductionReleaseReasonCodeV2.ATTESTATION_NOT_ISSUED
            )
            yield ProductionReleaseAttestationAdmission(
                authority=None,
                reason_code=reason,
            )
            return
        attestation_policy = self.accepted_store.report_store.get(
            result.policy_ref,
            ProductionAttestationStoreCodec.POLICY,
            ProductionReadinessAttestationPolicyV2,
        )
        if (
            attestation_policy.policy_version != current_policy.required_attestation_policy_version
            or result.version_set.system_version != current_policy.expected_system_version
            or result.version_set.base_contract_manifest_ref != current_policy.base_contract_manifest_ref
            or result.version_set.overlay_contract_manifest_ref
            != current_policy.overlay_contract_manifest_ref
            or result.version_set.schema_manifest_refs != current_policy.required_schema_manifest_refs
            or result.version_set.release_profile_decision_ref != current_policy.release_profile_decision_ref
            or result.version_set.policy_refs
            != _sorted_object_refs(
                (
                    *current_policy.baseline_attested_policy_refs,
                    current_policy.to_ref(),
                )
            )
        ):
            raise ProductionReleasePolicyError("issued attestation does not cover current production policy")
        accepted = self.accepted_store.material_store.get_acceptance(acceptance_key)
        projection = result.projection
        if (
            accepted.frozen_attestation_ref != projection.frozen_attestation_ref
            or accepted.current_projection_ref != projection.to_ref()
        ):
            raise ProductionReleaseIntegrityError("accepted attestation private closure is incomplete")
        with self.accepted_store.report_store.hold_current_projection(
            result.attestation_series_id
        ) as current:
            current_time = clock() if clock is not None else observed_at
            assert current_time is not None
            if current_time.tzinfo is None or current_time.utcoffset() is None:
                raise ProductionReleasePolicyError("production release clock must be timezone-aware")
            if current is None or current.to_ref() != projection.to_ref():
                reason = (
                    ProductionReleaseReasonCodeV2.ATTESTATION_EXPIRED
                    if current is not None and current.state is ProductionReadinessAttestationStateV2.EXPIRED
                    else ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT
                )
                yield ProductionReleaseAttestationAdmission(
                    authority=None,
                    reason_code=reason,
                )
                return
            if (
                current.state is not ProductionReadinessAttestationStateV2.ACTIVE
                or current_time < current.valid_from
                or current_time >= current.valid_until
            ):
                yield ProductionReleaseAttestationAdmission(
                    authority=None,
                    reason_code=(
                        ProductionReleaseReasonCodeV2.ATTESTATION_EXPIRED
                        if current_time >= current.valid_until
                        else ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT
                    ),
                )
                return
            invalidation, observed = ProductionAttestationEvaluator().evaluate_currentness(
                projection=current,
                prior_version_set=result.version_set,
                observed_version_set=result.version_set,
                observed_issuer_registry_ref=current.issuer_registry_ref,
                observed_at=current_time,
                audit=audit,
            )
            if invalidation is not None or observed.to_ref() != current.to_ref():
                yield ProductionReleaseAttestationAdmission(
                    authority=None,
                    reason_code=ProductionReleaseReasonCodeV2.ATTESTATION_NOT_CURRENT,
                )
                return
            authority = ProductionAttestationAuthorityV2.create(
                attestation_result_ref=result.to_ref(),
                current_projection_ref=current.to_ref(),
                frozen_attestation_ref=current.frozen_attestation_ref,
                version_set_ref=current.version_set_ref,
                prerequisite_ref=current.prerequisite_ref,
                attestation_series_id=current.attestation_series_id,
                attestation_version=current.attestation_version,
                valid_from=current.valid_from,
                valid_until=current.valid_until,
                verified_at=current_time,
                audit=audit,
            )
            yield ProductionReleaseAttestationAdmission(
                authority=authority,
                reason_code=ProductionReleaseReasonCodeV2.NONE,
            )


def _validate_release_profile(
    payload: bytes,
    policy: ProductionReleasePolicyV2,
) -> _ReleaseProfileDecisionV1:
    if hashlib.sha256(payload).hexdigest() != policy.release_profile_decision_sha256:
        raise ProductionReleasePolicyError("release-profile decision digest differs from policy")
    try:
        value = _ReleaseProfileDecisionV1.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise ProductionReleasePolicyError("release-profile decision violates approved semantics") from exc
    return value


def _validate_predecessor(
    approved: ReleaseProjectionResultV2,
    value: ProductionReleasePredecessor,
) -> ProductionReleasePredecessor:
    try:
        validate_release_projection_result_v2_identity(approved)
        decision = ReleaseDecisionV2.model_validate_json(value.decision.model_dump_json())
        item = EvaluationItemV2.model_validate_json(value.evaluation_item.model_dump_json())
        validate_release_decision_v2_identity(decision)
        validate_evaluation_item_v2_identity(item)
    except (ValidationError, ValueError) as exc:
        raise ProductionReleasePolicyError("production release predecessor is stale") from exc
    if value.source_nonproduction_publication_ref is None:
        if decision != approved.release_decision or item != approved.evaluation_item:
            raise ProductionReleasePolicyError(
                "direct production predecessor differs from approved authority"
            )
    else:
        if (
            value.source_nonproduction_publication_ref.object_type != "release-publication-result"
            or value.source_nonproduction_publication_ref.object_version != "v2"
            or decision.action is not ReleaseActionV2.PUBLISH
            or decision.state is not ReleaseStateV2.RELEASED
            or decision.channel is ReleaseChannel.PRODUCTION
            or decision.production_attestation_ref is not None
            or decision.previous_decision_ref != approved.release_decision_ref
            or decision.chain_id != approved.release_decision.chain_id
            or decision.item_id != approved.release_decision.item_id
            or decision.components != approved.release_decision.components
            or decision.release_subject_sha256 != approved.release_decision.release_subject_sha256
            or decision.package_manifest_ref != approved.release_decision.package_manifest_ref
            or decision.package_sha256 != approved.release_decision.package_sha256
            or item.release_decision_ref != release_decision_v2_ref(decision)
            or item.components != approved.evaluation_item.components
            or item.source_trace_refs != approved.evaluation_item.source_trace_refs
            or item.label_decision_refs != approved.evaluation_item.label_decision_refs
            or item.task_draft_ref != approved.evaluation_item.task_draft_ref
            or item.attachment_reconstruction_result_ref
            != approved.evaluation_item.attachment_reconstruction_result_ref
        ):
            raise ProductionReleasePolicyError("non-production predecessor differs from approved authority")
    return ProductionReleasePredecessor(
        decision=decision,
        evaluation_item=item,
        source_nonproduction_publication_ref=(value.source_nonproduction_publication_ref),
    )


def _production_decision(
    item: ProductionReleaseItemCompilation,
    *,
    registry: str,
    frozen_attestation_ref: ObjectRef,
    published_at: datetime,
    audit: ContractAudit,
) -> ReleaseDecisionV2:
    prior = item.predecessor.decision
    try:
        revision = int(prior.item_version) + 1
    except ValueError as exc:
        raise ProductionReleasePolicyError("release predecessor Item version is invalid") from exc
    refs = (
        item.item_manifest.approved_release_result_ref,
        item.item_manifest.release_subject_ref,
        item.item_manifest.predecessor_decision_ref,
        item.item_manifest.predecessor_evaluation_item_ref,
        item.item_manifest.attestation_authority_ref,
        *(
            (item.predecessor.source_nonproduction_publication_ref,)
            if item.predecessor.source_nonproduction_publication_ref
            else ()
        ),
    )
    pending = prior.model_copy(
        update={
            "release_decision_id": "release-decision://pending",
            "previous_decision_ref": release_decision_v2_ref(prior),
            "item_version": str(revision),
            "channel": ReleaseChannel.PRODUCTION,
            "registry": registry,
            "production_attestation_ref": frozen_attestation_ref,
            "action": ReleaseActionV2.PUBLISH,
            "state": ReleaseStateV2.RELEASED,
            "idempotency_key": (f"production-publish-{prior.release_subject_sha256[:24]}"),
            "actor": audit.created_by,
            "decided_at": published_at,
            "decision_sha256": "0" * 64,
            "audit": _audit_with_refs(audit, refs),
        }
    )
    try:
        pending = ReleaseDecisionV2.model_validate_json(pending.model_dump_json())
        digest = release_decision_v2_carried_sha256(pending)
        result = ReleaseDecisionV2.model_validate_json(
            pending.model_copy(
                update={
                    "release_decision_id": (f"release-decision://sha256/{digest}"),
                    "decision_sha256": digest,
                }
            ).model_dump_json()
        )
        validate_release_decision_v2_identity(result)
    except (ValidationError, ValueError) as exc:
        raise ProductionReleasePolicyError("production ReleaseDecision is invalid") from exc
    return result


def _production_item(
    source: ProductionReleaseItemCompilation,
    decision: ReleaseDecisionV2,
    *,
    audit: ContractAudit,
) -> EvaluationItemV2:
    prior = source.predecessor.evaluation_item
    decision_ref = release_decision_v2_ref(decision)
    refs = (
        source.item_manifest.predecessor_evaluation_item_ref,
        decision_ref,
    )
    pending = prior.model_copy(
        update={
            "evaluation_item_id": "evaluation-item://pending",
            "item_version": decision.item_version,
            "release_decision_ref": decision_ref,
            "item_sha256": "0" * 64,
            "audit": _audit_with_refs(audit, refs),
        }
    )
    try:
        pending = EvaluationItemV2.model_validate_json(pending.model_dump_json())
        digest = evaluation_item_v2_carried_sha256(pending)
        result = EvaluationItemV2.model_validate_json(
            pending.model_copy(
                update={
                    "evaluation_item_id": (f"evaluation-item://sha256/{digest}"),
                    "item_sha256": digest,
                }
            ).model_dump_json()
        )
        validate_evaluation_item_v2_identity(result)
    except (ValidationError, ValueError) as exc:
        raise ProductionReleasePolicyError("production EvaluationItem is invalid") from exc
    return result


def _facade_member(
    value: CandidatePackageInventoryEntryV2,
) -> LHWorkspaceExportMemberV2:
    member = value.inventory_member
    return LHWorkspaceExportMemberV2.create(
        normalized_path=member.normalized_path,
        member_type=LHWorkspaceExportMemberTypeV2(member.member_type),
        media_type=member.media_type,
        size_bytes=member.size_bytes,
        content_sha256=member.content_sha256,
        container_ref=(_facade_ref(value.container_ref) if value.container_ref is not None else None),
        output_ref=_facade_ref(value.output_ref),
    )


def _facade_ref(value: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _audit_with_refs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(
        update={
            "input_refs": tuple(
                sorted(
                    set(refs),
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.object_version,
                        value.object_sha256,
                    ),
                )
            )
        }
    )


def _sorted_object_refs(
    values: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )


def _validate_policy(
    value: ProductionReleasePolicyV2,
) -> ProductionReleasePolicyV2:
    try:
        current = ProductionReleasePolicyV2.model_validate_json(value.model_dump_json())
        validate_production_release_policy_v2_identity(current)
    except (ValidationError, ValueError) as exc:
        raise ProductionReleasePolicyError("production-release policy identity is stale") from exc
    return current


__all__ = [
    "R0_11_RELEASE_PROFILE_DECISION_SHA256",
    "R7_09_NONPRODUCTION_GOLD_SHA256",
    "R8_08_ACCEPTED_REPOSITORY_RESULT_SHA256",
    "ProductionReleaseAttestationAdmission",
    "ProductionReleaseAttestationGate",
    "ProductionReleaseAuthorizationError",
    "ProductionReleaseCompilation",
    "ProductionReleaseCompiler",
    "ProductionReleaseConflictError",
    "ProductionReleaseError",
    "ProductionReleaseIntegrityError",
    "ProductionReleaseItemCompilation",
    "ProductionReleaseManifestCompilation",
    "ProductionReleasePolicyError",
    "ProductionReleasePredecessor",
    "RepositoryPendingProductionReleaseEvidenceV1",
]
