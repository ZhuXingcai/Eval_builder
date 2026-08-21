from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

import yaml
from pydantic import ValidationError

from env_mock_agent.facade import (
    FacadeObjectRef,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
    lh_workspace_export_request_ref,
    validate_lh_workspace_export_result_identity,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    QueryPackagingChoice,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_record_carried_sha256,
    validate_user_decision_commit_result_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPhaseV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_carried_sha256,
    evaluation_item_v2_ref,
    release_decision_v2_carried_sha256,
    release_decision_v2_ref,
    release_projection_result_v2_ref,
    validate_release_projection_result_v2_identity,
)
from eval_factory.contracts.release_publication_v2 import (
    LHExportReceiptV2,
    LHReleaseItemManifestV2,
    NonProductionRegistryEntryV2,
    NonProductionReleaseManifestV2,
    PublishedItemProjectionV2,
    ReleasePublicationPolicyV2,
    ReleasePublicationResultV2,
)
from eval_factory.contracts.release_v2 import (
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task_v2 import (
    RubricSetV2,
    rubric_criterion_carried_sha256,
    rubric_reachability_carried_sha256,
    rubric_set_carried_sha256,
    rubric_set_ref,
)
from eval_factory.contracts.validation_v2 import (
    CandidatePackageInventoryEntryV2,
)


class ReleasePublicationPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReleasePublicationItemSource:
    approved_result: ReleaseProjectionResultV2
    rubric_set: RubricSetV2
    item_quality: ItemQualityCompilationResultV2
    query_packaging_decision: UserDecisionCommitResultV2 | None = None


@dataclass(frozen=True, slots=True)
class ReleasePublicationItemCompilation:
    source: ReleasePublicationItemSource
    item_manifest: LHReleaseItemManifestV2
    query_yaml_bytes: bytes
    rubrics_json_bytes: bytes
    workspace_request: LHWorkspaceExportRequestV2


@dataclass(frozen=True, slots=True)
class ReleasePublicationManifestCompilation:
    release_manifest: NonProductionReleaseManifestV2
    items: tuple[ReleasePublicationItemCompilation, ...]
    rejected_item_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseBundleFacts:
    bundle_sha256: str
    bundle_file_count: int
    bundle_total_bytes: int


@dataclass(frozen=True, slots=True)
class ReleasePublicationCompilation:
    result: ReleasePublicationResultV2


class ReleasePublicationCompiler:
    def compile_manifest(
        self,
        *,
        job_id: str,
        item_sources: tuple[ReleasePublicationItemSource, ...],
        rejected_item_ids: tuple[str, ...],
        policy: ReleasePublicationPolicyV2,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        release_profile_decision_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ReleasePublicationManifestCompilation:
        current_policy = _validate_policy(policy)
        if not item_sources:
            raise ReleasePublicationPolicyError("publication requires at least one approved Item")
        if len(item_sources) > current_policy.max_items:
            raise ReleasePublicationPolicyError("publication Item budget exceeded")
        sources = tuple(
            sorted(
                (_prepare_source(value, current_policy) for value in item_sources),
                key=lambda value: value.approved_result.release_subject.item_id,
            )
        )
        item_ids = tuple(source.approved_result.release_subject.item_id for source in sources)
        if len(item_ids) != len(set(item_ids)):
            raise ReleasePublicationPolicyError("publication Item sources must be unique")
        rejected = tuple(sorted(set(rejected_item_ids)))
        if len(rejected) != len(rejected_item_ids):
            raise ReleasePublicationPolicyError("rejected Item partition contains duplicates")
        if set(item_ids) & set(rejected):
            raise ReleasePublicationPolicyError("approved and rejected Item partitions overlap")
        if any(source.approved_result.release_subject.job_id != job_id for source in sources):
            raise ReleasePublicationPolicyError("publication sources cross Jobs")
        channels = {source.approved_result.release_decision.channel for source in sources}
        registries = {source.approved_result.release_decision.registry for source in sources}
        if len(channels) != 1 or len(registries) != 1:
            raise ReleasePublicationPolicyError("publication sources mix channel or registry")
        channel = next(iter(channels))
        registry = next(iter(registries))
        _validate_target(current_policy, channel, registry)

        item_compilations: list[ReleasePublicationItemCompilation] = []
        for source in sources:
            result = source.approved_result
            quality = source.item_quality
            package = quality.final_package_manifest
            provenance = quality.provenance_manifest
            environment = quality.environment_spec
            assert package is not None
            assert provenance is not None
            assert environment is not None
            query_bytes = render_lh_query_yaml(
                job_id=job_id,
                result=result,
                environment=environment,
            )
            rubric_bytes = render_lh_rubrics_json(source.rubric_set)
            members = tuple(_facade_member(entry) for entry in package.entries)
            if (
                len(members) > current_policy.max_workspace_members_per_item
                or sum(member.size_bytes for member in members) > current_policy.max_workspace_bytes_per_item
            ):
                raise ReleasePublicationPolicyError("publication workspace budget exceeded")
            item_manifest = LHReleaseItemManifestV2.create(
                job_id=job_id,
                item_id=result.release_subject.item_id,
                approved_release_result_ref=(release_projection_result_v2_ref(result)),
                approved_evaluation_item_ref=result.evaluation_item_ref,
                approved_release_decision_ref=result.release_decision_ref,
                release_subject_ref=result.release_subject_ref,
                query_spec_ref=result.query_spec_ref,
                rubric_set_ref=rubric_set_ref(source.rubric_set),
                environment_spec_ref=environment_spec_v2_ref(environment),
                provenance_manifest_ref=(provenance_manifest_v2_ref(provenance)),
                quality_report_ref=quality_report_v2_ref(quality.quality_report),
                final_package_manifest_ref=(final_package_manifest_ref(package)),
                package_sha256=package.package_sha256,
                query_yaml_sha256=_sha256(query_bytes),
                rubrics_json_sha256=_sha256(rubric_bytes),
                expected_workspace_members=members,
                channel=channel,
                registry=registry,
                audit=audit,
            )
            request = LHWorkspaceExportRequestV2.create(
                item_id=result.release_subject.item_id,
                final_package_manifest_ref=_facade_ref(final_package_manifest_ref(package)),
                package_sha256=package.package_sha256,
                output_refs=tuple(_facade_ref(ref) for ref in package.output_refs),
                members=members,
                max_member_count=(current_policy.max_workspace_members_per_item),
                max_total_bytes=(current_policy.max_workspace_bytes_per_item),
            )
            item_compilations.append(
                ReleasePublicationItemCompilation(
                    source=source,
                    item_manifest=item_manifest,
                    query_yaml_bytes=query_bytes,
                    rubrics_json_bytes=rubric_bytes,
                    workspace_request=request,
                )
            )
        release_manifest = NonProductionReleaseManifestV2.create(
            job_id=job_id,
            approved_item_ids=item_ids,
            rejected_item_ids=rejected,
            item_manifests=tuple(value.item_manifest for value in item_compilations),
            channel=channel,
            registry=registry,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            release_profile_decision_ref=release_profile_decision_ref,
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        manifest_ref_count = len(item_compilations) * 11 + 4
        if manifest_ref_count > current_policy.max_manifest_refs:
            raise ReleasePublicationPolicyError("publication manifest ref budget exceeded")
        return ReleasePublicationManifestCompilation(
            release_manifest=release_manifest,
            items=tuple(item_compilations),
            rejected_item_ids=rejected,
        )

    def compile_publication(
        self,
        *,
        manifest: ReleasePublicationManifestCompilation,
        workspace_results: tuple[LHWorkspaceExportResultV2, ...],
        bundle_facts: tuple[ReleaseBundleFacts, ...],
        policy: ReleasePublicationPolicyV2,
        published_at: datetime,
        audit: ContractAudit,
    ) -> ReleasePublicationCompilation:
        current_policy = _validate_policy(policy)
        if (
            manifest.release_manifest.policy_ref != current_policy.to_ref()
            or tuple(item.item_manifest for item in manifest.items)
            != manifest.release_manifest.item_manifests
        ):
            raise ReleasePublicationPolicyError("publication manifest policy is stale")
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ReleasePublicationPolicyError("publication timestamp must be timezone-aware")
        if len(workspace_results) != len(manifest.items) or len(bundle_facts) != len(manifest.items):
            raise ReleasePublicationPolicyError("publication export coverage is incomplete")
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
            raise ReleasePublicationPolicyError("publication workspace export is stale") from exc
        receipts = tuple(
            LHExportReceiptV2.create(
                release_manifest_ref=manifest.release_manifest.to_ref(),
                item_manifest_ref=item.item_manifest.to_ref(),
                workspace_export_result=workspace,
                query_yaml_sha256=(item.item_manifest.query_yaml_sha256),
                rubrics_json_sha256=(item.item_manifest.rubrics_json_sha256),
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
        decisions = tuple(
            _publish_decision(
                item.source.approved_result,
                decided_at=published_at,
                audit=audit,
            )
            for item in manifest.items
        )
        published_items = tuple(
            _published_item(
                item.source.approved_result,
                decision,
                audit=audit,
            )
            for item, decision in zip(
                manifest.items,
                decisions,
                strict=True,
            )
        )
        registry_entry = NonProductionRegistryEntryV2.create(
            release_manifest_ref=manifest.release_manifest.to_ref(),
            job_id=manifest.release_manifest.job_id,
            channel=manifest.release_manifest.channel,
            registry=manifest.release_manifest.registry,
            item_ids=manifest.release_manifest.approved_item_ids,
            item_manifest_refs=(manifest.release_manifest.item_manifest_refs),
            export_receipt_refs=tuple(value.to_ref() for value in receipts),
            published_release_decision_refs=tuple(release_decision_v2_ref(value) for value in decisions),
            published_evaluation_item_refs=tuple(evaluation_item_v2_ref(value) for value in published_items),
            bundle_sha256s=tuple(value.bundle_sha256 for value in bundle_facts),
            published_at=published_at.isoformat(),
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        projections = tuple(
            PublishedItemProjectionV2.create(
                job_id=manifest.release_manifest.job_id,
                item_id=item.source.approved_result.release_subject.item_id,
                chain_id=decision.chain_id,
                approved_result_ref=release_projection_result_v2_ref(item.source.approved_result),
                approved_projection_ref=(item.source.approved_result.item_projection_ref),
                release_manifest_ref=manifest.release_manifest.to_ref(),
                export_receipt_ref=receipt.to_ref(),
                registry_entry_ref=registry_entry.to_ref(),
                publish_decision_ref=release_decision_v2_ref(decision),
                published_evaluation_item_ref=(evaluation_item_v2_ref(published_item)),
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
        result = ReleasePublicationResultV2.create(
            release_manifest=manifest.release_manifest,
            export_receipts=receipts,
            published_decisions=decisions,
            published_items=published_items,
            item_projections=projections,
            registry_entry=registry_entry,
            approved_source_result_refs=tuple(
                release_projection_result_v2_ref(item.source.approved_result) for item in manifest.items
            ),
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        return ReleasePublicationCompilation(result=result)


def render_lh_query_yaml(
    *,
    job_id: str,
    result: ReleaseProjectionResultV2,
    environment: object,
) -> bytes:
    artifacts = getattr(environment, "artifacts", ())
    payload = {
        "task_id": result.release_subject.item_id,
        "request_id": (f"{job_id}:{result.release_subject.item_id}"),
        "task_name": (f"Eval Factory {result.release_subject.item_id}"),
        "prompt": result.query_spec.prompt,
        "dependencies": [
            {
                "dependency_id": artifact.attachment_dependency_id,
                "path": artifact.logical_path,
                "media_type": artifact.media_type,
                "sha256": artifact.content_sha256,
                "size_bytes": artifact.size_bytes,
            }
            for artifact in artifacts
        ],
        "profile": "LH",
        "profile_version": "v1",
    }
    return yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
    ).encode()


def render_lh_rubrics_json(value: RubricSetV2) -> bytes:
    payload = {
        "schema_version": "eval-factory/lh-rubrics/v1",
        "rubric_set_id": value.rubric_set_id,
        "rubric_version": value.rubric_version,
        "criteria": [
            {
                "criterion_id": item.criterion_id,
                "judged_object_id": (item.judged_object.judged_object_id),
                "judged_object_kind": item.judged_object.kind.value,
                "description": item.description,
                "weight": item.weight,
                "visibility": item.visibility.value,
                "evaluator_binding": item.evaluator_binding,
            }
            for item in value.criteria
        ],
        "total_weight": value.total_weight,
        "policy_version": value.policy_version,
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def validate_lh_release_source(
    value: ReleasePublicationItemSource,
) -> ReleasePublicationItemSource:
    try:
        result = ReleaseProjectionResultV2.model_validate(value.approved_result.model_dump(mode="python"))
        validate_release_projection_result_v2_identity(result)
        rubric = RubricSetV2.model_validate(value.rubric_set.model_dump(mode="python"))
        _validate_rubric_set(rubric)
        quality = ItemQualityCompilationResultV2.model_validate(value.item_quality.model_dump(mode="python"))
    except (ValidationError, ValueError) as exc:
        raise ReleasePublicationPolicyError("publication source is stale or malformed") from exc
    if (
        result.phase is not ReleaseProjectionPhaseV2.TERMINAL
        or result.release_decision.action is not ReleaseActionV2.APPROVE
        or result.release_decision.state is not ReleaseStateV2.APPROVED
        or result.item_projection.item_status.value != "APPROVED"
        or result.release_decision.production_attestation_ref is not None
        or result.release_decision.export_profile != "LH"
        or result.release_decision.export_profile_version != "v1"
    ):
        raise ReleasePublicationPolicyError("publication requires current non-production LH approval")
    package = quality.final_package_manifest
    provenance = quality.provenance_manifest
    environment = quality.environment_spec
    if (
        not quality.quality_report.approvable
        or package is None
        or provenance is None
        or environment is None
        or quality.package_sha256 is None
        or rubric_set_ref(rubric) != result.release_subject.components.rubric_set_ref
        or final_package_manifest_ref(package) != result.release_subject.package_manifest_ref
        or package.package_sha256 != result.release_subject.package_sha256
        or provenance_manifest_v2_ref(provenance) != result.release_subject.components.provenance_manifest_ref
        or environment_spec_v2_ref(environment) != result.release_subject.components.environment_spec_ref
        or quality_report_v2_ref(quality.quality_report)
        != result.release_subject.components.quality_report_ref
    ):
        raise ReleasePublicationPolicyError("publication package/component authority is stale")
    _validate_query_packaging(result, value.query_packaging_decision)
    return ReleasePublicationItemSource(
        approved_result=result,
        rubric_set=rubric,
        item_quality=quality,
        query_packaging_decision=value.query_packaging_decision,
    )


def _prepare_source(
    value: ReleasePublicationItemSource,
    policy: ReleasePublicationPolicyV2,
) -> ReleasePublicationItemSource:
    current = validate_lh_release_source(value)
    _validate_target(
        policy,
        current.approved_result.release_decision.channel,
        current.approved_result.release_decision.registry,
    )
    return current


def _validate_query_packaging(
    result: ReleaseProjectionResultV2,
    decision: UserDecisionCommitResultV2 | None,
) -> None:
    required = result.release_subject.required_checkpoints
    needs_environment = ApprovalCheckpoint.ENVIRONMENT_STRATEGY in required
    if not needs_environment:
        if decision is not None:
            raise ReleasePublicationPolicyError("disabled environment checkpoint cannot add a query decision")
        return
    if decision is not None:
        try:
            current = UserDecisionCommitResultV2.model_validate(decision.model_dump(mode="python"))
            validate_user_decision_commit_result_v2_identity(current)
        except (ValidationError, ValueError) as exc:
            raise ReleasePublicationPolicyError("LH v1 query decision authority is stale") from exc
        decision = current
    if (
        decision is None
        or decision.outcome is not UserDecisionCommitOutcomeV2.ACCEPTED
        or decision.decision_record.checkpoint is not ApprovalCheckpoint.ENVIRONMENT_STRATEGY
        or decision.decision_record.query_packaging is not QueryPackagingChoice.INCLUDE_QUERY_YAML
        or decision.job_id != result.release_subject.job_id
        or decision.decision_record.approval_policy_ref != result.release_subject.user_approval_policy_ref
        or decision.decision_record_ref not in result.evaluation_item.user_decision_record_refs
        or decision.decision_record.record_sha256
        != user_decision_record_carried_sha256(decision.decision_record)
    ):
        raise ReleasePublicationPolicyError("LH v1 requires current INCLUDE_QUERY_YAML authority")


def _validate_policy(
    value: ReleasePublicationPolicyV2,
) -> ReleasePublicationPolicyV2:
    try:
        current = ReleasePublicationPolicyV2.model_validate(value.model_dump(mode="python"))
        current.to_ref()
    except (ValidationError, ValueError) as exc:
        raise ReleasePublicationPolicyError("publication policy is stale") from exc
    return current


def _validate_target(
    policy: ReleasePublicationPolicyV2,
    channel: ReleaseChannel,
    registry: str,
) -> None:
    allowed = {
        ReleaseChannel.CANARY: policy.canary_registry_ids,
        ReleaseChannel.INTERNAL_REVIEW: (policy.internal_review_registry_ids),
    }.get(channel)
    if allowed is None or registry not in allowed:
        raise ReleasePublicationPolicyError("publication target is not an approved non-production registry")


def _publish_decision(
    approved: ReleaseProjectionResultV2,
    *,
    decided_at: datetime,
    audit: ContractAudit,
) -> ReleaseDecisionV2:
    prior = approved.release_decision
    revision = int(prior.item_version) + 1
    value = prior.model_copy(
        update={
            "release_decision_id": "release-decision://pending",
            "previous_decision_ref": approved.release_decision_ref,
            "item_version": str(revision),
            "action": ReleaseActionV2.PUBLISH,
            "state": ReleaseStateV2.RELEASED,
            "idempotency_key": (f"publish-{approved.release_subject.release_subject_sha256[:24]}"),
            "actor": audit.created_by,
            "decided_at": decided_at,
            "decision_sha256": "0" * 64,
            "audit": audit.model_copy(
                update={
                    "input_refs": tuple(
                        sorted(
                            {
                                approved.release_subject_ref,
                                approved.release_decision_ref,
                                approved.evaluation_item_ref,
                            },
                            key=_ref_key,
                        )
                    )
                }
            ),
        }
    )
    pending = ReleaseDecisionV2.model_validate(value.model_dump(mode="python"))
    digest = release_decision_v2_carried_sha256(pending)
    result = ReleaseDecisionV2.model_validate(
        pending.model_copy(
            update={
                "release_decision_id": (f"release-decision://sha256/{digest}"),
                "decision_sha256": digest,
            }
        ).model_dump(mode="python")
    )
    release_decision_v2_ref(result)
    return result


def _published_item(
    approved: ReleaseProjectionResultV2,
    decision: ReleaseDecisionV2,
    *,
    audit: ContractAudit,
) -> EvaluationItemV2:
    decision_ref = release_decision_v2_ref(decision)
    prior = approved.evaluation_item
    value = prior.model_copy(
        update={
            "evaluation_item_id": "evaluation-item://pending",
            "item_version": decision.item_version,
            "release_decision_ref": decision_ref,
            "item_sha256": "0" * 64,
            "audit": audit.model_copy(
                update={
                    "input_refs": tuple(
                        sorted(
                            {
                                approved.evaluation_item_ref,
                                decision_ref,
                            },
                            key=_ref_key,
                        )
                    )
                }
            ),
        }
    )
    pending = EvaluationItemV2.model_validate(value.model_dump(mode="python"))
    digest = evaluation_item_v2_carried_sha256(pending)
    result = EvaluationItemV2.model_validate(
        pending.model_copy(
            update={
                "evaluation_item_id": (f"evaluation-item://sha256/{digest}"),
                "item_sha256": digest,
            }
        ).model_dump(mode="python")
    )
    evaluation_item_v2_ref(result)
    return result


def _validate_rubric_set(value: RubricSetV2) -> None:
    for criterion in value.criteria:
        reachability_digest = rubric_reachability_carried_sha256(criterion.reachability)
        if (
            criterion.reachability.reachability_sha256 != reachability_digest
            or criterion.reachability.reachability_id != f"rubric-reachability://sha256/{reachability_digest}"
        ):
            raise ValueError("RubricSet reachability identity is stale")
        criterion_digest = rubric_criterion_carried_sha256(criterion)
        if (
            criterion.criterion_sha256 != criterion_digest
            or criterion.criterion_id != f"rubric-criterion://sha256/{criterion_digest}"
        ):
            raise ValueError("RubricSet criterion identity is stale")
    digest = rubric_set_carried_sha256(value)
    if value.rubric_set_sha256 != digest or value.rubric_set_id != f"rubric-set://sha256/{digest}":
        raise ValueError("RubricSet identity is stale")


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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "ReleaseBundleFacts",
    "ReleasePublicationCompilation",
    "ReleasePublicationCompiler",
    "ReleasePublicationItemCompilation",
    "ReleasePublicationItemSource",
    "ReleasePublicationManifestCompilation",
    "ReleasePublicationPolicyError",
    "render_lh_query_yaml",
    "render_lh_rubrics_json",
    "validate_lh_release_source",
]
