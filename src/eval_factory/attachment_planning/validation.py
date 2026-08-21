from __future__ import annotations

import asyncio
import hashlib
import json

from env_mock_agent.facade import (
    ATTACHMENT_VALIDATION_POLICY_VERSION,
    AttachmentInventoryMemberTypeV2,
    AttachmentValidationFacade,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingV2,
    AttachmentValidationFingerprintCategoryV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
    AttachmentValidationPiiRuleV2,
    AttachmentValidationRequestV2,
    AttachmentValidationResultV2,
    AttachmentValidationStatusV2,
    FacadeObjectRef,
    attachment_execution_request_ref,
    attachment_execution_result_ref,
    attachment_inventory_member_ref,
    attachment_validation_finding_ref,
    attachment_validation_request_carried_sha256,
    attachment_validation_request_ref,
    attachment_validation_result_ref,
    validate_attachment_validation_result_identity,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildResultOutcomeV2,
    ArtifactBuildResultV2,
    AttachmentReconstructionResultV2,
    artifact_build_result_v2_ref,
    artifact_build_spec_v2_ref,
    attachment_reconstruction_result_v2_ref,
    validate_artifact_build_result_v2_identity,
    validate_attachment_reconstruction_result_v2_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    attachment_candidate_revision_carried_sha256,
    attachment_candidate_revision_ref,
    candidate_artifact_version_carried_sha256,
    candidate_artifact_version_ref,
)
from eval_factory.contracts.safety import PackageInventoryMember
from eval_factory.contracts.task_v2 import (
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
    producer_task_view_ref,
    prompt_leakage_reference_set_ref,
)
from eval_factory.contracts.validation_v2 import (
    DETERMINISTIC_VALIDATION_POLICY_VERSION,
    ArtifactDeterministicValidationOutcomeV2,
    ArtifactDeterministicValidationResultV2,
    CandidatePackageInventoryEntryV2,
    CandidatePackageInventoryV2,
    DeterministicItemValidationResultV2,
    DeterministicValidationFindingCategoryV2,
    DeterministicValidationFindingCodeV2,
    DeterministicValidationFindingV2,
    artifact_deterministic_validation_result_carried_sha256,
    artifact_deterministic_validation_result_ref,
    candidate_package_inventory_carried_sha256,
    candidate_package_inventory_entry_carried_sha256,
    candidate_package_inventory_entry_ref,
    candidate_package_inventory_ref,
    deterministic_item_validation_outcome_v2,
    deterministic_item_validation_result_carried_sha256,
    deterministic_validation_finding_carried_sha256,
    deterministic_validation_finding_ref,
    validate_artifact_deterministic_validation_result_identity,
    validate_candidate_package_inventory_identity,
    validate_deterministic_item_validation_result_identity,
)
from eval_factory.provenance.redaction import ConfiguredPiiRule
from eval_factory.task_authoring.producer_view import (
    validate_producer_task_view_identity,
)
from eval_factory.task_authoring.prompt_safety import (
    validate_prompt_leakage_reference_set_identity,
)


class DeterministicValidationPolicyError(RuntimeError):
    pass


class DeterministicValidationCompiler:
    policy_version = DETERMINISTIC_VALIDATION_POLICY_VERSION

    async def compile(
        self,
        *,
        reconstruction_result: AttachmentReconstructionResultV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
        facade: AttachmentValidationFacade,
        audit: ContractAudit,
    ) -> DeterministicItemValidationResultV2:
        try:
            return await self._compile(
                reconstruction_result=reconstruction_result,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=configured_pii_rules,
                facade=facade,
                audit=audit,
            )
        except DeterministicValidationPolicyError:
            raise
        except ValueError as exc:
            raise DeterministicValidationPolicyError(
                f"deterministic validation compilation failed: {exc}"
            ) from exc

    def validate_current(
        self,
        result: DeterministicItemValidationResultV2,
        *,
        reconstruction_result: AttachmentReconstructionResultV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
    ) -> None:
        try:
            validated = DeterministicItemValidationResultV2.model_validate(result.model_dump(mode="python"))
            validate_deterministic_item_validation_result_identity(validated)
            _validate_sources(
                reconstruction_result=reconstruction_result,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=configured_pii_rules,
            )
            expected_reconstruction_ref = attachment_reconstruction_result_v2_ref(reconstruction_result)
            if (
                result.attachment_reconstruction_result_ref != expected_reconstruction_ref
                or result.producer_task_view_ref != producer_task_view_ref(producer_task_view)
                or result.leakage_reference_set_ref != prompt_leakage_reference_set_ref(leakage_reference_set)
                or result.configured_pii_rules_sha256 != _configured_pii_rules_sha256(configured_pii_rules)
            ):
                raise DeterministicValidationPolicyError(
                    "deterministic item validation source refs are stale"
                )
            _validate_artifact_currentness(
                result=result,
                reconstruction_result=reconstruction_result,
            )
        except DeterministicValidationPolicyError:
            raise
        except ValueError as exc:
            raise DeterministicValidationPolicyError(
                f"current deterministic validation failed: {exc}"
            ) from exc

    async def compile_revision(
        self,
        *,
        candidate_revision: AttachmentCandidateRevisionV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
        facade: AttachmentValidationFacade,
        audit: ContractAudit,
    ) -> DeterministicItemValidationResultV2:
        try:
            _validate_revision_sources(
                candidate_revision=candidate_revision,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=configured_pii_rules,
            )
            revision_ref = attachment_candidate_revision_ref(candidate_revision)
            requests = tuple(
                _revision_validation_request(
                    artifact_version=item,
                    producer_task_view=producer_task_view,
                    leakage_reference_set=leakage_reference_set,
                    configured_pii_rules=configured_pii_rules,
                )
                for item in candidate_revision.artifact_versions
            )
            facade_results = (
                tuple(await asyncio.gather(*(facade.validate(request) for request in requests)))
                if requests
                else ()
            )
            artifact_results = tuple(
                _compile_artifact_validation_result(
                    reconstruction_ref=(candidate_revision.base_reconstruction_result_ref),
                    artifact_subject_ref=_artifact_validation_subject_ref(artifact_version),
                    build_spec_ref=artifact_version.build_spec_ref,
                    artifact_id=artifact_version.artifact_id,
                    request=request,
                    facade_result=facade_result,
                    audit=audit,
                )
                for artifact_version, request, facade_result in zip(
                    candidate_revision.artifact_versions,
                    requests,
                    facade_results,
                    strict=True,
                )
            )
            inventory: CandidatePackageInventoryV2 | None = None
            inventory_failure_code: AttachmentValidationFailureCodeV2 | None = None
            if artifact_results and all(
                item.outcome is not ArtifactDeterministicValidationOutcomeV2.BLOCKED
                for item in artifact_results
            ):
                try:
                    inventory = _compile_revision_candidate_inventory(
                        revision_ref=revision_ref,
                        artifact_versions=candidate_revision.artifact_versions,
                        validation_results=artifact_results,
                        audit=audit,
                    )
                except _CandidateInventoryBlocked as exc:
                    inventory_failure_code = exc.code
            return _compile_item_validation_result(
                subject_ref=revision_ref,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=configured_pii_rules,
                artifact_results=artifact_results,
                upstream_incomplete=(),
                inventory=inventory,
                inventory_failure_code=inventory_failure_code,
                audit=audit,
            )
        except DeterministicValidationPolicyError:
            raise
        except ValueError as exc:
            raise DeterministicValidationPolicyError(
                f"revision deterministic validation failed: {exc}"
            ) from exc

    async def _compile(
        self,
        *,
        reconstruction_result: AttachmentReconstructionResultV2,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
        facade: AttachmentValidationFacade,
        audit: ContractAudit,
    ) -> DeterministicItemValidationResultV2:
        _validate_sources(
            reconstruction_result=reconstruction_result,
            producer_task_view=producer_task_view,
            leakage_reference_set=leakage_reference_set,
            configured_pii_rules=configured_pii_rules,
        )
        reconstruction_ref = attachment_reconstruction_result_v2_ref(reconstruction_result)
        producer_ref = producer_task_view_ref(producer_task_view)
        leakage_ref = prompt_leakage_reference_set_ref(leakage_reference_set)
        pii_sha256 = _configured_pii_rules_sha256(configured_pii_rules)
        successful = tuple(
            item
            for item in reconstruction_result.artifact_results
            if item.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED
        )
        upstream_incomplete = tuple(
            sorted(
                set(reconstruction_result.required_incomplete_artifact_ids)
                | set(reconstruction_result.optional_incomplete_artifact_ids)
            )
        )
        requests = tuple(
            _validation_request(
                artifact_result=item,
                producer_task_view=producer_task_view,
                leakage_reference_set=leakage_reference_set,
                configured_pii_rules=configured_pii_rules,
            )
            for item in successful
        )
        facade_results = (
            tuple(await asyncio.gather(*(facade.validate(request) for request in requests)))
            if requests
            else ()
        )
        artifact_results = tuple(
            _compile_artifact_validation_result(
                reconstruction_ref=reconstruction_ref,
                artifact_subject_ref=artifact_build_result_v2_ref(artifact_result),
                build_spec_ref=artifact_build_spec_v2_ref(artifact_result.route_entry.build_spec)
                if artifact_result.route_entry.build_spec is not None
                else None,
                artifact_id=artifact_result.route_entry.artifact_id,
                request=request,
                facade_result=facade_result,
                audit=audit,
            )
            for artifact_result, request, facade_result in zip(
                successful,
                requests,
                facade_results,
                strict=True,
            )
        )
        inventory: CandidatePackageInventoryV2 | None = None
        inventory_failure_code: AttachmentValidationFailureCodeV2 | None = None
        if artifact_results and all(
            item.outcome is not ArtifactDeterministicValidationOutcomeV2.BLOCKED for item in artifact_results
        ):
            try:
                inventory = _compile_candidate_inventory(
                    reconstruction_ref=reconstruction_ref,
                    source_artifact_results=successful,
                    validation_results=artifact_results,
                    audit=audit,
                )
            except _CandidateInventoryBlocked as exc:
                inventory_failure_code = exc.code

        result_refs = tuple(artifact_deterministic_validation_result_ref(item) for item in artifact_results)
        finding_refs = tuple(
            sorted(
                (finding_ref for item in artifact_results for finding_ref in item.finding_refs),
                key=_ref_key,
            )
        )
        candidate_inventory_ref = (
            candidate_package_inventory_ref(inventory) if inventory is not None else None
        )
        validated = _artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.PASSED},
        )
        repair_required = _artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR},
        )
        rejected = _artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.REJECTED},
        )
        blocked = _artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.BLOCKED},
        )
        outcome = deterministic_item_validation_outcome_v2(
            artifact_results,
            upstream_incomplete,
            inventory_failure_code,
        )
        audit_refs = [
            reconstruction_ref,
            producer_ref,
            leakage_ref,
            *result_refs,
            *finding_refs,
        ]
        if candidate_inventory_ref is not None:
            audit_refs.append(candidate_inventory_ref)
        result = DeterministicItemValidationResultV2(
            deterministic_item_validation_result_id=("deterministic-item-validation-result://pending"),
            attachment_reconstruction_result_ref=reconstruction_ref,
            producer_task_view_ref=producer_ref,
            leakage_reference_set_ref=leakage_ref,
            configured_pii_rules_sha256=pii_sha256,
            artifact_validation_results=artifact_results,
            artifact_validation_result_refs=result_refs,
            finding_refs=finding_refs,
            candidate_inventory=inventory,
            candidate_inventory_ref=candidate_inventory_ref,
            candidate_inventory_failure_code=inventory_failure_code,
            outcome=outcome,
            validated_artifact_ids=validated,
            repair_required_artifact_ids=repair_required,
            rejected_artifact_ids=rejected,
            validation_blocked_artifact_ids=blocked,
            upstream_incomplete_artifact_ids=upstream_incomplete,
            accepted_artifact_refs=(),
            environment_spec_ref=None,
            provenance_manifest_ref=None,
            quality_report_ref=None,
            package_sha256=None,
            input_state_only=None,
            policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
            deterministic_item_validation_result_sha256="0" * 64,
            audit=_safe_audit(audit, tuple(audit_refs)),
        )
        digest = deterministic_item_validation_result_carried_sha256(result)
        result = result.model_copy(
            update={
                "deterministic_item_validation_result_id": (
                    f"deterministic-item-validation-result://sha256/{digest}"
                ),
                "deterministic_item_validation_result_sha256": digest,
            }
        )
        validate_deterministic_item_validation_result_identity(result)
        return result


class _CandidateInventoryBlocked(RuntimeError):
    def __init__(
        self,
        code: AttachmentValidationFailureCodeV2,
    ) -> None:
        super().__init__(code.value)
        self.code = code


def _compile_item_validation_result(
    *,
    subject_ref: ObjectRef,
    producer_task_view: ProducerTaskViewV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...],
    artifact_results: tuple[
        ArtifactDeterministicValidationResultV2,
        ...,
    ],
    upstream_incomplete: tuple[str, ...],
    inventory: CandidatePackageInventoryV2 | None,
    inventory_failure_code: AttachmentValidationFailureCodeV2 | None,
    audit: ContractAudit,
) -> DeterministicItemValidationResultV2:
    producer_ref = producer_task_view_ref(producer_task_view)
    leakage_ref = prompt_leakage_reference_set_ref(leakage_reference_set)
    result_refs = tuple(artifact_deterministic_validation_result_ref(item) for item in artifact_results)
    finding_refs = tuple(
        sorted(
            (finding_ref for item in artifact_results for finding_ref in item.finding_refs),
            key=_ref_key,
        )
    )
    inventory_ref = candidate_package_inventory_ref(inventory) if inventory is not None else None
    result = DeterministicItemValidationResultV2(
        deterministic_item_validation_result_id=("deterministic-item-validation-result://pending"),
        attachment_reconstruction_result_ref=subject_ref,
        producer_task_view_ref=producer_ref,
        leakage_reference_set_ref=leakage_ref,
        configured_pii_rules_sha256=_configured_pii_rules_sha256(configured_pii_rules),
        artifact_validation_results=artifact_results,
        artifact_validation_result_refs=result_refs,
        finding_refs=finding_refs,
        candidate_inventory=inventory,
        candidate_inventory_ref=inventory_ref,
        candidate_inventory_failure_code=inventory_failure_code,
        outcome=deterministic_item_validation_outcome_v2(
            artifact_results,
            upstream_incomplete,
            inventory_failure_code,
        ),
        validated_artifact_ids=_artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.PASSED},
        ),
        repair_required_artifact_ids=_artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR},
        ),
        rejected_artifact_ids=_artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.REJECTED},
        ),
        validation_blocked_artifact_ids=_artifact_ids(
            artifact_results,
            {ArtifactDeterministicValidationOutcomeV2.BLOCKED},
        ),
        upstream_incomplete_artifact_ids=upstream_incomplete,
        accepted_artifact_refs=(),
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
        deterministic_item_validation_result_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                subject_ref,
                producer_ref,
                leakage_ref,
                *result_refs,
                *finding_refs,
                *((inventory_ref,) if inventory_ref is not None else ()),
            ),
        ),
    )
    digest = deterministic_item_validation_result_carried_sha256(result)
    result = result.model_copy(
        update={
            "deterministic_item_validation_result_id": (
                f"deterministic-item-validation-result://sha256/{digest}"
            ),
            "deterministic_item_validation_result_sha256": digest,
        }
    )
    validate_deterministic_item_validation_result_identity(result)
    return result


def _validate_sources(
    *,
    reconstruction_result: AttachmentReconstructionResultV2,
    producer_task_view: ProducerTaskViewV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...],
) -> None:
    expected_producer_ref = producer_task_view_ref(producer_task_view)
    if reconstruction_result.producer_task_view_ref != expected_producer_ref:
        raise DeterministicValidationPolicyError(
            "attachment reconstruction result has mismatched producer task view"
        )
    validate_attachment_reconstruction_result_v2_identity(reconstruction_result)
    for artifact_result in reconstruction_result.artifact_results:
        validate_artifact_build_result_v2_identity(artifact_result)
    validate_producer_task_view_identity(producer_task_view)
    validate_prompt_leakage_reference_set_identity(leakage_reference_set)
    rule_ids = tuple(item.rule_id for item in configured_pii_rules)
    if len(rule_ids) != len(set(rule_ids)):
        raise DeterministicValidationPolicyError("configured PII rule IDs must be unique")
    if rule_ids != tuple(sorted(rule_ids)):
        raise DeterministicValidationPolicyError("configured PII rules must be sorted by rule ID")
    expected_dependency_ids = tuple(item.dependency_id for item in producer_task_view.attachment_requirements)
    observed_dependency_ids = tuple(
        sorted(item.route_entry.attachment_dependency_id for item in reconstruction_result.artifact_results)
    )
    if expected_dependency_ids != observed_dependency_ids:
        raise DeterministicValidationPolicyError(
            "attachment reconstruction result does not exactly cover producer requirements"
        )


def _validate_revision_sources(
    *,
    candidate_revision: AttachmentCandidateRevisionV2,
    producer_task_view: ProducerTaskViewV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...],
) -> None:
    revision_digest = attachment_candidate_revision_carried_sha256(candidate_revision)
    if (
        candidate_revision.candidate_revision_sha256 != revision_digest
        or candidate_revision.attachment_candidate_revision_id
        != f"attachment-candidate-revision://sha256/{revision_digest}"
    ):
        raise DeterministicValidationPolicyError("candidate revision identity is stale")
    for artifact in candidate_revision.artifact_versions:
        artifact_digest = candidate_artifact_version_carried_sha256(artifact)
        if (
            artifact.artifact_version_sha256 != artifact_digest
            or artifact.candidate_artifact_version_id
            != f"candidate-artifact-version://sha256/{artifact_digest}"
        ):
            raise DeterministicValidationPolicyError("candidate artifact version identity is stale")
    validate_producer_task_view_identity(producer_task_view)
    validate_prompt_leakage_reference_set_identity(leakage_reference_set)
    rule_ids = tuple(item.rule_id for item in configured_pii_rules)
    if len(rule_ids) != len(set(rule_ids)) or rule_ids != tuple(sorted(rule_ids)):
        raise DeterministicValidationPolicyError("configured PII rules must be sorted and unique")
    expected_dependency_ids = tuple(item.dependency_id for item in producer_task_view.attachment_requirements)
    observed_dependency_ids = tuple(
        sorted(item.attachment_dependency_id for item in candidate_revision.artifact_versions)
    )
    if expected_dependency_ids != observed_dependency_ids:
        raise DeterministicValidationPolicyError(
            "candidate revision does not exactly cover producer requirements"
        )


def _validation_request(
    *,
    artifact_result: ArtifactBuildResultV2,
    producer_task_view: ProducerTaskViewV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...],
) -> AttachmentValidationRequestV2:
    entry = artifact_result.route_entry
    receipt = artifact_result.execution_receipt
    if (
        entry.build_spec is None
        or receipt is None
        or receipt.facade_request is None
        or receipt.facade_result is None
        or receipt.facade_result.output_ref is None
        or receipt.facade_result.output_sha256 is None
    ):
        raise DeterministicValidationPolicyError("successful artifact result lacks exact validation sources")
    build_spec = entry.build_spec
    facade_output_ref = receipt.facade_result.output_ref
    output_ref = _object_ref_from_facade(facade_output_ref)
    if output_ref.object_sha256 != receipt.facade_result.output_sha256:
        raise DeterministicValidationPolicyError("successful artifact output ref/hash mismatch")
    fingerprints = tuple(
        sorted(
            (
                AttachmentValidationFingerprintV2(
                    fingerprint_id=item.fingerprint_id,
                    category=(AttachmentValidationFingerprintCategoryV2(item.category.value)),
                    match_kind=(AttachmentValidationFingerprintMatchKindV2(item.match_kind.value)),
                    digest_sha256=item.digest_sha256,
                    token_count=item.token_count,
                    normalization_version=item.normalization_version,
                )
                for item in leakage_reference_set.fingerprints
            ),
            key=lambda item: (
                item.category.value,
                item.match_kind.value,
                item.digest_sha256,
                item.token_count,
                item.fingerprint_id,
            ),
        )
    )
    pii_rules = tuple(
        AttachmentValidationPiiRuleV2(
            rule_id=item.rule_id,
            pattern=item.pattern,
        )
        for item in configured_pii_rules
    )
    complete_categories = tuple(
        sorted(
            (
                AttachmentValidationFingerprintCategoryV2(item.value)
                for item in leakage_reference_set.complete_categories
            ),
            key=lambda item: item.value,
        )
    )
    seed = {
        "artifact_result_ref": _ref_payload(artifact_build_result_v2_ref(artifact_result)),
        "output_ref": _ref_payload(output_ref),
        "policy_version": DETERMINISTIC_VALIDATION_POLICY_VERSION,
    }
    idempotency_digest = _payload_sha256(seed)
    request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id=entry.artifact_id,
        artifact_result_ref=_facade_ref(artifact_build_result_v2_ref(artifact_result)),
        execution_request_ref=attachment_execution_request_ref(receipt.facade_request),
        execution_result_ref=attachment_execution_result_ref(receipt.facade_result),
        build_spec_ref=_facade_ref(artifact_build_spec_v2_ref(build_spec)),
        producer_task_view_ref=_facade_ref(producer_task_view_ref(producer_task_view)),
        output_ref=facade_output_ref,
        output_sha256=receipt.facade_result.output_sha256,
        logical_path=build_spec.build_spec.relative_path,
        media_type=build_spec.build_spec.media_type,
        declared_validator_ids=build_spec.build_spec.validator_ids,
        configured_pii_rules=pii_rules,
        leakage_reference_set_ref=_facade_ref(prompt_leakage_reference_set_ref(leakage_reference_set)),
        leakage_fingerprints=fingerprints,
        complete_leakage_categories=complete_categories,
        forbidden_output_values=tuple(sorted(producer_task_view.forbidden_outputs)),
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        idempotency_key=(f"attachment-validation-idempotency://sha256/{idempotency_digest}"),
        validation_request_sha256="0" * 64,
    )
    digest = attachment_validation_request_carried_sha256(request)
    return request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )


def _revision_validation_request(
    *,
    artifact_version: CandidateArtifactVersionV2,
    producer_task_view: ProducerTaskViewV2,
    leakage_reference_set: PromptLeakageReferenceSetV2,
    configured_pii_rules: tuple[ConfiguredPiiRule, ...],
) -> AttachmentValidationRequestV2:
    artifact_subject_ref = _artifact_validation_subject_ref(artifact_version)
    fingerprints = tuple(
        sorted(
            (
                AttachmentValidationFingerprintV2(
                    fingerprint_id=item.fingerprint_id,
                    category=AttachmentValidationFingerprintCategoryV2(item.category.value),
                    match_kind=AttachmentValidationFingerprintMatchKindV2(item.match_kind.value),
                    digest_sha256=item.digest_sha256,
                    token_count=item.token_count,
                    normalization_version=item.normalization_version,
                )
                for item in leakage_reference_set.fingerprints
            ),
            key=lambda item: (
                item.category.value,
                item.match_kind.value,
                item.digest_sha256,
                item.token_count,
                item.fingerprint_id,
            ),
        )
    )
    pii_rules = tuple(
        AttachmentValidationPiiRuleV2(
            rule_id=item.rule_id,
            pattern=item.pattern,
        )
        for item in configured_pii_rules
    )
    complete_categories = tuple(
        sorted(
            (
                AttachmentValidationFingerprintCategoryV2(item.value)
                for item in leakage_reference_set.complete_categories
            ),
            key=lambda item: item.value,
        )
    )
    idempotency_digest = _payload_sha256(
        {
            "artifact_result_ref": _ref_payload(artifact_subject_ref),
            "output_ref": _ref_payload(artifact_version.output_ref),
            "policy_version": DETERMINISTIC_VALIDATION_POLICY_VERSION,
        }
    )
    request = AttachmentValidationRequestV2(
        validation_request_id="attachment-validation-request://pending",
        artifact_id=artifact_version.artifact_id,
        artifact_result_ref=_facade_ref(artifact_subject_ref),
        execution_request_ref=_facade_ref(artifact_version.execution_request_ref),
        execution_result_ref=_facade_ref(artifact_version.execution_result_ref),
        build_spec_ref=_facade_ref(artifact_version.build_spec_ref),
        producer_task_view_ref=_facade_ref(producer_task_view_ref(producer_task_view)),
        output_ref=_facade_ref(artifact_version.output_ref),
        output_sha256=artifact_version.output_sha256,
        logical_path=artifact_version.logical_path,
        media_type=artifact_version.media_type,
        declared_validator_ids=artifact_version.declared_validator_ids,
        configured_pii_rules=pii_rules,
        leakage_reference_set_ref=_facade_ref(prompt_leakage_reference_set_ref(leakage_reference_set)),
        leakage_fingerprints=fingerprints,
        complete_leakage_categories=complete_categories,
        forbidden_output_values=tuple(sorted(producer_task_view.forbidden_outputs)),
        policy_version=ATTACHMENT_VALIDATION_POLICY_VERSION,
        idempotency_key=(f"attachment-validation-idempotency://sha256/{idempotency_digest}"),
        validation_request_sha256="0" * 64,
    )
    digest = attachment_validation_request_carried_sha256(request)
    return request.model_copy(
        update={
            "validation_request_id": (f"attachment-validation-request://sha256/{digest}"),
            "validation_request_sha256": digest,
        }
    )


def _artifact_validation_subject_ref(
    artifact_version: CandidateArtifactVersionV2,
) -> ObjectRef:
    if artifact_version.artifact_version == 1:
        return artifact_version.base_artifact_build_result_ref
    return candidate_artifact_version_ref(artifact_version)


def _compile_artifact_validation_result(
    *,
    reconstruction_ref: ObjectRef,
    artifact_subject_ref: ObjectRef,
    build_spec_ref: ObjectRef | None,
    artifact_id: str,
    request: AttachmentValidationRequestV2,
    facade_result: AttachmentValidationResultV2,
    audit: ContractAudit,
) -> ArtifactDeterministicValidationResultV2:
    validate_attachment_validation_result_identity(facade_result)
    request_ref = _object_ref_from_facade(attachment_validation_request_ref(request))
    facade_result_ref = _object_ref_from_facade(attachment_validation_result_ref(facade_result))
    if build_spec_ref is None:
        raise DeterministicValidationPolicyError("validated artifact lacks build spec")
    output_ref = _object_ref_from_facade(request.output_ref)
    if (
        _object_ref_from_facade(facade_result.validation_request_ref) != request_ref
        or facade_result.artifact_id != artifact_id
        or _object_ref_from_facade(facade_result.output_ref) != output_ref
        or facade_result.output_sha256 != request.output_sha256
    ):
        raise DeterministicValidationPolicyError("facade validation result is stale or cross-artifact")
    if facade_result.status is not AttachmentValidationStatusV2.BLOCKED:
        root_members = tuple(
            item
            for item in facade_result.inventory_members
            if item.container_ref is None and item.normalized_path == request.logical_path
        )
        if len(root_members) != 1:
            raise DeterministicValidationPolicyError(
                "facade validation inventory must contain the exact artifact root"
            )
        root_member = root_members[0]
        if (
            root_member.member_type is AttachmentInventoryMemberTypeV2.FILE
            and root_member.content_sha256 != request.output_sha256
        ):
            raise DeterministicValidationPolicyError("facade validation root file hash does not match output")
        if any(
            item.container_ref is None
            and item.normalized_path != request.logical_path
            and not item.normalized_path.startswith(f"{request.logical_path.rstrip('/')}/")
            for item in facade_result.inventory_members
        ):
            raise DeterministicValidationPolicyError(
                "facade validation inventory contains an out-of-root member"
            )
    member_refs = {
        item.inventory_member_id: _object_ref_from_facade(attachment_inventory_member_ref(item))
        for item in facade_result.inventory_members
    }
    findings = tuple(
        _factory_finding(
            artifact_result_ref=artifact_subject_ref,
            output_ref=output_ref,
            facade_result_ref=facade_result_ref,
            facade_finding=facade_finding,
            inventory_member_ref=(
                member_refs.get(facade_finding.inventory_member_id)
                if facade_finding.inventory_member_id is not None
                else None
            ),
            audit=audit,
        )
        for facade_finding in facade_result.findings
    )
    finding_refs = tuple(deterministic_validation_finding_ref(item) for item in findings)
    if facade_result.status is AttachmentValidationStatusV2.BLOCKED:
        outcome = ArtifactDeterministicValidationOutcomeV2.BLOCKED
    elif facade_result.status is AttachmentValidationStatusV2.PASSED:
        outcome = ArtifactDeterministicValidationOutcomeV2.PASSED
    elif any(item.non_waivable for item in findings):
        outcome = ArtifactDeterministicValidationOutcomeV2.REJECTED
    else:
        outcome = ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR
    audit_refs = (
        reconstruction_ref,
        artifact_subject_ref,
        build_spec_ref,
        output_ref,
        request_ref,
        facade_result_ref,
        *finding_refs,
    )
    result = ArtifactDeterministicValidationResultV2(
        artifact_validation_result_id=("artifact-deterministic-validation-result://pending"),
        attachment_reconstruction_result_ref=reconstruction_ref,
        artifact_build_result_ref=artifact_subject_ref,
        build_spec_ref=build_spec_ref,
        output_ref=output_ref,
        output_sha256=request.output_sha256,
        facade_validation_request_ref=request_ref,
        facade_validation_result=facade_result,
        outcome=outcome,
        findings=findings,
        finding_refs=finding_refs,
        inventory_members=facade_result.inventory_members,
        policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
        artifact_validation_result_sha256="0" * 64,
        audit=_safe_audit(audit, audit_refs),
    )
    digest = artifact_deterministic_validation_result_carried_sha256(result)
    result = result.model_copy(
        update={
            "artifact_validation_result_id": (f"artifact-deterministic-validation-result://sha256/{digest}"),
            "artifact_validation_result_sha256": digest,
        }
    )
    validate_artifact_deterministic_validation_result_identity(result)
    return result


def _factory_finding(
    *,
    artifact_result_ref: ObjectRef,
    output_ref: ObjectRef,
    facade_result_ref: ObjectRef,
    facade_finding: AttachmentValidationFindingV2,
    inventory_member_ref: ObjectRef | None,
    audit: ContractAudit,
) -> DeterministicValidationFindingV2:
    facade_finding_ref = _object_ref_from_facade(attachment_validation_finding_ref(facade_finding))
    audit_refs = [
        artifact_result_ref,
        output_ref,
        facade_result_ref,
        facade_finding_ref,
    ]
    if inventory_member_ref is not None:
        audit_refs.append(inventory_member_ref)
    finding = DeterministicValidationFindingV2(
        finding_id="deterministic-validation-finding://pending",
        artifact_build_result_ref=artifact_result_ref,
        output_ref=output_ref,
        facade_validation_result_ref=facade_result_ref,
        facade_finding_ref=facade_finding_ref,
        inventory_member_ref=inventory_member_ref,
        severity=Severity(facade_finding.severity.value),
        category=DeterministicValidationFindingCategoryV2(facade_finding.category.value),
        code=DeterministicValidationFindingCodeV2(facade_finding.code.value),
        rule_ids=facade_finding.rule_ids,
        matched_fingerprint_ids=(facade_finding.matched_fingerprint_ids),
        status="OPEN",
        non_waivable=facade_finding.non_waivable,
        policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
        finding_sha256="0" * 64,
        audit=_safe_audit(audit, tuple(audit_refs)),
    )
    digest = deterministic_validation_finding_carried_sha256(finding)
    return finding.model_copy(
        update={
            "finding_id": (f"deterministic-validation-finding://sha256/{digest}"),
            "finding_sha256": digest,
        }
    )


def _compile_candidate_inventory(
    *,
    reconstruction_ref: ObjectRef,
    source_artifact_results: tuple[ArtifactBuildResultV2, ...],
    validation_results: tuple[
        ArtifactDeterministicValidationResultV2,
        ...,
    ],
    audit: ContractAudit,
) -> CandidatePackageInventoryV2:
    sources_by_id = {item.route_entry.artifact_id: item for item in source_artifact_results}
    entries: list[CandidatePackageInventoryEntryV2] = []
    for validation_result in validation_results:
        artifact_id = validation_result.facade_validation_result.artifact_id
        try:
            source = sources_by_id[artifact_id]
        except KeyError as exc:
            raise DeterministicValidationPolicyError(
                "candidate inventory source artifact is missing"
            ) from exc
        if source.route_entry.build_spec is None:
            raise DeterministicValidationPolicyError("candidate inventory source lacks build spec")
        artifact_result_ref = artifact_build_result_v2_ref(source)
        validation_result_ref = artifact_deterministic_validation_result_ref(validation_result)
        build_spec_ref = artifact_build_spec_v2_ref(source.route_entry.build_spec)
        derivation_roots = _derivation_root_refs(source)
        for member in validation_result.inventory_members:
            facade_member_ref = attachment_inventory_member_ref(member)
            member_ref = _object_ref_from_facade(facade_member_ref)
            container_ref = (
                _object_ref_from_facade(member.container_ref) if member.container_ref is not None else None
            )
            frozen_member = PackageInventoryMember(
                normalized_path=member.normalized_path,
                member_type=member.member_type.value,
                media_type=member.media_type,
                size_bytes=member.size_bytes,
                content_sha256=member.content_sha256,
                container_ref=(container_ref.object_id if container_ref is not None else None),
            )
            entry = CandidatePackageInventoryEntryV2(
                inventory_member=frozen_member,
                inventory_member_ref=member_ref,
                container_ref=container_ref,
                artifact_build_result_ref=artifact_result_ref,
                artifact_validation_result_ref=validation_result_ref,
                output_ref=validation_result.output_ref,
                build_spec_ref=build_spec_ref,
                derivation_root_refs=derivation_roots,
                entry_sha256="0" * 64,
            )
            digest = candidate_package_inventory_entry_carried_sha256(entry)
            entries.append(entry.model_copy(update={"entry_sha256": digest}))
    entries = sorted(entries, key=_candidate_entry_key)
    _validate_inventory_paths(tuple(entries))
    validation_refs = tuple(artifact_deterministic_validation_result_ref(item) for item in validation_results)
    entry_refs = tuple(candidate_package_inventory_entry_ref(item) for item in entries)
    inventory = CandidatePackageInventoryV2(
        candidate_inventory_id="candidate-package-inventory://pending",
        attachment_reconstruction_result_ref=reconstruction_ref,
        artifact_validation_result_refs=validation_refs,
        entries=tuple(entries),
        exact_candidate_set_verified=True,
        final_package=False,
        provenance_manifest_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
        candidate_inventory_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                reconstruction_ref,
                *validation_refs,
                *entry_refs,
            ),
        ),
    )
    digest = candidate_package_inventory_carried_sha256(inventory)
    inventory = inventory.model_copy(
        update={
            "candidate_inventory_id": (f"candidate-package-inventory://sha256/{digest}"),
            "candidate_inventory_sha256": digest,
        }
    )
    validate_candidate_package_inventory_identity(inventory)
    return inventory


def _compile_revision_candidate_inventory(
    *,
    revision_ref: ObjectRef,
    artifact_versions: tuple[CandidateArtifactVersionV2, ...],
    validation_results: tuple[
        ArtifactDeterministicValidationResultV2,
        ...,
    ],
    audit: ContractAudit,
) -> CandidatePackageInventoryV2:
    sources_by_id = {item.artifact_id: item for item in artifact_versions}
    entries: list[CandidatePackageInventoryEntryV2] = []
    for validation_result in validation_results:
        artifact_id = validation_result.facade_validation_result.artifact_id
        try:
            source = sources_by_id[artifact_id]
        except KeyError as exc:
            raise DeterministicValidationPolicyError(
                "revision candidate inventory source artifact is missing"
            ) from exc
        subject_ref = _artifact_validation_subject_ref(source)
        validation_result_ref = artifact_deterministic_validation_result_ref(validation_result)
        for member in validation_result.inventory_members:
            member_ref = _object_ref_from_facade(attachment_inventory_member_ref(member))
            container_ref = (
                _object_ref_from_facade(member.container_ref) if member.container_ref is not None else None
            )
            frozen_member = PackageInventoryMember(
                normalized_path=member.normalized_path,
                member_type=member.member_type.value,
                media_type=member.media_type,
                size_bytes=member.size_bytes,
                content_sha256=member.content_sha256,
                container_ref=(container_ref.object_id if container_ref is not None else None),
            )
            entry = CandidatePackageInventoryEntryV2(
                inventory_member=frozen_member,
                inventory_member_ref=member_ref,
                container_ref=container_ref,
                artifact_build_result_ref=subject_ref,
                artifact_validation_result_ref=validation_result_ref,
                output_ref=validation_result.output_ref,
                build_spec_ref=source.build_spec_ref,
                derivation_root_refs=source.derivation_root_refs,
                entry_sha256="0" * 64,
            )
            digest = candidate_package_inventory_entry_carried_sha256(entry)
            entries.append(entry.model_copy(update={"entry_sha256": digest}))
    entries = sorted(entries, key=_candidate_entry_key)
    _validate_inventory_paths(tuple(entries))
    validation_refs = tuple(artifact_deterministic_validation_result_ref(item) for item in validation_results)
    entry_refs = tuple(candidate_package_inventory_entry_ref(item) for item in entries)
    inventory = CandidatePackageInventoryV2(
        candidate_inventory_id="candidate-package-inventory://pending",
        attachment_reconstruction_result_ref=revision_ref,
        artifact_validation_result_refs=validation_refs,
        entries=tuple(entries),
        exact_candidate_set_verified=True,
        final_package=False,
        provenance_manifest_ref=None,
        package_sha256=None,
        input_state_only=None,
        policy_version=DETERMINISTIC_VALIDATION_POLICY_VERSION,
        candidate_inventory_sha256="0" * 64,
        audit=_safe_audit(
            audit,
            (
                revision_ref,
                *validation_refs,
                *entry_refs,
            ),
        ),
    )
    digest = candidate_package_inventory_carried_sha256(inventory)
    inventory = inventory.model_copy(
        update={
            "candidate_inventory_id": (f"candidate-package-inventory://sha256/{digest}"),
            "candidate_inventory_sha256": digest,
        }
    )
    validate_candidate_package_inventory_identity(inventory)
    return inventory


def _derivation_root_refs(
    artifact_result: ArtifactBuildResultV2,
) -> tuple[ObjectRef, ...]:
    entry = artifact_result.route_entry
    receipt = artifact_result.execution_receipt
    if entry.build_spec is None or receipt is None:
        raise DeterministicValidationPolicyError(
            "candidate derivation roots require build and execution facts"
        )
    refs = [
        artifact_result.artifact_routing_plan_ref,
        entry.artifact_evidence_target_ref,
        artifact_build_spec_v2_ref(entry.build_spec),
        entry.build_spec.attachment_planning_context_ref,
        entry.build_spec.artifact_evidence_matrix_ref,
        entry.build_spec.artifact_build_contract_ref,
        entry.build_spec.artifact_routing_policy_ref,
        entry.build_spec.facade_route_request_ref,
        entry.build_spec.facade_route_decision_ref,
        artifact_result.artifact_execution_plan_ref,
        artifact_build_result_v2_ref(artifact_result),
    ]
    if entry.build_spec.source_evidence_set_ref is not None:
        refs.append(entry.build_spec.source_evidence_set_ref)
    if receipt.facade_request is not None:
        refs.append(_object_ref_from_facade(attachment_execution_request_ref(receipt.facade_request)))
    if receipt.facade_result is not None:
        refs.append(_object_ref_from_facade(attachment_execution_result_ref(receipt.facade_result)))
    return tuple(
        sorted(
            {ref for ref in refs if ref is not None},
            key=_ref_key,
        )
    )


def _validate_inventory_paths(
    entries: tuple[CandidatePackageInventoryEntryV2, ...],
) -> None:
    groups: dict[str, dict[str, str]] = {}
    casefold_keys: set[tuple[str, str]] = set()
    for entry in entries:
        container_id = entry.container_ref.object_id if entry.container_ref is not None else ""
        path = entry.inventory_member.normalized_path.rstrip("/")
        group = groups.setdefault(container_id, {})
        if path in group:
            raise _CandidateInventoryBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)
        casefold_key = (container_id, path.casefold())
        if casefold_key in casefold_keys:
            raise _CandidateInventoryBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)
        casefold_keys.add(casefold_key)
        group[path] = entry.inventory_member.member_type
    for group in groups.values():
        for path in group:
            parts = path.split("/")
            for index in range(1, len(parts)):
                parent = "/".join(parts[:index])
                if parent in group and group[parent] != "DIRECTORY":
                    raise _CandidateInventoryBlocked(AttachmentValidationFailureCodeV2.PACKAGE_PATH_COLLISION)


def _validate_artifact_currentness(
    *,
    result: DeterministicItemValidationResultV2,
    reconstruction_result: AttachmentReconstructionResultV2,
) -> None:
    successful = {
        item.route_entry.artifact_id: item
        for item in reconstruction_result.artifact_results
        if item.outcome is ArtifactBuildResultOutcomeV2.SUCCEEDED
    }
    observed_ids = tuple(
        item.facade_validation_result.artifact_id for item in result.artifact_validation_results
    )
    if observed_ids != tuple(sorted(successful)):
        raise DeterministicValidationPolicyError(
            "artifact validation results are stale for current candidates"
        )
    for validation_result in result.artifact_validation_results:
        artifact_id = validation_result.facade_validation_result.artifact_id
        source = successful[artifact_id]
        if source.route_entry.build_spec is None:
            raise DeterministicValidationPolicyError("current successful artifact lacks build spec")
        receipt = source.execution_receipt
        if (
            receipt is None
            or receipt.facade_result is None
            or receipt.facade_result.output_ref is None
            or validation_result.artifact_build_result_ref != artifact_build_result_v2_ref(source)
            or validation_result.build_spec_ref != artifact_build_spec_v2_ref(source.route_entry.build_spec)
            or validation_result.output_ref != _object_ref_from_facade(receipt.facade_result.output_ref)
        ):
            raise DeterministicValidationPolicyError("artifact validation result subject is stale")


def _configured_pii_rules_sha256(
    rules: tuple[ConfiguredPiiRule, ...],
) -> str:
    return _payload_sha256([item.model_dump(mode="json", exclude_none=False) for item in rules])


def _artifact_ids(
    results: tuple[ArtifactDeterministicValidationResultV2, ...],
    outcomes: set[ArtifactDeterministicValidationOutcomeV2],
) -> tuple[str, ...]:
    return tuple(
        sorted(item.facade_validation_result.artifact_id for item in results if item.outcome in outcomes)
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(item for item in audit.governing_versions if item.component != "deterministic-validation"),
        VersionBinding(
            component="deterministic-validation",
            version="r5-08",
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": governing,
            "input_refs": tuple(sorted(set(refs), key=_ref_key)),
        }
    )


def _candidate_entry_key(
    entry: CandidatePackageInventoryEntryV2,
) -> tuple[str, str, str]:
    return (
        entry.container_ref.object_id if entry.container_ref is not None else "",
        entry.inventory_member.normalized_path,
        entry.inventory_member.member_type,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
