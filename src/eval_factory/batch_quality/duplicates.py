from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from pydantic import ValidationError

from env_mock_agent.facade import (
    ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
    ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
    AttachmentDuplicateFingerprintFacade,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
    FacadeObjectRef,
    attachment_duplicate_fingerprint_request_carried_sha256,
    attachment_duplicate_fingerprint_request_ref,
    attachment_duplicate_fingerprint_result_ref,
    validate_attachment_duplicate_fingerprint_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.duplicate_v2 import (
    DUPLICATE_DETECTION_POLICY_VERSION,
    AttachmentDuplicateFingerprintV2,
    DuplicateClusterV2,
    DuplicateDetectionPolicyV2,
    DuplicateDetectionResultV2,
    DuplicateMatchKindV2,
    DuplicatePairEvidenceV2,
    DuplicateSubjectKindV2,
    TaskDuplicateFingerprintV2,
    attachment_duplicate_fingerprint_v2_ref,
    duplicate_cluster_v2_ref,
    duplicate_detection_policy_v2_ref,
    duplicate_pair_evidence_v2_ref,
    task_duplicate_fingerprint_v2_ref,
    validate_duplicate_detection_policy_v2_identity,
    validate_duplicate_detection_result_v2_identity,
)
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    resolved_job_work_graph_v2_ref,
    validate_resolved_job_work_graph_v2_identity,
)
from eval_factory.contracts.quality_v2 import (
    EnvironmentArtifactV2,
    ItemQualityCompilationResultV2,
    environment_artifact_ref,
    item_quality_compilation_result_carried_sha256,
    item_quality_compilation_result_ref,
)
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    TaskDraftV2,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
)

_TOKEN_PATTERN = re.compile(
    r"[\w]+(?:[._/-][\w]+)*",
    flags=re.UNICODE,
)


class DuplicateDetectionPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DuplicateDetectionItemSource:
    item_id: str
    task_draft: TaskDraftV2
    task_contract_set: R4TaskContractSetV2
    item_quality: ItemQualityCompilationResultV2


@dataclass(frozen=True, slots=True)
class _AdmittedSources:
    graph_ref: ObjectRef
    policy_ref: ObjectRef
    sources: tuple[DuplicateDetectionItemSource, ...]
    item_quality_refs: tuple[ObjectRef, ...]
    attachment_count: int
    task_pair_count: int
    attachment_pair_count: int


class DuplicateDetectionCompiler:
    async def compile(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: DuplicateDetectionPolicyV2,
        sources: tuple[DuplicateDetectionItemSource, ...],
        facade: AttachmentDuplicateFingerprintFacade,
        audit: ContractAudit,
    ) -> DuplicateDetectionResultV2:
        admitted = self._admit(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
        )
        task_fingerprints = self._task_fingerprints(
            admitted,
            policy=policy,
            audit=audit,
        )
        attachment_fingerprints: list[AttachmentDuplicateFingerprintV2] = []
        for source in admitted.sources:
            assert source.item_quality.environment_spec is not None
            for artifact in source.item_quality.environment_spec.artifacts:
                request = _attachment_request(
                    source=source,
                    artifact=artifact,
                    policy=policy,
                )
                provider_result = await facade.fingerprint(request)
                attachment_fingerprints.append(
                    _compile_attachment_fingerprint(
                        source=source,
                        artifact=artifact,
                        request=request,
                        provider_result=provider_result,
                        policy_ref=admitted.policy_ref,
                        audit=audit,
                    )
                )
        return self._compile_result(
            admitted=admitted,
            policy=policy,
            task_fingerprints=task_fingerprints,
            attachment_fingerprints=tuple(attachment_fingerprints),
            audit=audit,
        )

    def validate_current(
        self,
        result: DuplicateDetectionResultV2,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: DuplicateDetectionPolicyV2,
        sources: tuple[DuplicateDetectionItemSource, ...],
    ) -> None:
        try:
            validate_duplicate_detection_result_v2_identity(result)
        except ValueError as exc:
            raise DuplicateDetectionPolicyError("duplicate detection result identity is stale") from exc
        admitted = self._admit(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
        )
        task_fingerprints = self._task_fingerprints(
            admitted,
            policy=policy,
            audit=result.audit,
        )
        existing_by_subject = {
            value.environment_artifact_ref: value for value in result.attachment_fingerprints
        }
        attachment_fingerprints: list[AttachmentDuplicateFingerprintV2] = []
        for source in admitted.sources:
            assert source.item_quality.environment_spec is not None
            for artifact in source.item_quality.environment_spec.artifacts:
                artifact_ref = environment_artifact_ref(artifact)
                existing = existing_by_subject.get(artifact_ref)
                if existing is None:
                    raise DuplicateDetectionPolicyError(
                        "duplicate detection result omits an attachment subject"
                    )
                request = _attachment_request(
                    source=source,
                    artifact=artifact,
                    policy=policy,
                )
                if existing.facade_request_ref != _object_ref_from_facade(
                    attachment_duplicate_fingerprint_request_ref(request)
                ):
                    raise DuplicateDetectionPolicyError("attachment fingerprint request is stale")
                attachment_fingerprints.append(
                    _compile_attachment_fingerprint(
                        source=source,
                        artifact=artifact,
                        request=request,
                        provider_result=existing.facade_result,
                        policy_ref=admitted.policy_ref,
                        audit=result.audit,
                    )
                )
        if len(existing_by_subject) != len(attachment_fingerprints):
            raise DuplicateDetectionPolicyError("duplicate detection result has extra attachment subjects")
        rebuilt = self._compile_result(
            admitted=admitted,
            policy=policy,
            task_fingerprints=task_fingerprints,
            attachment_fingerprints=tuple(attachment_fingerprints),
            audit=result.audit,
        )
        if (
            rebuilt.duplicate_detection_result_id != result.duplicate_detection_result_id
            or rebuilt.result_sha256 != result.result_sha256
            or rebuilt.item_quality_result_refs != result.item_quality_result_refs
        ):
            raise DuplicateDetectionPolicyError("duplicate detection result is not current")

    def _admit(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: DuplicateDetectionPolicyV2,
        sources: tuple[DuplicateDetectionItemSource, ...],
    ) -> _AdmittedSources:
        try:
            graph = ResolvedJobWorkGraphV2.model_validate(resolved_job_work_graph.model_dump(mode="python"))
            validate_resolved_job_work_graph_v2_identity(graph)
        except (ValidationError, ValueError) as exc:
            raise DuplicateDetectionPolicyError("resolved Job work graph is stale or malformed") from exc
        try:
            parsed_policy = DuplicateDetectionPolicyV2.model_validate(policy.model_dump(mode="python"))
            validate_duplicate_detection_policy_v2_identity(parsed_policy)
        except (ValidationError, ValueError) as exc:
            raise DuplicateDetectionPolicyError("duplicate detection policy is stale or malformed") from exc
        if not sources:
            raise DuplicateDetectionPolicyError("duplicate detection requires at least one item")
        if len(sources) > policy.max_items:
            raise DuplicateDetectionPolicyError("duplicate detection item limit exceeded")
        item_ids = tuple(source.item_id for source in sources)
        if len(item_ids) != len(set(item_ids)):
            raise DuplicateDetectionPolicyError("duplicate item identity in duplicate detection sources")
        graph_item_ids = set(graph.item_ids)
        if any(item_id not in graph_item_ids for item_id in item_ids):
            raise DuplicateDetectionPolicyError("duplicate detection source belongs to another graph")

        admitted_sources = tuple(
            sorted(
                (self._admit_source(source) for source in sources),
                key=lambda source: source.item_id,
            )
        )
        task_subject_refs = tuple(task_draft_ref(source.task_draft) for source in admitted_sources)
        if len(task_subject_refs) != len(set(task_subject_refs)):
            raise DuplicateDetectionPolicyError("duplicate task subject ownership in source set")
        attachment_subject_values: list[ObjectRef] = []
        for source in admitted_sources:
            environment = source.item_quality.environment_spec
            assert environment is not None
            attachment_subject_values.extend(
                environment_artifact_ref(artifact) for artifact in environment.artifacts
            )
        attachment_subject_refs = tuple(attachment_subject_values)
        if len(attachment_subject_refs) != len(set(attachment_subject_refs)):
            raise DuplicateDetectionPolicyError("duplicate attachment subject ownership in source set")
        attachment_count = sum(
            len(source.item_quality.environment_spec.artifacts)
            for source in admitted_sources
            if source.item_quality.environment_spec is not None
        )
        if attachment_count > policy.max_attachments:
            raise DuplicateDetectionPolicyError("duplicate detection attachment limit exceeded")
        task_pair_count = len(admitted_sources) * (len(admitted_sources) - 1) // 2
        attachment_pair_count = 0
        for left_source_index, left_source in enumerate(admitted_sources):
            left_environment = left_source.item_quality.environment_spec
            assert left_environment is not None
            for right_source in admitted_sources[left_source_index + 1 :]:
                right_environment = right_source.item_quality.environment_spec
                assert right_environment is not None
                attachment_pair_count += len(left_environment.artifacts) * len(right_environment.artifacts)
        if task_pair_count + attachment_pair_count > policy.max_pair_comparisons:
            raise DuplicateDetectionPolicyError("duplicate detection pair budget exceeded")
        item_quality_refs = _sorted_refs(
            tuple(item_quality_compilation_result_ref(source.item_quality) for source in admitted_sources)
        )
        return _AdmittedSources(
            graph_ref=resolved_job_work_graph_v2_ref(graph),
            policy_ref=duplicate_detection_policy_v2_ref(parsed_policy),
            sources=admitted_sources,
            item_quality_refs=item_quality_refs,
            attachment_count=attachment_count,
            task_pair_count=task_pair_count,
            attachment_pair_count=attachment_pair_count,
        )

    def _admit_source(
        self,
        source: DuplicateDetectionItemSource,
    ) -> DuplicateDetectionItemSource:
        try:
            draft = TaskDraftV2.model_validate(source.task_draft.model_dump(mode="python"))
        except ValidationError as exc:
            raise DuplicateDetectionPolicyError("TaskDraft is malformed") from exc
        draft_digest = task_draft_carried_sha256(draft)
        if (
            draft.task_draft_sha256 != draft_digest
            or draft.task_draft_id != f"task-draft://sha256/{draft_digest}"
        ):
            raise DuplicateDetectionPolicyError("TaskDraft identity is stale")
        try:
            contract_set = R4TaskContractSetV2.model_validate(
                source.task_contract_set.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise DuplicateDetectionPolicyError("R4 task contract set is malformed") from exc
        contract_digest = r4_task_contract_set_carried_sha256(contract_set)
        if (
            contract_set.contract_set_sha256 != contract_digest
            or contract_set.contract_set_id != f"r4-task-contract-set://sha256/{contract_digest}"
            or contract_set.task_draft_ref != task_draft_ref(draft)
            or contract_set.task_prompt_safety_gate_ref != draft.prompt_safety_gate_ref
        ):
            raise DuplicateDetectionPolicyError("R4 task contract set is stale or cross-task")
        try:
            item_quality = ItemQualityCompilationResultV2.model_validate(
                source.item_quality.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise DuplicateDetectionPolicyError("Item quality result is malformed") from exc
        quality_digest = item_quality_compilation_result_carried_sha256(item_quality)
        if (
            item_quality.result_sha256 != quality_digest
            or item_quality.item_quality_compilation_result_id
            != f"item-quality-compilation-result://sha256/{quality_digest}"
            or not item_quality.quality_report.approvable
            or item_quality.environment_spec is None
            or item_quality.final_package_manifest is None
            or item_quality.provenance_manifest is None
            or item_quality.input_state_only is not True
            or item_quality.quality_report.r4_task_contract_set_ref != r4_task_contract_set_ref(contract_set)
        ):
            raise DuplicateDetectionPolicyError("Item quality result is non-approvable or stale")
        return DuplicateDetectionItemSource(
            item_id=source.item_id,
            task_draft=draft,
            task_contract_set=contract_set,
            item_quality=item_quality,
        )

    def _task_fingerprints(
        self,
        admitted: _AdmittedSources,
        *,
        policy: DuplicateDetectionPolicyV2,
        audit: ContractAudit,
    ) -> tuple[TaskDuplicateFingerprintV2, ...]:
        values = tuple(
            _task_fingerprint(
                source=source,
                policy=policy,
                policy_ref=admitted.policy_ref,
                audit=audit,
            )
            for source in admitted.sources
        )
        return tuple(
            sorted(
                values,
                key=lambda value: _ref_key(task_duplicate_fingerprint_v2_ref(value)),
            )
        )

    def _compile_result(
        self,
        *,
        admitted: _AdmittedSources,
        policy: DuplicateDetectionPolicyV2,
        task_fingerprints: tuple[TaskDuplicateFingerprintV2, ...],
        attachment_fingerprints: tuple[
            AttachmentDuplicateFingerprintV2,
            ...,
        ],
        audit: ContractAudit,
    ) -> DuplicateDetectionResultV2:
        pairs = (
            *_task_pairs(
                task_fingerprints,
                policy=policy,
                policy_ref=admitted.policy_ref,
                audit=audit,
            ),
            *_attachment_pairs(
                attachment_fingerprints,
                policy=policy,
                policy_ref=admitted.policy_ref,
                audit=audit,
            ),
        )
        pairs = tuple(
            sorted(
                pairs,
                key=lambda value: _ref_key(duplicate_pair_evidence_v2_ref(value)),
            )
        )
        clusters = (
            *_clusters(
                subject_kind=DuplicateSubjectKindV2.TASK,
                pairs=pairs,
                task_fingerprints=task_fingerprints,
                attachment_fingerprints=(),
                policy_ref=admitted.policy_ref,
                audit=audit,
            ),
            *_clusters(
                subject_kind=DuplicateSubjectKindV2.ATTACHMENT,
                pairs=pairs,
                task_fingerprints=(),
                attachment_fingerprints=attachment_fingerprints,
                policy_ref=admitted.policy_ref,
                audit=audit,
            ),
        )
        unsupported_refs = tuple(
            value.environment_artifact_ref
            for value in attachment_fingerprints
            if value.facade_result.outcome is AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED
        )
        result_refs = (
            admitted.graph_ref,
            admitted.policy_ref,
            *admitted.item_quality_refs,
            *(task_duplicate_fingerprint_v2_ref(value) for value in task_fingerprints),
            *(attachment_duplicate_fingerprint_v2_ref(value) for value in attachment_fingerprints),
            *unsupported_refs,
            *(duplicate_pair_evidence_v2_ref(value) for value in pairs),
            *(duplicate_cluster_v2_ref(value) for value in clusters),
        )
        return DuplicateDetectionResultV2.create(
            resolved_job_work_graph_ref=admitted.graph_ref,
            policy_ref=admitted.policy_ref,
            item_quality_result_refs=admitted.item_quality_refs,
            task_fingerprints=task_fingerprints,
            attachment_fingerprints=attachment_fingerprints,
            duplicate_pairs=pairs,
            duplicate_clusters=clusters,
            audit=_safe_audit(audit, result_refs),
        )


def _task_fingerprint(
    *,
    source: DuplicateDetectionItemSource,
    policy: DuplicateDetectionPolicyV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> TaskDuplicateFingerprintV2:
    prompt = source.task_draft.visible_prompt
    if len(prompt) > policy.max_task_characters:
        raise DuplicateDetectionPolicyError("TaskDraft exceeds duplicate normalization character limit")
    prompt_tokens = _normalize_tokens(prompt)
    shape_tokens = _task_shape_tokens(source.task_draft)
    tokens = (*prompt_tokens, *shape_tokens)
    if len(tokens) < policy.min_task_tokens or len(tokens) < policy.task_shingle_size:
        raise DuplicateDetectionPolicyError("TaskDraft has insufficient normalized content")
    exact_digest = _payload_sha256(
        {
            "prompt_tokens": prompt_tokens,
            "public_shape_tokens": shape_tokens,
            "normalization_version": (policy.task_normalization_version),
        }
    )
    similarity, shingle_count = _simhash(
        tokens,
        shingle_size=policy.task_shingle_size,
    )
    draft_ref = task_draft_ref(source.task_draft)
    contract_ref = r4_task_contract_set_ref(source.task_contract_set)
    return TaskDuplicateFingerprintV2.create(
        item_id=source.item_id,
        task_draft_ref=draft_ref,
        r4_task_contract_set_ref=contract_ref,
        exact_task_sha256=exact_digest,
        similarity_fingerprint=similarity,
        token_count=len(tokens),
        shingle_count=shingle_count,
        policy_ref=policy_ref,
        audit=_safe_audit(
            audit,
            (draft_ref, contract_ref, policy_ref),
        ),
    )


def _task_shape_tokens(draft: TaskDraftV2) -> tuple[str, ...]:
    values = [
        *(f"capability={_normalize_identifier(value)}" for value in sorted(draft.required_capabilities)),
        *(f"tool={_normalize_identifier(value)}" for value in sorted(draft.allowed_tools)),
    ]
    counts = Counter(
        (
            dependency.criticality.value,
            dependency.evidence_priority.value,
        )
        for dependency in draft.attachment_dependencies
    )
    values.extend(
        (f"attachment-shape={criticality.casefold()}/{priority.casefold()}/{count}")
        for (criticality, priority), count in sorted(counts.items())
    )
    return tuple(values)


def _attachment_request(
    *,
    source: DuplicateDetectionItemSource,
    artifact: EnvironmentArtifactV2,
    policy: DuplicateDetectionPolicyV2,
) -> AttachmentDuplicateFingerprintRequestV2:
    quality_ref = item_quality_compilation_result_ref(source.item_quality)
    artifact_ref = environment_artifact_ref(artifact)
    identity_seed = _payload_sha256(
        {
            "item_quality_result_ref": _ref_payload(quality_ref),
            "environment_artifact_ref": _ref_payload(artifact_ref),
            "policy_ref": _ref_payload(duplicate_detection_policy_v2_ref(policy)),
        }
    )
    pending = AttachmentDuplicateFingerprintRequestV2(
        fingerprint_request_id=("attachment-duplicate-fingerprint-request://pending"),
        item_quality_result_ref=_facade_ref(quality_ref),
        environment_artifact_ref=_facade_ref(artifact_ref),
        candidate_artifact_version_ref=_facade_ref(artifact.candidate_artifact_version_ref),
        output_ref=_facade_ref(artifact.output_ref),
        artifact_validation_result_ref=_facade_ref(artifact.artifact_validation_result_ref),
        logical_path=artifact.logical_path,
        media_type=artifact.media_type,
        content_sha256=artifact.content_sha256,
        size_bytes=artifact.size_bytes,
        limits=policy.attachment_fingerprint_limits,
        normalization_version=(ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION),
        policy_version=(ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION),
        idempotency_key=(f"attachment-duplicate-fingerprint-idempotency://sha256/{identity_seed}"),
        request_sha256="0" * 64,
    )
    digest = attachment_duplicate_fingerprint_request_carried_sha256(pending)
    return pending.model_copy(
        update={
            "fingerprint_request_id": (f"attachment-duplicate-fingerprint-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _compile_attachment_fingerprint(
    *,
    source: DuplicateDetectionItemSource,
    artifact: EnvironmentArtifactV2,
    request: AttachmentDuplicateFingerprintRequestV2,
    provider_result: AttachmentDuplicateFingerprintResultV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> AttachmentDuplicateFingerprintV2:
    try:
        validate_attachment_duplicate_fingerprint_result_identity(provider_result)
    except ValueError as exc:
        raise DuplicateDetectionPolicyError("attachment fingerprint facade result is stale") from exc
    if (
        provider_result.fingerprint_request_ref != attachment_duplicate_fingerprint_request_ref(request)
        or provider_result.environment_artifact_ref != request.environment_artifact_ref
        or provider_result.output_ref != request.output_ref
        or provider_result.exact_content_sha256 != request.content_sha256
        or provider_result.normalization_version != request.normalization_version
        or provider_result.policy_version != request.policy_version
    ):
        raise DuplicateDetectionPolicyError("attachment fingerprint facade result is cross-subject")
    if provider_result.outcome is AttachmentDuplicateFingerprintOutcomeV2.BLOCKED:
        raise DuplicateDetectionPolicyError("attachment fingerprint facade blocked the request")
    quality_ref = item_quality_compilation_result_ref(source.item_quality)
    artifact_ref = environment_artifact_ref(artifact)
    request_ref = _object_ref_from_facade(attachment_duplicate_fingerprint_request_ref(request))
    result_ref = _object_ref_from_facade(attachment_duplicate_fingerprint_result_ref(provider_result))
    refs = (
        quality_ref,
        artifact_ref,
        artifact.candidate_artifact_version_ref,
        artifact.output_ref,
        artifact.artifact_validation_result_ref,
        request_ref,
        result_ref,
        policy_ref,
    )
    return AttachmentDuplicateFingerprintV2.create(
        item_id=source.item_id,
        item_quality_result_ref=quality_ref,
        environment_artifact_ref=artifact_ref,
        candidate_artifact_version_ref=(artifact.candidate_artifact_version_ref),
        output_ref=artifact.output_ref,
        artifact_validation_result_ref=(artifact.artifact_validation_result_ref),
        facade_request_ref=request_ref,
        facade_result=provider_result,
        facade_result_ref=result_ref,
        policy_ref=policy_ref,
        audit=_safe_audit(audit, refs),
    )


def _task_pairs(
    values: tuple[TaskDuplicateFingerprintV2, ...],
    *,
    policy: DuplicateDetectionPolicyV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[DuplicatePairEvidenceV2, ...]:
    pairs: list[DuplicatePairEvidenceV2] = []
    for left, right in combinations(values, 2):
        if left.exact_task_sha256 == right.exact_task_sha256:
            match_kind = DuplicateMatchKindV2.EXACT
            similarity_bps = 10_000
        else:
            similarity_bps = _similarity_bps(
                left.similarity_fingerprint,
                right.similarity_fingerprint,
            )
            if similarity_bps < policy.task_near_threshold_bps:
                continue
            match_kind = DuplicateMatchKindV2.NEAR
        pairs.append(
            _pair(
                subject_kind=DuplicateSubjectKindV2.TASK,
                left_item_id=left.item_id,
                right_item_id=right.item_id,
                left_subject_ref=left.task_draft_ref,
                right_subject_ref=right.task_draft_ref,
                left_fingerprint_ref=(task_duplicate_fingerprint_v2_ref(left)),
                right_fingerprint_ref=(task_duplicate_fingerprint_v2_ref(right)),
                match_kind=match_kind,
                similarity_bps=similarity_bps,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    return tuple(pairs)


def _attachment_pairs(
    values: tuple[AttachmentDuplicateFingerprintV2, ...],
    *,
    policy: DuplicateDetectionPolicyV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[DuplicatePairEvidenceV2, ...]:
    pairs: list[DuplicatePairEvidenceV2] = []
    for left, right in combinations(values, 2):
        if left.item_id == right.item_id:
            continue
        if left.facade_result.exact_content_sha256 == right.facade_result.exact_content_sha256:
            match_kind = DuplicateMatchKindV2.EXACT
            similarity_bps = 10_000
        else:
            left_similarity = left.facade_result.similarity_fingerprint
            right_similarity = right.facade_result.similarity_fingerprint
            if left_similarity is None or right_similarity is None:
                continue
            similarity_bps = _similarity_bps(
                left_similarity,
                right_similarity,
            )
            if similarity_bps < policy.attachment_near_threshold_bps:
                continue
            match_kind = DuplicateMatchKindV2.NEAR
        pairs.append(
            _pair(
                subject_kind=DuplicateSubjectKindV2.ATTACHMENT,
                left_item_id=left.item_id,
                right_item_id=right.item_id,
                left_subject_ref=left.environment_artifact_ref,
                right_subject_ref=right.environment_artifact_ref,
                left_fingerprint_ref=(attachment_duplicate_fingerprint_v2_ref(left)),
                right_fingerprint_ref=(attachment_duplicate_fingerprint_v2_ref(right)),
                match_kind=match_kind,
                similarity_bps=similarity_bps,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    return tuple(pairs)


def _pair(
    *,
    subject_kind: DuplicateSubjectKindV2,
    left_item_id: str,
    right_item_id: str,
    left_subject_ref: ObjectRef,
    right_subject_ref: ObjectRef,
    left_fingerprint_ref: ObjectRef,
    right_fingerprint_ref: ObjectRef,
    match_kind: DuplicateMatchKindV2,
    similarity_bps: int,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> DuplicatePairEvidenceV2:
    refs = (
        left_subject_ref,
        right_subject_ref,
        left_fingerprint_ref,
        right_fingerprint_ref,
        policy_ref,
    )
    return DuplicatePairEvidenceV2.create(
        subject_kind=subject_kind,
        left_item_id=left_item_id,
        right_item_id=right_item_id,
        left_subject_ref=left_subject_ref,
        right_subject_ref=right_subject_ref,
        left_fingerprint_ref=left_fingerprint_ref,
        right_fingerprint_ref=right_fingerprint_ref,
        match_kind=match_kind,
        similarity_bps=similarity_bps,
        policy_ref=policy_ref,
        audit=_safe_audit(audit, refs),
    )


def _clusters(
    *,
    subject_kind: DuplicateSubjectKindV2,
    pairs: tuple[DuplicatePairEvidenceV2, ...],
    task_fingerprints: tuple[TaskDuplicateFingerprintV2, ...],
    attachment_fingerprints: tuple[
        AttachmentDuplicateFingerprintV2,
        ...,
    ],
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[DuplicateClusterV2, ...]:
    kind_pairs = tuple(pair for pair in pairs if pair.subject_kind is subject_kind)
    if not kind_pairs:
        return ()
    item_by_subject = (
        {value.task_draft_ref: value.item_id for value in task_fingerprints}
        if subject_kind is DuplicateSubjectKindV2.TASK
        else {value.environment_artifact_ref: value.item_id for value in attachment_fingerprints}
    )
    adjacency: dict[ObjectRef, set[ObjectRef]] = {}
    for pair in kind_pairs:
        adjacency.setdefault(pair.left_subject_ref, set()).add(pair.right_subject_ref)
        adjacency.setdefault(pair.right_subject_ref, set()).add(pair.left_subject_ref)
    remaining = set(adjacency)
    clusters: list[DuplicateClusterV2] = []
    while remaining:
        root = min(remaining, key=_ref_key)
        stack = [root]
        members: set[ObjectRef] = set()
        while stack:
            subject = stack.pop()
            if subject in members:
                continue
            members.add(subject)
            stack.extend(
                sorted(
                    adjacency.get(subject, ()),
                    key=_ref_key,
                    reverse=True,
                )
            )
        remaining.difference_update(members)
        member_refs = tuple(sorted(members, key=_ref_key))
        direct_pairs = tuple(
            pair
            for pair in kind_pairs
            if pair.left_subject_ref in members and pair.right_subject_ref in members
        )
        pair_refs = tuple(duplicate_pair_evidence_v2_ref(pair) for pair in direct_pairs)
        exact_count = sum(pair.match_kind is DuplicateMatchKindV2.EXACT for pair in direct_pairs)
        cluster_refs = (*member_refs, *pair_refs, policy_ref)
        clusters.append(
            DuplicateClusterV2.create(
                subject_kind=subject_kind,
                member_item_ids=tuple(item_by_subject[ref] for ref in member_refs),
                member_subject_refs=member_refs,
                direct_pair_refs=pair_refs,
                exact_pair_count=exact_count,
                near_pair_count=len(direct_pairs) - exact_count,
                policy_ref=policy_ref,
                audit=_safe_audit(audit, cluster_refs),
            )
        )
    return tuple(
        sorted(
            clusters,
            key=lambda value: _ref_key(duplicate_cluster_v2_ref(value)),
        )
    )


def _normalize_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(normalized))


def _normalize_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _simhash(
    tokens: tuple[str, ...],
    *,
    shingle_size: int,
) -> tuple[str, int]:
    shingle_count = len(tokens) - shingle_size + 1
    weights = [0] * 256
    for index in range(shingle_count):
        shingle = "\x1f".join(tokens[index : index + shingle_size])
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        for bit_index in range(256):
            byte = digest[bit_index // 8]
            mask = 1 << (7 - bit_index % 8)
            weights[bit_index] += 1 if byte & mask else -1
    value = 0
    for weight in weights:
        value = (value << 1) | int(weight >= 0)
    return value.to_bytes(32, byteorder="big").hex(), shingle_count


def _similarity_bps(left: str, right: str) -> int:
    distance = (int(left, 16) ^ int(right, 16)).bit_count()
    return (256 - distance) * 10_000 // 256


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(item for item in audit.governing_versions if item.component != "duplicate-detection"),
        VersionBinding(
            component="duplicate-detection",
            version=DUPLICATE_DETECTION_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": governing,
            "input_refs": _sorted_refs(refs),
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


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _facade_ref(value: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _object_ref_from_facade(value: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))
