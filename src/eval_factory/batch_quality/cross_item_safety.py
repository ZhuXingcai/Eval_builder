from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from pydantic import ValidationError

from env_mock_agent.facade import (
    ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
    AttachmentCrossItemSafetyScanFacade,
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    AttachmentValidationFingerprintCategoryV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
    FacadeObjectRef,
    attachment_cross_item_safety_scan_request_carried_sha256,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_ref,
    validate_attachment_cross_item_safety_scan_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.cross_item_safety_v2 import (
    CROSS_ITEM_SAFETY_POLICY_VERSION,
    AnswerReuseMatchKindV2,
    AnswerReusePairEvidenceV2,
    AttachmentCrossItemSafetyScanEvidenceV2,
    CrossItemSafetyClusterKindV2,
    CrossItemSafetyClusterV2,
    CrossItemSafetyEvidenceKindV2,
    CrossItemSafetyPolicyV2,
    CrossItemSafetyResultV2,
    CrossItemSafetySurfaceKindV2,
    CrossItemVisibleMatchEvidenceV2,
    answer_reuse_pair_evidence_v2_ref,
    attachment_cross_item_safety_scan_evidence_v2_ref,
    cross_item_safety_cluster_v2_ref,
    cross_item_safety_policy_v2_ref,
    cross_item_visible_match_evidence_v2_ref,
    validate_cross_item_safety_policy_v2_identity,
    validate_cross_item_safety_result_v2_identity,
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
    TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2,
    PromptLeakageCategoryV2,
    PromptLeakageFingerprintV2,
    PromptLeakageMatchKindV2,
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyGateStatusV2,
    TaskPromptSafetyGateV2,
    prompt_leakage_fingerprint_ref,
    prompt_leakage_reference_set_ref,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    task_prompt_safety_gate_ref,
    validate_prompt_leakage_fingerprint_identity,
)
from eval_factory.task_authoring.leakage_fingerprints import (
    match_prompt_leakage_fingerprints,
    normalize_prompt_leakage_tokens,
)
from eval_factory.task_authoring.prompt_safety import (
    validate_prompt_leakage_reference_set_identity,
    validate_task_prompt_safety_gate_identity,
)
from eval_factory.task_authoring.prompt_safety_models import (
    PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION,
    TASK_PROMPT_SAFETY_POLICY_VERSION,
)

_ANSWER_REUSE_CATEGORIES = frozenset(
    {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
        PromptLeakageCategoryV2.PRIVATE_REFERENCE,
    }
)
_CONTAMINATION_CATEGORIES = frozenset(
    {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
    }
)


class CrossItemSafetyPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CrossItemSafetyItemSource:
    item_id: str
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    leakage_reference_set: PromptLeakageReferenceSetV2
    task_contract_set: R4TaskContractSetV2
    item_quality: ItemQualityCompilationResultV2


@dataclass(frozen=True, slots=True)
class _FingerprintOwner:
    item_id: str
    reference_set_ref: ObjectRef
    fingerprint: PromptLeakageFingerprintV2


@dataclass(frozen=True, slots=True)
class _AnswerSource:
    item_id: str
    reference_set_ref: ObjectRef
    source_subject_ref: ObjectRef
    full_fingerprints: tuple[PromptLeakageFingerprintV2, ...]
    window_fingerprints: tuple[PromptLeakageFingerprintV2, ...]


@dataclass(frozen=True, slots=True)
class _AdmittedSources:
    graph_ref: ObjectRef
    policy_ref: ObjectRef
    sources: tuple[CrossItemSafetyItemSource, ...]
    fingerprint_owners: dict[str, _FingerprintOwner]
    answer_sources: tuple[_AnswerSource, ...]
    item_quality_refs: tuple[ObjectRef, ...]
    attachment_count: int
    prompt_foreign_set_count: int
    attachment_foreign_set_count: int
    answer_source_pair_count: int


class CrossItemSafetyCompiler:
    async def compile(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: CrossItemSafetyPolicyV2,
        sources: tuple[CrossItemSafetyItemSource, ...],
        facade: AttachmentCrossItemSafetyScanFacade,
        audit: ContractAudit,
    ) -> CrossItemSafetyResultV2:
        admitted = self._admit(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
        )
        scans: list[AttachmentCrossItemSafetyScanEvidenceV2] = []
        for source in admitted.sources:
            environment = source.item_quality.environment_spec
            assert environment is not None
            for artifact in environment.artifacts:
                request = _attachment_request(
                    source=source,
                    artifact=artifact,
                    admitted=admitted,
                    policy=policy,
                )
                provider_result = await facade.scan(request)
                scans.append(
                    _compile_attachment_scan(
                        source=source,
                        artifact=artifact,
                        request=request,
                        provider_result=provider_result,
                        admitted=admitted,
                        audit=audit,
                    )
                )
        return self._compile_result(
            admitted=admitted,
            policy=policy,
            attachment_scans=tuple(scans),
            audit=audit,
        )

    def validate_current(
        self,
        result: CrossItemSafetyResultV2,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: CrossItemSafetyPolicyV2,
        sources: tuple[CrossItemSafetyItemSource, ...],
    ) -> None:
        try:
            validate_cross_item_safety_result_v2_identity(result)
        except ValueError as exc:
            raise CrossItemSafetyPolicyError("cross-item safety result identity is stale") from exc
        admitted = self._admit(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
        )
        existing_by_subject = {
            value.environment_artifact_ref: value for value in result.attachment_scan_evidence
        }
        scans: list[AttachmentCrossItemSafetyScanEvidenceV2] = []
        for source in admitted.sources:
            environment = source.item_quality.environment_spec
            assert environment is not None
            for artifact in environment.artifacts:
                artifact_ref = environment_artifact_ref(artifact)
                existing = existing_by_subject.get(artifact_ref)
                if existing is None:
                    raise CrossItemSafetyPolicyError("cross-item safety result omits an attachment scan")
                request = _attachment_request(
                    source=source,
                    artifact=artifact,
                    admitted=admitted,
                    policy=policy,
                )
                if existing.facade_request_ref != _object_ref_from_facade(
                    attachment_cross_item_safety_scan_request_ref(request)
                ):
                    raise CrossItemSafetyPolicyError("attachment safety scan request is stale")
                scans.append(
                    _compile_attachment_scan(
                        source=source,
                        artifact=artifact,
                        request=request,
                        provider_result=existing.facade_result,
                        admitted=admitted,
                        audit=result.audit,
                    )
                )
        if len(existing_by_subject) != len(scans):
            raise CrossItemSafetyPolicyError("cross-item safety result has extra attachment scans")
        rebuilt = self._compile_result(
            admitted=admitted,
            policy=policy,
            attachment_scans=tuple(scans),
            audit=result.audit,
        )
        if (
            rebuilt.cross_item_safety_result_id != result.cross_item_safety_result_id
            or rebuilt.result_sha256 != result.result_sha256
        ):
            raise CrossItemSafetyPolicyError("cross-item safety result is not current")

    def _admit(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: CrossItemSafetyPolicyV2,
        sources: tuple[CrossItemSafetyItemSource, ...],
    ) -> _AdmittedSources:
        try:
            graph = ResolvedJobWorkGraphV2.model_validate(resolved_job_work_graph.model_dump(mode="python"))
            validate_resolved_job_work_graph_v2_identity(graph)
        except (ValidationError, ValueError) as exc:
            raise CrossItemSafetyPolicyError("resolved Job work graph is stale or malformed") from exc
        try:
            parsed_policy = CrossItemSafetyPolicyV2.model_validate(policy.model_dump(mode="python"))
            validate_cross_item_safety_policy_v2_identity(parsed_policy)
        except (ValidationError, ValueError) as exc:
            raise CrossItemSafetyPolicyError("cross-item safety policy is stale or malformed") from exc
        if len(sources) < 2:
            raise CrossItemSafetyPolicyError("cross-item safety requires at least two Items")
        if len(sources) > policy.max_items:
            raise CrossItemSafetyPolicyError("cross-item safety Item limit exceeded")
        item_ids = tuple(source.item_id for source in sources)
        if len(item_ids) != len(set(item_ids)):
            raise CrossItemSafetyPolicyError("duplicate Item identity in cross-item safety sources")
        if any(item_id not in set(graph.item_ids) for item_id in item_ids):
            raise CrossItemSafetyPolicyError("cross-item safety source belongs to another graph")
        admitted_sources = tuple(
            sorted(
                (
                    self._admit_source(
                        source,
                        policy=parsed_policy,
                    )
                    for source in sources
                ),
                key=lambda source: source.item_id,
            )
        )
        self._validate_subject_ownership(admitted_sources)
        if any(
            len(source.task_draft.visible_prompt) > parsed_policy.max_prompt_characters
            for source in admitted_sources
        ):
            raise CrossItemSafetyPolicyError("cross-item safety prompt character limit exceeded")
        attachment_count = sum(
            len(source.item_quality.environment_spec.artifacts)
            for source in admitted_sources
            if source.item_quality.environment_spec is not None
        )
        if attachment_count > policy.max_attachments:
            raise CrossItemSafetyPolicyError("cross-item safety attachment limit exceeded")
        fingerprint_owners = _fingerprint_owners(admitted_sources)
        if len(fingerprint_owners) > policy.max_total_fingerprints:
            raise CrossItemSafetyPolicyError("cross-item safety fingerprint limit exceeded")
        for source in admitted_sources:
            foreign_count = sum(
                len(candidate.leakage_reference_set.fingerprints)
                for candidate in admitted_sources
                if candidate.item_id != source.item_id
            )
            if foreign_count > policy.max_foreign_fingerprints_per_target:
                raise CrossItemSafetyPolicyError("foreign fingerprint target limit exceeded")
        prompt_scan_comparisons = _prompt_scan_comparison_count(admitted_sources)
        if prompt_scan_comparisons > policy.max_prompt_scan_comparisons:
            raise CrossItemSafetyPolicyError("cross-item prompt scan budget exceeded")
        answer_sources = _answer_sources(admitted_sources, policy=policy)
        answer_pairs = _answer_source_pair_count(answer_sources)
        if answer_pairs > policy.max_answer_source_pairs:
            raise CrossItemSafetyPolicyError("answer source pair budget exceeded")
        if answer_pairs > policy.max_reuse_pairs:
            raise CrossItemSafetyPolicyError("answer reuse pair budget exceeded")
        answer_window_comparisons = _answer_window_comparison_count(answer_sources)
        if answer_window_comparisons > policy.max_answer_window_comparisons:
            raise CrossItemSafetyPolicyError("answer window comparison budget exceeded")
        visible_match_upper_bound = _visible_match_upper_bound(admitted_sources)
        if visible_match_upper_bound > policy.max_visible_matches:
            raise CrossItemSafetyPolicyError("cross-item visible match budget exceeded")
        if visible_match_upper_bound + answer_pairs > policy.max_clusters:
            raise CrossItemSafetyPolicyError("cross-item safety cluster budget exceeded")
        item_quality_refs = tuple(
            item_quality_compilation_result_ref(source.item_quality) for source in admitted_sources
        )
        item_count = len(admitted_sources)
        return _AdmittedSources(
            graph_ref=resolved_job_work_graph_v2_ref(graph),
            policy_ref=cross_item_safety_policy_v2_ref(parsed_policy),
            sources=admitted_sources,
            fingerprint_owners=fingerprint_owners,
            answer_sources=answer_sources,
            item_quality_refs=item_quality_refs,
            attachment_count=attachment_count,
            prompt_foreign_set_count=item_count * (item_count - 1),
            attachment_foreign_set_count=attachment_count * (item_count - 1),
            answer_source_pair_count=answer_pairs,
        )

    def _admit_source(
        self,
        source: CrossItemSafetyItemSource,
        *,
        policy: CrossItemSafetyPolicyV2,
    ) -> CrossItemSafetyItemSource:
        try:
            draft = TaskDraftV2.model_validate(source.task_draft.model_dump(mode="python"))
            gate = TaskPromptSafetyGateV2.model_validate(
                source.task_prompt_safety_gate.model_dump(mode="python")
            )
            reference_set = PromptLeakageReferenceSetV2.model_validate(
                source.leakage_reference_set.model_dump(mode="python")
            )
            contract_set = R4TaskContractSetV2.model_validate(
                source.task_contract_set.model_dump(mode="python")
            )
            item_quality = ItemQualityCompilationResultV2.model_validate(
                source.item_quality.model_dump(mode="python")
            )
        except ValidationError as exc:
            raise CrossItemSafetyPolicyError("cross-item safety source is malformed") from exc
        draft_digest = task_draft_carried_sha256(draft)
        if (
            draft.task_draft_sha256 != draft_digest
            or draft.task_draft_id != f"task-draft://sha256/{draft_digest}"
        ):
            raise CrossItemSafetyPolicyError("TaskDraft identity is stale")
        try:
            validate_task_prompt_safety_gate_identity(gate)
            validate_prompt_leakage_reference_set_identity(reference_set)
            for fingerprint in reference_set.fingerprints:
                validate_prompt_leakage_fingerprint_identity(fingerprint)
        except ValueError as exc:
            raise CrossItemSafetyPolicyError("prompt safety source identity is stale") from exc
        if (
            gate.policy_version != TASK_PROMPT_SAFETY_POLICY_VERSION
            or reference_set.fingerprint_policy_version != PROMPT_LEAKAGE_FINGERPRINT_POLICY_VERSION
            or any(
                fingerprint.normalization_version != policy.normalization_version
                for fingerprint in reference_set.fingerprints
            )
        ):
            raise CrossItemSafetyPolicyError("prompt safety source policy version is stale")
        if (
            draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
            or gate.status is not TaskPromptSafetyGateStatusV2.PASSED
            or draft.prompt_safety_gate_ref != task_prompt_safety_gate_ref(gate)
            or draft.supersedes_task_draft_ref != gate.source_task_draft_ref
            or gate.leakage_reference_set_ref != prompt_leakage_reference_set_ref(reference_set)
            or gate.visible_prompt_sha256 != hashlib.sha256(draft.visible_prompt.encode()).hexdigest()
        ):
            raise CrossItemSafetyPolicyError("TaskDraft prompt safety chain is stale or cross-task")
        if reference_set.complete_categories != (TASK_PROMPT_SAFETY_REQUIRED_CATEGORIES_V2):
            raise CrossItemSafetyPolicyError("cross-item safety requires complete leakage categories")
        contract_digest = r4_task_contract_set_carried_sha256(contract_set)
        if (
            contract_set.contract_set_sha256 != contract_digest
            or contract_set.contract_set_id != f"r4-task-contract-set://sha256/{contract_digest}"
            or contract_set.task_draft_ref != task_draft_ref(draft)
            or contract_set.task_prompt_safety_gate_ref != task_prompt_safety_gate_ref(gate)
        ):
            raise CrossItemSafetyPolicyError("R4 task contract set is stale or cross-task")
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
            raise CrossItemSafetyPolicyError("Item quality result is non-approvable or stale")
        return CrossItemSafetyItemSource(
            item_id=source.item_id,
            task_draft=draft,
            task_prompt_safety_gate=gate,
            leakage_reference_set=reference_set,
            task_contract_set=contract_set,
            item_quality=item_quality,
        )

    def _validate_subject_ownership(
        self,
        sources: tuple[CrossItemSafetyItemSource, ...],
    ) -> None:
        _require_unique_refs(
            "TaskDraft ownership",
            tuple(task_draft_ref(source.task_draft) for source in sources),
        )
        _require_unique_refs(
            "leakage reference-set ownership",
            tuple(prompt_leakage_reference_set_ref(source.leakage_reference_set) for source in sources),
        )
        restricted_subject_owners: dict[ObjectRef, set[str]] = {}
        for source in sources:
            for fingerprint in source.leakage_reference_set.fingerprints:
                restricted_subject_owners.setdefault(
                    fingerprint.source_subject_ref,
                    set(),
                ).add(source.item_id)
        if any(len(item_ids) != 1 for item_ids in restricted_subject_owners.values()):
            raise CrossItemSafetyPolicyError("restricted source ownership crosses Items")
        attachment_refs: list[ObjectRef] = []
        output_refs: list[ObjectRef] = []
        for source in sources:
            environment = source.item_quality.environment_spec
            assert environment is not None
            attachment_refs.extend(environment_artifact_ref(artifact) for artifact in environment.artifacts)
            output_refs.extend(artifact.output_ref for artifact in environment.artifacts)
        _require_unique_refs(
            "environment artifact ownership",
            tuple(attachment_refs),
        )
        _require_unique_refs("attachment output ownership", tuple(output_refs))

    def _compile_result(
        self,
        *,
        admitted: _AdmittedSources,
        policy: CrossItemSafetyPolicyV2,
        attachment_scans: tuple[
            AttachmentCrossItemSafetyScanEvidenceV2,
            ...,
        ],
        audit: ContractAudit,
    ) -> CrossItemSafetyResultV2:
        prompt_matches = _prompt_visible_matches(
            admitted,
            audit=audit,
        )
        attachment_matches = tuple(
            evidence
            for scan in attachment_scans
            for evidence in _attachment_visible_matches(
                scan,
                admitted=admitted,
                audit=audit,
            )
        )
        visible_matches = _sorted_nested(
            (*prompt_matches, *attachment_matches),
            cross_item_visible_match_evidence_v2_ref,
        )
        answer_pairs = _answer_reuse_pairs(
            admitted.answer_sources,
            policy=policy,
            policy_ref=admitted.policy_ref,
            audit=audit,
        )
        if len(visible_matches) > policy.max_visible_matches:
            raise CrossItemSafetyPolicyError("cross-item visible match budget exceeded")
        if len(answer_pairs) > policy.max_reuse_pairs:
            raise CrossItemSafetyPolicyError("answer reuse pair budget exceeded")
        clusters = _clusters(
            visible_matches=visible_matches,
            answer_pairs=answer_pairs,
            policy_ref=admitted.policy_ref,
            audit=audit,
        )
        if len(clusters) > policy.max_clusters:
            raise CrossItemSafetyPolicyError("cross-item safety cluster budget exceeded")
        answer_bindings = tuple(
            sorted(
                ((source.item_id, source.source_subject_ref) for source in admitted.answer_sources),
                key=lambda value: (value[0], _ref_key(value[1])),
            )
        )
        refs = (
            admitted.graph_ref,
            admitted.policy_ref,
            *(task_draft_ref(source.task_draft) for source in admitted.sources),
            *(prompt_leakage_reference_set_ref(source.leakage_reference_set) for source in admitted.sources),
            *admitted.item_quality_refs,
            *(value[1] for value in answer_bindings),
            *(attachment_cross_item_safety_scan_evidence_v2_ref(value) for value in attachment_scans),
            *(cross_item_visible_match_evidence_v2_ref(value) for value in visible_matches),
            *(answer_reuse_pair_evidence_v2_ref(value) for value in answer_pairs),
            *(cross_item_safety_cluster_v2_ref(value) for value in clusters),
        )
        return CrossItemSafetyResultV2.create(
            resolved_job_work_graph_ref=admitted.graph_ref,
            policy_ref=admitted.policy_ref,
            item_ids=tuple(source.item_id for source in admitted.sources),
            task_draft_refs=tuple(task_draft_ref(source.task_draft) for source in admitted.sources),
            leakage_reference_set_refs=tuple(
                prompt_leakage_reference_set_ref(source.leakage_reference_set) for source in admitted.sources
            ),
            item_quality_result_refs=admitted.item_quality_refs,
            answer_source_item_ids=tuple(value[0] for value in answer_bindings),
            answer_source_subject_refs=tuple(value[1] for value in answer_bindings),
            attachment_scan_evidence=_sorted_nested(
                attachment_scans,
                attachment_cross_item_safety_scan_evidence_v2_ref,
            ),
            visible_matches=visible_matches,
            answer_reuse_pairs=answer_pairs,
            clusters=clusters,
            evaluated_prompt_foreign_set_count=(admitted.prompt_foreign_set_count),
            evaluated_attachment_foreign_set_count=(admitted.attachment_foreign_set_count),
            evaluated_answer_source_pair_count=(admitted.answer_source_pair_count),
            audit=_safe_audit(audit, refs),
        )


def _fingerprint_owners(
    sources: tuple[CrossItemSafetyItemSource, ...],
) -> dict[str, _FingerprintOwner]:
    values: dict[str, _FingerprintOwner] = {}
    for source in sources:
        reference_ref = prompt_leakage_reference_set_ref(source.leakage_reference_set)
        for fingerprint in source.leakage_reference_set.fingerprints:
            if fingerprint.fingerprint_id in values:
                raise CrossItemSafetyPolicyError("duplicate prompt leakage fingerprint ID")
            values[fingerprint.fingerprint_id] = _FingerprintOwner(
                item_id=source.item_id,
                reference_set_ref=reference_ref,
                fingerprint=fingerprint,
            )
    return values


def _answer_sources(
    sources: tuple[CrossItemSafetyItemSource, ...],
    *,
    policy: CrossItemSafetyPolicyV2,
) -> tuple[_AnswerSource, ...]:
    values: list[_AnswerSource] = []
    for source in sources:
        grouped: dict[
            ObjectRef,
            list[PromptLeakageFingerprintV2],
        ] = defaultdict(list)
        for fingerprint in source.leakage_reference_set.fingerprints:
            if fingerprint.category in _ANSWER_REUSE_CATEGORIES:
                grouped[fingerprint.source_subject_ref].append(fingerprint)
        reference_ref = prompt_leakage_reference_set_ref(source.leakage_reference_set)
        for subject_ref, fingerprints in grouped.items():
            full_by_digest: dict[str, PromptLeakageFingerprintV2] = {}
            windows_by_digest: dict[str, PromptLeakageFingerprintV2] = {}
            for fingerprint in fingerprints:
                if fingerprint.match_kind is PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT:
                    full_by_digest.setdefault(
                        fingerprint.digest_sha256,
                        fingerprint,
                    )
                elif (
                    fingerprint.match_kind is PromptLeakageMatchKindV2.TOKEN_WINDOW
                    and fingerprint.token_count == policy.answer_reuse_window_token_count
                ):
                    windows_by_digest.setdefault(
                        fingerprint.digest_sha256,
                        fingerprint,
                    )
            full = tuple(full_by_digest[digest] for digest in sorted(full_by_digest))
            windows = tuple(windows_by_digest[digest] for digest in sorted(windows_by_digest))
            if not full and not windows:
                continue
            if len(full) > 1:
                raise CrossItemSafetyPolicyError("answer source has duplicate full-text fingerprints")
            values.append(
                _AnswerSource(
                    item_id=source.item_id,
                    reference_set_ref=reference_ref,
                    source_subject_ref=subject_ref,
                    full_fingerprints=full,
                    window_fingerprints=windows,
                )
            )
    return tuple(
        sorted(
            values,
            key=lambda value: (
                value.item_id,
                _ref_key(value.source_subject_ref),
            ),
        )
    )


def _answer_source_pair_count(
    sources: tuple[_AnswerSource, ...],
) -> int:
    return sum(left.item_id != right.item_id for left, right in combinations(sources, 2))


def _answer_window_comparison_count(
    sources: tuple[_AnswerSource, ...],
) -> int:
    return sum(
        len(left.window_fingerprints) * len(right.window_fingerprints)
        for left, right in combinations(sources, 2)
        if left.item_id != right.item_id
    )


def _prompt_scan_comparison_count(
    sources: tuple[CrossItemSafetyItemSource, ...],
) -> int:
    total = 0
    for target in sources:
        tokens = normalize_prompt_leakage_tokens(target.task_draft.visible_prompt)
        for source in sources:
            if source.item_id == target.item_id:
                continue
            for fingerprint in source.leakage_reference_set.fingerprints:
                total += max(
                    0,
                    len(tokens) - fingerprint.token_count + 1,
                )
    return total


def _visible_match_upper_bound(
    sources: tuple[CrossItemSafetyItemSource, ...],
) -> int:
    source_group_counts = {
        source.item_id: len(
            {
                (
                    fingerprint.source_subject_ref,
                    fingerprint.category,
                )
                for fingerprint in source.leakage_reference_set.fingerprints
            }
        )
        for source in sources
    }
    total = 0
    for target in sources:
        environment = target.item_quality.environment_spec
        assert environment is not None
        target_surface_count = 1 + len(environment.artifacts)
        foreign_group_count = sum(
            count for item_id, count in source_group_counts.items() if item_id != target.item_id
        )
        total += target_surface_count * foreign_group_count
    return total


def _attachment_request(
    *,
    source: CrossItemSafetyItemSource,
    artifact: EnvironmentArtifactV2,
    admitted: _AdmittedSources,
    policy: CrossItemSafetyPolicyV2,
) -> AttachmentCrossItemSafetyScanRequestV2:
    foreign_sources = tuple(
        candidate for candidate in admitted.sources if candidate.item_id != source.item_id
    )
    foreign_reference_refs = tuple(
        _facade_ref(value)
        for value in _sorted_refs(
            tuple(
                prompt_leakage_reference_set_ref(candidate.leakage_reference_set)
                for candidate in foreign_sources
            )
        )
    )
    fingerprints = tuple(
        sorted(
            (
                AttachmentValidationFingerprintV2(
                    fingerprint_id=fingerprint.fingerprint_id,
                    category=AttachmentValidationFingerprintCategoryV2(fingerprint.category.value),
                    match_kind=(AttachmentValidationFingerprintMatchKindV2(fingerprint.match_kind.value)),
                    digest_sha256=fingerprint.digest_sha256,
                    token_count=fingerprint.token_count,
                    normalization_version=fingerprint.normalization_version,
                )
                for candidate in foreign_sources
                for fingerprint in candidate.leakage_reference_set.fingerprints
            ),
            key=lambda value: (
                value.category.value,
                value.match_kind.value,
                value.digest_sha256,
                value.token_count,
                value.fingerprint_id,
            ),
        )
    )
    quality_ref = item_quality_compilation_result_ref(source.item_quality)
    artifact_ref = environment_artifact_ref(artifact)
    identity_seed = _payload_sha256(
        {
            "target_item_id": source.item_id,
            "item_quality_result_ref": _ref_payload(quality_ref),
            "environment_artifact_ref": _ref_payload(artifact_ref),
            "foreign_reference_set_refs": [
                value.model_dump(mode="json", exclude_none=False) for value in foreign_reference_refs
            ],
            "policy_ref": _ref_payload(cross_item_safety_policy_v2_ref(policy)),
        }
    )
    pending = AttachmentCrossItemSafetyScanRequestV2(
        scan_request_id="attachment-cross-item-safety-scan-request://pending",
        target_item_id=source.item_id,
        item_quality_result_ref=_facade_ref(quality_ref),
        environment_artifact_ref=_facade_ref(artifact_ref),
        candidate_artifact_version_ref=_facade_ref(artifact.candidate_artifact_version_ref),
        output_ref=_facade_ref(artifact.output_ref),
        artifact_validation_result_ref=_facade_ref(artifact.artifact_validation_result_ref),
        logical_path=artifact.logical_path,
        media_type=artifact.media_type,
        content_sha256=artifact.content_sha256,
        size_bytes=artifact.size_bytes,
        foreign_reference_set_refs=foreign_reference_refs,
        foreign_fingerprints=fingerprints,
        limits=policy.attachment_scan_limits,
        normalization_version=policy.normalization_version,
        policy_version=ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
        idempotency_key=(f"attachment-cross-item-safety-idempotency://sha256/{identity_seed}"),
        request_sha256="0" * 64,
    )
    digest = attachment_cross_item_safety_scan_request_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_request_id": (f"attachment-cross-item-safety-scan-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _compile_attachment_scan(
    *,
    source: CrossItemSafetyItemSource,
    artifact: EnvironmentArtifactV2,
    request: AttachmentCrossItemSafetyScanRequestV2,
    provider_result: AttachmentCrossItemSafetyScanResultV2,
    admitted: _AdmittedSources,
    audit: ContractAudit,
) -> AttachmentCrossItemSafetyScanEvidenceV2:
    try:
        validate_attachment_cross_item_safety_scan_result_identity(provider_result)
    except ValueError as exc:
        raise CrossItemSafetyPolicyError("attachment safety facade result is stale") from exc
    if (
        provider_result.scan_request_ref != attachment_cross_item_safety_scan_request_ref(request)
        or provider_result.environment_artifact_ref != request.environment_artifact_ref
        or provider_result.output_ref != request.output_ref
        or provider_result.output_sha256 != request.content_sha256
        or provider_result.normalization_version != request.normalization_version
        or provider_result.policy_version != request.policy_version
    ):
        raise CrossItemSafetyPolicyError("attachment safety facade result is cross-subject")
    if provider_result.status is AttachmentCrossItemSafetyScanStatusV2.BLOCKED:
        raise CrossItemSafetyPolicyError("attachment safety facade blocked the request")
    if (
        provider_result.scanned_member_count > request.limits.max_inventory_members
        or len(provider_result.matched_fingerprint_ids) > request.limits.max_match_count
    ):
        raise CrossItemSafetyPolicyError("attachment safety facade exceeded request limits")
    foreign_ids = {value.fingerprint_id for value in request.foreign_fingerprints}
    if any(fingerprint_id not in foreign_ids for fingerprint_id in provider_result.matched_fingerprint_ids):
        raise CrossItemSafetyPolicyError("attachment safety facade returned an unknown fingerprint")
    quality_ref = item_quality_compilation_result_ref(source.item_quality)
    artifact_ref = environment_artifact_ref(artifact)
    request_ref = _object_ref_from_facade(attachment_cross_item_safety_scan_request_ref(request))
    result_ref = _object_ref_from_facade(attachment_cross_item_safety_scan_result_ref(provider_result))
    foreign_refs = tuple(_object_ref_from_facade(value) for value in request.foreign_reference_set_refs)
    refs = (
        quality_ref,
        artifact_ref,
        artifact.candidate_artifact_version_ref,
        artifact.output_ref,
        artifact.artifact_validation_result_ref,
        request_ref,
        result_ref,
        *foreign_refs,
        admitted.policy_ref,
    )
    return AttachmentCrossItemSafetyScanEvidenceV2.create(
        target_item_id=source.item_id,
        item_quality_result_ref=request.item_quality_result_ref,
        environment_artifact_ref=request.environment_artifact_ref,
        candidate_artifact_version_ref=(request.candidate_artifact_version_ref),
        output_ref=request.output_ref,
        artifact_validation_result_ref=(request.artifact_validation_result_ref),
        facade_request_ref=(attachment_cross_item_safety_scan_request_ref(request)),
        facade_result=provider_result,
        facade_result_ref=(attachment_cross_item_safety_scan_result_ref(provider_result)),
        foreign_reference_set_refs=request.foreign_reference_set_refs,
        policy_ref=admitted.policy_ref,
        audit=_safe_audit(audit, refs),
    )


def _prompt_visible_matches(
    admitted: _AdmittedSources,
    *,
    audit: ContractAudit,
) -> tuple[CrossItemVisibleMatchEvidenceV2, ...]:
    values: list[CrossItemVisibleMatchEvidenceV2] = []
    for target in admitted.sources:
        foreign_fingerprints = tuple(
            fingerprint
            for source in admitted.sources
            if source.item_id != target.item_id
            for fingerprint in source.leakage_reference_set.fingerprints
        )
        matches = match_prompt_leakage_fingerprints(
            target.task_draft.visible_prompt,
            foreign_fingerprints,
        )
        matched_ids = tuple(
            sorted({fingerprint_id for match in matches for fingerprint_id in match.fingerprint_ids})
        )
        values.extend(
            _visible_matches_from_ids(
                matched_ids,
                target_item_id=target.item_id,
                target_surface=CrossItemSafetySurfaceKindV2.PROMPT,
                target_subject_ref=task_draft_ref(target.task_draft),
                attachment_scan_evidence_ref=None,
                admitted=admitted,
                audit=audit,
            )
        )
    return _sorted_nested(
        tuple(values),
        cross_item_visible_match_evidence_v2_ref,
    )


def _attachment_visible_matches(
    scan: AttachmentCrossItemSafetyScanEvidenceV2,
    *,
    admitted: _AdmittedSources,
    audit: ContractAudit,
) -> tuple[CrossItemVisibleMatchEvidenceV2, ...]:
    return _visible_matches_from_ids(
        scan.facade_result.matched_fingerprint_ids,
        target_item_id=scan.target_item_id,
        target_surface=CrossItemSafetySurfaceKindV2.ATTACHMENT,
        target_subject_ref=scan.environment_artifact_ref,
        attachment_scan_evidence_ref=(attachment_cross_item_safety_scan_evidence_v2_ref(scan)),
        admitted=admitted,
        audit=audit,
    )


def _visible_matches_from_ids(
    matched_ids: tuple[str, ...],
    *,
    target_item_id: str,
    target_surface: CrossItemSafetySurfaceKindV2,
    target_subject_ref: ObjectRef,
    attachment_scan_evidence_ref: ObjectRef | None,
    admitted: _AdmittedSources,
    audit: ContractAudit,
) -> tuple[CrossItemVisibleMatchEvidenceV2, ...]:
    grouped: dict[
        tuple[
            str,
            ObjectRef,
            ObjectRef,
            PromptLeakageCategoryV2,
        ],
        list[PromptLeakageFingerprintV2],
    ] = defaultdict(list)
    for fingerprint_id in matched_ids:
        owner = admitted.fingerprint_owners.get(fingerprint_id)
        if owner is None or owner.item_id == target_item_id:
            raise CrossItemSafetyPolicyError("visible match fingerprint ownership is invalid")
        fingerprint = owner.fingerprint
        grouped[
            (
                owner.item_id,
                owner.reference_set_ref,
                fingerprint.source_subject_ref,
                fingerprint.category,
            )
        ].append(fingerprint)
    values: list[CrossItemVisibleMatchEvidenceV2] = []
    for (
        source_item_id,
        reference_set_ref,
        source_subject_ref,
        category,
    ), fingerprints in grouped.items():
        fingerprint_refs = tuple(prompt_leakage_fingerprint_ref(value) for value in fingerprints)
        refs = [
            reference_set_ref,
            source_subject_ref,
            target_subject_ref,
            *fingerprint_refs,
            admitted.policy_ref,
        ]
        if attachment_scan_evidence_ref is not None:
            refs.append(attachment_scan_evidence_ref)
        values.append(
            CrossItemVisibleMatchEvidenceV2.create(
                evidence_kind=_evidence_kind(category),
                target_surface=target_surface,
                source_item_id=source_item_id,
                target_item_id=target_item_id,
                source_reference_set_ref=reference_set_ref,
                source_subject_ref=source_subject_ref,
                target_subject_ref=target_subject_ref,
                category=category,
                matched_fingerprint_refs=fingerprint_refs,
                match_kinds=tuple(value.match_kind for value in fingerprints),
                attachment_scan_evidence_ref=attachment_scan_evidence_ref,
                policy_ref=admitted.policy_ref,
                audit=_safe_audit(audit, tuple(refs)),
            )
        )
    return _sorted_nested(
        tuple(values),
        cross_item_visible_match_evidence_v2_ref,
    )


def _answer_reuse_pairs(
    sources: tuple[_AnswerSource, ...],
    *,
    policy: CrossItemSafetyPolicyV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[AnswerReusePairEvidenceV2, ...]:
    values: list[AnswerReusePairEvidenceV2] = []
    for left, right in combinations(sources, 2):
        if left.item_id == right.item_id:
            continue
        full_left = {value.digest_sha256: value for value in left.full_fingerprints}
        full_right = {value.digest_sha256: value for value in right.full_fingerprints}
        shared_full = sorted(set(full_left) & set(full_right))
        if shared_full:
            digest = shared_full[0]
            left_refs = (prompt_leakage_fingerprint_ref(full_left[digest]),)
            right_refs = (prompt_leakage_fingerprint_ref(full_right[digest]),)
            values.append(
                _answer_pair(
                    left=left,
                    right=right,
                    left_refs=left_refs,
                    right_refs=right_refs,
                    match_kind=AnswerReuseMatchKindV2.FULL_SOURCE,
                    shared_window_count=0,
                    shorter_source_window_count=0,
                    coverage_bps=10_000,
                    policy_ref=policy_ref,
                    audit=audit,
                )
            )
            continue
        left_windows = {value.digest_sha256: value for value in left.window_fingerprints}
        right_windows = {value.digest_sha256: value for value in right.window_fingerprints}
        shared = tuple(sorted(set(left_windows) & set(right_windows)))
        shorter_count = min(len(left_windows), len(right_windows))
        if shorter_count == 0:
            continue
        coverage_bps = len(shared) * 10_000 // shorter_count
        if (
            len(shared) < policy.min_shared_answer_windows
            or coverage_bps < policy.min_answer_reuse_coverage_bps
        ):
            continue
        values.append(
            _answer_pair(
                left=left,
                right=right,
                left_refs=tuple(prompt_leakage_fingerprint_ref(left_windows[digest]) for digest in shared),
                right_refs=tuple(prompt_leakage_fingerprint_ref(right_windows[digest]) for digest in shared),
                match_kind=AnswerReuseMatchKindV2.PARTIAL_WINDOWS,
                shared_window_count=len(shared),
                shorter_source_window_count=shorter_count,
                coverage_bps=coverage_bps,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    return _sorted_nested(
        tuple(values),
        answer_reuse_pair_evidence_v2_ref,
    )


def _answer_pair(
    *,
    left: _AnswerSource,
    right: _AnswerSource,
    left_refs: tuple[ObjectRef, ...],
    right_refs: tuple[ObjectRef, ...],
    match_kind: AnswerReuseMatchKindV2,
    shared_window_count: int,
    shorter_source_window_count: int,
    coverage_bps: int,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> AnswerReusePairEvidenceV2:
    refs = (
        left.reference_set_ref,
        right.reference_set_ref,
        left.source_subject_ref,
        right.source_subject_ref,
        *left_refs,
        *right_refs,
        policy_ref,
    )
    return AnswerReusePairEvidenceV2.create(
        left_item_id=left.item_id,
        right_item_id=right.item_id,
        left_reference_set_ref=left.reference_set_ref,
        right_reference_set_ref=right.reference_set_ref,
        left_source_subject_ref=left.source_subject_ref,
        right_source_subject_ref=right.source_subject_ref,
        left_matched_fingerprint_refs=left_refs,
        right_matched_fingerprint_refs=right_refs,
        match_kind=match_kind,
        shared_window_count=shared_window_count,
        shorter_source_window_count=shorter_source_window_count,
        coverage_bps=coverage_bps,
        policy_ref=policy_ref,
        audit=_safe_audit(audit, refs),
    )


def _clusters(
    *,
    visible_matches: tuple[CrossItemVisibleMatchEvidenceV2, ...],
    answer_pairs: tuple[AnswerReusePairEvidenceV2, ...],
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[CrossItemSafetyClusterV2, ...]:
    edges_by_kind: dict[
        CrossItemSafetyClusterKindV2,
        list[tuple[ObjectRef, str, str]],
    ] = {kind: [] for kind in CrossItemSafetyClusterKindV2}
    for visible_value in visible_matches:
        kind = CrossItemSafetyClusterKindV2(visible_value.evidence_kind.value)
        edges_by_kind[kind].append(
            (
                cross_item_visible_match_evidence_v2_ref(visible_value),
                visible_value.source_item_id,
                visible_value.target_item_id,
            )
        )
    for reuse_value in answer_pairs:
        edges_by_kind[CrossItemSafetyClusterKindV2.ANSWER_REUSE].append(
            (
                answer_reuse_pair_evidence_v2_ref(reuse_value),
                reuse_value.left_item_id,
                reuse_value.right_item_id,
            )
        )
    clusters: list[CrossItemSafetyClusterV2] = []
    for kind, edges in edges_by_kind.items():
        for members, direct_refs in _edge_components(tuple(edges)):
            clusters.append(
                CrossItemSafetyClusterV2.create(
                    cluster_kind=kind,
                    member_item_ids=members,
                    direct_evidence_refs=direct_refs,
                    policy_ref=policy_ref,
                    audit=_safe_audit(
                        audit,
                        (*direct_refs, policy_ref),
                    ),
                )
            )
    return _sorted_nested(
        tuple(clusters),
        cross_item_safety_cluster_v2_ref,
    )


def _edge_components(
    edges: tuple[tuple[ObjectRef, str, str], ...],
) -> tuple[tuple[tuple[str, ...], tuple[ObjectRef, ...]], ...]:
    if not edges:
        return ()
    adjacency: dict[str, set[str]] = {}
    for _, left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    remaining = set(adjacency)
    components: list[tuple[tuple[str, ...], tuple[ObjectRef, ...]]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        members: set[str] = set()
        while stack:
            item = stack.pop()
            if item in members:
                continue
            members.add(item)
            stack.extend(
                sorted(
                    adjacency.get(item, ()),
                    reverse=True,
                )
            )
        remaining.difference_update(members)
        components.append(
            (
                tuple(sorted(members)),
                _sorted_refs(
                    tuple(ref for ref, left, right in edges if left in members and right in members)
                ),
            )
        )
    return tuple(
        sorted(
            components,
            key=lambda value: (
                value[0],
                tuple(_ref_key(ref) for ref in value[1]),
            ),
        )
    )


def _evidence_kind(
    category: PromptLeakageCategoryV2,
) -> CrossItemSafetyEvidenceKindV2:
    if category in _CONTAMINATION_CATEGORIES:
        return CrossItemSafetyEvidenceKindV2.CONTAMINATION
    return CrossItemSafetyEvidenceKindV2.LEAKAGE


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(item for item in audit.governing_versions if item.component != "cross-item-safety"),
        VersionBinding(
            component="cross-item-safety",
            version=CROSS_ITEM_SAFETY_POLICY_VERSION,
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


def _require_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if len(keys) != len(set(keys)):
        raise CrossItemSafetyPolicyError(f"{label} is duplicated")


def _sorted_nested[ValueT](
    values: tuple[ValueT, ...],
    ref_builder: Callable[[ValueT], ObjectRef],
) -> tuple[ValueT, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: _ref_key(ref_builder(value)),
        )
    )
