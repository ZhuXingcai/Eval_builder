from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2, LabelSpecV2
from eval_factory.contracts.statistics_v2 import (
    IndependentLabelPartitionManifestV2,
    IndependentLabelTestSetAccessPolicyV2,
    IndependentLabelTestSetFreezeRecordV2,
    IndependentLabelTestSetFreezeResultV2,
    IndependentLabelTestSetLabelPolicyV2,
    IndependentLabelTestSetManifestV2,
    IndependentLabelTestSetPolicyV2,
    LabelTestSetBalanceStatusV2,
    LabelTestSetClassCountV2,
    LabelTestSetEvaluationKindV2,
    LabelTestSetFreezeOutcomeV2,
    LabelTestSetLabelSummaryV2,
    LabelTestSetPartitionKindV2,
    LabelTestSetShortageReasonV2,
    LabelTestSetShortageV2,
    validate_independent_label_partition_manifest_v2_identity,
    validate_independent_label_test_set_access_policy_v2_identity,
    validate_independent_label_test_set_policy_v2_identity,
)
from eval_factory.statistics.models import (
    IndependentLabelCandidatePoolV1,
    IndependentLabelPartitionMaterialV1,
    IndependentLabelReferenceCandidateV1,
    IndependentLabelTestSetMaterialV1,
    independent_label_candidate_pool_v1_ref,
    independent_label_partition_material_v1_ref,
    independent_label_test_set_material_v1_ref,
    validate_independent_label_candidate_pool_v1_identity,
    validate_independent_label_partition_material_v1_identity,
)

_DECISIONS = (
    LabelDecisionValueV2.MATCH,
    LabelDecisionValueV2.NO_MATCH,
    LabelDecisionValueV2.ABSTAIN,
)


class IndependentLabelTestSetPolicyError(RuntimeError):
    pass


class IndependentLabelTestSetPendingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IndependentLabelTestSetFreezeCompilation:
    result: IndependentLabelTestSetFreezeResultV2
    selected_material: IndependentLabelTestSetMaterialV1 | None


class IndependentLabelTestSetCompiler:
    def compile(
        self,
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
        audit: ContractAudit,
    ) -> IndependentLabelTestSetFreezeCompilation:
        try:
            validate_independent_label_test_set_policy_v2_identity(policy)
            validate_independent_label_test_set_access_policy_v2_identity(access_policy)
            validate_independent_label_candidate_pool_v1_identity(candidate_pool)
            self._validate_partition(
                train_partition_manifest,
                train_partition_material,
                LabelTestSetPartitionKindV2.TRAIN,
            )
            self._validate_partition(
                development_partition_manifest,
                development_partition_material,
                LabelTestSetPartitionKindV2.DEVELOPMENT,
            )
            spec_by_ref = self._validate_specs(label_specs, policy)
            self._validate_limits(candidate_pool, policy)
            self._validate_independence(
                candidate_pool,
                train_partition_material,
                development_partition_material,
            )
            selections, summaries, shortages = self._select(
                candidate_pool,
                policy,
                spec_by_ref,
            )
        except IndependentLabelTestSetPolicyError:
            raise
        except ValueError as exc:
            raise IndependentLabelTestSetPolicyError(str(exc)) from exc

        candidate_pool_ref = independent_label_candidate_pool_v1_ref(candidate_pool)
        policy_ref = policy.to_ref()
        access_policy_ref = access_policy.to_ref()
        train_partition_ref = train_partition_manifest.to_ref()
        development_partition_ref = development_partition_manifest.to_ref()
        selected_material: IndependentLabelTestSetMaterialV1 | None = None
        manifest: IndependentLabelTestSetManifestV2 | None = None
        if not shortages:
            selected_material = IndependentLabelTestSetMaterialV1.create(
                dataset_series_id=dataset_series_id,
                dataset_version=dataset_version,
                members=selections,
                candidate_pool_ref=candidate_pool_ref,
                train_partition_ref=train_partition_ref,
                development_partition_ref=development_partition_ref,
                policy_ref=policy_ref,
                access_policy_ref=access_policy_ref,
                audit=audit,
            )
            if len(selected_material.canonical_json()) > policy.max_private_bytes:
                raise IndependentLabelTestSetPolicyError(
                    "selected test-set material exceeds max_private_bytes"
                )
            unique_traces = {item.source_trace_id for item in selections}
            unique_hashes = {item.raw_sha256 for item in selections}
            manifest = IndependentLabelTestSetManifestV2.create(
                dataset_series_id=dataset_series_id,
                dataset_version=dataset_version,
                private_material_ref=independent_label_test_set_material_v1_ref(selected_material),
                policy_ref=policy_ref,
                access_policy_ref=access_policy_ref,
                train_partition_ref=train_partition_ref,
                development_partition_ref=development_partition_ref,
                candidate_pool_ref=candidate_pool_ref,
                label_summaries=summaries,
                selected_member_count=len(selections),
                unique_trace_count=len(unique_traces),
                unique_raw_hash_count=len(unique_hashes),
                selection_algorithm=policy.selection_algorithm,
                selection_seed=policy.selection_seed,
                supersedes_dataset_ref=supersedes_dataset_ref,
                audit=audit,
            )
        outcome = (
            LabelTestSetFreezeOutcomeV2.FROZEN
            if manifest is not None
            else LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING
        )
        record = IndependentLabelTestSetFreezeRecordV2.create(
            dataset_series_id=dataset_series_id,
            requested_dataset_version=dataset_version,
            outcome=outcome,
            candidate_pool_ref=candidate_pool_ref,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            train_partition_ref=train_partition_ref,
            development_partition_ref=development_partition_ref,
            label_summaries=summaries,
            shortages=shortages,
            dataset_manifest_ref=None if manifest is None else manifest.to_ref(),
            supersedes_freeze_ref=supersedes_freeze_ref,
            audit=audit,
        )
        result = IndependentLabelTestSetFreezeResultV2.create(
            freeze_record=record,
            dataset_manifest=manifest,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            partition_refs=(train_partition_ref, development_partition_ref),
            audit=audit,
        )
        return IndependentLabelTestSetFreezeCompilation(
            result=result,
            selected_material=selected_material,
        )

    def validate_current(
        self,
        *,
        expected: IndependentLabelTestSetFreezeCompilation,
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
    ) -> None:
        observed = self.compile(
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
        if observed.result.to_ref() != expected.result.to_ref() or (
            None if observed.selected_material is None else observed.selected_material.to_ref()
        ) != (None if expected.selected_material is None else expected.selected_material.to_ref()):
            raise IndependentLabelTestSetPolicyError("independent label test-set compilation is stale")

    @staticmethod
    def _validate_partition(
        manifest: IndependentLabelPartitionManifestV2,
        material: IndependentLabelPartitionMaterialV1,
        kind: LabelTestSetPartitionKindV2,
    ) -> None:
        validate_independent_label_partition_manifest_v2_identity(manifest)
        validate_independent_label_partition_material_v1_identity(material)
        if (
            manifest.partition_kind is not kind
            or material.partition_kind is not kind
            or manifest.private_material_ref != independent_label_partition_material_v1_ref(material)
            or manifest.trace_count != len({item.source_trace_id for item in material.members})
            or manifest.unique_raw_hash_count != len({item.raw_sha256 for item in material.members})
            or manifest.source_authority_refs != material.source_authority_refs
        ):
            raise IndependentLabelTestSetPolicyError(
                f"{kind.value} partition manifest differs from private material"
            )

    @staticmethod
    def _validate_specs(
        label_specs: tuple[LabelSpecV2, ...],
        policy: IndependentLabelTestSetPolicyV2,
    ) -> dict[tuple[str, str, str, str], LabelSpecV2]:
        spec_by_ref: dict[tuple[str, str, str, str], LabelSpecV2] = {}
        for spec in label_specs:
            observed_digest = _label_spec_carried_sha256(spec)
            if observed_digest != spec.label_spec_sha256:
                raise IndependentLabelTestSetPolicyError("LabelSpec behavior hash is stale")
            ref = _label_spec_ref(spec)
            key = _ref_key(ref)
            if key in spec_by_ref:
                raise IndependentLabelTestSetPolicyError("duplicate LabelSpec authority")
            spec_by_ref[key] = spec
        expected = tuple(_ref_key(item.label_spec_ref) for item in policy.label_policies)
        if tuple(sorted(spec_by_ref)) != expected:
            raise IndependentLabelTestSetPolicyError("LabelSpec portfolio differs from freeze policy")
        for label_policy in policy.label_policies:
            spec = spec_by_ref[_ref_key(label_policy.label_spec_ref)]
            expected_kind = (
                LabelTestSetEvaluationKindV2.SEMANTIC
                if spec.semantic_residual is not None
                else LabelTestSetEvaluationKindV2.STRUCTURED
            )
            if label_policy.evaluation_kind is not expected_kind:
                raise IndependentLabelTestSetPolicyError(
                    "LabelSpec execution kind differs from freeze policy"
                )
        return spec_by_ref

    @staticmethod
    def _validate_limits(
        candidate_pool: IndependentLabelCandidatePoolV1,
        policy: IndependentLabelTestSetPolicyV2,
    ) -> None:
        if len(candidate_pool.candidates) > policy.max_candidates:
            raise IndependentLabelTestSetPolicyError("candidate pool exceeds max_candidates")
        if len(candidate_pool.canonical_json()) > policy.max_private_bytes:
            raise IndependentLabelTestSetPolicyError("candidate pool exceeds max_private_bytes")

    @staticmethod
    def _validate_independence(
        candidate_pool: IndependentLabelCandidatePoolV1,
        train: IndependentLabelPartitionMaterialV1,
        development: IndependentLabelPartitionMaterialV1,
    ) -> None:
        train_ids = {item.source_trace_id for item in train.members}
        train_hashes = {item.raw_sha256 for item in train.members}
        development_ids = {item.source_trace_id for item in development.members}
        development_hashes = {item.raw_sha256 for item in development.members}
        if train_ids & development_ids or train_hashes & development_hashes:
            raise IndependentLabelTestSetPolicyError("TRAIN and DEVELOPMENT partitions overlap")

        trace_to_hash: dict[str, str] = {}
        hash_to_trace: dict[str, str] = {}
        for candidate in candidate_pool.candidates:
            prior_hash = trace_to_hash.setdefault(
                candidate.source_trace_id,
                candidate.raw_sha256,
            )
            if prior_hash != candidate.raw_sha256:
                raise IndependentLabelTestSetPolicyError("source trace ID maps to multiple raw hashes")
            prior_trace = hash_to_trace.setdefault(
                candidate.raw_sha256,
                candidate.source_trace_id,
            )
            if prior_trace != candidate.source_trace_id:
                raise IndependentLabelTestSetPolicyError("raw hash maps to multiple source trace IDs")
            if candidate.source_trace_id in train_ids or candidate.raw_sha256 in train_hashes:
                raise IndependentLabelTestSetPolicyError("candidate overlaps TRAIN partition")
            if candidate.source_trace_id in development_ids or candidate.raw_sha256 in development_hashes:
                raise IndependentLabelTestSetPolicyError("candidate overlaps DEVELOPMENT partition")

    @staticmethod
    def _select(
        candidate_pool: IndependentLabelCandidatePoolV1,
        policy: IndependentLabelTestSetPolicyV2,
        spec_by_ref: dict[tuple[str, str, str, str], LabelSpecV2],
    ) -> tuple[
        tuple[IndependentLabelReferenceCandidateV1, ...],
        tuple[LabelTestSetLabelSummaryV2, ...],
        tuple[LabelTestSetShortageV2, ...],
    ]:
        allowed = set(spec_by_ref)
        grouped: dict[
            tuple[str, str, str, str],
            dict[LabelDecisionValueV2, list[IndependentLabelReferenceCandidateV1]],
        ] = {key: {decision: [] for decision in _DECISIONS} for key in allowed}
        for candidate in candidate_pool.candidates:
            key = _ref_key(candidate.label_spec_ref)
            if key not in allowed:
                raise IndependentLabelTestSetPolicyError("candidate label is outside freeze policy")
            grouped[key][candidate.expected_decision].append(candidate)
        selected: list[IndependentLabelReferenceCandidateV1] = []
        summaries: list[LabelTestSetLabelSummaryV2] = []
        shortages: list[LabelTestSetShortageV2] = []
        for label_policy in policy.label_policies:
            key = _ref_key(label_policy.label_spec_ref)
            ranked = {
                decision: sorted(
                    grouped[key][decision],
                    key=lambda candidate: _selection_rank(policy, candidate),
                )
                for decision in _DECISIONS
            }
            if label_policy.evaluation_kind is LabelTestSetEvaluationKindV2.STRUCTURED:
                label_selected, summary, label_shortages = _select_structured(
                    label_policy,
                    ranked,
                )
            else:
                label_selected, summary, label_shortages = _select_semantic(
                    label_policy,
                    ranked,
                )
            selected.extend(label_selected)
            summaries.append(summary)
            shortages.extend(label_shortages)
        ordered_selected = tuple(sorted(selected, key=_candidate_key))
        if len(ordered_selected) > policy.max_selected_members:
            raise IndependentLabelTestSetPolicyError("selected members exceed max_selected_members")
        return (
            ordered_selected,
            tuple(sorted(summaries, key=lambda item: _ref_key(item.label_spec_ref))),
            tuple(sorted(shortages, key=_shortage_key)),
        )


def _select_structured(
    policy: IndependentLabelTestSetLabelPolicyV2,
    ranked: dict[
        LabelDecisionValueV2,
        list[IndependentLabelReferenceCandidateV1],
    ],
) -> tuple[
    tuple[IndependentLabelReferenceCandidateV1, ...],
    LabelTestSetLabelSummaryV2,
    tuple[LabelTestSetShortageV2, ...],
]:
    required = {
        LabelDecisionValueV2.MATCH: policy.structured_positive_minimum,
        LabelDecisionValueV2.NO_MATCH: policy.structured_negative_minimum,
        LabelDecisionValueV2.ABSTAIN: 0,
    }
    selected_by_class = {decision: ranked[decision][: required[decision]] for decision in _DECISIONS}
    shortages = tuple(
        LabelTestSetShortageV2(
            label_spec_ref=policy.label_spec_ref,
            decision=decision,
            reason=LabelTestSetShortageReasonV2.CLASS_SHORTAGE,
            required_count=required[decision],
            available_count=len(ranked[decision]),
        )
        for decision in (
            LabelDecisionValueV2.MATCH,
            LabelDecisionValueV2.NO_MATCH,
        )
        if len(ranked[decision]) < required[decision]
    )
    counts = tuple(
        LabelTestSetClassCountV2(
            decision=decision,
            eligible_count=len(ranked[decision]),
            selected_count=len(selected_by_class[decision]),
            required_minimum=required[decision],
        )
        for decision in _DECISIONS
    )
    selected = tuple(candidate for decision in _DECISIONS for candidate in selected_by_class[decision])
    return (
        selected,
        LabelTestSetLabelSummaryV2(
            label_spec_ref=policy.label_spec_ref,
            evaluation_kind=policy.evaluation_kind,
            class_counts=counts,
            required_total=policy.required_total,
            eligible_total=sum(item.eligible_count for item in counts),
            selected_total=len(selected),
            balance_status=(
                LabelTestSetBalanceStatusV2.INSUFFICIENT
                if shortages
                else LabelTestSetBalanceStatusV2.ACHIEVED
            ),
        ),
        shortages,
    )


def _select_semantic(
    policy: IndependentLabelTestSetLabelPolicyV2,
    ranked: dict[
        LabelDecisionValueV2,
        list[IndependentLabelReferenceCandidateV1],
    ],
) -> tuple[
    tuple[IndependentLabelReferenceCandidateV1, ...],
    LabelTestSetLabelSummaryV2,
    tuple[LabelTestSetShortageV2, ...],
]:
    cursors = {decision: 0 for decision in _DECISIONS}
    selected_by_class: dict[
        LabelDecisionValueV2,
        list[IndependentLabelReferenceCandidateV1],
    ] = {decision: [] for decision in _DECISIONS}
    while sum(len(values) for values in selected_by_class.values()) < policy.semantic_total_minimum:
        advanced = False
        for decision in _DECISIONS:
            cursor = cursors[decision]
            if cursor >= len(ranked[decision]):
                continue
            selected_by_class[decision].append(ranked[decision][cursor])
            cursors[decision] += 1
            advanced = True
            if sum(len(values) for values in selected_by_class.values()) == policy.semantic_total_minimum:
                break
        if not advanced:
            break
    selected = tuple(candidate for decision in _DECISIONS for candidate in selected_by_class[decision])
    shortages = (
        (
            LabelTestSetShortageV2(
                label_spec_ref=policy.label_spec_ref,
                decision=None,
                reason=LabelTestSetShortageReasonV2.TOTAL_SHORTAGE,
                required_count=policy.semantic_total_minimum,
                available_count=sum(len(values) for values in ranked.values()),
            ),
        )
        if len(selected) < policy.semantic_total_minimum
        else ()
    )
    counts = tuple(
        LabelTestSetClassCountV2(
            decision=decision,
            eligible_count=len(ranked[decision]),
            selected_count=len(selected_by_class[decision]),
            required_minimum=0,
        )
        for decision in _DECISIONS
    )
    selected_counts = tuple(item.selected_count for item in counts)
    if shortages:
        balance = LabelTestSetBalanceStatusV2.INSUFFICIENT
    elif max(selected_counts) - min(selected_counts) <= 1:
        balance = LabelTestSetBalanceStatusV2.ACHIEVED
    else:
        balance = LabelTestSetBalanceStatusV2.SOURCE_POPULATION_LIMITED
    return (
        selected,
        LabelTestSetLabelSummaryV2(
            label_spec_ref=policy.label_spec_ref,
            evaluation_kind=policy.evaluation_kind,
            class_counts=counts,
            required_total=policy.required_total,
            eligible_total=sum(item.eligible_count for item in counts),
            selected_total=len(selected),
            balance_status=balance,
        ),
        shortages,
    )


def _selection_rank(
    policy: IndependentLabelTestSetPolicyV2,
    candidate: IndependentLabelReferenceCandidateV1,
) -> str:
    return _payload_sha256(
        {
            "policy_ref": _ref_payload(policy.to_ref()),
            "selection_seed": policy.selection_seed,
            "label_spec_ref": _ref_payload(candidate.label_spec_ref),
            "expected_decision": candidate.expected_decision.value,
            "trace_envelope_ref": _ref_payload(candidate.trace_envelope_ref),
            "source_trace_id": candidate.source_trace_id,
            "raw_sha256": candidate.raw_sha256,
            "annotation_ref": _ref_payload(candidate.annotation_ref),
        }
    )


def _label_spec_carried_sha256(value: LabelSpecV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"label_spec_id", "label_spec_sha256", "audit"},
            exclude_none=False,
        )
    )


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _candidate_key(
    value: IndependentLabelReferenceCandidateV1,
) -> tuple[tuple[str, str, str, str], str, str, tuple[str, str, str, str]]:
    return (
        _ref_key(value.label_spec_ref),
        value.expected_decision.value,
        value.source_trace_id,
        _ref_key(value.annotation_ref),
    )


def _shortage_key(
    value: LabelTestSetShortageV2,
) -> tuple[tuple[str, str, str, str], str, str]:
    return (
        _ref_key(value.label_spec_ref),
        "" if value.decision is None else value.decision.value,
        value.reason.value,
    )


def _ref_payload(value: ObjectRef) -> dict[str, str]:
    return {
        "object_type": value.object_type,
        "object_id": value.object_id,
        "object_version": value.object_version,
        "object_sha256": value.object_sha256,
    }


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
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
