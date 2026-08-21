from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    TypedAttribute,
)
from eval_factory.contracts.external_evidence_v2 import (
    ExternalEvidenceAdmissionOutcomeV2,
    ExternalEvidenceAdmissionReasonV2,
    ExternalEvidenceAdmissionReportV2,
    ExternalEvidencePackageManifestV2,
    ExternalLabelObservationSummaryV2,
    ExternalReferenceAuthoringModeV2,
    ExternalSc010Sc011EvidenceIndexV2,
)
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvaluationReportV2,
)
from eval_factory.contracts.labeling_v2 import (
    LabelExecutionStatus,
    LabelUnresolvedReason,
)
from eval_factory.contracts.statistics_v2 import (
    IndependentLabelTestSetFreezeResultV2,
)
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusInventoryCompilation,
    ExternalPartitionCompilation,
)
from eval_factory.readiness.external_evidence_models import (
    ExternalEvidenceMaterialClosureV1,
    ExternalPartitionKindV1,
)
from eval_factory.readiness.external_evidence_requirement import (
    ExternalRequirementCompilation,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationRecordV1,
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceRecordV1,
    ExternalReferenceSetV1,
    ExternalReferenceStateV1,
)


class ExternalEvidenceCompositionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExternalEvidencePackageCompilation:
    material_closure: ExternalEvidenceMaterialClosureV1
    package_manifest: ExternalEvidencePackageManifestV2
    admission_report: ExternalEvidenceAdmissionReportV2
    observation_summary: ExternalLabelObservationSummaryV2


class ExternalEvidenceComposer:
    def bind_label_material_policies(
        self,
        *,
        reference_set: ExternalReferenceSetV1,
        observation_set: BlindLabelObservationSetV1,
        reference_policy_ref: ObjectRef,
        observation_policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> tuple[
        ExternalReferenceSetV1,
        BlindLabelObservationSetV1,
    ]:
        if (
            reference_set.source_population_ref != observation_set.source_population_ref
            or reference_set.label_spec_refs != observation_set.label_spec_refs
        ):
            raise ExternalEvidenceCompositionError("external label material cannot share one policy binding")
        references = tuple(
            ExternalReferenceRecordV1.create(
                source_trace_id=record.source_trace_id,
                raw_sha256=record.raw_sha256,
                trace_envelope_ref=record.trace_envelope_ref,
                label_spec_ref=record.label_spec_ref,
                label_name=record.label_name,
                annotation_contract_ref=record.annotation_contract_ref,
                reference_policy_ref=reference_policy_ref,
                expected_decision=record.expected_decision,
                evidence_refs=record.evidence_refs,
                structured_capability_complete=(record.structured_capability_complete),
                author_kind=record.author_kind,
                reference_state=record.reference_state,
                rule_version=record.rule_version,
                model_profile=record.model_profile,
                prompt_version=record.prompt_version,
                audit=audit,
            )
            for record in reference_set.records
        )
        observations = tuple(
            BlindLabelObservationRecordV1.create(
                source_trace_id=record.source_trace_id,
                raw_sha256=record.raw_sha256,
                trace_envelope_ref=record.trace_envelope_ref,
                label_spec_ref=record.label_spec_ref,
                observation_policy_ref=observation_policy_ref,
                label_decision=record.label_decision,
                route=record.route,
                audit=audit,
            )
            for record in observation_set.records
        )
        return (
            ExternalReferenceSetV1.create(
                source_population_ref=reference_set.source_population_ref,
                label_spec_refs=reference_set.label_spec_refs,
                reference_policy_ref=reference_policy_ref,
                records=references,
                audit=audit,
            ),
            BlindLabelObservationSetV1.create(
                source_population_ref=(observation_set.source_population_ref),
                label_spec_refs=observation_set.label_spec_refs,
                observation_policy_ref=observation_policy_ref,
                records=observations,
                audit=audit,
            ),
        )

    def compile_package(
        self,
        *,
        requirement: ExternalRequirementCompilation,
        corpus: ExternalCorpusInventoryCompilation,
        partitions: ExternalPartitionCompilation,
        reference_set: ExternalReferenceSetV1,
        observation_set: BlindLabelObservationSetV1,
        audit: ContractAudit,
    ) -> ExternalEvidencePackageCompilation:
        inventory = corpus.inventory
        _validate_partitions(corpus, partitions)
        _validate_label_material(
            corpus=corpus,
            reference_set=reference_set,
            observation_set=observation_set,
        )
        private_values = (
            inventory,
            partitions.train,
            partitions.development,
            partitions.test,
            reference_set,
            observation_set,
        )
        closure = ExternalEvidenceMaterialClosureV1.create(
            inventory_ref=inventory.to_ref(),
            partition_refs=(
                partitions.train.to_ref(),
                partitions.development.to_ref(),
                partitions.test.to_ref(),
            ),
            reference_set_ref=reference_set.to_ref(),
            observation_set_ref=observation_set.to_ref(),
            source_count=inventory.source_count,
            total_pair_count=len(reference_set.records),
            material_canonical_bytes=sum(len(value.canonical_json()) for value in private_values),
            audit=audit,
        )
        runtime_counts = Counter(member.runtime.value for member in inventory.members)
        runtime_count_keys = (
            ("claude_curated",)
            if inventory.adapter_name == "curated_trajectory_v1"
            else ("claude_code", "codex", "hermes")
        )
        package = ExternalEvidencePackageManifestV2.create(
            requirement_source_ref=requirement.source_ref,
            requirement_spec_ref=requirement.requirement.to_ref(),
            source_authorization_ref=(inventory.source_authorization_ref),
            source_population_ref=corpus.source_population_ref,
            inventory_commitment_ref=inventory.to_ref(),
            train_partition_ref=partitions.train.to_public_ref(),
            development_partition_ref=(partitions.development.to_public_ref()),
            label_spec_refs=reference_set.label_spec_refs,
            reference_policy_ref=reference_set.reference_policy_ref,
            observation_policy_ref=(observation_set.observation_policy_ref),
            adapter_name=inventory.adapter_name,
            adapter_version=inventory.adapter_version,
            source_count=inventory.source_count,
            unique_trace_count=inventory.unique_trace_count,
            unique_raw_hash_count=inventory.unique_raw_hash_count,
            total_raw_bytes=inventory.total_raw_bytes,
            runtime_counts=tuple(
                TypedAttribute(key=key, value=runtime_counts[key]) for key in runtime_count_keys
            ),
            inventory_sha256=inventory.inventory_sha256,
            audit=audit,
        )
        admission = ExternalEvidenceAdmissionReportV2.create(
            package_manifest_ref=package.to_ref(),
            material_closure_ref=closure.to_ref(),
            outcome=ExternalEvidenceAdmissionOutcomeV2.ADMITTED,
            reason_codes=(ExternalEvidenceAdmissionReasonV2.NONE,),
            source_count=inventory.source_count,
            admitted_source_count=inventory.source_count,
            rejected_source_count=0,
            unique_trace_count=inventory.unique_trace_count,
            unique_raw_hash_count=inventory.unique_raw_hash_count,
            duplicate_source_count=0,
            overlap_source_count=0,
            total_raw_bytes=inventory.total_raw_bytes,
            manifest_verified=True,
            material_verified=True,
            audit=audit,
        )
        reference_counts = Counter(record.reference_state for record in reference_set.records)
        complete_observations = sum(
            record.label_decision.execution_status is LabelExecutionStatus.FINAL
            for record in observation_set.records
        )
        blocked_observations = sum(
            record.label_decision.execution_status is not LabelExecutionStatus.FINAL
            for record in observation_set.records
        )
        summary = ExternalLabelObservationSummaryV2.create(
            package_manifest_ref=package.to_ref(),
            source_population_ref=corpus.source_population_ref,
            reference_closure_ref=reference_set.to_ref(),
            observation_closure_ref=observation_set.to_ref(),
            label_spec_refs=reference_set.label_spec_refs,
            reference_policy_ref=reference_set.reference_policy_ref,
            observation_policy_ref=(observation_set.observation_policy_ref),
            authoring_mode=(ExternalReferenceAuthoringModeV2.AUTOMATIC_INDEPENDENT),
            reference_authority_committed=True,
            observation_reference_access_denied=True,
            total_pair_count=len(reference_set.records),
            reference_ready_count=reference_counts[ExternalReferenceStateV1.REFERENCE_READY],
            reference_abstained_count=reference_counts[ExternalReferenceStateV1.ABSTAINED],
            reference_blocked_count=reference_counts[ExternalReferenceStateV1.BLOCKED],
            observation_complete_count=complete_observations,
            observation_missing_count=0,
            observation_blocked_count=blocked_observations,
            audit=audit,
        )
        return ExternalEvidencePackageCompilation(
            material_closure=closure,
            package_manifest=package,
            admission_report=admission,
            observation_summary=summary,
        )

    def compile_index(
        self,
        *,
        package: ExternalEvidencePackageCompilation,
        freeze_result: IndependentLabelTestSetFreezeResultV2,
        label_quality_report: LabelQualityEvaluationReportV2,
        stability_report: ExternalRealTraceStabilityReportV2,
        development_evidence_count: int,
        synthetic_evidence_count: int,
        audit: ContractAudit,
    ) -> ExternalSc010Sc011EvidenceIndexV2:
        manifest = package.package_manifest
        if (
            package.admission_report.package_manifest_ref != manifest.to_ref()
            or package.observation_summary.package_manifest_ref != manifest.to_ref()
            or package.observation_summary.reference_closure_ref != package.material_closure.reference_set_ref
            or package.observation_summary.observation_closure_ref
            != package.material_closure.observation_set_ref
        ):
            raise ExternalEvidenceCompositionError("external package public/private closure differs")
        if (
            label_quality_report.prerequisite.freeze_result_ref != freeze_result.to_ref()
            or label_quality_report.prerequisite.dataset_manifest_ref
            != (None if freeze_result.dataset_manifest is None else freeze_result.dataset_manifest.to_ref())
        ):
            raise ExternalEvidenceCompositionError("label quality report differs from freeze authority")
        if (
            stability_report.package_manifest_ref != manifest.to_ref()
            or stability_report.private_inventory_ref != manifest.inventory_commitment_ref
            or stability_report.unique_real_trace_count != manifest.source_count
        ):
            raise ExternalEvidenceCompositionError("stability report differs from external package")
        if development_evidence_count < 0 or synthetic_evidence_count < 0:
            raise ExternalEvidenceCompositionError("external evidence class counts must be non-negative")
        return ExternalSc010Sc011EvidenceIndexV2.create(
            package_manifest_ref=manifest.to_ref(),
            admission_report_ref=package.admission_report.to_ref(),
            requirement_spec_ref=manifest.requirement_spec_ref,
            observation_summary_ref=(package.observation_summary.to_ref()),
            freeze_result_ref=freeze_result.to_ref(),
            dataset_manifest_ref=(
                None if freeze_result.dataset_manifest is None else freeze_result.dataset_manifest.to_ref()
            ),
            label_quality_report_ref=label_quality_report.to_ref(),
            label_quality_outcome=label_quality_report.outcome,
            label_evidence_class=label_quality_report.evidence_class,
            label_report_satisfies_sc_010=(label_quality_report.satisfies_sc_010),
            stability_report_ref=stability_report.to_ref(),
            stability_outcome=stability_report.outcome,
            unique_real_trace_count=(stability_report.unique_real_trace_count),
            observed_stability_threshold_met=(stability_report.observed_threshold_met),
            evidence_class_counts=(
                TypedAttribute(
                    key="DEVELOPMENT",
                    value=development_evidence_count,
                ),
                TypedAttribute(
                    key="REAL",
                    value=stability_report.unique_real_trace_count,
                ),
                TypedAttribute(
                    key="REPLAY",
                    value=sum(summary.replay_count for summary in stability_report.case_summaries),
                ),
                TypedAttribute(
                    key="SYNTHETIC",
                    value=synthetic_evidence_count,
                ),
            ),
            audit=audit,
        )


def _validate_partitions(
    corpus: ExternalCorpusInventoryCompilation,
    partitions: ExternalPartitionCompilation,
) -> None:
    observed = {
        partition.partition_kind: partition
        for partition in (
            partitions.train,
            partitions.development,
            partitions.test,
        )
    }
    if set(observed) != set(ExternalPartitionKindV1):
        raise ExternalEvidenceCompositionError("external partition kinds are not exact")
    all_members = tuple(member for partition in observed.values() for member in partition.members)
    expected = {(member.source_trace_id, member.raw_sha256) for member in corpus.inventory.members}
    actual = {(member.source_trace_id, member.raw_sha256) for member in all_members}
    if (
        actual != expected
        or len(all_members) != len(expected)
        or partitions.eligible_source_count != len(partitions.test.members)
        or partitions.excluded_source_count
        != (len(partitions.train.members) + len(partitions.development.members))
    ):
        raise ExternalEvidenceCompositionError("external partition closure is not exact")


def _validate_label_material(
    *,
    corpus: ExternalCorpusInventoryCompilation,
    reference_set: ExternalReferenceSetV1,
    observation_set: BlindLabelObservationSetV1,
) -> None:
    if (
        reference_set.source_population_ref != corpus.source_population_ref
        or observation_set.source_population_ref != corpus.source_population_ref
        or reference_set.label_spec_refs != observation_set.label_spec_refs
        or len(reference_set.label_spec_refs) != 3
    ):
        raise ExternalEvidenceCompositionError("external label material portfolio differs")
    expected_pairs = {
        (member.source_trace_id, label_ref)
        for member in corpus.inventory.members
        for label_ref in reference_set.label_spec_refs
    }
    reference_pairs = {(record.source_trace_id, record.label_spec_ref) for record in reference_set.records}
    observation_pairs = {
        (record.source_trace_id, record.label_spec_ref) for record in observation_set.records
    }
    if (
        reference_pairs != expected_pairs
        or observation_pairs != expected_pairs
        or len(reference_set.records) != len(expected_pairs)
        or len(observation_set.records) != len(expected_pairs)
    ):
        raise ExternalEvidenceCompositionError("external label pair closure is not exact")
    inventory = {member.source_trace_id: member.raw_sha256 for member in corpus.inventory.members}
    reference_bindings = {
        (
            record.source_trace_id,
            record.label_spec_ref,
        ): (
            record.raw_sha256,
            record.trace_envelope_ref,
        )
        for record in reference_set.records
    }
    for record in observation_set.records:
        key = (record.source_trace_id, record.label_spec_ref)
        expected = reference_bindings[key]
        if record.raw_sha256 != inventory[record.source_trace_id] or expected != (
            record.raw_sha256,
            record.trace_envelope_ref,
        ):
            raise ExternalEvidenceCompositionError("reference and observation source bindings differ")
    if any(
        LabelUnresolvedReason.MODEL_UNAVAILABLE in record.label_decision.unresolved_reasons
        and record.label_decision.execution_status is LabelExecutionStatus.FINAL
        for record in observation_set.records
    ):
        raise ExternalEvidenceCompositionError("model-unavailable observation cannot be final")


__all__ = [
    "ExternalEvidenceComposer",
    "ExternalEvidenceCompositionError",
    "ExternalEvidencePackageCompilation",
]
