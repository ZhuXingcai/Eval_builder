from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelSpecV2,
    PredicateOperatorV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.statistics_v2 import (
    INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION,
    IndependentLabelPartitionManifestV2,
    IndependentLabelTestSetAccessPolicyV2,
    IndependentLabelTestSetLabelPolicyV2,
    IndependentLabelTestSetPolicyV2,
    IndependentLabelTestSetPrincipalGrantV2,
    LabelTestSetAccessPurposeV2,
    LabelTestSetBalanceStatusV2,
    LabelTestSetEvaluationKindV2,
    LabelTestSetFreezeOutcomeV2,
    LabelTestSetPartitionKindV2,
)
from eval_factory.statistics.freeze import (
    IndependentLabelTestSetCompiler,
    IndependentLabelTestSetPolicyError,
)
from eval_factory.statistics.models import (
    IndependentLabelCandidatePoolV1,
    IndependentLabelPartitionMaterialV1,
    IndependentLabelReferenceCandidateV1,
    independent_label_partition_material_v1_ref,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-01/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r8-01-freeze-test",
        governing_versions=(
            VersionBinding(
                component="independent-label-test-set",
                version=INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
    )


def _predicate(suffix: str) -> StructuredPredicateV2:
    return StructuredPredicateV2(
        predicate_id=f"predicate://r8-01/{suffix}",
        fact_type="tool-call-record",
        field_path="tool_family",
        operator=PredicateOperatorV2.EQUALS,
        expected_value=suffix,
        required_capability="tool-events",
        rule_version="r8-01-test",
    )


def _label_spec(suffix: str, *, semantic: bool) -> LabelSpecV2:
    residual = (
        SemanticResidualSpecV2(
            residual_id=f"semantic-residual://r8-01/{suffix}",
            question="Does the authorized evidence satisfy this semantic label?",
            evidence_bundle_purpose="r8-label-test",
            allowed_evidence_types=("interaction-segment",),
            abstain_conditions=("Evidence is incomplete.",),
            model_profile="internal-semantic-labeler-v1",
            prompt_version="r8-01-semantic/v1",
        )
        if semantic
        else None
    )
    pending = LabelSpecV2(
        label_spec_id=f"label-spec://r8-01/{suffix}",
        label_version="v2",
        name=suffix,
        requirement=f"Test requirement for {suffix}.",
        prerequisite_predicates=(),
        positive_predicates=(_predicate(suffix),),
        negative_predicates=(),
        semantic_residual=residual,
        decision_threshold=0.85,
        review_threshold=0.7,
        label_plan_ref=None,
        policy_version="labeling/r8-01-test",
        label_spec_sha256="0" * 64,
        audit=_audit(),
    )
    digest = _payload_sha256(
        pending.model_dump(
            mode="python",
            exclude={"label_spec_id", "label_spec_sha256", "audit"},
            exclude_none=False,
        )
    )
    return pending.model_copy(update={"label_spec_sha256": digest})


def _label_ref(spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=spec.label_spec_id,
        object_version=spec.label_version,
        object_sha256=spec.label_spec_sha256,
    )


def _policy(specs: tuple[LabelSpecV2, ...]) -> IndependentLabelTestSetPolicyV2:
    policies = tuple(
        IndependentLabelTestSetLabelPolicyV2(
            label_spec_ref=_label_ref(spec),
            evaluation_kind=(
                LabelTestSetEvaluationKindV2.SEMANTIC
                if spec.semantic_residual is not None
                else LabelTestSetEvaluationKindV2.STRUCTURED
            ),
            structured_positive_minimum=0 if spec.semantic_residual is not None else 50,
            structured_negative_minimum=0 if spec.semantic_residual is not None else 50,
            semantic_total_minimum=100 if spec.semantic_residual is not None else 0,
            semantic_classes=(
                (
                    LabelDecisionValueV2.MATCH,
                    LabelDecisionValueV2.NO_MATCH,
                    LabelDecisionValueV2.ABSTAIN,
                )
                if spec.semantic_residual is not None
                else ()
            ),
        )
        for spec in sorted(specs, key=lambda item: item.label_spec_id)
    )
    return IndependentLabelTestSetPolicyV2.create(
        label_policies=policies,
        selection_seed="approved-r8-01-test-seed",
        max_candidates=10_000,
        max_selected_members=1_000,
        max_private_bytes=50_000_000,
        audit=_audit(),
    )


def _access_policy() -> IndependentLabelTestSetAccessPolicyV2:
    principal = _ref("statistical-principal", "evaluator", version="v1")
    return IndependentLabelTestSetAccessPolicyV2.create(
        principal_grants=(
            IndependentLabelTestSetPrincipalGrantV2(
                principal_ref=principal,
                purposes=frozenset(
                    {
                        LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
                    }
                ),
            ),
        ),
        max_members_per_read=1_000,
        audit=_audit(),
    )


def _candidate(
    spec: LabelSpecV2,
    *,
    index: int,
    decision: LabelDecisionValueV2,
    raw_sha256: str | None = None,
    source_trace_id: str | None = None,
) -> IndependentLabelReferenceCandidateV1:
    source_id = source_trace_id or f"source-trace://r8-01/{spec.name}/{decision.value}/{index:03d}"
    raw_hash = raw_sha256 or _digest(source_id)
    return IndependentLabelReferenceCandidateV1(
        trace_envelope_ref=ObjectRef(
            object_type="trace-envelope",
            object_id=f"trace-envelope://r8-01/{spec.name}/{decision.value}/{index:03d}",
            object_version="v1",
            object_sha256=_digest(f"envelope:{source_id}"),
        ),
        source_trace_id=source_id,
        raw_sha256=raw_hash,
        label_spec_ref=_label_ref(spec),
        annotation_ref=_ref(
            "independent-label-annotation",
            f"{spec.name}/{decision.value}/{index:03d}",
            version="v1",
        ),
        annotation_contract_ref=_ref(
            "annotation-contract-manifest",
            "independent-label/v1",
            version="v1",
        ),
        expected_decision=decision,
        reference_state="REFERENCE_READY",
        reference_policy_version="independent-label-reference/v1",
    )


def _candidate_pool(
    candidates: tuple[IndependentLabelReferenceCandidateV1, ...],
) -> IndependentLabelCandidatePoolV1:
    return IndependentLabelCandidatePoolV1.create(
        candidates=candidates,
        source_population_ref=_ref(
            "independent-label-source-population",
            "approved",
            version="v1",
        ),
        audit=_audit(),
    )


def _partition(
    kind: LabelTestSetPartitionKindV2,
    *,
    members: tuple[tuple[str, str], ...] = (),
) -> tuple[IndependentLabelPartitionManifestV2, IndependentLabelPartitionMaterialV1]:
    source_ref = _ref("partition-source-authority", kind.value.casefold(), version="v1")
    material = IndependentLabelPartitionMaterialV1.create(
        partition_kind=kind,
        members=members,
        source_authority_refs=(source_ref,),
        audit=_audit(),
    )
    manifest = IndependentLabelPartitionManifestV2.create(
        partition_kind=kind,
        private_material_ref=independent_label_partition_material_v1_ref(material),
        trace_count=len({item[0] for item in members}),
        unique_raw_hash_count=len({item[1] for item in members}),
        source_authority_refs=(source_ref,),
        audit=_audit(),
    )
    return manifest, material


def _compile(
    specs: tuple[LabelSpecV2, ...],
    candidates: tuple[IndependentLabelReferenceCandidateV1, ...],
    *,
    train_members: tuple[tuple[str, str], ...] = (),
    development_members: tuple[tuple[str, str], ...] = (),
):
    train_manifest, train_material = _partition(
        LabelTestSetPartitionKindV2.TRAIN,
        members=train_members,
    )
    development_manifest, development_material = _partition(
        LabelTestSetPartitionKindV2.DEVELOPMENT,
        members=development_members,
    )
    return IndependentLabelTestSetCompiler().compile(
        dataset_series_id="independent-label-test-set://first-labels",
        dataset_version="2026-08-02.v1",
        label_specs=specs,
        policy=_policy(specs),
        access_policy=_access_policy(),
        candidate_pool=_candidate_pool(candidates),
        train_partition_manifest=train_manifest,
        train_partition_material=train_material,
        development_partition_manifest=development_manifest,
        development_partition_material=development_material,
        supersedes_dataset_ref=None,
        supersedes_freeze_ref=None,
        audit=_audit(),
    )


def test_compiler_freezes_structured_and_semantic_minimums() -> None:
    search = _label_spec("search", semantic=False)
    shell = _label_spec("powershell", semantic=False)
    recovery = _label_spec("recovery", semantic=True)
    candidates = tuple(
        (
            *(_candidate(search, index=index, decision=LabelDecisionValueV2.MATCH) for index in range(60)),
            *(_candidate(search, index=index, decision=LabelDecisionValueV2.NO_MATCH) for index in range(60)),
            *(_candidate(shell, index=index, decision=LabelDecisionValueV2.MATCH) for index in range(55)),
            *(_candidate(shell, index=index, decision=LabelDecisionValueV2.NO_MATCH) for index in range(55)),
            *(_candidate(recovery, index=index, decision=LabelDecisionValueV2.MATCH) for index in range(50)),
            *(
                _candidate(recovery, index=index, decision=LabelDecisionValueV2.NO_MATCH)
                for index in range(50)
            ),
            *(
                _candidate(recovery, index=index, decision=LabelDecisionValueV2.ABSTAIN)
                for index in range(50)
            ),
        )
    )

    compilation = _compile((search, shell, recovery), candidates)

    assert compilation.result.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.FROZEN
    assert compilation.result.dataset_manifest is not None
    assert compilation.selected_material is not None
    summaries = {
        item.label_spec_ref.object_id: item for item in compilation.result.freeze_record.label_summaries
    }
    assert summaries[search.label_spec_id].selected_total == 100
    assert summaries[shell.label_spec_id].selected_total == 100
    assert summaries[recovery.label_spec_id].selected_total == 100
    assert summaries[recovery.label_spec_id].balance_status is LabelTestSetBalanceStatusV2.ACHIEVED
    semantic_counts = summaries[recovery.label_spec_id].class_counts
    assert tuple(item.selected_count for item in semantic_counts) == (34, 33, 33)


def test_compiler_keeps_91_trace_semantic_population_pending() -> None:
    recovery = _label_spec("recovery", semantic=True)
    candidates = tuple(
        (
            *(_candidate(recovery, index=index, decision=LabelDecisionValueV2.MATCH) for index in range(46)),
            *(
                _candidate(recovery, index=index, decision=LabelDecisionValueV2.NO_MATCH)
                for index in range(45)
            ),
        )
    )

    compilation = _compile((recovery,), candidates)

    assert compilation.result.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING
    assert compilation.result.dataset_manifest is None
    assert compilation.selected_material is None
    assert compilation.result.freeze_record.shortages[0].required_count == 100
    assert compilation.result.freeze_record.shortages[0].available_count == 91


def test_compiler_freezes_source_limited_semantic_population() -> None:
    recovery = _label_spec("recovery", semantic=True)
    candidates = tuple(
        (
            *(
                _candidate(
                    recovery,
                    index=index,
                    decision=LabelDecisionValueV2.MATCH,
                )
                for index in range(70)
            ),
            *(
                _candidate(
                    recovery,
                    index=index,
                    decision=LabelDecisionValueV2.NO_MATCH,
                )
                for index in range(70)
            ),
        )
    )

    compilation = _compile((recovery,), candidates)

    assert compilation.result.dataset_manifest is not None
    summary = compilation.result.freeze_record.label_summaries[0]
    assert summary.balance_status is LabelTestSetBalanceStatusV2.SOURCE_POPULATION_LIMITED
    assert tuple(item.selected_count for item in summary.class_counts) == (
        50,
        50,
        0,
    )


def test_compiler_rejects_partition_overlap_and_raw_hash_aliases() -> None:
    search = _label_spec("search", semantic=False)
    first = _candidate(search, index=1, decision=LabelDecisionValueV2.MATCH)
    negatives = tuple(
        _candidate(search, index=index, decision=LabelDecisionValueV2.NO_MATCH) for index in range(50)
    )

    with pytest.raises(IndependentLabelTestSetPolicyError, match="TRAIN"):
        _compile(
            (search,),
            (first, *negatives),
            train_members=((first.source_trace_id, first.raw_sha256),),
        )

    alias = _candidate(
        search,
        index=2,
        decision=LabelDecisionValueV2.MATCH,
        raw_sha256=first.raw_sha256,
    )
    with pytest.raises(IndependentLabelTestSetPolicyError, match="raw hash"):
        _compile((search,), (first, alias, *negatives))


def test_compiler_is_order_and_audit_time_stable() -> None:
    recovery = _label_spec("recovery", semantic=True)
    candidates = tuple(
        _candidate(
            recovery,
            index=index,
            decision=(
                LabelDecisionValueV2.MATCH
                if index % 3 == 0
                else (LabelDecisionValueV2.NO_MATCH if index % 3 == 1 else LabelDecisionValueV2.ABSTAIN)
            ),
        )
        for index in range(120)
    )

    first = _compile((recovery,), candidates)
    second = _compile((recovery,), tuple(reversed(candidates)))

    assert first.result.to_ref() == second.result.to_ref()
    assert first.selected_material is not None
    assert second.selected_material is not None
    assert first.selected_material.to_ref() == second.selected_material.to_ref()


def test_compiler_rejects_stale_label_spec_behavior_hash() -> None:
    search = _label_spec("search", semantic=False)
    stale = search.model_copy(update={"requirement": "Changed without rehash."})
    candidates = tuple(
        (
            *(_candidate(search, index=index, decision=LabelDecisionValueV2.MATCH) for index in range(50)),
            *(_candidate(search, index=index, decision=LabelDecisionValueV2.NO_MATCH) for index in range(50)),
        )
    )

    with pytest.raises(IndependentLabelTestSetPolicyError, match="LabelSpec"):
        _compile((stale,), candidates)


def test_private_candidate_boundary_rejects_duplicates_and_forged_authority() -> None:
    search = _label_spec("search", semantic=False)
    candidate = _candidate(
        search,
        index=1,
        decision=LabelDecisionValueV2.MATCH,
    )

    with pytest.raises(ValidationError, match="duplicate trace-label pair"):
        _candidate_pool((candidate, candidate))
    with pytest.raises(ValidationError, match="annotation_ref"):
        IndependentLabelReferenceCandidateV1.model_validate(
            {
                **candidate.model_dump(mode="python"),
                "annotation_ref": _ref("label-annotation", "forged"),
            }
        )
    with pytest.raises(ValidationError):
        IndependentLabelReferenceCandidateV1.model_validate(
            {
                **candidate.model_dump(mode="python"),
                "raw_trace": "forbidden",
            }
        )
