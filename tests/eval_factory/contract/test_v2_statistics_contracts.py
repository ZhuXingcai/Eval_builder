from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.contracts.statistics_v2 import (
    INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION,
    IndependentLabelPartitionManifestV2,
    IndependentLabelTestSetAccessPolicyV2,
    IndependentLabelTestSetAccessReceiptV2,
    IndependentLabelTestSetFreezeRecordV2,
    IndependentLabelTestSetFreezeResultV2,
    IndependentLabelTestSetLabelPolicyV2,
    IndependentLabelTestSetManifestV2,
    IndependentLabelTestSetPolicyV2,
    IndependentLabelTestSetPrincipalGrantV2,
    LabelTestSetAccessOutcomeV2,
    LabelTestSetAccessPurposeV2,
    LabelTestSetAccessReasonV2,
    LabelTestSetBalanceStatusV2,
    LabelTestSetClassCountV2,
    LabelTestSetEvaluationKindV2,
    LabelTestSetFreezeOutcomeV2,
    LabelTestSetLabelSummaryV2,
    LabelTestSetPartitionKindV2,
    LabelTestSetSelectionAlgorithmV2,
    LabelTestSetShortageReasonV2,
    LabelTestSetShortageV2,
    independent_label_test_set_freeze_result_v2_ref,
    independent_label_test_set_manifest_v2_ref,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[3]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


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
        created_by="r8-01-contract-test",
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


def _label_ref(suffix: str) -> ObjectRef:
    return _ref("label-spec", suffix)


def _label_policies() -> tuple[IndependentLabelTestSetLabelPolicyV2, ...]:
    return (
        IndependentLabelTestSetLabelPolicyV2(
            label_spec_ref=_label_ref("powershell"),
            evaluation_kind=LabelTestSetEvaluationKindV2.STRUCTURED,
            structured_positive_minimum=50,
            structured_negative_minimum=50,
            semantic_total_minimum=0,
            semantic_classes=(),
        ),
        IndependentLabelTestSetLabelPolicyV2(
            label_spec_ref=_label_ref("recovery"),
            evaluation_kind=LabelTestSetEvaluationKindV2.SEMANTIC,
            structured_positive_minimum=0,
            structured_negative_minimum=0,
            semantic_total_minimum=100,
            semantic_classes=(
                LabelDecisionValueV2.MATCH,
                LabelDecisionValueV2.NO_MATCH,
                LabelDecisionValueV2.ABSTAIN,
            ),
        ),
        IndependentLabelTestSetLabelPolicyV2(
            label_spec_ref=_label_ref("search"),
            evaluation_kind=LabelTestSetEvaluationKindV2.STRUCTURED,
            structured_positive_minimum=50,
            structured_negative_minimum=50,
            semantic_total_minimum=0,
            semantic_classes=(),
        ),
    )


def _policy() -> IndependentLabelTestSetPolicyV2:
    return IndependentLabelTestSetPolicyV2.create(
        label_policies=_label_policies(),
        selection_seed="r8-01-approved-seed",
        max_candidates=10_000,
        max_selected_members=1_000,
        max_private_bytes=50_000_000,
        audit=_audit(),
    )


def _principal() -> ObjectRef:
    return _ref("statistical-principal", "evaluator", version="v1")


def _access_policy() -> IndependentLabelTestSetAccessPolicyV2:
    grant = IndependentLabelTestSetPrincipalGrantV2(
        principal_ref=_principal(),
        purposes=frozenset(
            {
                LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
                LabelTestSetAccessPurposeV2.R8_LABEL_TEST_SET_INTEGRITY_AUDIT,
            }
        ),
    )
    return IndependentLabelTestSetAccessPolicyV2.create(
        principal_grants=(grant,),
        max_members_per_read=1_000,
        audit=_audit(),
    )


def _partition(kind: LabelTestSetPartitionKindV2) -> IndependentLabelPartitionManifestV2:
    return IndependentLabelPartitionManifestV2.create(
        partition_kind=kind,
        private_material_ref=_ref(
            "independent-label-partition-material",
            kind.value.casefold(),
            version="private-v1",
        ),
        trace_count=10,
        unique_raw_hash_count=10,
        source_authority_refs=(_ref("partition-source-authority", kind.value.casefold()),),
        audit=_audit(),
    )


def _class_count(
    decision: LabelDecisionValueV2,
    *,
    eligible: int,
    selected: int,
    required: int,
) -> LabelTestSetClassCountV2:
    return LabelTestSetClassCountV2(
        decision=decision,
        eligible_count=eligible,
        selected_count=selected,
        required_minimum=required,
    )


def _summaries() -> tuple[LabelTestSetLabelSummaryV2, ...]:
    structured = (
        _class_count(LabelDecisionValueV2.MATCH, eligible=70, selected=50, required=50),
        _class_count(LabelDecisionValueV2.NO_MATCH, eligible=80, selected=50, required=50),
        _class_count(LabelDecisionValueV2.ABSTAIN, eligible=3, selected=0, required=0),
    )
    semantic = (
        _class_count(LabelDecisionValueV2.MATCH, eligible=50, selected=34, required=0),
        _class_count(LabelDecisionValueV2.NO_MATCH, eligible=50, selected=33, required=0),
        _class_count(LabelDecisionValueV2.ABSTAIN, eligible=50, selected=33, required=0),
    )
    return (
        LabelTestSetLabelSummaryV2(
            label_spec_ref=_label_ref("powershell"),
            evaluation_kind=LabelTestSetEvaluationKindV2.STRUCTURED,
            class_counts=structured,
            required_total=100,
            eligible_total=153,
            selected_total=100,
            balance_status=LabelTestSetBalanceStatusV2.ACHIEVED,
        ),
        LabelTestSetLabelSummaryV2(
            label_spec_ref=_label_ref("recovery"),
            evaluation_kind=LabelTestSetEvaluationKindV2.SEMANTIC,
            class_counts=semantic,
            required_total=100,
            eligible_total=150,
            selected_total=100,
            balance_status=LabelTestSetBalanceStatusV2.ACHIEVED,
        ),
        LabelTestSetLabelSummaryV2(
            label_spec_ref=_label_ref("search"),
            evaluation_kind=LabelTestSetEvaluationKindV2.STRUCTURED,
            class_counts=structured,
            required_total=100,
            eligible_total=153,
            selected_total=100,
            balance_status=LabelTestSetBalanceStatusV2.ACHIEVED,
        ),
    )


def _frozen_result() -> IndependentLabelTestSetFreezeResultV2:
    policy = _policy()
    access = _access_policy()
    train = _partition(LabelTestSetPartitionKindV2.TRAIN)
    development = _partition(LabelTestSetPartitionKindV2.DEVELOPMENT)
    summaries = _summaries()
    candidate_pool_ref = _ref(
        "independent-label-candidate-pool",
        "pool",
        version="private-v1",
    )
    manifest = IndependentLabelTestSetManifestV2.create(
        dataset_series_id="independent-label-test-set://first-labels",
        dataset_version="2026-08-02.v1",
        private_material_ref=_ref(
            "independent-label-test-set-material",
            "dataset",
            version="private-v1",
        ),
        policy_ref=policy.to_ref(),
        access_policy_ref=access.to_ref(),
        train_partition_ref=train.to_ref(),
        development_partition_ref=development.to_ref(),
        candidate_pool_ref=candidate_pool_ref,
        label_summaries=summaries,
        selected_member_count=300,
        unique_trace_count=250,
        unique_raw_hash_count=250,
        selection_algorithm=LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1,
        selection_seed=policy.selection_seed,
        supersedes_dataset_ref=None,
        audit=_audit(),
    )
    record = IndependentLabelTestSetFreezeRecordV2.create(
        dataset_series_id=manifest.dataset_series_id,
        requested_dataset_version=manifest.dataset_version,
        outcome=LabelTestSetFreezeOutcomeV2.FROZEN,
        candidate_pool_ref=candidate_pool_ref,
        policy_ref=policy.to_ref(),
        access_policy_ref=access.to_ref(),
        train_partition_ref=train.to_ref(),
        development_partition_ref=development.to_ref(),
        label_summaries=summaries,
        shortages=(),
        dataset_manifest_ref=manifest.to_ref(),
        supersedes_freeze_ref=None,
        audit=_audit(),
    )
    return IndependentLabelTestSetFreezeResultV2.create(
        freeze_record=record,
        dataset_manifest=manifest,
        policy_ref=policy.to_ref(),
        access_policy_ref=access.to_ref(),
        partition_refs=(train.to_ref(), development.to_ref()),
        audit=_audit(),
    )


def test_statistics_contracts_are_strict_frozen_and_content_addressed() -> None:
    result = _frozen_result()

    assert result.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.FROZEN
    assert result.dataset_manifest is not None
    assert independent_label_test_set_freeze_result_v2_ref(result).object_sha256 == result.result_sha256
    assert (
        independent_label_test_set_manifest_v2_ref(result.dataset_manifest).object_sha256
        == result.dataset_manifest.dataset_sha256
    )
    with pytest.raises(ValidationError):
        result.result_sha256 = "f" * 64
    with pytest.raises(ValidationError):
        IndependentLabelTestSetFreezeResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "unknown": True,
            }
        )


def test_policy_requires_exact_nfr_008_thresholds_and_label_kinds() -> None:
    with pytest.raises(ValidationError, match="50"):
        _label_policies()[0].model_copy(update={"structured_positive_minimum": 49}).__class__.model_validate(
            {
                **_label_policies()[0].model_dump(mode="python"),
                "structured_positive_minimum": 49,
            }
        )
    with pytest.raises(ValidationError, match="100"):
        IndependentLabelTestSetLabelPolicyV2.model_validate(
            {
                **_label_policies()[1].model_dump(mode="python"),
                "semantic_total_minimum": 99,
            }
        )
    with pytest.raises(ValidationError, match="semantic classes"):
        IndependentLabelTestSetLabelPolicyV2.model_validate(
            {
                **_label_policies()[1].model_dump(mode="python"),
                "semantic_classes": (
                    LabelDecisionValueV2.MATCH,
                    LabelDecisionValueV2.NO_MATCH,
                ),
            }
        )


def test_pending_record_requires_shortage_and_forbids_dataset_authority() -> None:
    frozen = _frozen_result()
    manifest = frozen.dataset_manifest
    assert manifest is not None
    shortage = LabelTestSetShortageV2(
        label_spec_ref=_label_ref("recovery"),
        decision=None,
        reason=LabelTestSetShortageReasonV2.TOTAL_SHORTAGE,
        required_count=100,
        available_count=91,
    )
    pending = IndependentLabelTestSetFreezeRecordV2.create(
        dataset_series_id=manifest.dataset_series_id,
        requested_dataset_version="2026-08-02.pending",
        outcome=LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING,
        candidate_pool_ref=frozen.freeze_record.candidate_pool_ref,
        policy_ref=frozen.policy_ref,
        access_policy_ref=frozen.access_policy_ref,
        train_partition_ref=frozen.partition_refs[0],
        development_partition_ref=frozen.partition_refs[1],
        label_summaries=frozen.freeze_record.label_summaries,
        shortages=(shortage,),
        dataset_manifest_ref=None,
        supersedes_freeze_ref=frozen.freeze_record.to_ref(),
        audit=_audit(),
    )
    result = IndependentLabelTestSetFreezeResultV2.create(
        freeze_record=pending,
        dataset_manifest=None,
        policy_ref=frozen.policy_ref,
        access_policy_ref=frozen.access_policy_ref,
        partition_refs=frozen.partition_refs,
        audit=_audit(),
    )

    assert result.dataset_manifest is None
    with pytest.raises(ValidationError, match="forbids dataset"):
        IndependentLabelTestSetFreezeRecordV2.model_validate(
            pending.model_copy(update={"dataset_manifest_ref": manifest.to_ref()}).model_dump(mode="python")
        )
    with pytest.raises(ValidationError, match="requires shortage"):
        IndependentLabelTestSetFreezeRecordV2.model_validate(
            pending.model_copy(update={"shortages": ()}).model_dump(mode="python")
        )


def test_access_receipt_never_carries_private_members() -> None:
    result = _frozen_result()
    manifest = result.dataset_manifest
    assert manifest is not None
    granted = IndependentLabelTestSetAccessReceiptV2.create(
        dataset_manifest_ref=manifest.to_ref(),
        principal_ref=_principal(),
        purpose=LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
        access_policy_ref=result.access_policy_ref,
        outcome=LabelTestSetAccessOutcomeV2.GRANTED,
        reason=LabelTestSetAccessReasonV2.AUTHORIZED,
        returned_member_count=manifest.selected_member_count,
        material_verified=True,
        audit=_audit(),
    )
    denied = IndependentLabelTestSetAccessReceiptV2.create(
        dataset_manifest_ref=manifest.to_ref(),
        principal_ref=_ref("statistical-principal", "denied", version="v1"),
        purpose=LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
        access_policy_ref=result.access_policy_ref,
        outcome=LabelTestSetAccessOutcomeV2.DENIED,
        reason=LabelTestSetAccessReasonV2.PRINCIPAL_NOT_AUTHORIZED,
        returned_member_count=0,
        material_verified=False,
        audit=_audit(),
    )

    assert granted.material_verified is True
    assert denied.returned_member_count == 0
    for forbidden in (
        "annotation_ref",
        "expected_decision",
        "private_members",
        "raw_trace",
        "physical_path",
        "credential",
    ):
        assert forbidden not in granted.model_dump_json()
    with pytest.raises(ValidationError, match="denied access"):
        IndependentLabelTestSetAccessReceiptV2.model_validate(
            denied.model_copy(update={"returned_member_count": 1}).model_dump(mode="python")
        )


def test_audit_timestamp_does_not_change_behavior_identity() -> None:
    first = _policy()
    later = IndependentLabelTestSetPolicyV2.create(
        label_policies=first.label_policies,
        selection_seed=first.selection_seed,
        max_candidates=first.max_candidates,
        max_selected_members=first.max_selected_members,
        max_private_bytes=first.max_private_bytes,
        audit=_audit(created_at=datetime(2026, 8, 3, tzinfo=UTC)),
    )

    assert first.to_ref() == later.to_ref()


def test_generated_statistics_schemas_are_closed_and_gold_is_safe() -> None:
    manifest = json.loads(
        (ROOT / "specs/002-eval-dataset-factory/contracts/v2/manifest.json").read_text(encoding="utf-8")
    )
    contracts = [
        value
        for value in manifest["contracts"]
        if value["python_type"].startswith("eval_factory.contracts.statistics_v2.")
    ]
    assert len(contracts) == 12
    assert {value["owner"] for value in contracts} == {"factory"}
    for contract in contracts:
        schema = json.loads(
            (ROOT / "specs/002-eval-dataset-factory/contracts/v2" / contract["schema_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert schema["additionalProperties"] is False

    gold = json.loads(
        (
            ROOT / "evals/golden/eval_factory/statistics" / "r8-01-independent-label-test-set-freeze-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert gold["claim_scope"] == "MECHANISM_VALIDATION_ONLY"
    assert gold["real_corpus"]["outcome"] == "STATISTICAL_GATE_PENDING"
    assert gold["real_corpus"]["production_statistical_authority"] is False
    serialized = json.dumps(gold, ensure_ascii=False).casefold()
    for forbidden in (
        "annotation_ref",
        "credential_value",
        "expected_decision",
        "physical_path",
        "source_trace_id",
    ):
        assert forbidden not in serialized
