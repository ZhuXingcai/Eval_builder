from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    label_decision_ref,
)
from eval_factory.contracts.statistics_v2 import (
    LabelTestSetAccessPurposeV2,
)
from eval_factory.statistics.models import (
    IndependentLabelReferenceCandidateV1,
    IndependentLabelTestSetAccessRequestV1,
    TrustedIndependentLabelTestSetPrincipalV1,
)


class LabelQualityPairStatusV1(StrEnum):
    FINAL_MATCH = "FINAL_MATCH"
    FINAL_NO_MATCH = "FINAL_NO_MATCH"
    VALID_ABSTAIN = "VALID_ABSTAIN"
    ABSTAIN = "ABSTAIN"
    INCOMPLETE = "INCOMPLETE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MISSING = "MISSING"


class LabelQualityObservationSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-observation-set/private-v1"] = (
        "eval-factory/label-quality-observation-set/private-v1"
    )
    observation_set_id: Identifier
    dataset_manifest_ref: ObjectRef
    label_specs: tuple[LabelSpecV2, ...] = Field(min_length=1, max_length=100)
    decisions: tuple[LabelDecisionV2, ...] = Field(max_length=10_000_000)
    observation_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_observation_set(self) -> Self:
        _require_ref(
            self.dataset_manifest_ref,
            "independent-label-test-set-manifest",
            "v2",
            "dataset_manifest_ref",
        )
        spec_refs = tuple(_label_spec_ref(value) for value in self.label_specs)
        _require_sorted_unique_refs(spec_refs, "label_specs")
        for spec in self.label_specs:
            if spec.label_spec_sha256 != _label_spec_carried_sha256(spec):
                raise ValueError("observation set contains stale LabelSpec")
        decision_keys = tuple(_decision_key(value) for value in self.decisions)
        if decision_keys != tuple(sorted(decision_keys)):
            raise ValueError("observation decisions must be sorted")
        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError("observation set contains duplicate trace-label pair")
        decision_refs = tuple(label_decision_ref(value) for value in self.decisions)
        if len(decision_refs) != len(set(decision_refs)):
            raise ValueError("observation set contains duplicate decision ref")
        allowed_specs = set(spec_refs)
        for decision, ref in zip(self.decisions, decision_refs, strict=True):
            if decision.label_spec_ref not in allowed_specs:
                raise ValueError("observation decision label is outside portfolio")
            if (
                decision.label_decision_id != f"label-decision://sha256/{decision.decision_sha256}"
                or ref.object_sha256 != decision.decision_sha256
            ):
                raise ValueError("observation decision identity is stale")
        refs = (
            self.dataset_manifest_ref,
            *spec_refs,
            *decision_refs,
        )
        _require_audit(self.audit, refs, "label quality observation set")
        _validate_identity(
            self.observation_set_id,
            self.observation_set_sha256,
            "label-quality-observation-set",
            label_quality_observation_set_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_manifest_ref: ObjectRef,
        label_specs: tuple[LabelSpecV2, ...],
        decisions: tuple[LabelDecisionV2, ...],
        audit: ContractAudit,
    ) -> LabelQualityObservationSetV1:
        specs = tuple(sorted(label_specs, key=lambda value: _ref_key(_label_spec_ref(value))))
        ordered_decisions = tuple(sorted(decisions, key=_decision_key))
        refs = (
            dataset_manifest_ref,
            *(_label_spec_ref(value) for value in specs),
            *(label_decision_ref(value) for value in ordered_decisions),
        )
        value = cls(
            observation_set_id="label-quality-observation-set://pending",
            dataset_manifest_ref=dataset_manifest_ref,
            label_specs=specs,
            decisions=ordered_decisions,
            observation_set_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "observation_set_id",
            "observation_set_sha256",
            "label-quality-observation-set",
            label_quality_observation_set_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_observation_set_v1_ref(self)


class LabelQualityTrustedPrincipalV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-trusted-principal/private-v1"] = (
        "eval-factory/label-quality-trusted-principal/private-v1"
    )
    principal_ref: ObjectRef
    allowed_purposes: tuple[LabelTestSetAccessPurposeV2, ...] = Field(min_length=1)
    allowed_dataset_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    max_members: int = Field(ge=1, le=10_000_000)
    audit_case_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def validate_principal(self) -> Self:
        _require_ref(
            self.principal_ref,
            "statistical-principal",
            "v1",
            "principal_ref",
        )
        if self.allowed_purposes != tuple(sorted(set(self.allowed_purposes), key=lambda value: value.value)):
            raise ValueError("principal purposes must be sorted and unique")
        _require_sorted_unique_refs(
            self.allowed_dataset_refs,
            "allowed_dataset_refs",
        )
        for ref in self.allowed_dataset_refs:
            _require_ref(
                ref,
                "independent-label-test-set-manifest",
                "v2",
                "allowed_dataset_refs",
            )
        if (
            LabelTestSetAccessPurposeV2.R8_LABEL_TEST_SET_INTEGRITY_AUDIT in self.allowed_purposes
            and self.audit_case_ref is None
        ):
            raise ValueError("integrity-audit principal requires audit_case_ref")
        return self

    @classmethod
    def from_domain(
        cls,
        value: TrustedIndependentLabelTestSetPrincipalV1,
    ) -> LabelQualityTrustedPrincipalV1:
        return cls(
            principal_ref=value.principal_ref,
            allowed_purposes=tuple(sorted(value.allowed_purposes, key=lambda item: item.value)),
            allowed_dataset_refs=tuple(sorted(value.allowed_dataset_refs, key=_ref_key)),
            max_members=value.max_members,
            audit_case_ref=value.audit_case_ref,
        )

    def to_domain(self) -> TrustedIndependentLabelTestSetPrincipalV1:
        return TrustedIndependentLabelTestSetPrincipalV1(
            principal_ref=self.principal_ref,
            allowed_purposes=frozenset(self.allowed_purposes),
            allowed_dataset_refs=frozenset(self.allowed_dataset_refs),
            max_members=self.max_members,
            audit_case_ref=self.audit_case_ref,
        )


class LabelQualityFrozenEvaluationRequestV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-frozen-evaluation-request/private-v1"] = (
        "eval-factory/label-quality-frozen-evaluation-request/private-v1"
    )
    expected_freeze_result_ref: ObjectRef
    dataset_series_id: Identifier
    label_specs: tuple[LabelSpecV2, ...] = Field(min_length=1, max_length=100)
    principal: LabelQualityTrustedPrincipalV1
    access_request: IndependentLabelTestSetAccessRequestV1
    observation_set: LabelQualityObservationSetV1
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.expected_freeze_result_ref,
            "independent-label-test-set-freeze-result",
            "v2",
            "expected_freeze_result_ref",
        )
        if self.access_request.purpose is not LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION:
            raise ValueError("label quality access purpose is invalid")
        if self.observation_set.dataset_manifest_ref != self.access_request.dataset_manifest_ref:
            raise ValueError("observation set and access request use different datasets")
        expected_specs = tuple(
            sorted(
                (_label_spec_ref(value) for value in self.label_specs),
                key=_ref_key,
            )
        )
        observed_specs = tuple(_label_spec_ref(value) for value in self.observation_set.label_specs)
        if expected_specs != observed_specs:
            raise ValueError("frozen request LabelSpecs differ from observations")
        refs = (
            self.expected_freeze_result_ref,
            self.access_request.dataset_manifest_ref,
            self.access_request.access_policy_ref,
            self.principal.principal_ref,
            self.observation_set.to_ref(),
            *expected_specs,
        )
        _require_audit(self.audit, refs, "label quality frozen request")
        return self

    @classmethod
    def create(
        cls,
        *,
        expected_freeze_result_ref: ObjectRef,
        dataset_series_id: str,
        label_specs: tuple[LabelSpecV2, ...],
        principal: LabelQualityTrustedPrincipalV1,
        access_request: IndependentLabelTestSetAccessRequestV1,
        observation_set: LabelQualityObservationSetV1,
        audit: ContractAudit,
    ) -> LabelQualityFrozenEvaluationRequestV1:
        specs = tuple(sorted(label_specs, key=lambda value: _ref_key(_label_spec_ref(value))))
        refs = (
            expected_freeze_result_ref,
            access_request.dataset_manifest_ref,
            access_request.access_policy_ref,
            principal.principal_ref,
            observation_set.to_ref(),
            *(_label_spec_ref(value) for value in specs),
        )
        return cls(
            expected_freeze_result_ref=expected_freeze_result_ref,
            dataset_series_id=dataset_series_id,
            label_specs=specs,
            principal=principal,
            access_request=access_request,
            observation_set=observation_set,
            audit=_safe_audit(audit, refs),
        )


class LabelQualityPairResultV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-pair-result/private-v1"] = (
        "eval-factory/label-quality-pair-result/private-v1"
    )
    pair_result_id: Identifier
    reference: IndependentLabelReferenceCandidateV1
    decision: LabelDecisionV2 | None = None
    status: LabelQualityPairStatusV1
    pair_result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.status is LabelQualityPairStatusV1.MISSING:
            if self.decision is not None:
                raise ValueError("missing pair cannot carry a decision")
        else:
            if self.decision is None:
                raise ValueError("observed pair requires a decision")
            if (
                self.decision.trace_envelope_ref != self.reference.trace_envelope_ref
                or self.decision.label_spec_ref != self.reference.label_spec_ref
            ):
                raise ValueError("pair decision differs from reference authority")
            if self.decision.label_decision_id != f"label-decision://sha256/{self.decision.decision_sha256}":
                raise ValueError("pair decision identity is stale")
            if self.status is LabelQualityPairStatusV1.FINAL_MATCH and (
                self.decision.decision is not LabelDecisionValueV2.MATCH
                or self.decision.execution_status is not LabelExecutionStatus.FINAL
            ):
                raise ValueError("FINAL_MATCH pair classification is invalid")
            if self.status is LabelQualityPairStatusV1.FINAL_NO_MATCH and (
                self.decision.decision is not LabelDecisionValueV2.NO_MATCH
                or self.decision.execution_status is not LabelExecutionStatus.FINAL
            ):
                raise ValueError("FINAL_NO_MATCH pair classification is invalid")
            if (
                self.status
                in {
                    LabelQualityPairStatusV1.VALID_ABSTAIN,
                    LabelQualityPairStatusV1.ABSTAIN,
                    LabelQualityPairStatusV1.MODEL_UNAVAILABLE,
                }
                and self.decision.decision is not LabelDecisionValueV2.ABSTAIN
            ):
                raise ValueError("abstain pair classification is invalid")
        refs = (
            self.reference.trace_envelope_ref,
            self.reference.label_spec_ref,
            self.reference.annotation_ref,
            *(() if self.decision is None else (label_decision_ref(self.decision),)),
        )
        _require_audit(self.audit, refs, "label quality pair result")
        _validate_identity(
            self.pair_result_id,
            self.pair_result_sha256,
            "label-quality-pair-result",
            label_quality_pair_result_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        reference: IndependentLabelReferenceCandidateV1,
        decision: LabelDecisionV2 | None,
        status: LabelQualityPairStatusV1,
        audit: ContractAudit,
    ) -> LabelQualityPairResultV1:
        refs = (
            reference.trace_envelope_ref,
            reference.label_spec_ref,
            reference.annotation_ref,
            *((label_decision_ref(decision),) if decision is not None else ()),
        )
        value = cls(
            pair_result_id="label-quality-pair-result://pending",
            reference=reference,
            decision=decision,
            status=status,
            pair_result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "pair_result_id",
            "pair_result_sha256",
            "label-quality-pair-result",
            label_quality_pair_result_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_pair_result_v1_ref(self)


class LabelQualityResultSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-result-set/private-v1"] = (
        "eval-factory/label-quality-result-set/private-v1"
    )
    result_set_id: Identifier
    dataset_manifest_ref: ObjectRef
    observation_set_ref: ObjectRef
    pair_results: tuple[LabelQualityPairResultV1, ...] = Field(
        min_length=1,
        max_length=10_000_000,
    )
    result_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result_set(self) -> Self:
        _require_ref(
            self.dataset_manifest_ref,
            "independent-label-test-set-manifest",
            "v2",
            "dataset_manifest_ref",
        )
        _require_ref(
            self.observation_set_ref,
            "label-quality-observation-set",
            "private-v1",
            "observation_set_ref",
        )
        keys = tuple(_pair_result_key(value) for value in self.pair_results)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("label quality pair results must be sorted and unique")
        refs = (
            self.dataset_manifest_ref,
            self.observation_set_ref,
            *(value.to_ref() for value in self.pair_results),
        )
        _require_audit(self.audit, refs, "label quality result set")
        _validate_identity(
            self.result_set_id,
            self.result_set_sha256,
            "label-quality-result-set",
            label_quality_result_set_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_manifest_ref: ObjectRef,
        observation_set_ref: ObjectRef,
        pair_results: tuple[LabelQualityPairResultV1, ...],
        audit: ContractAudit,
    ) -> LabelQualityResultSetV1:
        ordered = tuple(sorted(pair_results, key=_pair_result_key))
        refs = (
            dataset_manifest_ref,
            observation_set_ref,
            *(value.to_ref() for value in ordered),
        )
        value = cls(
            result_set_id="label-quality-result-set://pending",
            dataset_manifest_ref=dataset_manifest_ref,
            observation_set_ref=observation_set_ref,
            pair_results=ordered,
            result_set_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_set_id",
            "result_set_sha256",
            "label-quality-result-set",
            label_quality_result_set_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_result_set_v1_ref(self)


class LabelQualityAcceptedRunV1(ContractModelV2):
    schema_version: Literal["eval-factory/label-quality-accepted-run/private-v1"] = (
        "eval-factory/label-quality-accepted-run/private-v1"
    )
    accepted_run_id: Identifier
    acceptance_key: Identifier
    request_sha256: Sha256
    report_ref: ObjectRef
    observation_set_ref: ObjectRef | None = None
    result_set_ref: ObjectRef | None = None
    accepted_run_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        _require_ref(
            self.report_ref,
            "label-quality-evaluation-report",
            "v2",
            "report_ref",
        )
        if (self.observation_set_ref is None) != (self.result_set_ref is None):
            raise ValueError("accepted run private closure must be all-or-none")
        if self.observation_set_ref is not None:
            _require_ref(
                self.observation_set_ref,
                "label-quality-observation-set",
                "private-v1",
                "observation_set_ref",
            )
            if self.result_set_ref is None:
                raise ValueError("accepted run result-set ref is missing")
            _require_ref(
                self.result_set_ref,
                "label-quality-result-set",
                "private-v1",
                "result_set_ref",
            )
        refs = (
            self.report_ref,
            *((self.observation_set_ref,) if self.observation_set_ref is not None else ()),
            *((self.result_set_ref,) if self.result_set_ref is not None else ()),
        )
        _require_audit(self.audit, refs, "label quality accepted run")
        _validate_identity(
            self.accepted_run_id,
            self.accepted_run_sha256,
            "label-quality-accepted-run",
            label_quality_accepted_run_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        acceptance_key: str,
        request_sha256: str,
        report_ref: ObjectRef,
        observation_set_ref: ObjectRef | None,
        result_set_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> LabelQualityAcceptedRunV1:
        refs = (
            report_ref,
            *((observation_set_ref,) if observation_set_ref is not None else ()),
            *((result_set_ref,) if result_set_ref is not None else ()),
        )
        value = cls(
            accepted_run_id="label-quality-accepted-run://pending",
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            report_ref=report_ref,
            observation_set_ref=observation_set_ref,
            result_set_ref=result_set_ref,
            accepted_run_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "accepted_run_id",
            "accepted_run_sha256",
            "label-quality-accepted-run",
            label_quality_accepted_run_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return label_quality_accepted_run_v1_ref(self)


def label_quality_observation_set_v1_carried_sha256(
    value: LabelQualityObservationSetV1,
) -> str:
    return _carried(
        value,
        {"observation_set_id", "observation_set_sha256", "audit"},
    )


def label_quality_pair_result_v1_carried_sha256(
    value: LabelQualityPairResultV1,
) -> str:
    return _carried(value, {"pair_result_id", "pair_result_sha256", "audit"})


def label_quality_result_set_v1_carried_sha256(
    value: LabelQualityResultSetV1,
) -> str:
    return _carried(value, {"result_set_id", "result_set_sha256", "audit"})


def label_quality_accepted_run_v1_carried_sha256(
    value: LabelQualityAcceptedRunV1,
) -> str:
    return _carried(value, {"accepted_run_id", "accepted_run_sha256", "audit"})


def label_quality_observation_set_v1_ref(
    value: LabelQualityObservationSetV1,
) -> ObjectRef:
    _validate_identity(
        value.observation_set_id,
        value.observation_set_sha256,
        "label-quality-observation-set",
        label_quality_observation_set_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "label-quality-observation-set",
        value.observation_set_id,
        "private-v1",
        value.observation_set_sha256,
    )


def label_quality_pair_result_v1_ref(
    value: LabelQualityPairResultV1,
) -> ObjectRef:
    _validate_identity(
        value.pair_result_id,
        value.pair_result_sha256,
        "label-quality-pair-result",
        label_quality_pair_result_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "label-quality-pair-result",
        value.pair_result_id,
        "private-v1",
        value.pair_result_sha256,
    )


def label_quality_result_set_v1_ref(
    value: LabelQualityResultSetV1,
) -> ObjectRef:
    _validate_identity(
        value.result_set_id,
        value.result_set_sha256,
        "label-quality-result-set",
        label_quality_result_set_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "label-quality-result-set",
        value.result_set_id,
        "private-v1",
        value.result_set_sha256,
    )


def label_quality_accepted_run_v1_ref(
    value: LabelQualityAcceptedRunV1,
) -> ObjectRef:
    _validate_identity(
        value.accepted_run_id,
        value.accepted_run_sha256,
        "label-quality-accepted-run",
        label_quality_accepted_run_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "label-quality-accepted-run",
        value.accepted_run_id,
        "private-v1",
        value.accepted_run_sha256,
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


def _decision_key(
    value: LabelDecisionV2,
) -> tuple[tuple[str, str, str, str], tuple[str, str, str, str]]:
    return (_ref_key(value.label_spec_ref), _ref_key(value.trace_envelope_ref))


def _pair_result_key(
    value: LabelQualityPairResultV1,
) -> tuple[tuple[str, str, str, str], tuple[str, str, str, str]]:
    return (
        _ref_key(value.reference.label_spec_ref),
        _ref_key(value.reference.trace_envelope_ref),
    )


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _ref(
    object_type: str,
    object_id: str,
    version: str,
    digest: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=version,
        object_sha256=digest,
    )


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


def _carried(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")
