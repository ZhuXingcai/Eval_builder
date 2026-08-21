from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import ApprovalCheckpoint
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel
from eval_factory.contracts.release_v2 import (
    CheckpointDecisionBinding,
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec

RELEASE_PROJECTION_POLICY_VERSION: Literal["release-projection/r7-08-v1"] = "release-projection/r7-08-v1"

_CHECKPOINT_ORDER = {value: index for index, value in enumerate(ApprovalCheckpoint)}
_DENIED_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "completed-deliverable",
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-condition",
        "hidden-pass-condition",
        "private-reference",
        "provider-payload",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "secret",
        "trace-raw",
    }
)


class ReleaseProjectionPhaseV2(StrEnum):
    CANDIDATE = "CANDIDATE"
    TERMINAL = "TERMINAL"


class ReleaseProjectionPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/release-projection-policy/v2"] = (
        "eval-factory/release-projection-policy/v2"
    )
    release_projection_policy_id: Identifier
    max_source_trace_refs: int = Field(ge=1, le=100_000)
    max_label_decision_refs: int = Field(ge=1, le=100_000)
    max_user_decision_refs: int = Field(ge=1, le=100_000)
    max_revalidation_reports: int = Field(ge=1, le=100_000)
    max_current_head_refs: int = Field(ge=1, le=1_000_000)
    max_chain_depth: int = Field(ge=1, le=100_000)
    allowed_channels: frozenset[ReleaseChannel] = Field(min_length=1)
    allowed_export_profiles: frozenset[Literal["LH"]] = Field(min_length=1)
    policy_version: Literal["release-projection/r7-08-v1"] = RELEASE_PROJECTION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("allowed_channels", mode="before")
    @classmethod
    def parse_channels(cls, value: object) -> frozenset[ReleaseChannel]:
        if not isinstance(value, (set, frozenset, tuple, list)):
            raise TypeError("allowed_channels must be a collection")
        return frozenset(item if isinstance(item, ReleaseChannel) else ReleaseChannel(item) for item in value)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if ReleaseChannel.PRODUCTION in self.allowed_channels:
            raise ValueError("R7-08 policy cannot allow production")
        if self.allowed_export_profiles != frozenset({"LH"}):
            raise ValueError("R7-08 policy allows only the LH export profile")
        _validate_audit(self.audit, (), "release projection policy")
        _validate_identity(
            object_id=self.release_projection_policy_id,
            object_sha256=self.policy_sha256,
            prefix="release-projection-policy",
            observed=release_projection_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_source_trace_refs: int,
        max_label_decision_refs: int,
        max_user_decision_refs: int,
        max_revalidation_reports: int,
        max_current_head_refs: int,
        max_chain_depth: int,
        allowed_channels: frozenset[ReleaseChannel],
        allowed_export_profiles: frozenset[Literal["LH"]],
        audit: ContractAudit,
    ) -> ReleaseProjectionPolicyV2:
        value = cls(
            release_projection_policy_id="release-projection-policy://pending",
            max_source_trace_refs=max_source_trace_refs,
            max_label_decision_refs=max_label_decision_refs,
            max_user_decision_refs=max_user_decision_refs,
            max_revalidation_reports=max_revalidation_reports,
            max_current_head_refs=max_current_head_refs,
            max_chain_depth=max_chain_depth,
            allowed_channels=allowed_channels,
            allowed_export_profiles=allowed_export_profiles,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        digest = release_projection_policy_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "release_projection_policy_id": (f"release-projection-policy://sha256/{digest}"),
                "policy_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        return release_projection_policy_v2_ref(self)


class EvaluationItemReleaseSubjectV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluation-item-release-subject/v2"] = (
        "eval-factory/evaluation-item-release-subject/v2"
    )
    release_subject_id: Identifier
    job_id: Identifier
    item_id: Identifier
    source_trace_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    task_draft_ref: ObjectRef
    attachment_reconstruction_result_ref: ObjectRef
    components: EvaluationItemComponents
    item_quality_result_ref: ObjectRef
    batch_quality_report_ref: ObjectRef
    package_manifest_ref: ObjectRef
    package_sha256: Sha256
    user_approval_policy_ref: ObjectRef
    required_checkpoints: tuple[ApprovalCheckpoint, ...] = ()
    not_required_checkpoints: tuple[ApprovalCheckpoint, ...] = ()
    satisfied_checkpoint_bindings: tuple[CheckpointDecisionBinding, ...] = ()
    user_decision_record_refs: tuple[ObjectRef, ...] = ()
    relevant_revalidation_application_refs: tuple[ObjectRef, ...] = ()
    revalidation_report_refs: tuple[ObjectRef, ...] = ()
    current_head_refs: tuple[ObjectRef, ...] = ()
    channel: ReleaseChannel
    registry: Identifier
    export_profile: Literal["LH"]
    export_profile_version: str = Field(min_length=1, max_length=128)
    policy_ref: ObjectRef
    policy_version: Literal["release-projection/r7-08-v1"] = RELEASE_PROJECTION_POLICY_VERSION
    release_subject_sha256: Sha256
    audit: ContractAudit

    @field_validator("required_checkpoints", mode="before")
    @classmethod
    def parse_checkpoints(
        cls,
        value: object,
    ) -> tuple[ApprovalCheckpoint, ...]:
        if not isinstance(value, (set, frozenset, tuple, list)):
            raise TypeError("required_checkpoints must be a collection")
        return tuple(
            item if isinstance(item, ApprovalCheckpoint) else ApprovalCheckpoint(item) for item in value
        )

    @field_validator("channel", mode="before")
    @classmethod
    def parse_channel(cls, value: object) -> ReleaseChannel:
        if isinstance(value, ReleaseChannel):
            return value
        if isinstance(value, str):
            return ReleaseChannel(value)
        raise TypeError("channel must be a ReleaseChannel")

    @model_validator(mode="after")
    def validate_subject(self) -> Self:
        for label, refs in (
            ("release source trace refs", self.source_trace_refs),
            ("release label decision refs", self.label_decision_refs),
            (
                "release user decision record refs",
                self.user_decision_record_refs,
            ),
            (
                "release relevant revalidation application refs",
                self.relevant_revalidation_application_refs,
            ),
            ("release revalidation report refs", self.revalidation_report_refs),
            ("release current head refs", self.current_head_refs),
        ):
            _require_sorted_unique_refs(label, refs)
            _require_safe_refs(refs, label)
        for ref in self.source_trace_refs:
            _require_ref_type(ref, "trace-source", "source_trace_refs")
        for ref in self.label_decision_refs:
            _require_ref_type(ref, "label-decision", "label_decision_refs")
        for ref in self.user_decision_record_refs:
            _require_ref(
                ref,
                "user-decision-record",
                "v2",
                "user_decision_record_refs",
            )
        for ref in self.relevant_revalidation_application_refs:
            _require_ref(ref, "user-plan-application", "v2", "relevant application")
        for ref in self.revalidation_report_refs:
            _require_ref(
                ref,
                "directed-revalidation-report",
                "v2",
                "revalidation_report_refs",
            )
        if len(self.relevant_revalidation_application_refs) != len(self.revalidation_report_refs):
            raise ValueError("revalidation report coverage must match relevant applications")
        for ref, object_type, version, field_name in (
            (self.task_draft_ref, "task-draft", "v2", "task_draft_ref"),
            (
                self.attachment_reconstruction_result_ref,
                "attachment-reconstruction-result",
                "v2",
                "attachment_reconstruction_result_ref",
            ),
            (
                self.item_quality_result_ref,
                "item-quality-compilation-result",
                "v2",
                "item_quality_result_ref",
            ),
            (
                self.batch_quality_report_ref,
                "batch-quality-report",
                "v2",
                "batch_quality_report_ref",
            ),
            (
                self.package_manifest_ref,
                "final-package-manifest",
                "v2",
                "package_manifest_ref",
            ),
            (
                self.user_approval_policy_ref,
                "user-approval-policy",
                "v2",
                "user_approval_policy_ref",
            ),
            (
                self.policy_ref,
                "release-projection-policy",
                "v2",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, version, field_name)
            _require_safe_refs((ref,), field_name)
        _validate_components(self.components)
        _require_canonical_checkpoints(
            "release required checkpoints",
            self.required_checkpoints,
        )
        _require_canonical_checkpoints(
            "release not-required checkpoints",
            self.not_required_checkpoints,
        )
        if set(self.required_checkpoints).intersection(self.not_required_checkpoints):
            raise ValueError("required and not-required checkpoints must be disjoint")
        if any(
            checkpoint is not ApprovalCheckpoint.ENVIRONMENT_STRATEGY
            for checkpoint in self.not_required_checkpoints
        ):
            raise ValueError("only Environment Strategy may be not required")
        binding_checkpoints = tuple(binding.checkpoint for binding in self.satisfied_checkpoint_bindings)
        _require_canonical_checkpoints(
            "release satisfied checkpoints",
            binding_checkpoints,
        )
        if not set(binding_checkpoints).issubset(self.required_checkpoints):
            raise ValueError("satisfied checkpoints must be required by policy")
        for binding in self.satisfied_checkpoint_bindings:
            _require_ref(
                binding.request_ref,
                "user-approval-request",
                "v2",
                "checkpoint request",
            )
            _require_ref(
                binding.decision_record_ref,
                "user-decision-record",
                "v2",
                "checkpoint decision",
            )
        if self.channel is ReleaseChannel.PRODUCTION:
            raise ValueError("R7-08 release subject cannot target production")
        refs = _release_subject_refs(self)
        _validate_audit(self.audit, refs, "evaluation Item release subject")
        _validate_identity(
            object_id=self.release_subject_id,
            object_sha256=self.release_subject_sha256,
            prefix="evaluation-item-release-subject",
            observed=evaluation_item_release_subject_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_id: str,
        source_trace_refs: tuple[ObjectRef, ...],
        label_decision_refs: tuple[ObjectRef, ...],
        task_draft_ref: ObjectRef,
        attachment_reconstruction_result_ref: ObjectRef,
        components: EvaluationItemComponents,
        item_quality_result_ref: ObjectRef,
        batch_quality_report_ref: ObjectRef,
        package_manifest_ref: ObjectRef,
        package_sha256: str,
        user_approval_policy_ref: ObjectRef,
        required_checkpoints: tuple[ApprovalCheckpoint, ...],
        not_required_checkpoints: tuple[ApprovalCheckpoint, ...] = (),
        satisfied_checkpoint_bindings: tuple[CheckpointDecisionBinding, ...],
        user_decision_record_refs: tuple[ObjectRef, ...] = (),
        relevant_revalidation_application_refs: tuple[ObjectRef, ...],
        revalidation_report_refs: tuple[ObjectRef, ...],
        current_head_refs: tuple[ObjectRef, ...],
        channel: ReleaseChannel,
        registry: str,
        export_profile: Literal["LH"],
        export_profile_version: str,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> EvaluationItemReleaseSubjectV2:
        applications = _sorted_refs(relevant_revalidation_application_refs)
        reports = _sorted_refs(revalidation_report_refs)
        value = cls(
            release_subject_id="evaluation-item-release-subject://pending",
            job_id=job_id,
            item_id=item_id,
            source_trace_refs=_sorted_refs(source_trace_refs),
            label_decision_refs=_sorted_refs(label_decision_refs),
            task_draft_ref=task_draft_ref,
            attachment_reconstruction_result_ref=(attachment_reconstruction_result_ref),
            components=components,
            item_quality_result_ref=item_quality_result_ref,
            batch_quality_report_ref=batch_quality_report_ref,
            package_manifest_ref=package_manifest_ref,
            package_sha256=package_sha256,
            user_approval_policy_ref=user_approval_policy_ref,
            required_checkpoints=_sorted_checkpoints(required_checkpoints),
            not_required_checkpoints=_sorted_checkpoints(not_required_checkpoints),
            satisfied_checkpoint_bindings=_sorted_bindings(satisfied_checkpoint_bindings),
            user_decision_record_refs=_sorted_refs(user_decision_record_refs),
            relevant_revalidation_application_refs=applications,
            revalidation_report_refs=reports,
            current_head_refs=_sorted_refs(current_head_refs),
            channel=channel,
            registry=registry,
            export_profile=export_profile,
            export_profile_version=export_profile_version,
            policy_ref=policy_ref,
            release_subject_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    *source_trace_refs,
                    *label_decision_refs,
                    task_draft_ref,
                    attachment_reconstruction_result_ref,
                    *_component_refs(components),
                    item_quality_result_ref,
                    batch_quality_report_ref,
                    package_manifest_ref,
                    user_approval_policy_ref,
                    *(
                        ref
                        for binding in satisfied_checkpoint_bindings
                        for ref in (
                            binding.request_ref,
                            binding.decision_record_ref,
                        )
                    ),
                    *user_decision_record_refs,
                    *applications,
                    *reports,
                    *current_head_refs,
                    policy_ref,
                ),
            ),
        )
        digest = evaluation_item_release_subject_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "release_subject_id": (f"evaluation-item-release-subject://sha256/{digest}"),
                "release_subject_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        return evaluation_item_release_subject_v2_ref(self)


class ItemReleaseProjectionV2(ContractModelV2):
    schema_version: Literal["eval-factory/item-release-projection/v2"] = (
        "eval-factory/item-release-projection/v2"
    )
    projection_id: Identifier
    job_id: Identifier
    item_id: Identifier
    projection_revision: int = Field(ge=1)
    chain_id: Identifier
    previous_projection_ref: ObjectRef | None = None
    release_subject_ref: ObjectRef
    evaluation_item_ref: ObjectRef
    release_decision_ref: ObjectRef
    release_state: ReleaseStateV2
    pending_checkpoints: tuple[ApprovalCheckpoint, ...] = ()
    item_status: ItemStatus
    policy_version: Literal["release-projection/r7-08-v1"] = RELEASE_PROJECTION_POLICY_VERSION
    projection_sha256: Sha256
    audit: ContractAudit

    @field_validator("release_state", mode="before")
    @classmethod
    def parse_release_state(cls, value: object) -> ReleaseStateV2:
        if isinstance(value, ReleaseStateV2):
            return value
        if isinstance(value, str):
            return ReleaseStateV2(value)
        raise TypeError("release_state must be a ReleaseStateV2")

    @field_validator("item_status", mode="before")
    @classmethod
    def parse_item_status(cls, value: object) -> ItemStatus:
        if isinstance(value, ItemStatus):
            return value
        if isinstance(value, str):
            return ItemStatus(value)
        raise TypeError("item_status must be an ItemStatus")

    @field_validator("pending_checkpoints", mode="before")
    @classmethod
    def parse_pending(
        cls,
        value: object,
    ) -> tuple[ApprovalCheckpoint, ...]:
        if not isinstance(value, (set, frozenset, tuple, list)):
            raise TypeError("pending_checkpoints must be a collection")
        return tuple(
            item if isinstance(item, ApprovalCheckpoint) else ApprovalCheckpoint(item) for item in value
        )

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.release_state not in {
            ReleaseStateV2.CANDIDATE,
            ReleaseStateV2.APPROVED,
            ReleaseStateV2.REJECTED,
        }:
            raise ValueError("R7-08 projection cannot claim released or revoked state")
        if self.projection_revision == 1 and self.previous_projection_ref is not None:
            raise ValueError("first Item projection cannot have a predecessor")
        if self.projection_revision > 1 and self.previous_projection_ref is None:
            raise ValueError("later Item projection requires a predecessor")
        if self.previous_projection_ref is not None:
            _require_ref(
                self.previous_projection_ref,
                "item-release-projection",
                "v2",
                "previous_projection_ref",
            )
        for ref, object_type, field_name in (
            (
                self.release_subject_ref,
                "evaluation-item-release-subject",
                "release_subject_ref",
            ),
            (
                self.evaluation_item_ref,
                "evaluation-item",
                "evaluation_item_ref",
            ),
            (
                self.release_decision_ref,
                "release-decision",
                "release_decision_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        _require_canonical_checkpoints(
            "projection pending checkpoints",
            self.pending_checkpoints,
        )
        expected_status = _projection_item_status(
            self.release_state,
            self.pending_checkpoints,
        )
        if self.item_status is not expected_status:
            raise ValueError("Item projection status must match release state and pending checkpoints")
        refs = (
            *((self.previous_projection_ref,) if self.previous_projection_ref else ()),
            self.release_subject_ref,
            self.evaluation_item_ref,
            self.release_decision_ref,
        )
        _validate_audit(self.audit, refs, "Item release projection")
        _validate_identity(
            object_id=self.projection_id,
            object_sha256=self.projection_sha256,
            prefix="item-release-projection",
            observed=item_release_projection_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        item_id: str,
        projection_revision: int,
        chain_id: str,
        previous_projection_ref: ObjectRef | None,
        release_subject_ref: ObjectRef,
        evaluation_item_ref: ObjectRef,
        release_decision_ref: ObjectRef,
        release_state: ReleaseStateV2,
        pending_checkpoints: tuple[ApprovalCheckpoint, ...],
        audit: ContractAudit,
        item_status: ItemStatus | None = None,
    ) -> ItemReleaseProjectionV2:
        pending = _sorted_checkpoints(pending_checkpoints)
        status = item_status or _projection_item_status(release_state, pending)
        refs = (
            *((previous_projection_ref,) if previous_projection_ref else ()),
            release_subject_ref,
            evaluation_item_ref,
            release_decision_ref,
        )
        value = cls(
            projection_id="item-release-projection://pending",
            job_id=job_id,
            item_id=item_id,
            projection_revision=projection_revision,
            chain_id=chain_id,
            previous_projection_ref=previous_projection_ref,
            release_subject_ref=release_subject_ref,
            evaluation_item_ref=evaluation_item_ref,
            release_decision_ref=release_decision_ref,
            release_state=release_state,
            pending_checkpoints=pending,
            item_status=status,
            projection_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        digest = item_release_projection_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "projection_id": f"item-release-projection://sha256/{digest}",
                "projection_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        return item_release_projection_v2_ref(self)


class ReleaseProjectionResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/release-projection-result/v2"] = (
        "eval-factory/release-projection-result/v2"
    )
    result_id: Identifier
    phase: ReleaseProjectionPhaseV2
    release_subject: EvaluationItemReleaseSubjectV2
    release_subject_ref: ObjectRef
    query_spec: QuerySpec
    query_spec_ref: ObjectRef
    release_decision: ReleaseDecisionV2
    release_decision_ref: ObjectRef
    evaluation_item: EvaluationItemV2
    evaluation_item_ref: ObjectRef
    item_projection: ItemReleaseProjectionV2
    item_projection_ref: ObjectRef
    previous_result_ref: ObjectRef | None = None
    policy_ref: ObjectRef
    policy_version: Literal["release-projection/r7-08-v1"] = RELEASE_PROJECTION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("phase", mode="before")
    @classmethod
    def parse_phase(cls, value: object) -> ReleaseProjectionPhaseV2:
        if isinstance(value, ReleaseProjectionPhaseV2):
            return value
        if isinstance(value, str):
            return ReleaseProjectionPhaseV2(value)
        raise TypeError("phase must be a ReleaseProjectionPhaseV2")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        validate_evaluation_item_release_subject_v2_identity(self.release_subject)
        validate_query_spec_identity(self.query_spec)
        validate_release_decision_v2_identity(self.release_decision)
        validate_evaluation_item_v2_identity(self.evaluation_item)
        validate_item_release_projection_v2_identity(self.item_projection)
        if self.release_subject_ref != evaluation_item_release_subject_v2_ref(self.release_subject):
            raise ValueError("release subject ref does not match nested subject")
        if self.query_spec_ref != query_spec_ref(self.query_spec):
            raise ValueError("QuerySpec ref does not match complete nested QuerySpec")
        if self.release_decision_ref != release_decision_v2_ref(self.release_decision):
            raise ValueError("ReleaseDecision ref does not match nested decision")
        if self.evaluation_item_ref != evaluation_item_v2_ref(self.evaluation_item):
            raise ValueError("EvaluationItem ref does not match nested Item")
        if self.item_projection_ref != item_release_projection_v2_ref(self.item_projection):
            raise ValueError("Item projection ref does not match nested projection")
        _require_ref(
            self.policy_ref,
            "release-projection-policy",
            "v2",
            "policy_ref",
        )
        if self.previous_result_ref is not None:
            _require_ref(
                self.previous_result_ref,
                "release-projection-result",
                "v2",
                "previous_result_ref",
            )
        if self.phase is ReleaseProjectionPhaseV2.CANDIDATE:
            if (
                self.release_decision.action is not ReleaseActionV2.REQUEST_RELEASE
                or self.release_decision.state is not ReleaseStateV2.CANDIDATE
                or self.item_projection.release_state is not ReleaseStateV2.CANDIDATE
            ):
                raise ValueError("candidate result requires candidate release values")
        else:
            if self.previous_result_ref is None:
                raise ValueError("terminal result requires a previous candidate result")
            if self.release_decision.action not in {
                ReleaseActionV2.APPROVE,
                ReleaseActionV2.REJECT,
            } or self.release_decision.state not in {
                ReleaseStateV2.APPROVED,
                ReleaseStateV2.REJECTED,
            }:
                raise ValueError("terminal result requires approve or reject decision")
            if self.item_projection.release_state is not self.release_decision.state:
                raise ValueError("terminal Item projection must match release decision")
        if self.release_decision.release_subject_sha256 != (self.release_subject.release_subject_sha256):
            raise ValueError("ReleaseDecision does not bind the release subject")
        if self.release_decision.components != self.release_subject.components:
            raise ValueError("ReleaseDecision components differ from release subject")
        if (
            self.release_decision.package_manifest_ref != (self.release_subject.package_manifest_ref)
            or self.release_decision.package_sha256 != self.release_subject.package_sha256
        ):
            raise ValueError("ReleaseDecision package differs from release subject")
        if self.release_decision.batch_quality_report_ref != (self.release_subject.batch_quality_report_ref):
            raise ValueError("ReleaseDecision batch report differs from release subject")
        if self.evaluation_item.release_decision_ref != self.release_decision_ref:
            raise ValueError("EvaluationItem does not bind the nested ReleaseDecision")
        if (
            self.evaluation_item.components != self.release_subject.components
            or self.evaluation_item.source_trace_refs != self.release_subject.source_trace_refs
            or self.evaluation_item.label_decision_refs != self.release_subject.label_decision_refs
            or self.evaluation_item.task_draft_ref != self.release_subject.task_draft_ref
            or self.evaluation_item.attachment_reconstruction_result_ref
            != self.release_subject.attachment_reconstruction_result_ref
            or self.evaluation_item.user_approval_policy_ref != self.release_subject.user_approval_policy_ref
        ):
            raise ValueError("EvaluationItem differs from release subject")
        if (
            self.release_subject.components.query_spec_ref != self.query_spec_ref
            or self.item_projection.release_subject_ref != self.release_subject_ref
            or self.item_projection.evaluation_item_ref != self.evaluation_item_ref
            or self.item_projection.release_decision_ref != self.release_decision_ref
            or self.item_projection.job_id != self.release_subject.job_id
            or self.item_projection.item_id != self.release_subject.item_id
        ):
            raise ValueError("release result nested authority is inconsistent")
        refs = _release_result_refs(self)
        _validate_audit(self.audit, refs, "release projection result")
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            prefix="release-projection-result",
            observed=release_projection_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        phase: ReleaseProjectionPhaseV2,
        release_subject: EvaluationItemReleaseSubjectV2,
        query_spec: QuerySpec,
        release_decision: ReleaseDecisionV2,
        evaluation_item: EvaluationItemV2,
        item_projection: ItemReleaseProjectionV2,
        previous_result_ref: ObjectRef | None,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> ReleaseProjectionResultV2:
        subject_ref = evaluation_item_release_subject_v2_ref(release_subject)
        query_ref = query_spec_ref(query_spec)
        decision_ref = release_decision_v2_ref(release_decision)
        item_ref = evaluation_item_v2_ref(evaluation_item)
        projection_ref = item_release_projection_v2_ref(item_projection)
        refs = (
            subject_ref,
            query_ref,
            decision_ref,
            item_ref,
            projection_ref,
            *((previous_result_ref,) if previous_result_ref else ()),
            policy_ref,
        )
        value = cls(
            result_id="release-projection-result://pending",
            phase=phase,
            release_subject=release_subject,
            release_subject_ref=subject_ref,
            query_spec=query_spec,
            query_spec_ref=query_ref,
            release_decision=release_decision,
            release_decision_ref=decision_ref,
            evaluation_item=evaluation_item,
            evaluation_item_ref=item_ref,
            item_projection=item_projection,
            item_projection_ref=projection_ref,
            previous_result_ref=previous_result_ref,
            policy_ref=policy_ref,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        digest = release_projection_result_v2_carried_sha256(value)
        return value.model_copy(
            update={
                "result_id": f"release-projection-result://sha256/{digest}",
                "result_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        return release_projection_result_v2_ref(self)


def release_projection_policy_v2_carried_sha256(
    value: ReleaseProjectionPolicyV2,
) -> str:
    return _carried_sha256(
        value,
        {"release_projection_policy_id", "policy_sha256", "audit"},
    )


def release_projection_policy_v2_ref(
    value: ReleaseProjectionPolicyV2,
) -> ObjectRef:
    validate_release_projection_policy_v2_identity(value)
    return _object_ref(
        "release-projection-policy",
        value.release_projection_policy_id,
        value.policy_sha256,
    )


def evaluation_item_release_subject_v2_carried_sha256(
    value: EvaluationItemReleaseSubjectV2,
) -> str:
    return _carried_sha256(
        value,
        {"release_subject_id", "release_subject_sha256", "audit"},
    )


def evaluation_item_release_subject_v2_ref(
    value: EvaluationItemReleaseSubjectV2,
) -> ObjectRef:
    validate_evaluation_item_release_subject_v2_identity(value)
    return _object_ref(
        "evaluation-item-release-subject",
        value.release_subject_id,
        value.release_subject_sha256,
    )


def item_release_projection_v2_carried_sha256(
    value: ItemReleaseProjectionV2,
) -> str:
    return _carried_sha256(
        value,
        {"projection_id", "projection_sha256", "audit"},
    )


def item_release_projection_v2_ref(
    value: ItemReleaseProjectionV2,
) -> ObjectRef:
    validate_item_release_projection_v2_identity(value)
    return _object_ref(
        "item-release-projection",
        value.projection_id,
        value.projection_sha256,
    )


def release_projection_result_v2_carried_sha256(
    value: ReleaseProjectionResultV2,
) -> str:
    return _carried_sha256(
        value,
        {"result_id", "result_sha256", "audit"},
    )


def release_projection_result_v2_ref(
    value: ReleaseProjectionResultV2,
) -> ObjectRef:
    validate_release_projection_result_v2_identity(value)
    return _object_ref(
        "release-projection-result",
        value.result_id,
        value.result_sha256,
    )


def query_spec_carried_sha256(value: QuerySpec) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"query_spec_id", "audit"},
            exclude_none=False,
        )
    )


def query_spec_ref(value: QuerySpec) -> ObjectRef:
    validate_query_spec_identity(value)
    return _object_ref("query-spec", value.query_spec_id, query_spec_carried_sha256(value))


def release_decision_v2_carried_sha256(value: ReleaseDecisionV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={
                "release_decision_id",
                "decision_sha256",
                "decided_at",
                "audit",
            },
            exclude_none=False,
        )
    )


def release_decision_v2_ref(value: ReleaseDecisionV2) -> ObjectRef:
    validate_release_decision_v2_identity(value)
    return _object_ref(
        "release-decision",
        value.release_decision_id,
        value.decision_sha256,
    )


def evaluation_item_v2_carried_sha256(value: EvaluationItemV2) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"evaluation_item_id", "item_sha256", "audit"},
            exclude_none=False,
        )
    )


def evaluation_item_v2_ref(value: EvaluationItemV2) -> ObjectRef:
    validate_evaluation_item_v2_identity(value)
    return _object_ref(
        "evaluation-item",
        value.evaluation_item_id,
        value.item_sha256,
    )


def validate_release_projection_policy_v2_identity(
    value: ReleaseProjectionPolicyV2,
) -> None:
    _validate_identity(
        object_id=value.release_projection_policy_id,
        object_sha256=value.policy_sha256,
        prefix="release-projection-policy",
        observed=release_projection_policy_v2_carried_sha256(value),
    )


def validate_evaluation_item_release_subject_v2_identity(
    value: EvaluationItemReleaseSubjectV2,
) -> None:
    _validate_identity(
        object_id=value.release_subject_id,
        object_sha256=value.release_subject_sha256,
        prefix="evaluation-item-release-subject",
        observed=evaluation_item_release_subject_v2_carried_sha256(value),
    )


def validate_item_release_projection_v2_identity(
    value: ItemReleaseProjectionV2,
) -> None:
    _validate_identity(
        object_id=value.projection_id,
        object_sha256=value.projection_sha256,
        prefix="item-release-projection",
        observed=item_release_projection_v2_carried_sha256(value),
    )


def validate_release_projection_result_v2_identity(
    value: ReleaseProjectionResultV2,
) -> None:
    _validate_identity(
        object_id=value.result_id,
        object_sha256=value.result_sha256,
        prefix="release-projection-result",
        observed=release_projection_result_v2_carried_sha256(value),
    )


def validate_query_spec_identity(value: QuerySpec) -> None:
    expected_prompt_sha256 = hashlib.sha256(value.prompt.encode()).hexdigest()
    if value.prompt_sha256 != expected_prompt_sha256:
        raise ValueError("QuerySpec prompt hash is stale")
    digest = query_spec_carried_sha256(value)
    if value.query_spec_id != f"query-spec://sha256/{digest}":
        raise ValueError("QuerySpec identity is stale")


def validate_release_decision_v2_identity(value: ReleaseDecisionV2) -> None:
    digest = release_decision_v2_carried_sha256(value)
    if value.decision_sha256 != digest or value.release_decision_id != f"release-decision://sha256/{digest}":
        raise ValueError("ReleaseDecision identity is stale")


def validate_evaluation_item_v2_identity(value: EvaluationItemV2) -> None:
    digest = evaluation_item_v2_carried_sha256(value)
    if value.item_sha256 != digest or value.evaluation_item_id != f"evaluation-item://sha256/{digest}":
        raise ValueError("EvaluationItem identity is stale")


def _release_subject_refs(
    value: EvaluationItemReleaseSubjectV2,
) -> tuple[ObjectRef, ...]:
    return (
        *value.source_trace_refs,
        *value.label_decision_refs,
        value.task_draft_ref,
        value.attachment_reconstruction_result_ref,
        *_component_refs(value.components),
        value.item_quality_result_ref,
        value.batch_quality_report_ref,
        value.package_manifest_ref,
        value.user_approval_policy_ref,
        *value.user_decision_record_refs,
        *(
            ref
            for binding in value.satisfied_checkpoint_bindings
            for ref in (binding.request_ref, binding.decision_record_ref)
        ),
        *value.relevant_revalidation_application_refs,
        *value.revalidation_report_refs,
        *value.current_head_refs,
        value.policy_ref,
    )


def _release_result_refs(
    value: ReleaseProjectionResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.release_subject_ref,
        value.query_spec_ref,
        value.release_decision_ref,
        value.evaluation_item_ref,
        value.item_projection_ref,
        *((value.previous_result_ref,) if value.previous_result_ref else ()),
        value.policy_ref,
    )


def _component_refs(value: EvaluationItemComponents) -> tuple[ObjectRef, ...]:
    return (
        value.query_spec_ref,
        value.environment_spec_ref,
        value.rubric_set_ref,
        value.evaluator_spec_ref,
        value.reference_policy_ref,
        value.tool_policy_ref,
        value.provenance_manifest_ref,
        value.quality_report_ref,
    )


def _validate_components(value: EvaluationItemComponents) -> None:
    for ref, object_type, field_name in (
        (value.query_spec_ref, "query-spec", "query_spec_ref"),
        (value.environment_spec_ref, "environment-spec", "environment_spec_ref"),
        (value.rubric_set_ref, "rubric-set", "rubric_set_ref"),
        (value.evaluator_spec_ref, "evaluator-spec", "evaluator_spec_ref"),
        (value.reference_policy_ref, "reference-policy", "reference_policy_ref"),
        (value.tool_policy_ref, "tool-policy", "tool_policy_ref"),
        (
            value.provenance_manifest_ref,
            "provenance-manifest",
            "provenance_manifest_ref",
        ),
        (value.quality_report_ref, "quality-report", "quality_report_ref"),
    ):
        _require_ref_type(ref, object_type, field_name)
        _require_safe_refs((ref,), field_name)


def _projection_item_status(
    state: ReleaseStateV2,
    pending: tuple[ApprovalCheckpoint, ...],
) -> ItemStatus:
    status = {
        (ReleaseStateV2.CANDIDATE, False): ItemStatus.RUNNING,
        (ReleaseStateV2.CANDIDATE, True): ItemStatus.NEEDS_REVIEW,
        (ReleaseStateV2.APPROVED, False): ItemStatus.APPROVED,
        (ReleaseStateV2.REJECTED, False): ItemStatus.REJECTED,
    }.get((state, bool(pending)))
    if status is None:
        raise ValueError("R7-08 release state and pending checkpoint combination is invalid")
    return status


def _sorted_bindings(
    values: tuple[CheckpointDecisionBinding, ...],
) -> tuple[CheckpointDecisionBinding, ...]:
    return tuple(sorted(values, key=lambda value: _CHECKPOINT_ORDER[value.checkpoint]))


def _sorted_checkpoints(
    values: tuple[ApprovalCheckpoint, ...],
) -> tuple[ApprovalCheckpoint, ...]:
    return tuple(sorted(set(values), key=lambda value: _CHECKPOINT_ORDER[value]))


def _require_canonical_checkpoints(
    label: str,
    values: tuple[ApprovalCheckpoint, ...],
) -> None:
    if values != _sorted_checkpoints(values):
        raise ValueError(f"{label} must be sorted and unique")


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


def _require_sorted_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    if values != _sorted_refs(values) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_safe_refs(values: tuple[ObjectRef, ...], label: str) -> None:
    for value in values:
        normalized = f"{value.object_type}:{value.object_id}".casefold().replace("_", "-")
        if any(marker in normalized for marker in _DENIED_REF_MARKERS):
            raise ValueError(f"{label} contains a restricted reference")


def _require_ref_type(
    value: ObjectRef,
    object_type: str,
    field_name: str,
) -> None:
    if value.object_type != object_type:
        raise ValueError(f"{field_name} must reference {object_type}")


def _require_ref(
    value: ObjectRef,
    object_type: str,
    version: str,
    field_name: str,
) -> None:
    _require_ref_type(value, object_type, field_name)
    if value.object_version != version:
        raise ValueError(f"{field_name} must reference {object_type} {version}")


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
) -> None:
    if object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit input refs are not exact")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _object_ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _carried_sha256(value: ContractModelV2, exclude: set[str]) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
