from __future__ import annotations

from typing import Literal

from eval_factory.attachment_planning.models import (
    AttachmentPlanningPolicyError,
)
from eval_factory.contracts.attachment_v2 import (
    AttachmentPlanningContextV2,
    attachment_planning_context_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    producer_storage_authorization_ref,
    producer_task_view_ref,
)
from eval_factory.provenance.bundles import (
    EvidenceCompilationPolicyError,
    evidence_bundle_ref,
    validate_evidence_bundle_identity,
)
from eval_factory.task_authoring.producer_models import (
    ProducerTaskViewPolicyError,
)
from eval_factory.task_authoring.producer_view import (
    validate_producer_task_view_identity,
    validate_storage_authorization_identity,
)

ATTACHMENT_PLANNING_POLICY_VERSION: Literal["attachment-planning/r5-01-v1"] = "attachment-planning/r5-01-v1"
_DENIED_PROJECTED_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "configured-pii",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-pii",
        "secret",
        "sensitive-pii",
        "trace-raw",
    }
)


class AttachmentPlanningBridge:
    policy_version = ATTACHMENT_PLANNING_POLICY_VERSION

    def compile(
        self,
        *,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        audit: ContractAudit,
    ) -> AttachmentPlanningContextV2:
        self._validate_inputs(
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
        )
        view_ref = producer_task_view_ref(producer_task_view)
        authorization_ref = producer_storage_authorization_ref(storage_authorization)
        bundle_ref = evidence_bundle_ref(evidence_bundle)
        context = AttachmentPlanningContextV2(
            attachment_planning_context_id=("attachment-planning-context://pending"),
            producer_task_view_ref=view_ref,
            producer_storage_authorization_ref=authorization_ref,
            safe_evidence_bundle_ref=bundle_ref,
            projection_policy_ref=evidence_bundle.projection_policy_ref,
            source_trace_id=evidence_bundle.source_trace_id,
            trace_ir_version_id=evidence_bundle.trace_ir_version_id,
            producer_principal_id=storage_authorization.producer_principal_id,
            source_task_draft_sha256=(producer_task_view.source_task_draft_sha256),
            source_contract_chain_sha256=(producer_task_view.source_contract_chain_sha256),
            policy_version=self.policy_version,
            attachment_planning_context_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (
                    view_ref,
                    authorization_ref,
                    bundle_ref,
                    evidence_bundle.projection_policy_ref,
                ),
            ),
        )
        digest = attachment_planning_context_carried_sha256(context)
        return context.model_copy(
            update={
                "attachment_planning_context_id": (f"attachment-planning-context://sha256/{digest}"),
                "attachment_planning_context_sha256": digest,
            }
        )

    def validate_current(
        self,
        *,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
        attachment_planning_context: AttachmentPlanningContextV2,
    ) -> None:
        try:
            _validate_context_identity(attachment_planning_context)
            rebuilt = self.compile(
                producer_task_view=producer_task_view,
                storage_authorization=storage_authorization,
                evidence_bundle=evidence_bundle,
                audit=attachment_planning_context.audit,
            )
            if not _same_context_except_audit_actor_time(
                rebuilt,
                attachment_planning_context,
            ):
                raise AttachmentPlanningPolicyError(
                    "attachment planning context does not match authoritative inputs"
                )
        except AttachmentPlanningPolicyError as exc:
            raise AttachmentPlanningPolicyError(
                f"current attachment planning context validation failed: {exc}"
            ) from exc

    def _validate_inputs(
        self,
        *,
        producer_task_view: ProducerTaskViewV2,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
    ) -> None:
        try:
            validate_producer_task_view_identity(producer_task_view)
            validate_storage_authorization_identity(storage_authorization)
            validate_evidence_bundle_identity(evidence_bundle)
        except (
            ProducerTaskViewPolicyError,
            EvidenceCompilationPolicyError,
        ) as exc:
            raise AttachmentPlanningPolicyError(
                "attachment planning source identity is stale or invalid"
            ) from exc

        authorization_ref = producer_storage_authorization_ref(storage_authorization)
        if producer_task_view.storage_authorization_ref != authorization_ref:
            raise AttachmentPlanningPolicyError("ProducerTaskView storage authorization ref is mismatched")

        bundle_ref = evidence_bundle_ref(evidence_bundle)
        expected_bundle_refs = (bundle_ref,)
        if (
            producer_task_view.safe_evidence_bundle_refs != expected_bundle_refs
            or storage_authorization.evidence_bundle_refs != expected_bundle_refs
        ):
            raise AttachmentPlanningPolicyError(
                "producer view and authorization must bind the exact EvidenceBundle"
            )

        if (
            producer_task_view.projection_policy_ref != evidence_bundle.projection_policy_ref
            or storage_authorization.projection_policy_ref != evidence_bundle.projection_policy_ref
        ):
            raise AttachmentPlanningPolicyError(
                "producer view, authorization, and bundle projection policy must match"
            )

        if producer_task_view.source_task_draft_sha256 != storage_authorization.source_task_draft_sha256:
            raise AttachmentPlanningPolicyError("producer view and authorization TaskDraft hashes must match")

        if (
            storage_authorization.purpose != "ATTACHMENT_PRODUCTION"
            or evidence_bundle.consumer_stage != "attachment-producer"
            or evidence_bundle.purpose != "attachment-production"
        ):
            raise AttachmentPlanningPolicyError("producer evidence bundle stage or purpose is invalid")

        if evidence_bundle.returned_characters > evidence_bundle.max_characters:
            raise AttachmentPlanningPolicyError("producer EvidenceBundle exceeds its authorized budget")
        if evidence_bundle.model_dump(mode="python")["tainted_content_included"] is not False:
            raise AttachmentPlanningPolicyError("producer EvidenceBundle cannot include tainted content")

        if any(
            (
                storage_authorization.raw_store_access,
                storage_authorization.canonical_store_access,
                storage_authorization.quarantine_store_access,
                storage_authorization.private_reference_store_access,
                storage_authorization.credentials_issued,
            )
        ):
            raise AttachmentPlanningPolicyError(
                "attachment planning cannot use direct store access or credentials"
            )

        self._validate_evidence_inventory(
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
        )

    def _validate_evidence_inventory(
        self,
        *,
        storage_authorization: ProducerStorageAuthorizationV2,
        evidence_bundle: EvidenceBundle,
    ) -> None:
        evidence_ids = tuple(item.evidence_ref_id for item in evidence_bundle.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise AttachmentPlanningPolicyError("duplicate evidence reference ID in EvidenceBundle")
        if evidence_ids != tuple(sorted(evidence_ids)):
            raise AttachmentPlanningPolicyError("evidence references must be sorted by evidence_ref_id")

        subject_refs = tuple(item.subject_ref for item in evidence_bundle.evidence)
        subject_keys = tuple(_ref_key(ref) for ref in subject_refs)
        if len(subject_keys) != len(set(subject_keys)):
            raise AttachmentPlanningPolicyError("duplicate evidence subject in EvidenceBundle")
        if any(not _is_safe_projected_ref(ref) for ref in subject_refs):
            raise AttachmentPlanningPolicyError("EvidenceBundle subjects must be safe projected refs")
        expected_subject_refs = tuple(sorted(subject_refs, key=_ref_key))
        if expected_subject_refs != storage_authorization.authorized_subject_refs:
            raise AttachmentPlanningPolicyError(
                "EvidenceBundle subjects must exactly match authorization subjects"
            )

        excluded_keys = tuple(_ref_key(ref) for ref in evidence_bundle.excluded_subject_refs)
        if len(excluded_keys) != len(set(excluded_keys)):
            raise AttachmentPlanningPolicyError("duplicate excluded subject in EvidenceBundle")
        if excluded_keys != tuple(sorted(excluded_keys)):
            raise AttachmentPlanningPolicyError("excluded subject refs must be sorted")

        for item in evidence_bundle.evidence:
            if any(span.source_trace_id != evidence_bundle.source_trace_id for span in item.source_spans):
                raise AttachmentPlanningPolicyError("evidence source span must match bundle source trace")


def _validate_context_identity(
    context: AttachmentPlanningContextV2,
) -> None:
    digest = attachment_planning_context_carried_sha256(context)
    if (
        context.attachment_planning_context_sha256 != digest
        or context.attachment_planning_context_id != f"attachment-planning-context://sha256/{digest}"
    ):
        raise AttachmentPlanningPolicyError("attachment planning context identity is stale or invalid")


def _same_context_except_audit_actor_time(
    expected: AttachmentPlanningContextV2,
    observed: AttachmentPlanningContextV2,
) -> bool:
    if expected.model_dump(
        mode="json",
        exclude={"audit"},
    ) != observed.model_dump(
        mode="json",
        exclude={"audit"},
    ):
        return False
    return (
        expected.audit.governing_versions == observed.audit.governing_versions
        and expected.audit.input_refs == observed.audit.input_refs
    )


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=tuple(sorted(set(refs), key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _is_safe_projected_ref(ref: ObjectRef) -> bool:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace(
        "_",
        "-",
    )
    return ref.object_type.endswith("-projection") and not any(
        marker in normalized for marker in _DENIED_PROJECTED_REF_MARKERS
    )
