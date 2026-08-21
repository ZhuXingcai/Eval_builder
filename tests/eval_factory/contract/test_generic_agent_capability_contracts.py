from __future__ import annotations

from datetime import UTC, datetime
from typing import get_origin

import pytest
from pydantic import ValidationError

from eval_factory.contracts.agent_system_v2 import PlanKindV2
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    AttachmentQualityCapabilityRequestV1,
    AttachmentReconstructionCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    RequirementPlanningCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.manifest import (
    build_generic_agent_trace_pack,
)

HASH = "a" * 64

REQUEST_MODELS = (
    RequirementPlanningCapabilityRequestV1,
    TraceIngestionCapabilityRequestV1,
    TaskAuthoringCapabilityRequestV1,
    AttachmentReconstructionCapabilityRequestV1,
    CriteriaRubricCapabilityRequestV1,
    GradingDesignCapabilityRequestV1,
    AttachmentQualityCapabilityRequestV1,
    BatchQualityCapabilityRequestV1,
    PlanReviewCapabilityRequestV1,
    DeliveryCapabilityRequestV1,
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 18, tzinfo=UTC),
        created_by="generic-capability-contract-test",
        governing_versions=(
            VersionBinding(
                component="eval-harness-stage2",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _ref(
    object_type: str,
    suffix: str = "example",
    *,
    version: str = "v1",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/{version}",
        object_version=version,
        object_sha256=HASH,
    )


def test_pack_request_bundles_are_strict_frozen_and_dict_free() -> None:
    for model_type in REQUEST_MODELS:
        assert model_type.model_config["extra"] == "forbid"
        assert model_type.model_config["frozen"] is True
        assert model_type.model_json_schema()["additionalProperties"] is False
        assert all(
            get_origin(field.annotation) is not dict and field.annotation is not dict
            for field in model_type.model_fields.values()
        )


def test_static_pack_registers_ten_unique_request_schemas() -> None:
    registration = build_generic_agent_trace_pack(audit=_audit())

    assert len(registration.capability_definitions) == len(REQUEST_MODELS)
    assert len(
        {value.request_schema_ref for value in registration.capability_definitions},
    ) == len(REQUEST_MODELS)
    assert all(
        value.request_schema_ref in registration.manifest.artifact_schema_refs
        for value in registration.capability_definitions
    )


def test_request_bundle_rejects_unknown_fields_and_wrong_schema_version() -> None:
    request = PlanReviewCapabilityRequestV1.create(
        run_id="factory-run://capability-contract",
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        plan_ref=_ref(
            "attachment-generation-plan",
            version="v2",
        ),
        audit=_audit(),
    )
    payload = request.model_dump(mode="json")
    payload["unknown"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        PlanReviewCapabilityRequestV1.model_validate(payload)

    version_payload = request.model_dump(mode="json")
    version_payload["schema_version"] = "generic-agent-trace/plan-review-request/v2"
    with pytest.raises(ValidationError, match="literal"):
        PlanReviewCapabilityRequestV1.model_validate(version_payload)


def test_plan_review_request_supports_global_plan_and_requires_v2_ref() -> None:
    request = PlanReviewCapabilityRequestV1.create(
        run_id="factory-run://capability-contract",
        plan_kind=PlanKindV2.GLOBAL_BUILD,
        plan_ref=_ref(
            "dataset-build-plan",
            version="v2",
        ),
        audit=_audit(),
    )
    assert request.plan_kind is PlanKindV2.GLOBAL_BUILD

    with pytest.raises(ValidationError, match="object version v2"):
        PlanReviewCapabilityRequestV1.create(
            run_id="factory-run://capability-contract",
            plan_kind=PlanKindV2.GLOBAL_BUILD,
            plan_ref=_ref(
                "dataset-build-plan",
                version="v1",
            ),
            audit=_audit(),
        )
