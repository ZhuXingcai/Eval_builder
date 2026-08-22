from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from env_mock_agent.facade import (
    AttachmentCrossItemSafetyScanLimitsV2,
    AttachmentDuplicateFingerprintLimitsV2,
    FacadeObjectRef,
)
from eval_factory.agent_system.candidate_output import (
    CandidateDatasetOutputAssembler,
)
from eval_factory.agent_system.core_material import (
    FactoryCoreMaterialStore,
)
from eval_factory.agent_system.criteria_registry import (
    CriteriaRubricAgentRegistryConfig,
    build_criteria_rubric_agent_registry,
)
from eval_factory.agent_system.dataset_release_fixture import (
    FactoryDatasetReleaseRuntimeConfigV1,
)
from eval_factory.agent_system.dataset_runtime_fixture import (
    FactoryDatasetAttachmentRuntimeConfigV1,
    FactoryDatasetCoreRuntimeConfigV1,
    FactoryDatasetPlanningFixtureV1,
    FactoryDatasetR4RuntimeConfigV1,
    FactoryDatasetRuntimeConfigV1,
)
from eval_factory.agent_system.dataset_specialist_fixture import (
    FactoryDatasetJobStoreRuntimeConfigV1,
    FactoryDatasetSpecialistRuntimeConfigV1,
)
from eval_factory.agent_system.grading_registry import (
    GradingDesignAgentRegistryConfig,
    build_grading_design_agent_registry,
)
from eval_factory.agent_system.plan_review import (
    PlanReviewDecisionSubmissionV1,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
)
from eval_factory.agent_system.planner import DatasetBuildPlanCompiler
from eval_factory.agent_system.registry import AgentRegistry
from eval_factory.agent_system.store import FactoryControlStore
from eval_factory.approval.requests import UserApprovalPolicyCompiler
from eval_factory.attachment_planning import (
    SemanticReviewContextSources,
)
from eval_factory.cli import app
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    DatasetBuildPlanTaskV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    PlanDecisionKindV2,
    PlanKindV2,
    TraceCandidateDispositionV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelCapabilityProfileV2,
    ModelHealthSnapshotV2,
    ModelPriceScheduleV2,
    ModelQualityBaselineV2,
    ModelRoutingPolicyV2,
    PromptTemplateV2,
)
from eval_factory.contracts.approval import (
    ApprovalMode,
    FinalReviewScope,
    UserApprovalPolicy,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactRoutingPolicyV2,
    PublicSourceRetrievalPolicyV2,
    artifact_routing_policy_carried_sha256,
    public_source_retrieval_policy_carried_sha256,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.cross_item_safety_v2 import (
    CrossItemSafetyPolicyV2,
)
from eval_factory.contracts.dataset_runtime_v2 import (
    FactoryDatasetRunRequestV2,
    FactoryItemStageV2,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
)
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ResourceBudget,
    StageRunStatus,
)
from eval_factory.contracts.orchestration_v2 import (
    StageNameV2,
    resolved_work_unit_v2_ref,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPolicyV2,
)
from eval_factory.contracts.review_v2 import (
    SemanticReviewPolicyV2,
    SemanticReviewRoundV2,
)
from eval_factory.contracts.task_v2 import EvaluatorExecutionModeV2
from eval_factory.contracts.trace import ToolFamily
from eval_factory.orchestration.job_store import JobStore
from eval_factory.task_authoring import (
    EvaluatorBindingDefinition,
    ToolCapabilityCatalog,
    ToolCapabilityDefinition,
    tool_capability_catalog_carried_sha256,
    tool_capability_definition_carried_sha256,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 7, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT.parent / "raw_traj"
requires_private_corpus = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _ref(object_type: str, suffix: str = "example") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}/v2",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="factory-dataset-cli-test",
        governing_versions=(
            VersionBinding(
                component="factory-dataset-runtime",
                version="v1",
                sha256=HASH,
            ),
        ),
    )


def _requirement() -> EvaluationRequirementSpecV2:
    return EvaluationRequirementSpecV2.create(
        requirement_spec_id="evaluation-requirement-spec://runtime",
        run_id="factory-run://dataset/runtime",
        source_ref=_ref("evaluation-requirement-source"),
        goals=("Build safe source-grounded candidate items.",),
        constraints=("Do not expose restricted trace material.",),
        assumptions=("Provider boundary is deterministic in tests.",),
        open_questions=(),
        requirement_version=1,
        audit=_audit(),
    )


def _policy(
    *,
    include_attachment: bool = False,
    include_specialists: bool = False,
) -> FactoryRunPolicyV2:
    return FactoryRunPolicyV2.create(
        policy_id="factory-run-policy://runtime",
        allowed_task_kinds=tuple(
            sorted(
                (
                    "task-rewrite",
                    "trace-extraction",
                    *(("attachment-mock",) if include_attachment else ()),
                    *(
                        (
                            "criteria-rubric",
                            "grading-design",
                        )
                        if include_specialists
                        else ()
                    ),
                )
            )
        ),
        max_transitions=512,
        max_plan_revisions=8,
        max_agent_attempts=2,
        max_model_requests=1_000,
        max_model_tokens=2_000_000,
        max_cost_micro_usd=25_000_000,
        audit=_audit(),
    )


def _request(
    manifest_sha256: str = HASH,
    *,
    include_attachment: bool = False,
    include_specialists: bool = False,
) -> FactoryDatasetRunRequestV2:
    return FactoryDatasetRunRequestV2.create(
        dataset_run_id=_requirement().run_id,
        requirement_spec_ref=_requirement().to_ref(),
        manifest_ref=ObjectRef(
            object_type="trace-manifest",
            object_id=(f"trace-manifest://sha256/{manifest_sha256}"),
            object_version="v2",
            object_sha256=manifest_sha256,
        ),
        source_authorization_ref=_ref("trace-source-authorization"),
        factory_policy_ref=_policy(
            include_attachment=include_attachment,
            include_specialists=include_specialists,
        ).to_ref(),
        pipeline_policy_refs=(
            _ref("batch-quality-policy"),
            _ref("release-projection-policy"),
        ),
        gateway_registry_refs=(
            _ref("agent-registry"),
            _ref("model-catalog"),
            _ref("prompt-registry"),
        ),
        output_target_ref=_ref("candidate-output-target"),
        idempotency_key="dataset-runtime-request",
        max_transitions=512,
        audit=_audit(),
    )


def _prompt(
    task_kind: str = "planning",
) -> PromptTemplateV2:
    input_type = "evaluation-requirement-spec" if task_kind == "planning" else "extracted-user-prompt"
    return PromptTemplateV2.create(
        prompt_template_id=f"prompt-template://{task_kind}/runtime",
        agent_role=f"{task_kind}-agent",
        task_kind=task_kind,
        system_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-system",
        ),
        instruction_template_ref=_ref(
            "prompt-template-content",
            f"{task_kind}-instruction",
        ),
        required_input_object_types=(input_type,),
        output_schema_ref=_ref("json-schema", task_kind),
        allowed_tool_ids=(),
        required_model_capabilities=tuple(sorted((task_kind, "structured-output"))),
        injection_policy_ref=_ref("prompt-injection-policy"),
        template_version=1,
        audit=_audit(),
    )


def _profile(
    task_kind: str = "planning",
) -> ModelCapabilityProfileV2:
    return ModelCapabilityProfileV2.create(
        model_profile_id=f"model-profile://{task_kind}/runtime",
        provider_id="provider://offline-runtime",
        model_id=f"model://{task_kind}/runtime",
        model_version="v1",
        capabilities=tuple(sorted((task_kind, "structured-output"))),
        supported_data_classifications=(
            "INTERNAL_DERIVED",
            "RESTRICTED_EVALUATOR_CONTROL",
            "RESTRICTED_TRACE_DERIVED",
        ),
        supported_residencies=("LOCAL",),
        context_limit_tokens=200_000,
        output_limit_tokens=16_000,
        availability="AVAILABLE",
        audit=_audit(),
    )


def _registry() -> AgentRegistry:
    extract = AgentCapabilityV2.create(
        capability_id="agent-capability://trace-extraction",
        task_kinds=("trace-extraction",),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        model_capabilities=("structured-output",),
        tool_ids=("trace-query",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    rewrite = AgentCapabilityV2.create(
        capability_id="agent-capability://task-rewrite",
        task_kinds=("task-rewrite",),
        input_object_types=("extracted-user-prompt",),
        output_object_types=("task-rewrite-candidate",),
        model_capabilities=("structured-output",),
        tool_ids=("task-rewrite",),
        data_purposes=("evaluation-dataset-construction",),
        data_classifications=("RESTRICTED_TRACE_DERIVED",),
        audit=_audit(),
    )
    definitions = tuple(
        AgentDefinitionV2.create(
            agent_definition_id=f"agent-definition://{role}",
            agent_role=role,
            agent_version="v1",
            capability_refs=(capability.to_ref(),),
            prompt_template_ref=_ref("prompt-template", role),
            model_policy_ref=_ref("model-routing-policy"),
            tool_ids=capability.tool_ids,
            data_purpose="evaluation-dataset-construction",
            allowed_data_classifications=("RESTRICTED_TRACE_DERIVED",),
            validator_refs=(_ref("validator", role),),
            max_attempts=2,
            max_model_requests=2,
            max_model_tokens=4_096,
            max_cost_micro_usd=100_000,
            workspace_isolated=True,
            network_allowed=False,
            audit=_audit(),
        )
        for role, capability in (
            ("trace-extraction-agent", extract),
            ("task-rewrite-agent", rewrite),
        )
    )
    return AgentRegistry(
        capabilities=(extract, rewrite),
        definitions=definitions,
    )


def _config(
    *,
    include_core: bool = False,
    include_r4: bool = False,
    include_attachment: bool = False,
    specialist_config: (FactoryDatasetSpecialistRuntimeConfigV1 | None) = None,
) -> FactoryDatasetRuntimeConfigV1:
    registry = _registry()
    task_kinds = (
        (
            "extraction",
            "planning",
            "rewrite",
            *(
                (
                    "rubric-authoring",
                    "task-draft-authoring",
                    "task-episode-grouping",
                    "task-prompt-safety",
                )
                if include_r4
                else ()
            ),
            *(("attachment-planning",) if include_attachment else ()),
            *(
                (
                    "attachment-solvability",
                    "criteria-rubric-planning",
                    "criteria-rubric",
                    "grading-design-planning",
                    "grading-design",
                )
                if specialist_config is not None
                else ()
            ),
        )
        if include_core
        else ("planning",)
    )
    profiles = tuple(_profile(task_kind) for task_kind in task_kinds)
    prompts = {task_kind: _prompt(task_kind) for task_kind in task_kinds}
    planning_agent_ref = _ref(
        "agent-definition",
        "planning",
    )
    return FactoryDatasetRuntimeConfigV1(
        capabilities=registry.capabilities,
        definitions=registry.definitions,
        model_profiles=profiles,
        quality_baselines=(
            *(
                ModelQualityBaselineV2.create(
                    baseline_id=(f"model-quality-baseline://{task_kind}/runtime"),
                    model_profile_ref=profile.to_ref(),
                    task_kind=task_kind,
                    benchmark_ref=_ref(
                        "model-quality-benchmark",
                        task_kind,
                    ),
                    quality_basis_points=9_000,
                    minimum_sample_count=100,
                    audit=_audit(),
                )
                for task_kind, profile in zip(
                    task_kinds,
                    profiles,
                    strict=True,
                )
            ),
        ),
        health_snapshots=tuple(
            ModelHealthSnapshotV2.create(
                health_snapshot_id=(f"model-health-snapshot://{task_kind}/runtime"),
                model_profile_ref=profile.to_ref(),
                observed_at=_audit().created_at,
                availability="HEALTHY",
                success_basis_points=9_900,
                latency_milliseconds=10,
                audit=_audit(),
            )
            for task_kind, profile in zip(
                task_kinds,
                profiles,
                strict=True,
            )
        ),
        price_schedules=tuple(
            ModelPriceScheduleV2.create(
                price_schedule_id=(f"model-price-schedule://{task_kind}/runtime"),
                model_profile_ref=profile.to_ref(),
                input_micro_usd_per_million_tokens=1,
                output_micro_usd_per_million_tokens=1,
                effective_from=_audit().created_at,
                audit=_audit(),
            )
            for task_kind, profile in zip(
                task_kinds,
                profiles,
                strict=True,
            )
        ),
        routing_policy=ModelRoutingPolicyV2.create(
            routing_policy_id=("model-routing-policy://planning/runtime"),
            allowed_provider_ids=("provider://offline-runtime",),
            allowed_agent_definition_refs=tuple(
                sorted(
                    (
                        planning_agent_ref,
                        *(
                            (
                                _ref(
                                    "agent-definition",
                                    "extraction",
                                ),
                                _ref(
                                    "agent-definition",
                                    "rewrite",
                                ),
                            )
                            if include_core
                            else ()
                        ),
                        *(
                            (
                                _ref(
                                    "agent-definition",
                                    "rubric-authoring",
                                ),
                                _ref(
                                    "agent-definition",
                                    "task-draft-authoring",
                                ),
                                _ref(
                                    "agent-definition",
                                    "task-episode-grouping",
                                ),
                                _ref(
                                    "agent-definition",
                                    "task-prompt-safety",
                                ),
                            )
                            if include_r4
                            else ()
                        ),
                        *(
                            (
                                _ref(
                                    "agent-definition",
                                    "attachment-planning",
                                ),
                            )
                            if include_attachment
                            else ()
                        ),
                        *(
                            (
                                specialist_config.solvability_agent_definition_ref,
                                specialist_config.criteria_planning_agent_definition_ref,
                                specialist_config.criteria_agent_definition_ref,
                                specialist_config.grading_planning_agent_definition_ref,
                                specialist_config.grading_agent_definition_ref,
                            )
                            if specialist_config is not None
                            else ()
                        ),
                    ),
                    key=lambda value: value.object_id,
                )
            ),
            allowed_model_profile_refs=tuple(
                sorted(
                    (profile.to_ref() for profile in profiles),
                    key=lambda value: value.object_id,
                )
            ),
            quality_weight=100,
            health_weight=10,
            latency_weight=1,
            cost_weight=1,
            audit=_audit(),
        ),
        planning_prompt=prompts["planning"],
        planning_agent_definition_ref=planning_agent_ref,
        allowed_model_profile_refs=tuple(
            sorted(
                (profile.to_ref() for profile in profiles),
                key=lambda value: value.object_id,
            )
        ),
        budget_reservation_ref=_ref(
            "work-model-reservation",
            "planning",
        ),
    )


def _core_config(
    *,
    blocked_extracted_prompt_refs: tuple[ObjectRef, ...] = (),
) -> FactoryDatasetCoreRuntimeConfigV1:
    profiles = tuple(
        sorted(
            (
                _profile("extraction").to_ref(),
                _profile("rewrite").to_ref(),
            ),
            key=lambda value: value.object_id,
        )
    )
    return FactoryDatasetCoreRuntimeConfigV1(
        intent_prompt=_prompt("extraction"),
        rewrite_prompt=_prompt("rewrite"),
        intent_agent_definition_ref=_ref(
            "agent-definition",
            "extraction",
        ),
        rewrite_agent_definition_ref=_ref(
            "agent-definition",
            "rewrite",
        ),
        allowed_model_profile_refs=profiles,
        intent_budget_reservation_ref=_ref(
            "work-model-reservation",
            "extraction",
        ),
        rewrite_budget_reservation_ref=_ref(
            "work-model-reservation",
            "rewrite",
        ),
        blocked_extracted_prompt_refs=blocked_extracted_prompt_refs,
    )


def _r4_config() -> FactoryDatasetR4RuntimeConfigV1:
    task_kinds = (
        "rubric-authoring",
        "task-draft-authoring",
        "task-episode-grouping",
        "task-prompt-safety",
    )
    prompts = {value: _prompt(value) for value in task_kinds}
    return FactoryDatasetR4RuntimeConfigV1(
        episode_prompt=prompts["task-episode-grouping"],
        draft_prompt=prompts["task-draft-authoring"],
        safety_prompt=prompts["task-prompt-safety"],
        rubric_prompt=prompts["rubric-authoring"],
        episode_agent_definition_ref=_ref(
            "agent-definition",
            "task-episode-grouping",
        ),
        draft_agent_definition_ref=_ref(
            "agent-definition",
            "task-draft-authoring",
        ),
        safety_agent_definition_ref=_ref(
            "agent-definition",
            "task-prompt-safety",
        ),
        rubric_agent_definition_ref=_ref(
            "agent-definition",
            "rubric-authoring",
        ),
        allowed_model_profile_refs=tuple(
            sorted(
                (_profile(value).to_ref() for value in task_kinds),
                key=lambda value: value.object_id,
            )
        ),
        episode_budget_reservation_ref=_ref(
            "work-model-reservation",
            "task-episode-grouping",
        ),
        draft_budget_reservation_ref=_ref(
            "work-model-reservation",
            "task-draft-authoring",
        ),
        safety_budget_reservation_ref=_ref(
            "work-model-reservation",
            "task-prompt-safety",
        ),
        rubric_budget_reservation_ref=_ref(
            "work-model-reservation",
            "rubric-authoring",
        ),
    )


def _attachment_config() -> FactoryDatasetAttachmentRuntimeConfigV1:
    retrieval_policy = PublicSourceRetrievalPolicyV2(
        public_source_retrieval_policy_id=("public-source-retrieval-policy://pending"),
        approved_search_provider_ids=(),
        approved_fetch_provider_ids=("fetch-provider://disabled",),
        allowed_schemes=("https",),
        allowed_host_suffixes=("example.gov",),
        max_search_results=1,
        max_fetch_bytes=1_024,
        query_egress_policy_ref=_ref(
            "query-egress-policy",
            "attachment-runtime",
        ),
        network_policy_ref=_ref(
            "network-policy",
            "attachment-runtime",
        ),
        source_usage_policy_ref=_ref(
            "source-usage-policy",
            "attachment-runtime",
        ),
        safety_scan_policy_ref=_ref(
            "safety-scan-policy",
            "attachment-runtime",
        ),
        license_policy_ref=_ref(
            "source-license-policy",
            "attachment-runtime",
        ),
        policy_version="public-source-retrieval/r5-04-v1",
        public_source_retrieval_policy_sha256="0" * 64,
        audit=_audit(),
    )
    retrieval_digest = public_source_retrieval_policy_carried_sha256(retrieval_policy)
    retrieval_policy = retrieval_policy.model_copy(
        update={
            "public_source_retrieval_policy_id": (
                f"public-source-retrieval-policy://sha256/{retrieval_digest}"
            ),
            "public_source_retrieval_policy_sha256": retrieval_digest,
        }
    )
    model_refs = (
        _ref("model-profile", "attachment-sdk"),
        _ref("model-profile", "attachment-cli"),
        _ref("model-profile", "attachment-pi"),
    )
    provider_policy_ref = _ref(
        "provider-capability-policy",
        "attachment-runtime",
    )
    runtime_policy_ref = _ref(
        "runtime-capability-policy",
        "attachment-runtime",
    )
    validator_policy_ref = _ref(
        "validator-policy",
        "attachment-runtime",
    )
    routing_audit = _audit().model_copy(
        update={
            "input_refs": tuple(
                sorted(
                    (
                        *model_refs,
                        provider_policy_ref,
                        runtime_policy_ref,
                        validator_policy_ref,
                    ),
                    key=lambda value: (
                        value.object_type,
                        value.object_id,
                        value.object_version,
                        value.object_sha256,
                    ),
                )
            )
        }
    )
    routing_policy = ArtifactRoutingPolicyV2(
        artifact_routing_policy_id=("artifact-routing-policy://pending"),
        deterministic_provider_first=True,
        approved_provider_ids=("text",),
        runtime_order=(
            "claude_agent_sdk",
            "claude_code_cli",
            "pi_rpc",
        ),
        runtime_model_profile_refs=model_refs,
        provider_capability_policy_ref=provider_policy_ref,
        runtime_capability_policy_ref=runtime_policy_ref,
        validator_policy_ref=validator_policy_ref,
        silent_degradation_allowed=False,
        policy_version="artifact-routing/r5-05-v1",
        artifact_routing_policy_sha256="0" * 64,
        audit=routing_audit,
    )
    routing_digest = artifact_routing_policy_carried_sha256(routing_policy)
    routing_policy = routing_policy.model_copy(
        update={
            "artifact_routing_policy_id": (f"artifact-routing-policy://sha256/{routing_digest}"),
            "artifact_routing_policy_sha256": routing_digest,
        }
    )
    ledger_sha256 = hashlib.sha256(b"attachment-cli-world-ledger").hexdigest()
    return FactoryDatasetAttachmentRuntimeConfigV1(
        planning_prompt=_prompt("attachment-planning"),
        planning_agent_definition_ref=_ref(
            "agent-definition",
            "attachment-planning",
        ),
        allowed_model_profile_refs=(_profile("attachment-planning").to_ref(),),
        planning_budget_reservation_ref=_ref(
            "work-model-reservation",
            "attachment-planning",
        ),
        retrieval_policy=retrieval_policy,
        routing_policy=routing_policy,
        world_ledger_ref=FacadeObjectRef(
            object_type="world-ledger",
            object_id=("world-ledger://attachment-cli/runtime"),
            object_version="v2",
            object_sha256=ledger_sha256,
        ),
        mock_prompt_ref=_ref(
            "prompt-template",
            "attachment-mock",
        ),
        quality_prompt_ref=_ref(
            "prompt-template",
            "attachment-quality",
        ),
        solvability_prompt_ref=_ref(
            "prompt-template",
            "attachment-solvability",
        ),
        mock_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-mock",
        ),
        quality_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-quality",
        ),
        solvability_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-solvability",
        ),
    )


def _tool_catalog() -> ToolCapabilityCatalog:
    definition = ToolCapabilityDefinition(
        definition_id="tool-capability-definition://pending",
        tool_id="file-read",
        tool_family=ToolFamily.FILE_READ,
        capability_ref=_ref(
            "tool-capability",
            "specialist-file-read",
        ),
        enforcement_profile_ref=_ref(
            "tool-enforcement-profile",
            "specialist-file-read",
        ),
        constraint_profile_ref=_ref(
            "tool-constraint-profile",
            "specialist-file-read",
        ),
        contestant_descriptor_ref=_ref(
            "contestant-tool-descriptor",
            "specialist-file-read",
        ),
        contestant_constraint_profile_ref=_ref(
            "contestant-tool-constraint-profile",
            "specialist-file-read",
        ),
        contestant_eligible=True,
        definition_sha256="0" * 64,
    )
    definition_digest = tool_capability_definition_carried_sha256(definition)
    definition = definition.model_copy(
        update={
            "definition_id": (f"tool-capability-definition://sha256/{definition_digest}"),
            "definition_sha256": definition_digest,
        }
    )
    catalog = ToolCapabilityCatalog(
        catalog_id="tool-capability-catalog://pending",
        catalog_version="tool-capability-catalog/r4-07-v1",
        definitions=(definition,),
        catalog_sha256="0" * 64,
        audit=_audit(),
    )
    catalog_digest = tool_capability_catalog_carried_sha256(catalog)
    return catalog.model_copy(
        update={
            "catalog_id": (f"tool-capability-catalog://sha256/{catalog_digest}"),
            "catalog_sha256": catalog_digest,
        }
    )


def _semantic_review_policy() -> SemanticReviewPolicyV2:
    return SemanticReviewPolicyV2.create(
        model_profile_refs={
            round_: _ref(
                "model-profile",
                f"semantic-{round_.value.casefold()}",
            )
            for round_ in SemanticReviewRoundV2
        },
        prompt_versions={
            round_: (f"semantic-review/{round_.value.casefold()}/v1") for round_ in SemanticReviewRoundV2
        },
        projection_policy_refs={
            round_: _ref(
                "projection-policy",
                f"semantic-{round_.value.casefold()}",
            )
            for round_ in SemanticReviewRoundV2
        },
    )


def _semantic_context_sources() -> SemanticReviewContextSources:
    evidence_bundle_ref = _ref(
        "evidence-bundle",
        "specialist-safe",
    ).model_copy(
        update={
            "object_version": "v1",
        }
    )
    return SemanticReviewContextSources(
        query_public_view_ref=_ref(
            "query-spec-public-view",
            "specialist",
        ),
        candidate_manifest_ref=_ref(
            "candidate-package-inventory",
            "specialist",
        ),
        public_rubric_requirement_refs=(
            _ref(
                "public-rubric-requirements",
                "specialist",
            ),
        ),
        safe_evidence_bundle_refs=(evidence_bundle_ref,),
        world_ledger_snapshot_ref=_ref(
            "world-ledger-snapshot",
            "specialist",
        ),
        approved_source_evidence_refs=(
            _ref(
                "source-evidence",
                "specialist",
            ),
        ),
        taint_lineage_projection_refs=(
            _ref(
                "taint-lineage-projection",
                "specialist",
            ),
        ),
        evaluator_contract_ref=_ref(
            "evaluator-contract-projection",
            "specialist",
        ),
        contestant_tool_policy_ref=_ref(
            "contestant-tool-policy",
            "specialist",
        ),
        leakage_reference_set_ref=_ref(
            "prompt-leakage-reference-set",
            "specialist",
        ),
        executability_result_refs=(
            _ref(
                "deterministic-executability-result",
                "specialist",
            ),
        ),
    )


def _specialist_config() -> FactoryDatasetSpecialistRuntimeConfigV1:
    criteria_prompt = _prompt("criteria-rubric")
    criteria_model_policy_ref = _ref(
        "model-routing-policy",
        "criteria-rubric",
    )
    criteria_registry = build_criteria_rubric_agent_registry(
        config=CriteriaRubricAgentRegistryConfig(
            prompt_ref=criteria_prompt.to_ref(),
            model_policy_ref=criteria_model_policy_ref,
        ),
        audit=_audit(),
    )
    grading_prompt = _prompt("grading-design")
    grading_model_policy_ref = _ref(
        "model-routing-policy",
        "grading-design",
    )
    grading_registry = build_grading_design_agent_registry(
        config=GradingDesignAgentRegistryConfig(
            prompt_ref=grading_prompt.to_ref(),
            model_policy_ref=grading_model_policy_ref,
        ),
        audit=_audit(),
    )
    criteria_profile_ref = _profile("criteria-rubric").to_ref()
    grading_profile_ref = _profile("grading-design").to_ref()
    return FactoryDatasetSpecialistRuntimeConfigV1(
        job_store=FactoryDatasetJobStoreRuntimeConfigV1(
            requested_stages=(
                StageNameV2.TRACE_INDEX,
                StageNameV2.SAFETY,
                StageNameV2.LABEL,
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
                StageNameV2.RELEASE,
            ),
            privacy_profile="trusted-monitored-local",
            model_profiles=(),
            budget=ResourceBudget(
                max_model_requests=100,
                max_model_tokens=1_000_000,
                max_processes=4,
                max_renderers=4,
                max_network_requests=100,
                max_storage_bytes=10_000_000,
            ),
            concurrency=ConcurrencyLimit(
                model_requests=2,
                processes=2,
                renderers=2,
                network_requests=2,
                artifacts_per_item=4,
                items=4,
            ),
            approval_policy=(
                UserApprovalPolicyCompiler().compile(
                    mode=ApprovalMode.NONE,
                    custom_checkpoints=frozenset(),
                    final_review_scope=FinalReviewScope.NONE,
                    explicit_choice=True,
                    audit=_audit(),
                )
            ),
            export_target=ExportTarget(
                profile="LH",
                profile_version="v1",
                channel="CANARY",
                registry="registry://factory-candidate",
            ),
        ),
        review_policy=_semantic_review_policy(),
        context_sources=_semantic_context_sources(),
        evaluator_bindings=(
            EvaluatorBindingDefinition(
                evaluator_binding_id=("evaluator-binding://specialist/default"),
                evaluator_type="contract-evaluator",
                execution_mode=(EvaluatorExecutionModeV2.DETERMINISTIC),
                evaluator_version="specialist-fixture/v1",
                input_contract_ref=_ref(
                    "evaluator-input-contract",
                    "specialist",
                ),
                output_contract_ref=_ref(
                    "evaluator-output-contract",
                    "specialist",
                ),
                evaluator_principal_id=("principal://evaluator/specialist"),
                model_profile_ref=None,
                reference_refs=(),
                timeout_seconds=300,
            ),
        ),
        tool_catalog=_tool_catalog(),
        evaluated_at=NOW,
        solvability_prompt=_prompt("attachment-solvability"),
        solvability_agent_definition_ref=_ref(
            "agent-definition",
            "attachment-solvability",
        ),
        solvability_allowed_model_profile_refs=tuple(
            sorted(
                (
                    _profile("attachment-planning").to_ref(),
                    _profile("attachment-solvability").to_ref(),
                ),
                key=lambda value: value.object_id,
            )
        ),
        solvability_budget_reservation_ref=_ref(
            "work-model-reservation",
            "attachment-solvability",
        ),
        generator_model_profile_ref=_profile("attachment-planning").to_ref(),
        criteria_planning_prompt=_prompt("criteria-rubric-planning"),
        criteria_prompt=criteria_prompt,
        criteria_planning_agent_definition_ref=_ref(
            "agent-definition",
            "criteria-rubric-planning",
        ),
        criteria_agent_definition_ref=(
            criteria_registry.resolve(
                "criteria-rubric-agent",
                "criteria-rubric",
            ).to_ref()
        ),
        criteria_planning_allowed_model_profile_refs=(_profile("criteria-rubric-planning").to_ref(),),
        criteria_allowed_model_profile_refs=(criteria_profile_ref,),
        criteria_planning_budget_reservation_ref=_ref(
            "work-model-reservation",
            "criteria-rubric-planning",
        ),
        criteria_budget_reservation_ref=_ref(
            "work-model-reservation",
            "criteria-rubric",
        ),
        criteria_model_policy_ref=criteria_model_policy_ref,
        grading_planning_prompt=_prompt("grading-design-planning"),
        grading_prompt=grading_prompt,
        grading_planning_agent_definition_ref=_ref(
            "agent-definition",
            "grading-design-planning",
        ),
        grading_agent_definition_ref=(
            grading_registry.resolve(
                "grading-design-agent",
                "grading-design",
            ).to_ref()
        ),
        grading_planning_allowed_model_profile_refs=(_profile("grading-design-planning").to_ref(),),
        grading_allowed_model_profile_refs=tuple(
            sorted(
                (
                    criteria_profile_ref,
                    grading_profile_ref,
                ),
                key=lambda value: value.object_id,
            )
        ),
        grading_planning_budget_reservation_ref=_ref(
            "work-model-reservation",
            "grading-design-planning",
        ),
        grading_budget_reservation_ref=_ref(
            "work-model-reservation",
            "grading-design",
        ),
        grading_model_policy_ref=grading_model_policy_ref,
        judge_input_schema_ref=_ref(
            "json-schema",
            "grading-input",
        ),
        judge_output_schema_ref=_ref(
            "json-schema",
            "grading-output",
        ),
    )


def _release_config(
    *,
    approval_policy: UserApprovalPolicy,
) -> FactoryDatasetReleaseRuntimeConfigV1:
    duplicate_limits = AttachmentDuplicateFingerprintLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        min_token_count=4,
        shingle_size=3,
        fingerprint_bits=256,
    )
    duplicate_policy = DuplicateDetectionPolicyV2.create(
        task_near_threshold_bps=9_000,
        attachment_near_threshold_bps=9_000,
        task_shingle_size=3,
        attachment_shingle_size=3,
        fingerprint_bits=256,
        min_task_tokens=4,
        min_attachment_tokens=4,
        max_task_characters=100_000,
        max_attachment_characters=1_000_000,
        max_items=100,
        max_attachments=1_000,
        max_pair_comparisons=1_000_000,
        attachment_fingerprint_limits=duplicate_limits,
        audit=_audit(),
    )
    cross_limits = AttachmentCrossItemSafetyScanLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        max_fingerprint_count=100_000,
        max_match_count=10_000,
    )
    cross_policy = CrossItemSafetyPolicyV2.create(
        min_shared_answer_windows=2,
        min_answer_reuse_coverage_bps=8_000,
        max_items=100,
        max_attachments=1_000,
        max_total_fingerprints=100_000,
        max_foreign_fingerprints_per_target=100_000,
        max_prompt_characters=100_000,
        max_prompt_scan_comparisons=10_000_000,
        max_answer_source_pairs=1_000_000,
        max_answer_window_comparisons=10_000_000,
        max_visible_matches=100_000,
        max_reuse_pairs=100_000,
        max_clusters=100_000,
        attachment_scan_limits=cross_limits,
        audit=_audit(),
    )
    batch_policy = BatchQualityPolicyV2.create(
        max_items=100,
        max_label_decisions=10_000,
        max_package_members=100_000,
        max_lineage_checks=10_000,
        max_direct_evidence_refs=100_000,
        max_findings=100_000,
        max_finding_subject_refs=1_000_000,
        max_report_revisions=100,
        max_blocked_items=100,
        audit=_audit(),
    )
    release_policy = ReleaseProjectionPolicyV2.create(
        max_source_trace_refs=8,
        max_label_decision_refs=32,
        max_user_decision_refs=32,
        max_revalidation_reports=32,
        max_current_head_refs=256,
        max_chain_depth=32,
        allowed_channels=frozenset(
            {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
        ),
        allowed_export_profiles=frozenset({"LH"}),
        audit=_audit(),
    )
    return FactoryDatasetReleaseRuntimeConfigV1(
        duplicate_policy=duplicate_policy,
        cross_item_policy=cross_policy,
        batch_policy=batch_policy,
        approval_policy=approval_policy,
        release_policy=release_policy,
        not_required_checkpoints=(),
        output_target_ref=_ref("candidate-output-target"),
        max_files=1_000,
        max_total_bytes=10_000_000,
    )


def _fixture() -> FactoryDatasetPlanningFixtureV1:
    extract = DatasetBuildPlanTaskV2(
        task_key="extract",
        stage="core",
        task_kind="trace-extraction",
        agent_role="trace-extraction-agent",
        dependency_task_keys=(),
        input_object_types=("evaluation-requirement-spec",),
        output_object_types=("extracted-user-prompt",),
        required_capability_ids=("agent-capability://trace-extraction",),
        acceptance_check_refs=(_ref("acceptance-check", "extract"),),
        plan_review_kind=PlanKindV2.GLOBAL_BUILD,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4_096,
        max_cost_micro_usd=100_000,
    )
    rewrite = DatasetBuildPlanTaskV2(
        task_key="rewrite",
        stage="core",
        task_kind="task-rewrite",
        agent_role="task-rewrite-agent",
        dependency_task_keys=("extract",),
        input_object_types=("extracted-user-prompt",),
        output_object_types=("task-rewrite-candidate",),
        required_capability_ids=("agent-capability://task-rewrite",),
        acceptance_check_refs=(_ref("acceptance-check", "rewrite"),),
        plan_review_kind=PlanKindV2.TASK_REWRITE,
        max_attempts=2,
        max_model_requests=1,
        max_model_tokens=4_096,
        max_cost_micro_usd=100_000,
    )
    return FactoryDatasetPlanningFixtureV1(
        stage_order=("core",),
        tasks=(extract, rewrite),
        required_review_kinds=(
            PlanKindV2.GLOBAL_BUILD,
            PlanKindV2.TASK_REWRITE,
        ),
        total_model_requests=2,
        total_model_tokens=8_192,
        total_cost_micro_usd=200_000,
    )


def _write(path: Path, value: object) -> None:
    if hasattr(value, "model_dump_json"):
        payload = value.model_dump_json(indent=2)
    else:
        payload = json.dumps(value, indent=2)
    path.write_text(payload + "\n", encoding="utf-8")


def _fresh_process_output_probe(
    tmp_path: Path,
    *,
    hash_seed: str,
) -> dict[str, object]:
    script = """
import json
import sys
from dataclasses import asdict
from pathlib import Path

from eval_factory.agent_system.candidate_output import CandidateDatasetOutputAssembler
from eval_factory.agent_system.store import FactoryControlStore

store = FactoryControlStore(Path(sys.argv[1]))
run_id = sys.argv[3]
factory_rebuild = store.rebuild_current_heads()
assembler = CandidateDatasetOutputAssembler(
    store=store,
    root=Path(sys.argv[2]),
)
output_rebuild = assembler.rebuild_current_heads()
output = assembler.get(run_id)
bindings = store.list_item_bindings(run_id)
payload = {
    "aggregate_ref": store.get_dataset_aggregate(run_id).to_ref().model_dump(mode="json"),
    "binding_refs": [
        value.to_ref().model_dump(mode="json")
        for value in bindings
    ],
    "bundle_sha256": output.inventory.bundle_sha256,
    "factory_rebuild": asdict(factory_rebuild),
    "inventory_ref": output.inventory.to_ref().model_dump(mode="json"),
    "manifest_ref": output.manifest.to_ref().model_dump(mode="json"),
    "output_rebuild": list(output_rebuild),
    "run_ref": store.get_run(run_id).to_ref().model_dump(mode="json"),
    "stage_head_refs": [
        head.to_ref().model_dump(mode="json")
        for binding in bindings
        for head in store.list_item_stage_heads(binding.item_id)
    ],
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""
    environment = {
        **os.environ,
        "PYTHONHASHSEED": hash_seed,
        "PYTHONPATH": str(ROOT / "src"),
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            script,
            str(tmp_path / "factory.sqlite3"),
            str(tmp_path / "candidate-output"),
            _requirement().run_id,
        ),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _args(
    tmp_path: Path,
    *,
    core_manifest: Path | None = None,
    core_manifest_sha256: str = HASH,
    include_r4: bool = False,
    include_attachment: bool = False,
    include_specialists: bool = False,
    include_release: bool = False,
    blocked_core_prompt_refs: tuple[ObjectRef, ...] = (),
) -> list[str]:
    include_core = core_manifest is not None
    if include_attachment and not include_r4:
        raise ValueError("attachment CLI fixture requires R4")
    if include_specialists and not include_attachment:
        raise ValueError("specialist CLI fixture requires attachment")
    if include_release and not include_specialists:
        raise ValueError("release CLI fixture requires specialists")
    specialist_config = _specialist_config() if include_specialists else None
    values = {
        "request": _request(
            core_manifest_sha256,
            include_attachment=include_attachment,
            include_specialists=include_specialists,
        ),
        "policy": _policy(
            include_attachment=include_attachment,
            include_specialists=include_specialists,
        ),
        "requirement": _requirement(),
        "runtime-config": _config(
            include_core=include_core,
            include_r4=include_r4,
            include_attachment=include_attachment,
            specialist_config=specialist_config,
        ),
        "provider-fixture": _fixture(),
    }
    arguments = [
        "agent",
        "run",
        "--factory-store",
        str(tmp_path / "factory.sqlite3"),
        "--private-store",
        str(tmp_path / "private"),
        "--gateway-store",
        str(tmp_path / "gateway.sqlite3"),
        "--direct-runtime-compatibility",
    ]
    for option, value in values.items():
        path = tmp_path / f"{option}.json"
        _write(path, value)
        arguments.extend((f"--{option}", str(path)))
    if core_manifest is not None:
        core_config_path = tmp_path / "core-runtime-config.json"
        _write(
            core_config_path,
            _core_config(
                blocked_extracted_prompt_refs=(blocked_core_prompt_refs),
            ),
        )
        arguments.extend(
            (
                "--core-runtime-config",
                str(core_config_path),
                "--manifest",
                str(core_manifest),
                "--raw-root",
                str(core_manifest.parent),
                "--core-workspace",
                str(tmp_path / "core-workspace"),
            )
        )
    if include_r4:
        r4_config_path = tmp_path / ("r4-runtime-config.json")
        _write(r4_config_path, _r4_config())
        arguments.extend(
            (
                "--r4-runtime-config",
                str(r4_config_path),
            )
        )
    if include_attachment:
        attachment_config_path = tmp_path / "attachment-runtime-config.json"
        _write(attachment_config_path, _attachment_config())
        arguments.extend(
            (
                "--attachment-runtime-config",
                str(attachment_config_path),
                "--attachment-workspace",
                str(tmp_path / "attachment-workspace"),
            )
        )
    if specialist_config is not None:
        specialist_config_path = tmp_path / "specialist-runtime-config.json"
        _write(specialist_config_path, specialist_config)
        arguments.extend(
            (
                "--specialist-runtime-config",
                str(specialist_config_path),
                "--job-store",
                str(tmp_path / "job-store.sqlite3"),
                "--specialist-workspace",
                str(tmp_path / "specialist-workspace"),
            )
        )
    if include_release:
        assert specialist_config is not None
        release_config_path = tmp_path / "release-runtime-config.json"
        _write(
            release_config_path,
            _release_config(
                approval_policy=(specialist_config.job_store.approval_policy),
            ),
        )
        arguments.extend(
            (
                "--release-runtime-config",
                str(release_config_path),
                "--candidate-output",
                str(tmp_path / "candidate-output"),
            )
        )
    arguments.extend(("--user", "user://dataset-owner", "--json"))
    return arguments


def _real_trace_root(
    tmp_path: Path,
    *,
    instance_ids: tuple[str, ...],
) -> tuple[Path, str]:
    root = tmp_path / "raw"
    root.mkdir()
    with (RAW_ROOT / "manifest.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        rows_by_id = {row["instance_id"]: row for row in reader if row["instance_id"] in instance_ids}
        fieldnames = reader.fieldnames
    assert fieldnames is not None
    assert set(rows_by_id) == set(instance_ids)
    rows = tuple(rows_by_id[instance_id] for instance_id in instance_ids)
    manifest = root / "manifest.csv"
    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        source_name = f"{row['instance_id']}_{row['sid']}.jsonl"
        (root / source_name).write_bytes((RAW_ROOT / source_name).read_bytes())
    return (
        manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )


def _one_trace_root(
    tmp_path: Path,
) -> tuple[Path, str]:
    return _real_trace_root(
        tmp_path,
        instance_ids=("LH_077",),
    )


def _two_trace_root(
    tmp_path: Path,
) -> tuple[Path, str]:
    return _real_trace_root(
        tmp_path,
        instance_ids=("LH_077", "LH_098"),
    )


def _three_trace_root(
    tmp_path: Path,
) -> tuple[Path, str]:
    return _real_trace_root(
        tmp_path,
        instance_ids=("LH_006", "LH_077", "LH_098"),
    )


def _lh_006_extracted_prompt_ref() -> ObjectRef:
    digest = "fc22ce173f80e8953598497826287f689da507296465a188f5b7d57f2cadab53"
    return ObjectRef(
        object_type="extracted-user-prompt",
        object_id=f"extracted-user-prompt://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _approve_and_resume(
    reviews: PlanReviewService,
    review_id: str,
    *,
    suffix: str,
) -> None:
    reason_prefix = suffix.upper().replace("-", "_")
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://dataset-owner",
            reason_code=f"{reason_prefix}_PLAN_APPROVED",
            idempotency_key=f"approve-cli-{suffix}",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://dataset-owner",
            idempotency_key=f"resume-cli-{suffix}",
        ),
        audit=_audit(),
    )


def _approve_review_batch(
    reviews: PlanReviewService,
    payload: dict[str, object],
    *,
    plan_kind: PlanKindV2,
    expected_plan_refs: set[ObjectRef],
    suffix: str,
) -> None:
    pending_refs = payload["pending_review_refs"]
    assert isinstance(pending_refs, list)
    assert len(pending_refs) == len(expected_plan_refs)
    observed_run_refs = set()
    observed_plan_refs = set()
    for index, reference in enumerate(pending_refs):
        assert isinstance(reference, dict)
        review_id = str(reference["object_id"])
        review = reviews.show(review_id)
        assert review.request.plan_kind is plan_kind
        observed_run_refs.add(review.request.run_ref)
        observed_plan_refs.add(review.request.plan_ref)
        _approve_and_resume(
            reviews,
            review_id,
            suffix=f"{suffix}-{index}",
        )
    assert len(observed_run_refs) == len(expected_plan_refs)
    assert observed_plan_refs == expected_plan_refs


def test_agent_run_cli_opens_and_replays_global_review(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    arguments = _args(tmp_path)

    first = runner.invoke(app, arguments)
    replay = runner.invoke(app, arguments)

    assert first.exit_code == replay.exit_code == 0
    first_payload = json.loads(first.stdout)
    assert json.loads(replay.stdout) == first_payload
    assert first_payload["status"] == "WAITING_REVIEW"
    assert first_payload["next_action"] == "REVIEW_PLAN"
    assert len(first_payload["pending_review_refs"]) == 1
    assert "DEVELOPMENT_FIXTURE_ONLY" not in first.stdout
    forbidden = (
        "credential",
        "final_answer",
        "grader_rule",
        "hidden_condition",
        "private_reference",
        "raw_trace",
    )
    assert not any(value in first.stdout.casefold() for value in forbidden)

    with sqlite3.connect(tmp_path / "factory.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM factory_dataset_run_requests").fetchone() == (1,)
        assert connection.execute("SELECT COUNT(*) FROM plan_review_requests").fetchone() == (1,)
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (1,)


def test_agent_run_cli_never_falls_back_to_direct_runtime(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    arguments.remove("--direct-runtime-compatibility")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert payload["message"] == ("input does not satisfy the required contract")
    assert not (tmp_path / "factory.sqlite3").exists()


def test_agent_run_cli_rejects_overlapping_store_roots(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    private_index = arguments.index("--private-store") + 1
    arguments[private_index] = str(tmp_path / "factory.sqlite3")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.stderr


@requires_private_corpus
def test_agent_run_cli_rejects_partial_core_admission(
    tmp_path: Path,
) -> None:
    manifest, _manifest_sha256 = _one_trace_root(tmp_path)
    arguments = _args(tmp_path)
    arguments.extend(("--manifest", str(manifest)))

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.stderr
    assert not (tmp_path / "factory.sqlite3").exists()


def test_agent_run_cli_rejects_partial_attachment_admission(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    arguments.extend(
        (
            "--attachment-workspace",
            str(tmp_path / "attachment-workspace"),
        )
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.stderr
    assert not (tmp_path / "factory.sqlite3").exists()


def test_agent_run_cli_rejects_partial_specialist_admission(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    specialist_path = tmp_path / "specialist.json"
    _write(specialist_path, _specialist_config())
    arguments.extend(
        (
            "--specialist-runtime-config",
            str(specialist_path),
        )
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.stderr
    assert not (tmp_path / "factory.sqlite3").exists()


def test_agent_run_cli_rejects_partial_release_admission(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    arguments.extend(
        (
            "--candidate-output",
            str(tmp_path / "candidate-output"),
        )
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error_code"] == "INVALID_INPUT"
    assert str(tmp_path) not in result.stderr
    assert not (tmp_path / "factory.sqlite3").exists()


def test_agent_run_cli_rejects_stale_expected_version_before_provider(
    tmp_path: Path,
) -> None:
    arguments = _args(tmp_path)
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0

    stale = runner.invoke(
        app,
        (
            *arguments,
            "--expected-run-version",
            "0",
        ),
    )

    assert stale.exit_code == 2
    payload = json.loads(stale.stderr)
    assert payload["error_code"] == "CONCURRENCY_CONFLICT"
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (1,)


@requires_private_corpus
def test_agent_run_cli_advances_one_real_trace_after_review(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256 = _one_trace_root(tmp_path)
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
    )
    runner = CliRunner()

    waiting = runner.invoke(app, arguments)
    assert waiting.exit_code == 0
    waiting_payload = json.loads(waiting.stdout)
    review_id = waiting_payload["pending_review_refs"][0]["object_id"]
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    reviews.decide(
        review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://dataset-owner",
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-cli-core-plan",
        ),
        audit=_audit(),
    )
    reviews.resume(
        review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://dataset-owner",
            idempotency_key="resume-cli-core-plan",
        ),
        audit=_audit(),
    )

    advanced = runner.invoke(app, arguments)
    replay = runner.invoke(app, arguments)

    assert advanced.exit_code == 0, advanced.stderr
    assert replay.exit_code == 0, replay.stderr
    advanced_payload = json.loads(advanced.stdout)
    assert json.loads(replay.stdout) == advanced_payload
    assert len(advanced_payload["item_binding_refs"]) == 1
    assert advanced_payload["incomplete_count"] == 1
    binding = store.list_item_bindings(_requirement().run_id)[0]
    head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.CORE_SELECTION,
    )
    assert head.result_ref == binding.rewrite_candidate_ref
    task_head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.TASK_AUTHORING,
    )
    assert task_head.result_ref.object_type == ("r4-task-contract-set")
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (7,)


@requires_private_corpus
def test_agent_run_cli_reviews_executes_and_replays_attachment(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256 = _one_trace_root(tmp_path)
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
        include_attachment=True,
    )
    runner = CliRunner()

    waiting = runner.invoke(app, arguments)
    assert waiting.exit_code == 0, waiting.stderr
    waiting_payload = json.loads(waiting.stdout)
    global_review_id = waiting_payload["pending_review_refs"][0]["object_id"]
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    reviews.decide(
        global_review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://dataset-owner",
            reason_code="GLOBAL_PLAN_APPROVED",
            idempotency_key="approve-cli-attachment-global",
        ),
        audit=_audit(),
    )
    reviews.resume(
        global_review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://dataset-owner",
            idempotency_key="resume-cli-attachment-global",
        ),
        audit=_audit(),
    )

    attachment_waiting = runner.invoke(app, arguments)
    attachment_replay = runner.invoke(app, arguments)
    assert attachment_waiting.exit_code == 0, attachment_waiting.stderr
    assert attachment_replay.exit_code == 0, attachment_replay.stderr
    attachment_payload = json.loads(attachment_waiting.stdout)
    assert json.loads(attachment_replay.stdout) == (attachment_payload)
    assert attachment_payload["status"] == "WAITING_REVIEW", (
        attachment_payload["next_action"],
        attachment_payload["pending_review_refs"],
        attachment_payload["item_binding_refs"],
    )
    assert attachment_payload["next_action"] == "REVIEW_PLAN"
    attachment_review_id = attachment_payload["pending_review_refs"][0]["object_id"]
    assert reviews.show(attachment_review_id).request.plan_kind is PlanKindV2.ATTACHMENT_GENERATION
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (8,)

    reviews.decide(
        attachment_review_id,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=1,
            decision=PlanDecisionKindV2.APPROVE,
            decided_by="user://dataset-owner",
            reason_code="ATTACHMENT_PLAN_APPROVED",
            idempotency_key="approve-cli-attachment-plan",
        ),
        audit=_audit(),
    )
    reviews.resume(
        attachment_review_id,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=1,
            resumed_by="user://dataset-owner",
            idempotency_key="resume-cli-attachment-plan",
        ),
        audit=_audit(),
    )

    executed = runner.invoke(app, arguments)
    executed_replay = runner.invoke(app, arguments)
    assert executed.exit_code == 0, executed.stderr
    assert executed_replay.exit_code == 0, executed_replay.stderr
    assert json.loads(executed_replay.stdout) == json.loads(executed.stdout)
    binding = store.list_item_bindings(_requirement().run_id)[0]
    attachment_head = store.get_item_stage_head(
        binding.item_id,
        FactoryItemStageV2.ATTACHMENT,
    )
    assert attachment_head.outcome.value == "SUCCEEDED"
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (8,)


@requires_private_corpus
def test_agent_run_cli_reviews_and_executes_specialist_chain(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256 = _one_trace_root(tmp_path)
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
        include_attachment=True,
        include_specialists=True,
    )
    runner = CliRunner()

    global_waiting = runner.invoke(app, arguments)
    assert global_waiting.exit_code == 0, global_waiting.stderr
    global_payload = json.loads(global_waiting.stdout)
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    _approve_and_resume(
        reviews,
        global_payload["pending_review_refs"][0]["object_id"],
        suffix="specialist-global",
    )

    attachment_waiting = runner.invoke(app, arguments)
    assert attachment_waiting.exit_code == 0, attachment_waiting.stderr
    attachment_payload = json.loads(attachment_waiting.stdout)
    attachment_review_id = attachment_payload["pending_review_refs"][0]["object_id"]
    assert reviews.show(attachment_review_id).request.plan_kind is PlanKindV2.ATTACHMENT_GENERATION
    _approve_and_resume(
        reviews,
        attachment_review_id,
        suffix="specialist-attachment",
    )

    criteria_waiting = runner.invoke(app, arguments)
    assert criteria_waiting.exit_code == 0, criteria_waiting.stderr
    criteria_payload = json.loads(criteria_waiting.stdout)
    criteria_review_id = criteria_payload["pending_review_refs"][0]["object_id"]
    assert reviews.show(criteria_review_id).request.plan_kind is PlanKindV2.CRITERIA_RUBRIC
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (10,)
    _approve_and_resume(
        reviews,
        criteria_review_id,
        suffix="specialist-criteria",
    )

    grading_waiting = runner.invoke(app, arguments)
    assert grading_waiting.exit_code == 0, grading_waiting.stderr
    grading_payload = json.loads(grading_waiting.stdout)
    grading_review_id = grading_payload["pending_review_refs"][0]["object_id"]
    assert reviews.show(grading_review_id).request.plan_kind is PlanKindV2.GRADING_DESIGN
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (12,)
    _approve_and_resume(
        reviews,
        grading_review_id,
        suffix="specialist-grading",
    )

    executed = runner.invoke(app, arguments)
    replay = runner.invoke(app, arguments)
    assert executed.exit_code == 0, executed.stderr
    assert replay.exit_code == 0, replay.stderr
    assert json.loads(replay.stdout) == json.loads(executed.stdout)
    binding = store.list_item_bindings(_requirement().run_id)[0]
    for stage in (
        FactoryItemStageV2.ITEM_QUALITY,
        FactoryItemStageV2.CRITERIA_RUBRIC,
        FactoryItemStageV2.GRADING_DESIGN,
    ):
        head = store.get_item_stage_head(
            binding.item_id,
            stage,
        )
        assert head.outcome.value == "SUCCEEDED", (
            stage,
            head.reason_codes,
        )
    job_store = JobStore(tmp_path / "job-store.sqlite3")
    quality_runs = job_store.list_stage_runs(
        job_id=_requirement().run_id,
        item_id=binding.item_id,
        stage=StageNameV2.ITEM_QUALITY,
    )
    assert len(quality_runs) == 3
    assert all(
        stage_run.status is StageRunStatus.SUCCEEDED
        and job_store.get_stage_result_for_run(stage_run.stage_run_id) is not None
        for stage_run in quality_runs
    )
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (13,)


@requires_private_corpus
def test_agent_run_cli_delivers_two_candidates_and_exactly_replays(
    tmp_path: Path,
) -> None:
    manifest, manifest_sha256 = _two_trace_root(tmp_path)
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
        include_attachment=True,
        include_specialists=True,
        include_release=True,
    )
    runner = CliRunner()

    global_waiting = runner.invoke(app, arguments)
    assert global_waiting.exit_code == 0, global_waiting.stderr
    global_payload = json.loads(global_waiting.stdout)
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    _approve_and_resume(
        reviews,
        global_payload["pending_review_refs"][0]["object_id"],
        suffix="release-global",
    )

    attachment_waiting = runner.invoke(app, arguments)
    assert attachment_waiting.exit_code == 0, attachment_waiting.stderr
    bindings = store.list_item_bindings(_requirement().run_id)
    assert len(bindings) == 2
    child_run_ids = {store.get_item_run(binding).run_id for binding in bindings}
    _approve_review_batch(
        reviews,
        json.loads(attachment_waiting.stdout),
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="release-attachment",
    )

    criteria_waiting = runner.invoke(app, arguments)
    assert criteria_waiting.exit_code == 0, criteria_waiting.stderr
    _approve_review_batch(
        reviews,
        json.loads(criteria_waiting.stdout),
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.CRITERIA_RUBRIC,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="release-criteria",
    )

    grading_waiting = runner.invoke(app, arguments)
    assert grading_waiting.exit_code == 0, grading_waiting.stderr
    _approve_review_batch(
        reviews,
        json.loads(grading_waiting.stdout),
        plan_kind=PlanKindV2.GRADING_DESIGN,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.GRADING_DESIGN,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="release-grading",
    )

    delivery_waiting = runner.invoke(app, arguments)
    assert delivery_waiting.exit_code == 0, delivery_waiting.stderr
    delivery_payload = json.loads(delivery_waiting.stdout)
    assert delivery_payload["candidate_count"] == 2
    assert delivery_payload["incomplete_count"] == 0
    pending_delivery = delivery_payload["pending_review_refs"]
    assert len(pending_delivery) == 1
    delivery_review_id = pending_delivery[0]["object_id"]
    delivery_review = reviews.show(delivery_review_id)
    assert delivery_review.request.plan_kind is PlanKindV2.FINAL_DELIVERY
    assert (
        delivery_review.request.plan_ref
        == store.get_domain_plan(
            _requirement().run_id,
            PlanKindV2.FINAL_DELIVERY,
        ).plan_ref
    )

    aggregate = store.get_dataset_aggregate(_requirement().run_id)
    assert aggregate.candidate_count == 2
    assert len(aggregate.candidate_binding_refs) == 2
    for binding in bindings:
        release_head = store.get_item_stage_head(
            binding.item_id,
            FactoryItemStageV2.RELEASE_CANDIDATE,
        )
        assert release_head.outcome.value == "SUCCEEDED"
        assert release_head.result_ref.object_type == ("release-projection-result")

    job_store = JobStore(tmp_path / "job-store.sqlite3")
    graph = job_store.get_job_work_graph(_requirement().run_id)
    leases = {
        lease.work_unit_ref: lease
        for lease in job_store.list_work_leases(
            job_id=_requirement().run_id,
        )
    }
    witness_units = tuple(
        unit
        for unit in graph.work_units
        if unit.stage
        in {
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
        }
    )
    assert len(witness_units) == 3
    for unit in witness_units:
        lease = leases[resolved_work_unit_v2_ref(unit)]
        assert lease.stage_run_ref is not None
        stage_run = job_store.get_stage_run(
            lease.stage_run_ref.object_id,
        )
        result = job_store.get_stage_result_for_run(
            stage_run.stage_run_id,
        )
        assert stage_run.status is StageRunStatus.SUCCEEDED
        assert result is not None
        assert result.status is StageRunStatus.SUCCEEDED
        if unit.stage is StageNameV2.ITEM_QUALITY:
            assert unit.item_id is not None
            quality_head = store.get_item_stage_head(
                unit.item_id,
                FactoryItemStageV2.ITEM_QUALITY,
            )
            criteria_head = store.get_item_stage_head(
                unit.item_id,
                FactoryItemStageV2.CRITERIA_RUBRIC,
            )
            release_head = store.get_item_stage_head(
                unit.item_id,
                FactoryItemStageV2.RELEASE_CANDIDATE,
            )
            reviewed_quality_refs = tuple(
                ref
                for ref in release_head.dependency_result_refs
                if ref.object_type == "item-quality-compilation-result"
            )
            assert criteria_head.dependency_result_refs == (quality_head.result_ref,)
            assert len(reviewed_quality_refs) == 1
            assert result.output_refs == reviewed_quality_refs
        else:
            assert result.output_refs == (aggregate.batch_quality_ref,)

    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (25,)

    _approve_and_resume(
        reviews,
        delivery_review_id,
        suffix="release-final-delivery",
    )
    completed = runner.invoke(app, arguments)
    assert completed.exit_code == 0, completed.stderr
    completed_payload = json.loads(completed.stdout)
    assert completed_payload["status"] == "COMPLETED"
    assert completed_payload["next_action"] == "INSPECT_OUTPUT"
    assert completed_payload["candidate_count"] == 2
    assert completed_payload["delivery_manifest_ref"] is not None

    assembler = CandidateDatasetOutputAssembler(
        store=store,
        root=tmp_path / "candidate-output",
    )
    output = assembler.get(_requirement().run_id)
    assert output.bundle_path.is_dir()
    assert output.manifest.item_count == 2
    assert len(output.manifest.candidate_item_refs) == 2
    assert output.manifest.inventory_ref == output.inventory.to_ref()
    assert (
        output.bundle_path / "dataset-manifest.json"
    ).read_bytes() == output.manifest.canonical_json() + b"\n"
    assert len((output.bundle_path / "candidate-items.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    replay = runner.invoke(app, arguments)
    assert replay.exit_code == 0, replay.stderr
    assert json.loads(replay.stdout) == completed_payload
    replayed_output = assembler.get(_requirement().run_id)
    assert replayed_output.manifest == output.manifest
    assert replayed_output.inventory == output.inventory
    assert replayed_output.bundle_path == output.bundle_path
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (25,)
    seed_one = _fresh_process_output_probe(
        tmp_path,
        hash_seed="1",
    )
    seed_321 = _fresh_process_output_probe(
        tmp_path,
        hash_seed="321",
    )
    assert seed_321 == seed_one
    assert seed_one["manifest_ref"] == output.manifest.to_ref().model_dump(
        mode="json",
    )
    assert seed_one["inventory_ref"] == output.inventory.to_ref().model_dump(
        mode="json",
    )


@requires_private_corpus
def test_agent_run_cli_isolates_one_blocked_trace_and_delivers_two_candidates(
    tmp_path: Path,
) -> None:
    blocked_prompt_ref = _lh_006_extracted_prompt_ref()
    manifest, manifest_sha256 = _three_trace_root(tmp_path)
    arguments = _args(
        tmp_path,
        core_manifest=manifest,
        core_manifest_sha256=manifest_sha256,
        include_r4=True,
        include_attachment=True,
        include_specialists=True,
        include_release=True,
        blocked_core_prompt_refs=(blocked_prompt_ref,),
    )
    runner = CliRunner()

    global_waiting = runner.invoke(app, arguments)
    assert global_waiting.exit_code == 0, global_waiting.stderr
    store = FactoryControlStore(tmp_path / "factory.sqlite3")
    reviews = PlanReviewService(
        store,
        compiler=DatasetBuildPlanCompiler(_registry()),
    )
    _approve_and_resume(
        reviews,
        json.loads(global_waiting.stdout)["pending_review_refs"][0]["object_id"],
        suffix="blocked-global",
    )

    attachment_waiting = runner.invoke(app, arguments)
    assert attachment_waiting.exit_code == 0, attachment_waiting.stderr
    core_execution = FactoryCoreMaterialStore(store).get(
        _requirement().run_id,
    )
    blocked_decisions = tuple(
        value
        for value in core_execution.decisions
        if value.disposition is TraceCandidateDispositionV2.BLOCKED
    )
    assert len(blocked_decisions) == 1
    assert (
        blocked_decisions[0].source_trace_id
        == "source-trace://LH_006/0217819360791830b3f405af7fd729b2ba1cacbbbecc68de452d9"
    )
    assert blocked_decisions[0].reason_codes == ("TRACE_PROCESSING_FAILED",)
    assert (
        sum(value.disposition is TraceCandidateDispositionV2.CANDIDATE for value in core_execution.decisions)
        == 2
    )
    bindings = store.list_item_bindings(_requirement().run_id)
    assert len(bindings) == 2
    child_run_ids = {store.get_item_run(binding).run_id for binding in bindings}
    _approve_review_batch(
        reviews,
        json.loads(attachment_waiting.stdout),
        plan_kind=PlanKindV2.ATTACHMENT_GENERATION,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.ATTACHMENT_GENERATION,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="blocked-attachment",
    )

    criteria_waiting = runner.invoke(app, arguments)
    assert criteria_waiting.exit_code == 0, criteria_waiting.stderr
    _approve_review_batch(
        reviews,
        json.loads(criteria_waiting.stdout),
        plan_kind=PlanKindV2.CRITERIA_RUBRIC,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.CRITERIA_RUBRIC,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="blocked-criteria",
    )

    grading_waiting = runner.invoke(app, arguments)
    assert grading_waiting.exit_code == 0, grading_waiting.stderr
    _approve_review_batch(
        reviews,
        json.loads(grading_waiting.stdout),
        plan_kind=PlanKindV2.GRADING_DESIGN,
        expected_plan_refs={
            store.get_domain_plan(
                child_run_id,
                PlanKindV2.GRADING_DESIGN,
            ).plan_ref
            for child_run_id in child_run_ids
        },
        suffix="blocked-grading",
    )

    delivery_waiting = runner.invoke(app, arguments)
    assert delivery_waiting.exit_code == 0, delivery_waiting.stderr
    delivery_payload = json.loads(delivery_waiting.stdout)
    assert delivery_payload["candidate_count"] == 2
    assert delivery_payload["blocked_count"] == 0
    assert delivery_payload["incomplete_count"] == 0
    pending_delivery = delivery_payload["pending_review_refs"]
    assert len(pending_delivery) == 1
    delivery_review_id = pending_delivery[0]["object_id"]
    delivery_review = reviews.show(delivery_review_id)
    assert delivery_review.request.plan_kind is PlanKindV2.FINAL_DELIVERY

    aggregate = store.get_dataset_aggregate(_requirement().run_id)
    assert aggregate.candidate_count == 2
    assert aggregate.blocked_count == 0
    assert aggregate.item_count == 2
    for binding in bindings:
        assert (
            store.get_item_stage_head(
                binding.item_id,
                FactoryItemStageV2.RELEASE_CANDIDATE,
            ).outcome.value
            == "SUCCEEDED"
        )

    job_store = JobStore(tmp_path / "job-store.sqlite3")
    graph = job_store.get_job_work_graph(_requirement().run_id)
    witness_units = tuple(
        unit
        for unit in graph.work_units
        if unit.stage
        in {
            StageNameV2.ITEM_QUALITY,
            StageNameV2.BATCH_QUALITY,
        }
    )
    assert len(witness_units) == 3
    leases = {
        lease.work_unit_ref: lease
        for lease in job_store.list_work_leases(
            job_id=_requirement().run_id,
        )
    }
    for unit in witness_units:
        lease = leases[resolved_work_unit_v2_ref(unit)]
        assert lease.stage_run_ref is not None
        result = job_store.get_stage_result_for_run(
            lease.stage_run_ref.object_id,
        )
        assert result is not None
        assert result.status is StageRunStatus.SUCCEEDED

    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (26,)

    _approve_and_resume(
        reviews,
        delivery_review_id,
        suffix="blocked-final-delivery",
    )
    completed = runner.invoke(app, arguments)
    assert completed.exit_code == 0, completed.stderr
    completed_payload = json.loads(completed.stdout)
    assert completed_payload["status"] == "COMPLETED"
    assert completed_payload["candidate_count"] == 2
    assert completed_payload["blocked_count"] == 0

    output = CandidateDatasetOutputAssembler(
        store=store,
        root=tmp_path / "candidate-output",
    ).get(_requirement().run_id)
    assert output.manifest.item_count == 2
    assert output.manifest.blocked_binding_refs == ()
    assert len((output.bundle_path / "candidate-items.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (output.bundle_path / "blocked-bindings.jsonl").read_text(encoding="utf-8") == ""

    replay = runner.invoke(app, arguments)
    assert replay.exit_code == 0, replay.stderr
    assert json.loads(replay.stdout) == completed_payload
    replayed_output = CandidateDatasetOutputAssembler(
        store=store,
        root=tmp_path / "candidate-output",
    ).get(_requirement().run_id)
    assert replayed_output.manifest == output.manifest
    assert replayed_output.inventory == output.inventory
    assert (
        FactoryCoreMaterialStore(store).get(
            _requirement().run_id,
        )
        == core_execution
    )
    with sqlite3.connect(tmp_path / "gateway.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM gateway_invocation_records").fetchone() == (26,)
