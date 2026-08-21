from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import BaseModel, ConfigDict, ValidationError

from eval_factory.agent_system import (
    AgentRegistry,
    AttachmentGenerationPlanAdapter,
    AttachmentGenerationPlanCompiler,
    AttachmentPlanReviewEditSubmissionV1,
    CoreOutputAssembler,
    CoreOutputConflictError,
    CoreOutputError,
    CoreOutputIntegrityError,
    CriteriaRubricPlanAdapter,
    CriteriaRubricPlanCompiler,
    CriteriaRubricPlanReviewEditSubmissionV1,
    DatasetBuildPlanCompiler,
    DatasetDeliveryPlanAdapter,
    DatasetDeliveryPlanCompiler,
    DatasetDeliveryPlanReviewEditSubmissionV1,
    FactoryControlConflictError,
    FactoryControlIntegrityError,
    FactoryControlNotFoundError,
    FactoryControlStore,
    GlobalBuildPlanAdapter,
    GradingDesignPlanAdapter,
    GradingDesignPlanCompiler,
    GradingDesignPlanReviewEditSubmissionV1,
    PlanReviewConflictError,
    PlanReviewDecisionSubmissionV1,
    PlanReviewError,
    PlanReviewIntegrityError,
    PlanReviewNotResumableError,
    PlanReviewResumeSubmissionV1,
    PlanReviewService,
    ReviewablePlanAdapterRegistry,
)
from eval_factory.agent_system.plan_adapters import erased_adapter
from eval_factory.approval import (
    UserCheckpointAuthenticationError,
    UserCheckpointInteractionConflictError,
    UserCheckpointInteractionError,
    UserCheckpointInteractionIntegrityError,
    UserCheckpointInteractionNotResumableError,
    UserCheckpointInteractionService,
    UserCheckpointMaterialError,
    UserCheckpointMaterialIntegrityError,
    UserCheckpointMaterialStore,
    UserDecisionConflictError,
)
from eval_factory.approval.interaction_models import (
    TrustedAuthenticatedUserContextV2,
    UserCheckpointSourceContextV2,
)
from eval_factory.console_api.contracts import (
    PlanReviewEditPayloadV1,
    compile_plan_review_edit,
)
from eval_factory.contracts.agent_system_v2 import (
    AgentCapabilityV2,
    AgentDefinitionV2,
    EvaluationRequirementSpecV2,
    FactoryRunPolicyV2,
    PlanDecisionKindV2,
    PlanReviewStateV2,
)
from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
)
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionOutcomeV2,
    CanaryRegressionPolicyV2,
    CanaryRegressionReportV2,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointDecisionSubmissionV2,
    UserCheckpointInteractionPolicyV2,
)
from eval_factory.contracts.cli_v2 import (
    PipelineControlConfigV2,
)
from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentOutcomeV2,
    ConcurrencyExperimentPolicyV2,
    ConcurrencyExperimentReportV2,
    validate_concurrency_experiment_report_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    LabelQualityEvaluationPolicyV2,
    LabelQualityEvaluationReportV2,
    LabelQualityEvidenceClassV2,
    LabelQualityOutcomeV2,
)
from eval_factory.contracts.observability_v2 import MetricScopeV2
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2
from eval_factory.contracts.production_attestation_v2 import (
    ProductionAttestationEvidenceClassV2,
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationPolicyV2,
    ProductionReadinessAttestationResultV2,
)
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewEvidenceClassV2,
    ProductionReadinessReviewOutcomeV2,
    ProductionReadinessReviewPolicyV2,
    ProductionReadinessReviewReportV2,
)
from eval_factory.contracts.production_release_v2 import (
    ProductionReleaseEvidenceClassV2,
    ProductionReleaseOutcomeV2,
    ProductionReleasePolicyV2,
    ProductionReleaseResultV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityOutcomeV2,
    RealTraceStabilityPolicyV2,
    RealTraceStabilityReportV2,
)
from eval_factory.contracts.scheduler_load_v2 import (
    SchedulerLoadPolicyV2,
    SchedulerLoadReportOutcomeV2,
    SchedulerLoadReportV2,
)
from eval_factory.orchestration import (
    BatchObservabilityIntegrityError,
    BatchObservabilityPolicyError,
    ConcurrencyConflictError,
    IdempotencyConflictError,
    IllegalTransitionError,
    ImmutableResultError,
    JobPlanningPolicyError,
    JobStore,
    JobStoreError,
    ModelControlPolicyError,
    PipelineCheckpointRequiredError,
    PipelineLifecyclePolicyError,
    PipelineLifecycleService,
    PipelineObservabilityRefreshRequiredError,
    RecordNotFoundError,
    ResourceControlPolicyError,
    WorkControlPolicyError,
    WorkFanoutPolicyError,
)
from eval_factory.orchestration.canary_errors import (
    CanaryDriverError,
    CanaryProfileError,
)
from eval_factory.orchestration.stage_object_store import (
    StageObjectStoreError,
)
from eval_factory.trace import TraceSourceRegistry
from eval_factory.trace.query.cli import run_app, trace_app
from eval_factory.trace.storage import TraceIndexStore

if TYPE_CHECKING:
    from eval_factory.readiness.concurrency_experiment_models import (
        PreparedConcurrencyRecoveryCase,
    )
    from eval_factory.readiness.concurrency_experiment_store import (
        ConcurrencyExperimentMaterialStore,
        ConcurrencyExperimentReportStore,
    )

app = typer.Typer(help="Eval Factory evidence and pipeline tools.")
pipeline_app = typer.Typer(help="Bounded R6 pipeline lifecycle commands.")
checkpoint_app = typer.Typer(help="Bounded R7 user checkpoint interaction commands.")
agent_app = typer.Typer(help="Graph-engineered multi-Agent control commands.")
agent_plan_app = typer.Typer(help="Shared plan review and resume commands.")
app.add_typer(trace_app, name="trace")
app.add_typer(run_app, name="run")
app.add_typer(pipeline_app, name="pipeline")
pipeline_app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(agent_plan_app, name="plan")

_MAX_PIPELINE_INPUT_BYTES = 8 * 1024 * 1024
_MAX_CHECKPOINT_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_CHECKPOINT_PRESENTATION_BYTES = 64 * 1024 * 1024


class _PlanReviewRegistryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capabilities: tuple[AgentCapabilityV2, ...]
    definitions: tuple[AgentDefinitionV2, ...]


@agent_app.command("output")
def agent_output(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    run: Annotated[str, typer.Option("--run")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: CoreOutputAssembler(
            store=FactoryControlStore(factory_store),
            root=output_root,
        ).get(run)
    )


@agent_app.command("run")
def agent_run(
    factory_store: Annotated[
        Path,
        typer.Option("--factory-store"),
    ],
    private_store: Annotated[
        Path,
        typer.Option("--private-store"),
    ],
    gateway_store: Annotated[
        Path,
        typer.Option("--gateway-store"),
    ],
    request: Annotated[Path, typer.Option("--request")],
    policy: Annotated[Path, typer.Option("--policy")],
    requirement: Annotated[Path, typer.Option("--requirement")],
    runtime_config: Annotated[
        Path,
        typer.Option("--runtime-config"),
    ],
    provider_fixture: Annotated[
        Path,
        typer.Option("--provider-fixture"),
    ],
    user: Annotated[str, typer.Option("--user")],
    expected_run_version: Annotated[
        int | None,
        typer.Option("--expected-run-version", min=0),
    ] = None,
    direct_runtime_compatibility: Annotated[
        bool,
        typer.Option(
            "--direct-runtime-compatibility",
            help=(
                "Use the legacy direct Dataset Runtime explicitly. "
                "The default product path requires Factory Graph authority."
            ),
        ),
    ] = False,
    graph_runtime_config: Annotated[
        Path | None,
        typer.Option("--graph-runtime-config"),
    ] = None,
    harness_store: Annotated[
        Path | None,
        typer.Option("--harness-store"),
    ] = None,
    team_store: Annotated[
        Path | None,
        typer.Option("--team-store"),
    ] = None,
    capability_request_store: Annotated[
        Path | None,
        typer.Option("--capability-request-store"),
    ] = None,
    graph_journal: Annotated[
        Path | None,
        typer.Option("--graph-journal"),
    ] = None,
    graph_checkpoint: Annotated[
        Path | None,
        typer.Option("--graph-checkpoint"),
    ] = None,
    core_runtime_config: Annotated[
        Path | None,
        typer.Option("--core-runtime-config"),
    ] = None,
    r4_runtime_config: Annotated[
        Path | None,
        typer.Option("--r4-runtime-config"),
    ] = None,
    attachment_runtime_config: Annotated[
        Path | None,
        typer.Option("--attachment-runtime-config"),
    ] = None,
    specialist_runtime_config: Annotated[
        Path | None,
        typer.Option("--specialist-runtime-config"),
    ] = None,
    release_runtime_config: Annotated[
        Path | None,
        typer.Option("--release-runtime-config"),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest"),
    ] = None,
    raw_root: Annotated[
        Path | None,
        typer.Option("--raw-root"),
    ] = None,
    core_workspace: Annotated[
        Path | None,
        typer.Option("--core-workspace"),
    ] = None,
    attachment_workspace: Annotated[
        Path | None,
        typer.Option("--attachment-workspace"),
    ] = None,
    job_store: Annotated[
        Path | None,
        typer.Option("--job-store"),
    ] = None,
    specialist_workspace: Annotated[
        Path | None,
        typer.Option("--specialist-workspace"),
    ] = None,
    candidate_output: Annotated[
        Path | None,
        typer.Option("--candidate-output"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_dataset_run(
            factory_store=factory_store,
            private_store=private_store,
            gateway_store=gateway_store,
            request_path=request,
            policy_path=policy,
            requirement_path=requirement,
            runtime_config_path=runtime_config,
            provider_fixture_path=provider_fixture,
            user=user,
            expected_run_version=expected_run_version,
            direct_runtime_compatibility=(direct_runtime_compatibility),
            graph_runtime_config_path=graph_runtime_config,
            harness_store=harness_store,
            team_store=team_store,
            capability_request_store=capability_request_store,
            graph_journal=graph_journal,
            graph_checkpoint=graph_checkpoint,
            core_runtime_config_path=core_runtime_config,
            r4_runtime_config_path=r4_runtime_config,
            attachment_runtime_config_path=(attachment_runtime_config),
            specialist_runtime_config_path=(specialist_runtime_config),
            release_runtime_config_path=(release_runtime_config),
            manifest_path=manifest,
            raw_root=raw_root,
            core_workspace=core_workspace,
            attachment_workspace=attachment_workspace,
            job_store=job_store,
            specialist_workspace=specialist_workspace,
            candidate_output=candidate_output,
        )
    )


@agent_app.command("serve")
def agent_serve(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    registry: Annotated[Path, typer.Option("--registry")],
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1024, max=65535)] = 8174,
) -> None:
    import uvicorn

    from eval_factory.console_api import create_app

    service = _plan_review_service(
        factory_store,
        registry_path=registry,
    )
    uvicorn.run(create_app(service), host=host, port=port)


@pipeline_app.command("plan")
def pipeline_plan(
    spec: Annotated[Path, typer.Option("--spec")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(lambda: _plan(spec))


@pipeline_app.command("create")
def pipeline_create(
    job_store: Annotated[Path, typer.Option("--job-store")],
    spec: Annotated[Path, typer.Option("--spec")],
    control: Annotated[Path, typer.Option("--control")],
    idempotency_key: Annotated[
        str,
        typer.Option("--idempotency-key"),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _create(
            job_store=job_store,
            spec_path=spec,
            control_path=control,
            idempotency_key=idempotency_key,
        )
    )


@pipeline_app.command("status")
def pipeline_status(
    job_store: Annotated[Path, typer.Option("--job-store")],
    job: Annotated[str, typer.Option("--job")],
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _service(job_store).status(
            job_id=job,
            offset=offset,
            limit=limit,
        )
    )


@pipeline_app.command("resume")
def pipeline_resume(
    job_store: Annotated[Path, typer.Option("--job-store")],
    job: Annotated[str, typer.Option("--job")],
    expected_job_version: Annotated[
        int,
        typer.Option("--expected-job-version"),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option("--idempotency-key"),
    ],
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _service(job_store).resume(
            job_id=job,
            expected_job_version=expected_job_version,
            idempotency_key=idempotency_key,
            offset=offset,
            limit=limit,
        )
    )


@checkpoint_app.command("open")
def pipeline_checkpoint_open(
    job_store: Annotated[Path, typer.Option("--job-store")],
    checkpoint_store: Annotated[Path, typer.Option("--checkpoint-store")],
    context: Annotated[Path, typer.Option("--context")],
    policy: Annotated[Path, typer.Option("--policy")],
    expected_job_version: Annotated[
        int,
        typer.Option("--expected-job-version"),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option("--idempotency-key"),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _checkpoint_open(
            job_store=job_store,
            checkpoint_store=checkpoint_store,
            context_path=context,
            policy_path=policy,
            expected_job_version=expected_job_version,
            idempotency_key=idempotency_key,
        )
    )


@checkpoint_app.command("list")
def pipeline_checkpoint_list(
    job_store: Annotated[Path, typer.Option("--job-store")],
    checkpoint_store: Annotated[Path, typer.Option("--checkpoint-store")],
    job: Annotated[str, typer.Option("--job")],
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _checkpoint_service(
            job_store,
            checkpoint_store,
        ).list_interactions(
            job,
            offset=offset,
            limit=limit,
        )
    )


@checkpoint_app.command("show")
def pipeline_checkpoint_show(
    job_store: Annotated[Path, typer.Option("--job-store")],
    checkpoint_store: Annotated[Path, typer.Option("--checkpoint-store")],
    interaction: Annotated[str, typer.Option("--interaction")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _checkpoint_service(
            job_store,
            checkpoint_store,
        ).show_interaction(interaction)
    )


@checkpoint_app.command("decide")
def pipeline_checkpoint_decide(
    job_store: Annotated[Path, typer.Option("--job-store")],
    checkpoint_store: Annotated[Path, typer.Option("--checkpoint-store")],
    interaction: Annotated[str, typer.Option("--interaction")],
    submission: Annotated[Path, typer.Option("--submission")],
    authentication: Annotated[Path, typer.Option("--authentication")],
    expected_job_version: Annotated[
        int,
        typer.Option("--expected-job-version"),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _checkpoint_decide(
            job_store=job_store,
            checkpoint_store=checkpoint_store,
            interaction_id=interaction,
            submission_path=submission,
            authentication_path=authentication,
            expected_job_version=expected_job_version,
        )
    )


@checkpoint_app.command("resume")
def pipeline_checkpoint_resume(
    job_store: Annotated[Path, typer.Option("--job-store")],
    checkpoint_store: Annotated[Path, typer.Option("--checkpoint-store")],
    job: Annotated[str, typer.Option("--job")],
    expected_job_version: Annotated[
        int,
        typer.Option("--expected-job-version"),
    ],
    idempotency_key: Annotated[
        str,
        typer.Option("--idempotency-key"),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _checkpoint_service(
            job_store,
            checkpoint_store,
        ).resume_checkpoint(
            job_id=job,
            expected_job_version=expected_job_version,
            idempotency_key=idempotency_key,
            audit=JobStore(job_store).get_job_spec(job).audit,
        )
    )


@agent_plan_app.command("pending")
def agent_plan_pending(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    run: Annotated[str | None, typer.Option("--run")] = None,
    state: Annotated[PlanReviewStateV2 | None, typer.Option("--state")] = (PlanReviewStateV2.PENDING_REVIEW),
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _plan_review_service(factory_store).list_reviews(
            run_id=run,
            state=state,
            offset=offset,
            limit=limit,
        )
    )


@agent_plan_app.command("show")
def agent_plan_show(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(lambda: _plan_review_service(factory_store).show(review))


@agent_plan_app.command("export")
def agent_plan_export(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(lambda: _plan_review_service(factory_store).export(review))


@agent_plan_app.command("edit")
def agent_plan_edit(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    submission: Annotated[Path, typer.Option("--submission")],
    registry: Annotated[Path, typer.Option("--registry")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_edit(
            factory_store=factory_store,
            review=review,
            submission_path=submission,
            registry_path=registry,
        )
    )


@agent_plan_app.command("approve")
def agent_plan_approve(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    expected_plan_version: Annotated[int, typer.Option("--expected-plan-version")],
    user: Annotated[str, typer.Option("--user")],
    reason_code: Annotated[str, typer.Option("--reason-code")],
    idempotency_key: Annotated[str, typer.Option("--idempotency-key")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_decide(
            factory_store=factory_store,
            review=review,
            expected_plan_version=expected_plan_version,
            decision=PlanDecisionKindV2.APPROVE,
            user=user,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )
    )


@agent_plan_app.command("reject")
def agent_plan_reject(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    expected_plan_version: Annotated[int, typer.Option("--expected-plan-version")],
    user: Annotated[str, typer.Option("--user")],
    reason_code: Annotated[str, typer.Option("--reason-code")],
    idempotency_key: Annotated[str, typer.Option("--idempotency-key")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_decide(
            factory_store=factory_store,
            review=review,
            expected_plan_version=expected_plan_version,
            decision=PlanDecisionKindV2.REJECT,
            user=user,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )
    )


@agent_plan_app.command("defer")
def agent_plan_defer(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    expected_plan_version: Annotated[int, typer.Option("--expected-plan-version")],
    user: Annotated[str, typer.Option("--user")],
    reason_code: Annotated[str, typer.Option("--reason-code")],
    idempotency_key: Annotated[str, typer.Option("--idempotency-key")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_decide(
            factory_store=factory_store,
            review=review,
            expected_plan_version=expected_plan_version,
            decision=PlanDecisionKindV2.DEFER,
            user=user,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )
    )


@agent_plan_app.command("request-more")
def agent_plan_request_more(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    expected_plan_version: Annotated[int, typer.Option("--expected-plan-version")],
    user: Annotated[str, typer.Option("--user")],
    reason_code: Annotated[str, typer.Option("--reason-code")],
    idempotency_key: Annotated[str, typer.Option("--idempotency-key")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_decide(
            factory_store=factory_store,
            review=review,
            expected_plan_version=expected_plan_version,
            decision=PlanDecisionKindV2.REQUEST_MORE,
            user=user,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )
    )


@agent_plan_app.command("resume")
def agent_plan_resume(
    factory_store: Annotated[Path, typer.Option("--factory-store")],
    review: Annotated[str, typer.Option("--review")],
    expected_plan_version: Annotated[int, typer.Option("--expected-plan-version")],
    user: Annotated[str, typer.Option("--user")],
    idempotency_key: Annotated[str, typer.Option("--idempotency-key")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _agent_plan_resume(
            factory_store=factory_store,
            review=review,
            expected_plan_version=expected_plan_version,
            user=user,
            idempotency_key=idempotency_key,
        )
    )


@pipeline_app.command("cancel")
def pipeline_cancel(
    job_store: Annotated[Path, typer.Option("--job-store")],
    job: Annotated[str, typer.Option("--job")],
    reason_code: Annotated[str, typer.Option("--reason-code")],
    idempotency_key: Annotated[
        str,
        typer.Option("--idempotency-key"),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _cancel(
            job_store=job_store,
            job_id=job,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        )
    )


@pipeline_app.command("events")
def pipeline_events(
    job_store: Annotated[Path, typer.Option("--job-store")],
    job: Annotated[str, typer.Option("--job")],
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    idempotency_key: Annotated[
        str | None,
        typer.Option("--idempotency-key"),
    ] = None,
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _events(
            job_store=job_store,
            job_id=job,
            refresh=refresh,
            idempotency_key=idempotency_key,
            offset=offset,
            limit=limit,
        )
    )


@pipeline_app.command("metrics")
def pipeline_metrics(
    job_store: Annotated[Path, typer.Option("--job-store")],
    job: Annotated[str, typer.Option("--job")],
    item: Annotated[str | None, typer.Option("--item")] = None,
    scope: Annotated[
        MetricScopeV2 | None,
        typer.Option("--scope"),
    ] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    idempotency_key: Annotated[
        str | None,
        typer.Option("--idempotency-key"),
    ] = None,
    offset: Annotated[int, typer.Option("--offset")] = 0,
    limit: Annotated[int, typer.Option("--limit")] = 100,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _metrics(
            job_store=job_store,
            job_id=job,
            item_id=item,
            scope=scope,
            refresh=refresh,
            idempotency_key=idempotency_key,
            offset=offset,
            limit=limit,
        )
    )


@pipeline_app.command("canary-run")
def pipeline_canary_run(
    manifest: Annotated[Path, typer.Option("--manifest")],
    job_store: Annotated[Path, typer.Option("--job-store")],
    source_registry: Annotated[
        Path,
        typer.Option("--source-registry"),
    ],
    trace_store: Annotated[Path, typer.Option("--trace-store")],
    stage_store: Annotated[Path, typer.Option("--stage-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    _execute(
        lambda: _canary_run(
            manifest_path=manifest,
            job_store=job_store,
            source_registry=source_registry,
            trace_store=trace_store,
            stage_store=stage_store,
        )
    )


@pipeline_app.command("canary-regression")
def pipeline_canary_regression(
    cohort: Annotated[Path, typer.Option("--cohort")],
    raw_root: Annotated[Path, typer.Option("--raw-root")],
    template: Annotated[Path, typer.Option("--template")],
    policy: Annotated[Path, typer.Option("--policy")],
    run_root: Annotated[Path, typer.Option("--run-root")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _canary_regression(
            cohort_path=cohort,
            raw_root=raw_root,
            template_path=template,
            policy_path=policy,
            run_root=run_root,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, CanaryRegressionReportV2)
        and report.outcome is CanaryRegressionOutcomeV2.INCOMPLETE
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("real-trace-stability")
def pipeline_real_trace_stability(
    source_manifest: Annotated[Path, typer.Option("--source-manifest")],
    raw_root: Annotated[Path, typer.Option("--raw-root")],
    policy: Annotated[Path, typer.Option("--policy")],
    run_root: Annotated[Path, typer.Option("--run-root")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _real_trace_stability(
            source_manifest=source_manifest,
            raw_root=raw_root,
            policy_path=policy,
            run_root=run_root,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, RealTraceStabilityReportV2)
        and report.outcome is RealTraceStabilityOutcomeV2.INCOMPLETE
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("scheduler-load")
def pipeline_scheduler_load(
    policy: Annotated[Path, typer.Option("--policy")],
    job_store: Annotated[Path, typer.Option("--job-store")],
    run_root: Annotated[Path, typer.Option("--run-root")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _scheduler_load(
            policy_path=policy,
            job_store=job_store,
            run_root=run_root,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, SchedulerLoadReportV2)
        and report.outcome is SchedulerLoadReportOutcomeV2.INCOMPLETE
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("concurrency-experiment")
def pipeline_concurrency_experiment(
    policy: Annotated[Path, typer.Option("--policy")],
    canary_manifest: Annotated[
        Path,
        typer.Option("--canary-manifest"),
    ],
    raw_root: Annotated[Path, typer.Option("--raw-root")],
    template: Annotated[Path, typer.Option("--template")],
    run_root: Annotated[Path, typer.Option("--run-root")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _concurrency_experiment(
            policy_path=policy,
            canary_manifest=canary_manifest,
            raw_root=raw_root,
            template_path=template,
            run_root=run_root,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, ConcurrencyExperimentReportV2)
        and report.outcome is not ConcurrencyExperimentOutcomeV2.NON_MODEL_RECOMMENDED
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("label-quality-evaluation")
def pipeline_label_quality_evaluation(
    policy: Annotated[Path, typer.Option("--policy")],
    r8_01_evidence: Annotated[Path, typer.Option("--r8-01-evidence")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    evaluation_request: Annotated[
        Path | None,
        typer.Option("--evaluation-request"),
    ] = None,
    r8_01_database: Annotated[
        Path | None,
        typer.Option("--r8-01-database"),
    ] = None,
    r8_01_material_store: Annotated[
        Path | None,
        typer.Option("--r8-01-material-store"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _label_quality_evaluation(
            policy_path=policy,
            evidence_path=r8_01_evidence,
            evaluation_request_path=evaluation_request,
            r8_01_database=r8_01_database,
            r8_01_material_store=r8_01_material_store,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, LabelQualityEvaluationReportV2)
        and report.outcome is not LabelQualityOutcomeV2.PASSED
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("readiness-review")
def pipeline_readiness_review(
    policy: Annotated[Path, typer.Option("--policy")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    repository_evidence: Annotated[
        Path | None,
        typer.Option("--repository-evidence"),
    ] = None,
    review_request: Annotated[
        Path | None,
        typer.Option("--review-request"),
    ] = None,
    approval_registry: Annotated[
        Path | None,
        typer.Option("--approval-registry"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    report = _execute_result(
        lambda: _production_readiness_review(
            policy_path=policy,
            repository_evidence_path=repository_evidence,
            review_request_path=review_request,
            approval_registry_path=approval_registry,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(report.model_dump_json(indent=2))
    if (
        isinstance(report, ProductionReadinessReviewReportV2)
        and report.outcome is not ProductionReadinessReviewOutcomeV2.APPROVED
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("readiness-attestation")
def pipeline_readiness_attestation(
    policy: Annotated[Path, typer.Option("--policy")],
    private_store: Annotated[Path, typer.Option("--private-store")],
    report_store: Annotated[Path, typer.Option("--report-store")],
    repository_evidence: Annotated[
        Path | None,
        typer.Option("--repository-evidence"),
    ] = None,
    attestation_request: Annotated[
        Path | None,
        typer.Option("--attestation-request"),
    ] = None,
    issuer_registry: Annotated[
        Path | None,
        typer.Option("--issuer-registry"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    result = _execute_result(
        lambda: _production_readiness_attestation(
            policy_path=policy,
            repository_evidence_path=repository_evidence,
            attestation_request_path=attestation_request,
            issuer_registry_path=issuer_registry,
            private_store=private_store,
            report_store=report_store,
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if (
        isinstance(result, ProductionReadinessAttestationResultV2)
        and result.outcome is not ProductionReadinessAttestationOutcomeV2.ISSUED
    ):
        raise typer.Exit(code=3)


@pipeline_app.command("production-release")
def pipeline_production_release(
    policy: Annotated[Path, typer.Option("--policy")],
    repository_evidence: Annotated[
        Path,
        typer.Option("--repository-evidence"),
    ],
    report_store: Annotated[Path, typer.Option("--report-store")],
    json_output: Annotated[bool, typer.Option("--json")] = True,
) -> None:
    del json_output
    result = _execute_result(
        lambda: _production_release(
            policy_path=policy,
            repository_evidence_path=repository_evidence,
            report_store=report_store,
        )
    )
    typer.echo(result.model_dump_json(indent=2))
    if (
        isinstance(result, ProductionReleaseResultV2)
        and result.outcome is not ProductionReleaseOutcomeV2.PUBLISHED
    ):
        raise typer.Exit(code=3)


def _agent_dataset_run(
    *,
    factory_store: Path,
    private_store: Path,
    gateway_store: Path,
    request_path: Path,
    policy_path: Path,
    requirement_path: Path,
    runtime_config_path: Path,
    provider_fixture_path: Path,
    user: str,
    expected_run_version: int | None,
    direct_runtime_compatibility: bool,
    graph_runtime_config_path: Path | None,
    harness_store: Path | None,
    team_store: Path | None,
    capability_request_store: Path | None,
    graph_journal: Path | None,
    graph_checkpoint: Path | None,
    core_runtime_config_path: Path | None,
    r4_runtime_config_path: Path | None,
    attachment_runtime_config_path: Path | None,
    specialist_runtime_config_path: Path | None,
    release_runtime_config_path: Path | None,
    manifest_path: Path | None,
    raw_root: Path | None,
    core_workspace: Path | None,
    attachment_workspace: Path | None,
    job_store: Path | None,
    specialist_workspace: Path | None,
    candidate_output: Path | None,
) -> BaseModel:
    from eval_factory.agent_system.dataset_release_fixture import (
        FactoryDatasetReleaseRuntimeConfigV1,
    )
    from eval_factory.agent_system.dataset_runtime import (
        FactoryDatasetCoreInput,
    )
    from eval_factory.agent_system.dataset_runtime_fixture import (
        FactoryDatasetAttachmentRuntimeConfigV1,
        FactoryDatasetCoreRuntimeConfigV1,
        FactoryDatasetPlanningFixtureV1,
        FactoryDatasetR4RuntimeConfigV1,
        FactoryDatasetRuntimeConfigV1,
        build_fixture_dataset_components,
    )
    from eval_factory.agent_system.dataset_specialist_fixture import (
        FactoryDatasetSpecialistRuntimeConfigV1,
    )
    from eval_factory.contracts.dataset_runtime_v2 import (
        FactoryDatasetRunRequestV2,
    )
    from eval_factory.harness.session_store import HarnessSessionStore
    from eval_factory.packs.generic_agent_trace.product_builder import (
        GenericAgentFirstPartyGraphBuilder,
        GenericAgentGraphProductPaths,
        GenericAgentGraphRuntimeConfigV1,
    )

    request = _load_model(
        request_path,
        FactoryDatasetRunRequestV2,
    )
    policy = _load_model(policy_path, FactoryRunPolicyV2)
    requirement = _load_model(
        requirement_path,
        EvaluationRequirementSpecV2,
    )
    runtime_config = _load_model(
        runtime_config_path,
        FactoryDatasetRuntimeConfigV1,
    )
    provider_fixture = _load_model(
        provider_fixture_path,
        FactoryDatasetPlanningFixtureV1,
    )
    core_values = (
        core_runtime_config_path,
        manifest_path,
        raw_root,
        core_workspace,
    )
    if any(value is not None for value in core_values) and not all(
        value is not None for value in core_values
    ):
        raise ValueError("core runtime config, manifest, raw root, and workspace must appear together")
    if r4_runtime_config_path is not None and not all(value is not None for value in core_values):
        raise ValueError("R4 runtime config requires complete core admission")
    attachment_values = (
        attachment_runtime_config_path,
        attachment_workspace,
    )
    if any(value is not None for value in attachment_values) and not all(
        value is not None for value in attachment_values
    ):
        raise ValueError("attachment runtime config and workspace must appear together")
    if attachment_runtime_config_path is not None and r4_runtime_config_path is None:
        raise ValueError("attachment runtime config requires R4 runtime config")
    specialist_values = (
        specialist_runtime_config_path,
        job_store,
        specialist_workspace,
    )
    if any(value is not None for value in specialist_values) and not all(
        value is not None for value in specialist_values
    ):
        raise ValueError("specialist runtime config, JobStore, and workspace must appear together")
    if specialist_runtime_config_path is not None and attachment_runtime_config_path is None:
        raise ValueError("specialist runtime config requires attachment runtime config")
    release_values = (
        release_runtime_config_path,
        candidate_output,
    )
    if any(value is not None for value in release_values) and not all(
        value is not None for value in release_values
    ):
        raise ValueError("release runtime config and candidate output must appear together")
    if release_runtime_config_path is not None and specialist_runtime_config_path is None:
        raise ValueError("release runtime config requires specialist runtime config")
    core_config = (
        _load_model(
            core_runtime_config_path,
            FactoryDatasetCoreRuntimeConfigV1,
        )
        if core_runtime_config_path is not None
        else None
    )
    r4_config = (
        _load_model(
            r4_runtime_config_path,
            FactoryDatasetR4RuntimeConfigV1,
        )
        if r4_runtime_config_path is not None
        else None
    )
    attachment_config = (
        _load_model(
            attachment_runtime_config_path,
            FactoryDatasetAttachmentRuntimeConfigV1,
        )
        if attachment_runtime_config_path is not None
        else None
    )
    specialist_config = (
        _load_model(
            specialist_runtime_config_path,
            FactoryDatasetSpecialistRuntimeConfigV1,
        )
        if specialist_runtime_config_path is not None
        else None
    )
    release_config = (
        _load_model(
            release_runtime_config_path,
            FactoryDatasetReleaseRuntimeConfigV1,
        )
        if release_runtime_config_path is not None
        else None
    )
    if release_config is not None and release_config.output_target_ref != request.output_target_ref:
        raise ValueError("release output target differs from run request")
    core_input = (
        FactoryDatasetCoreInput(
            manifest_path=manifest_path,
            raw_root=raw_root,
            expected_manifest_sha256=(request.manifest_ref.object_sha256),
        )
        if manifest_path is not None and raw_root is not None
        else None
    )
    graph_values = (
        graph_runtime_config_path,
        harness_store,
        team_store,
        capability_request_store,
        graph_journal,
        graph_checkpoint,
    )
    if not direct_runtime_compatibility:
        if any(value is None for value in graph_values):
            raise ValueError(
                "default Factory Graph requires graph config and all authority stores",
            )
        if (
            core_input is None
            or core_config is None
            or r4_config is None
            or attachment_config is None
            or specialist_config is None
            or release_config is None
            or candidate_output is None
        ):
            raise ValueError(
                "Pack 1.2.0 Graph requires the complete owner runtime",
            )
    components = build_fixture_dataset_components(
        factory_store_path=factory_store,
        private_store_path=private_store,
        gateway_store_path=gateway_store,
        config=runtime_config,
        fixture=provider_fixture,
        requested_by=user,
        core_config=core_config,
        r4_config=r4_config,
        core_workspace_path=core_workspace,
        attachment_config=attachment_config,
        attachment_workspace_path=attachment_workspace,
        specialist_config=specialist_config,
        job_store_path=job_store,
        specialist_workspace_path=specialist_workspace,
        release_config=release_config,
        candidate_output_path=candidate_output,
    )
    runtime = components.runtime
    if expected_run_version is not None:
        from eval_factory.agent_system.store import (
            FactoryControlConcurrencyError,
            FactoryControlNotFoundError,
        )

        try:
            current_run = runtime.store.get_run(
                request.dataset_run_id,
            )
        except FactoryControlNotFoundError:
            raise FactoryControlConcurrencyError("expected Factory run does not exist") from None
        if current_run.run_version != expected_run_version:
            raise FactoryControlConcurrencyError("expected Factory run version is stale")
    if direct_runtime_compatibility:
        return asyncio.run(
            runtime.advance(
                request=request,
                policy=policy,
                requirement=requirement,
                core_input=core_input,
                audit=request.audit,
            )
        )
    assert core_input is not None
    assert graph_runtime_config_path is not None
    assert harness_store is not None
    assert team_store is not None
    assert capability_request_store is not None
    assert graph_journal is not None
    assert graph_checkpoint is not None
    assert candidate_output is not None
    product = GenericAgentFirstPartyGraphBuilder.build(
        components=components,
        session_store=HarnessSessionStore(harness_store),
        config=_load_model(
            graph_runtime_config_path,
            GenericAgentGraphRuntimeConfigV1,
        ),
        paths=GenericAgentGraphProductPaths(
            team_store=team_store,
            request_store=capability_request_store,
            graph_journal=graph_journal,
            graph_checkpoint=graph_checkpoint,
        ),
        core_input=core_input,
        request=request,
        policy=policy,
        requirement=requirement,
        candidate_output_root=candidate_output,
        audit=request.audit,
    )
    return asyncio.run(product.advance())


def _plan(spec_path: Path) -> BaseModel:
    spec = _load_model(spec_path, DatasetJobSpecV2)
    return PipelineLifecycleService.plan(
        job_spec=spec,
        audit=spec.audit,
    )


def _create(
    *,
    job_store: Path,
    spec_path: Path,
    control_path: Path,
    idempotency_key: str,
) -> BaseModel:
    spec = _load_model(spec_path, DatasetJobSpecV2)
    control = _load_model(
        control_path,
        PipelineControlConfigV2,
    )
    return _service(job_store).create(
        job_spec=spec,
        control=control,
        audit=spec.audit,
        idempotency_key=idempotency_key,
    )


def _checkpoint_open(
    *,
    job_store: Path,
    checkpoint_store: Path,
    context_path: Path,
    policy_path: Path,
    expected_job_version: int,
    idempotency_key: str,
) -> BaseModel:
    context = _load_model(
        context_path,
        UserCheckpointSourceContextV2,
    )
    policy = _load_model(
        policy_path,
        UserCheckpointInteractionPolicyV2,
    )
    return _checkpoint_service(
        job_store,
        checkpoint_store,
    ).open_checkpoint(
        source_context=context,
        policy=policy,
        expected_job_version=expected_job_version,
        idempotency_key=idempotency_key,
        audit=context.job_spec.audit,
    )


def _checkpoint_decide(
    *,
    job_store: Path,
    checkpoint_store: Path,
    interaction_id: str,
    submission_path: Path,
    authentication_path: Path,
    expected_job_version: int,
) -> BaseModel:
    submission = _load_model(
        submission_path,
        UserCheckpointDecisionSubmissionV2,
    )
    authentication = _load_model(
        authentication_path,
        TrustedAuthenticatedUserContextV2,
    )
    service = _checkpoint_service(
        job_store,
        checkpoint_store,
    )
    interaction = service.get_interaction(interaction_id)
    return service.decide(
        interaction_id=interaction_id,
        submission=submission,
        authentication=authentication.to_domain(),
        expected_job_version=expected_job_version,
        audit=JobStore(job_store).get_job_spec(interaction.job_id).audit,
    )


def _checkpoint_service(
    job_store: Path,
    checkpoint_store: Path,
) -> UserCheckpointInteractionService:
    return UserCheckpointInteractionService(
        JobStore(job_store),
        UserCheckpointMaterialStore(
            checkpoint_store,
            max_source_context_bytes=_MAX_CHECKPOINT_SOURCE_BYTES,
            max_presentation_bytes=_MAX_CHECKPOINT_PRESENTATION_BYTES,
        ),
    )


def _canary_run(
    *,
    manifest_path: Path,
    job_store: Path,
    source_registry: Path,
    trace_store: Path,
    stage_store: Path,
) -> BaseModel:
    manifest = _load_model(
        manifest_path,
        R6CanaryExecutionManifestV2,
    )
    from eval_factory.orchestration.canary_driver import (
        CanaryPipelineDriver,
        create_canary_stage_store,
    )

    return asyncio.run(
        CanaryPipelineDriver(
            job_store=JobStore(job_store),
            source_registry=TraceSourceRegistry(source_registry),
            trace_store=TraceIndexStore(trace_store),
            stage_store=create_canary_stage_store(stage_store),
        ).run(manifest)
    )


def _canary_regression(
    *,
    cohort_path: Path,
    raw_root: Path,
    template_path: Path,
    policy_path: Path,
    run_root: Path,
    report_store: Path,
) -> CanaryRegressionReportV2:
    if run_root.expanduser().resolve() == report_store.expanduser().resolve():
        raise ValueError("canary regression run and report roots must be distinct")
    from eval_factory.readiness import (
        CanaryRegressionReportStore,
        CanaryRegressionRunner,
        CanaryRegressionTemplateV1,
        FrozenCanaryCohortLoader,
    )

    policy = _load_model(policy_path, CanaryRegressionPolicyV2)
    template = _load_model(template_path, CanaryRegressionTemplateV1)
    cohort = FrozenCanaryCohortLoader().load(
        cohort_path,
        policy=policy,
    )
    report = asyncio.run(
        CanaryRegressionRunner().run(
            cohort=cohort,
            template=template,
            policy=policy,
            raw_root=raw_root,
            run_root=run_root,
            audit=policy.audit,
        )
    )
    CanaryRegressionReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
        max_case_refs=policy.max_case_refs,
    ).put(report)
    return report


def _real_trace_stability(
    *,
    source_manifest: Path,
    raw_root: Path,
    policy_path: Path,
    run_root: Path,
    private_store: Path,
    report_store: Path,
) -> RealTraceStabilityReportV2:
    roots = {
        run_root.expanduser().resolve(),
        private_store.expanduser().resolve(),
        report_store.expanduser().resolve(),
    }
    if len(roots) != 3:
        raise ValueError("real-trace stability run, private, and report roots must be distinct")
    from eval_factory.readiness import (
        RealTraceStabilityBuilder,
        RealTraceStabilityMaterialStore,
        RealTraceStabilityReportStore,
        RealTraceStabilityRunner,
    )

    policy = _load_model(policy_path, RealTraceStabilityPolicyV2)
    builder = RealTraceStabilityBuilder()
    compilation = builder.compile_inventory(
        source_manifest=source_manifest,
        raw_root=raw_root,
        policy=policy,
        audit=policy.audit,
    )
    material = RealTraceStabilityMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=policy.max_sources,
    )
    material.put_inventory(compilation.inventory)
    result = RealTraceStabilityRunner(builder=builder).run(
        compilation=compilation,
        policy=policy,
        run_root=run_root,
        audit=policy.audit,
    )
    for case in result.case_results:
        material.put_case_result(case)
    material.put_result_set(result.result_set)
    RealTraceStabilityReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
        max_cases=policy.max_sources,
    ).put(result.report)
    return result.report


def _scheduler_load(
    *,
    policy_path: Path,
    job_store: Path,
    run_root: Path,
    private_store: Path,
    report_store: Path,
) -> SchedulerLoadReportV2:
    roots = (
        policy_path.expanduser().resolve(),
        job_store.expanduser().resolve(),
        run_root.expanduser().resolve(),
        private_store.expanduser().resolve(),
        report_store.expanduser().resolve(),
    )
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise ValueError("scheduler load policy, JobStore, run, private, and report paths must not overlap")
    from eval_factory.readiness import (
        SchedulerLoadBuilder,
        SchedulerLoadMaterialStore,
        SchedulerLoadReportStore,
        SchedulerLoadRunner,
    )

    policy = _load_model(policy_path, SchedulerLoadPolicyV2)
    builder = SchedulerLoadBuilder()
    workload = builder.compile_workload(
        policy=policy,
        audit=policy.audit,
    )
    material = SchedulerLoadMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=policy.required_job_count,
    )
    material.put_workload(workload)
    result = SchedulerLoadRunner(builder=builder).run(
        workload=workload,
        policy=policy,
        job_store_path=job_store,
        run_root=run_root,
        audit=policy.audit,
    )
    material.put_run_ledger(result.run_ledger)
    for fault in result.fault_observations:
        material.put_fault_observation(fault)
    for case in result.case_results:
        material.put_case_result(case)
    material.put_result_set(result.result_set)
    SchedulerLoadReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
        max_cases=policy.required_job_count,
    ).put(result.report)
    return result.report


def _concurrency_experiment(
    *,
    policy_path: Path,
    canary_manifest: Path,
    raw_root: Path,
    template_path: Path,
    run_root: Path,
    private_store: Path,
    report_store: Path,
) -> ConcurrencyExperimentReportV2:
    paths = (
        policy_path.expanduser().resolve(),
        canary_manifest.expanduser().resolve(),
        raw_root.expanduser().resolve(),
        template_path.expanduser().resolve(),
        run_root.expanduser().resolve(),
        private_store.expanduser().resolve(),
        report_store.expanduser().resolve(),
    )
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(paths)
        for right in paths[index + 1 :]
    ):
        raise ValueError("concurrency experiment inputs and stores must not overlap")
    from eval_factory.readiness import (
        CanaryRegressionTemplateV1,
        ConcurrencyExperimentBuilder,
        ConcurrencyExperimentMaterialStore,
        ConcurrencyExperimentReportStore,
        ConcurrencyExperimentRunner,
    )

    policy = _load_model(policy_path, ConcurrencyExperimentPolicyV2)
    template = _load_model(template_path, CanaryRegressionTemplateV1)
    builder = ConcurrencyExperimentBuilder()
    cohort = builder.load_cohort(
        canary_manifest,
        template=template,
    )
    host = builder.observe_host(audit=policy.audit)
    workload = builder.compile_workload(
        policy=policy,
        host_summary=host,
        cohort=cohort,
        template=template,
        audit=policy.audit,
    )
    material = ConcurrencyExperimentMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=1_000,
    )
    reports = ConcurrencyExperimentReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
    )
    material.put_host(host)
    material.put_workload(workload)
    accepted = _load_accepted_concurrency_report(
        run_root=run_root,
        policy=policy,
        workload_ref=workload.to_ref(),
        material_store=material,
        report_store=reports,
    )
    if accepted is not None:
        return accepted
    preparation_root = run_root / "preparation"
    prepared_trials = {
        trial.trial_id: builder.prepare_trial(
            workload=workload,
            trial=trial,
            cohort=cohort,
            template=template,
            raw_root=raw_root,
            preparation_root=preparation_root,
            audit=policy.audit,
        )
        for trial in workload.trials
    }

    def prepare_recovery(
        worker_count: int,
    ) -> tuple[PreparedConcurrencyRecoveryCase, ...]:
        return builder.prepare_recovery_cases(
            workload=workload,
            worker_count=worker_count,
            cohort=cohort,
            template=template,
            raw_root=raw_root,
            preparation_root=preparation_root,
            audit=policy.audit,
        )

    result = ConcurrencyExperimentRunner(
        material_store=material,
    ).run(
        workload=workload,
        policy=policy,
        host_summary=host,
        prepared_trials=prepared_trials,
        recovery_cases=(),
        run_root=run_root,
        audit=policy.audit,
        recovery_case_factory=prepare_recovery,
    )
    _admit_concurrency_report(run_root, result.report)
    reports.put(result.report)
    return result.report


def _label_quality_evaluation(
    *,
    policy_path: Path,
    evidence_path: Path,
    evaluation_request_path: Path | None,
    r8_01_database: Path | None,
    r8_01_material_store: Path | None,
    private_store: Path,
    report_store: Path,
) -> LabelQualityEvaluationReportV2:
    _require_pipeline_file(policy_path, "label quality policy")
    _require_pipeline_file(evidence_path, "R8-01 evidence")
    if evaluation_request_path is not None:
        _require_pipeline_file(
            evaluation_request_path,
            "label quality evaluation request",
        )
    if r8_01_database is not None:
        _require_pipeline_file(r8_01_database, "R8-01 database")
    if r8_01_material_store is not None:
        _require_pipeline_directory(
            r8_01_material_store,
            "R8-01 material store",
            require_existing=True,
        )
    _require_pipeline_directory(private_store, "label quality private store")
    _require_pipeline_directory(report_store, "label quality report store")
    paths = (
        policy_path,
        evidence_path,
        private_store,
        report_store,
        *((evaluation_request_path,) if evaluation_request_path is not None else ()),
        *((r8_01_database,) if r8_01_database is not None else ()),
        *((r8_01_material_store,) if r8_01_material_store is not None else ()),
    )
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ValueError("label quality inputs and stores must not overlap")
    from eval_factory.contracts.statistics_v2 import (
        IndependentLabelTestSetFreezeResultV2,
        LabelTestSetFreezeOutcomeV2,
    )
    from eval_factory.statistics import (
        IndependentLabelMaterialStore,
        IndependentLabelTestSetPersistenceService,
        LabelQualityAcceptedRunStore,
        LabelQualityEvaluationBuilder,
        LabelQualityEvaluator,
        LabelQualityFrozenEvaluationRequestV1,
        LabelQualityMaterialStore,
        LabelQualityReportStore,
        LabelQualityStoreConflictError,
    )

    policy_bytes = _read_pipeline_input(policy_path)
    evidence_bytes = _read_pipeline_input(evidence_path)
    policy = LabelQualityEvaluationPolicyV2.model_validate_json(policy_bytes)
    evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
    frozen: IndependentLabelTestSetFreezeResultV2 | None = None
    if evidence_sha256 == LABEL_QUALITY_REPOSITORY_PENDING_SHA256:
        if policy.evidence_class is not LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ValueError("repository pending evidence class differs from policy")
        if any(
            value is not None
            for value in (
                evaluation_request_path,
                r8_01_database,
                r8_01_material_store,
            )
        ):
            raise ValueError("repository-pending evaluation forbids frozen-only inputs")
        request_bytes = b""
    else:
        frozen = IndependentLabelTestSetFreezeResultV2.model_validate_json(evidence_bytes)
        if policy.evidence_class is LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ValueError("R8-01 freeze evidence class differs from policy")
        if frozen.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.FROZEN:
            if evaluation_request_path is None or r8_01_database is None or r8_01_material_store is None:
                raise ValueError("frozen evaluation requires all private R8-01 inputs")
            request_bytes = _read_pipeline_input(evaluation_request_path)
        else:
            if any(
                value is not None
                for value in (
                    evaluation_request_path,
                    r8_01_database,
                    r8_01_material_store,
                )
            ):
                raise ValueError("pending evaluation forbids frozen-only inputs")
            request_bytes = b""
    request_sha256 = hashlib.sha256(policy_bytes + b"\0" + evidence_bytes + b"\0" + request_bytes).hexdigest()
    acceptance_digest = hashlib.sha256(f"{policy.policy_sha256}:{evidence_sha256}".encode()).hexdigest()
    acceptance_key = f"label-quality-acceptance://sha256/{acceptance_digest}"
    material = LabelQualityMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=policy.max_members,
    )
    reports = LabelQualityReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
    )
    accepted_store = LabelQualityAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    existing = material.find_acceptance(acceptance_key)
    if existing is not None:
        if existing.request_sha256 != request_sha256:
            raise LabelQualityStoreConflictError("label quality accepted request changed")
        return accepted_store.get_accepted_report(acceptance_key)

    builder = LabelQualityEvaluationBuilder()
    if evidence_sha256 == LABEL_QUALITY_REPOSITORY_PENDING_SHA256:
        compilation = builder.compile_repository_pending(
            payload=evidence_bytes,
            policy=policy,
        )
    elif (
        frozen is not None
        and frozen.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING
    ):
        compilation = builder.compile_pending_result(
            expected_result=frozen,
            policy=policy,
        )
    else:
        if (
            frozen is None
            or evaluation_request_path is None
            or r8_01_database is None
            or r8_01_material_store is None
        ):
            raise ValueError("frozen evaluation inputs are incomplete")
        request = LabelQualityFrozenEvaluationRequestV1.model_validate_json(request_bytes)
        r8_material = IndependentLabelMaterialStore(
            r8_01_material_store,
            max_candidate_pool_bytes=policy.max_private_bytes,
            max_partition_bytes=policy.max_private_bytes,
            max_test_set_bytes=policy.max_private_bytes,
            max_members=policy.max_members,
        )
        persistence = IndependentLabelTestSetPersistenceService(
            r8_01_database,
            material_store=r8_material,
        )
        compilation = builder.compile_frozen(
            persistence=persistence,
            expected_result=frozen,
            policy=policy,
            request=request,
            audit=policy.audit,
        )
    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=policy.audit,
    )
    accepted_store.persist(
        acceptance_key=acceptance_key,
        request_sha256=request_sha256,
        compilation=compilation,
        report=report,
        audit=policy.audit,
    )
    return report


def _production_release(
    *,
    policy_path: Path,
    repository_evidence_path: Path,
    report_store: Path,
) -> ProductionReleaseResultV2:
    _require_pipeline_file(policy_path, "production release policy")
    _require_pipeline_file(
        repository_evidence_path,
        "repository production release evidence",
    )
    _require_pipeline_directory(
        report_store,
        "production release report store",
    )
    resolved = tuple(
        path.expanduser().resolve()
        for path in (
            policy_path,
            repository_evidence_path,
            report_store,
        )
    )
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ValueError("production release inputs and store must not overlap")
    policy_bytes = _read_pipeline_input(policy_path)
    evidence_bytes = _read_pipeline_input(repository_evidence_path)
    policy = ProductionReleasePolicyV2.model_validate_json(policy_bytes)
    if policy.evidence_class is not ProductionReleaseEvidenceClassV2.REPOSITORY_PENDING_ONLY:
        raise ValueError("production release CLI requires repository-pending policy")
    repository_evidence_ref = policy.repository_evidence_ref
    if repository_evidence_ref is None:
        raise ValueError("production release repository policy requires evidence")
    from eval_factory.dataset.production_release import ProductionReleaseCompiler
    from eval_factory.dataset.production_release_store import (
        ProductionReleaseReportStore,
    )

    compilation = ProductionReleaseCompiler().compile_repository_blocked(
        payload=evidence_bytes,
        policy=policy,
    )
    request_sha256 = hashlib.sha256(policy_bytes + b"\0" + evidence_bytes).hexdigest()
    acceptance_digest = hashlib.sha256(
        f"{policy.policy_sha256}:{repository_evidence_ref.object_sha256}".encode()
    ).hexdigest()
    acceptance_key = f"production-release-acceptance://sha256/{acceptance_digest}"
    store = ProductionReleaseReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
    )
    store.persist_repository(
        acceptance_key=acceptance_key,
        request_sha256=request_sha256,
        payload=evidence_bytes,
        compilation=compilation,
    )
    return store.get_accepted_result(acceptance_key)


def _production_readiness_attestation(
    *,
    policy_path: Path,
    repository_evidence_path: Path | None,
    attestation_request_path: Path | None,
    issuer_registry_path: Path | None,
    private_store: Path,
    report_store: Path,
) -> ProductionReadinessAttestationResultV2:
    _require_pipeline_file(policy_path, "production attestation policy")
    pending_mode = repository_evidence_path is not None
    full_mode = attestation_request_path is not None or issuer_registry_path is not None
    if pending_mode == full_mode:
        raise ValueError("production attestation requires exactly one pending or full input mode")
    if pending_mode:
        assert repository_evidence_path is not None
        _require_pipeline_file(
            repository_evidence_path,
            "repository attestation evidence",
        )
    else:
        if attestation_request_path is None or issuer_registry_path is None:
            raise ValueError("full production attestation requires request and issuer registry")
        _require_pipeline_file(
            attestation_request_path,
            "production attestation request",
        )
        _require_pipeline_file(
            issuer_registry_path,
            "attestation issuer registry",
        )
    _require_pipeline_directory(
        private_store,
        "production attestation private store",
    )
    _require_pipeline_directory(
        report_store,
        "production attestation report store",
    )
    paths = (
        policy_path,
        private_store,
        report_store,
        *((repository_evidence_path,) if repository_evidence_path is not None else ()),
        *((attestation_request_path,) if attestation_request_path is not None else ()),
        *((issuer_registry_path,) if issuer_registry_path is not None else ()),
    )
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ValueError("production attestation inputs and stores must not overlap")
    from eval_factory.readiness import (
        REPOSITORY_ATTESTATION_SERIES_ID,
        ProductionAttestationAcceptedRunStore,
        ProductionAttestationBuilder,
        ProductionAttestationEvaluator,
        ProductionAttestationMaterialStore,
        ProductionAttestationReportStore,
        ProductionAttestationStoreConflictError,
        ProductionReadinessAttestationRequestV1,
        TrustedAttestationIssuerRegistryV1,
    )

    policy_bytes = _read_pipeline_input(policy_path)
    policy = ProductionReadinessAttestationPolicyV2.model_validate_json(policy_bytes)
    repository_bytes = (
        _read_pipeline_input(repository_evidence_path) if repository_evidence_path is not None else b""
    )
    request_bytes = (
        _read_pipeline_input(attestation_request_path) if attestation_request_path is not None else b""
    )
    registry_bytes = _read_pipeline_input(issuer_registry_path) if issuer_registry_path is not None else b""
    if pending_mode:
        if policy.evidence_class is not (ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY):
            raise ValueError("repository attestation evidence class differs from policy")
        series_id = REPOSITORY_ATTESTATION_SERIES_ID
        version = 1
        request = None
        registry = None
    else:
        if policy.evidence_class is not (ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED):
            raise ValueError("full attestation evidence class differs from policy")
        request = ProductionReadinessAttestationRequestV1.model_validate_json(request_bytes)
        registry = TrustedAttestationIssuerRegistryV1.model_validate_json(registry_bytes)
        series_id = request.attestation_series_id
        version = request.attestation_version
    request_sha256 = hashlib.sha256(
        policy_bytes + b"\0" + repository_bytes + b"\0" + request_bytes + b"\0" + registry_bytes
    ).hexdigest()
    acceptance_key = _production_attestation_acceptance_key(
        policy.policy_sha256,
        series_id,
        version,
    )
    previous_acceptance_key = (
        _production_attestation_acceptance_key(
            policy.policy_sha256,
            series_id,
            version - 1,
        )
        if version > 1
        else None
    )
    material = ProductionAttestationMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=max(16, policy.minimum_issuer_count + 16),
    )
    reports = ProductionAttestationReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
    )
    accepted = ProductionAttestationAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    existing = material.find_acceptance(acceptance_key)
    if existing is not None:
        if existing.request_sha256 != request_sha256:
            raise ProductionAttestationStoreConflictError("accepted production attestation request changed")
        return accepted.replay_accepted_result(
            acceptance_key,
            previous_acceptance_key=previous_acceptance_key,
        )
    builder = ProductionAttestationBuilder()
    if pending_mode:
        compilation = builder.compile_repository_pending(
            payload=repository_bytes,
            policy=policy,
        )
    else:
        if request is None or registry is None:
            raise ValueError("full production attestation inputs are incomplete")
        compilation = builder.compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )
    result, frozen = ProductionAttestationEvaluator().evaluate(
        compilation=compilation,
        audit=policy.audit,
    )
    accepted.persist(
        acceptance_key=acceptance_key,
        request_sha256=request_sha256,
        compilation=compilation,
        result=result,
        frozen=frozen,
        audit=policy.audit,
        previous_acceptance_key=previous_acceptance_key,
    )
    return result


def _production_attestation_acceptance_key(
    policy_sha256: str,
    attestation_series_id: str,
    attestation_version: int,
) -> str:
    digest = hashlib.sha256(
        f"{policy_sha256}:{attestation_series_id}:{attestation_version}".encode()
    ).hexdigest()
    return f"production-attestation-acceptance://sha256/{digest}"


def _production_readiness_review(
    *,
    policy_path: Path,
    repository_evidence_path: Path | None,
    review_request_path: Path | None,
    approval_registry_path: Path | None,
    private_store: Path,
    report_store: Path,
) -> ProductionReadinessReviewReportV2:
    _require_pipeline_file(policy_path, "production readiness review policy")
    pending_mode = repository_evidence_path is not None
    full_mode = review_request_path is not None or approval_registry_path is not None
    if pending_mode == full_mode:
        raise ValueError("readiness review requires exactly one pending or full input mode")
    if pending_mode:
        if repository_evidence_path is None:
            raise ValueError("repository readiness evidence is missing")
        _require_pipeline_file(
            repository_evidence_path,
            "repository readiness evidence",
        )
    else:
        if review_request_path is None or approval_registry_path is None:
            raise ValueError("full readiness review requires request and trusted registry")
        _require_pipeline_file(
            review_request_path,
            "production readiness review request",
        )
        _require_pipeline_file(
            approval_registry_path,
            "production approval registry",
        )
    _require_pipeline_directory(
        private_store,
        "production readiness private store",
    )
    _require_pipeline_directory(
        report_store,
        "production readiness report store",
    )
    paths = (
        policy_path,
        private_store,
        report_store,
        *((repository_evidence_path,) if repository_evidence_path is not None else ()),
        *((review_request_path,) if review_request_path is not None else ()),
        *((approval_registry_path,) if approval_registry_path is not None else ()),
    )
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if any(
        left == right or left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ValueError("production readiness inputs and stores must not overlap")
    from eval_factory.readiness import (
        REPOSITORY_REVIEW_SERIES_ID,
        ProductionReadinessAcceptedRunStore,
        ProductionReadinessMaterialStore,
        ProductionReadinessReportStore,
        ProductionReadinessReviewBuilder,
        ProductionReadinessReviewer,
        ProductionReadinessReviewRequestV1,
        ProductionReadinessStoreConflictError,
        TrustedProductionApprovalRegistryV1,
    )

    policy_bytes = _read_pipeline_input(policy_path)
    policy = ProductionReadinessReviewPolicyV2.model_validate_json(policy_bytes)
    repository_bytes = (
        _read_pipeline_input(repository_evidence_path) if repository_evidence_path is not None else b""
    )
    request_bytes = _read_pipeline_input(review_request_path) if review_request_path is not None else b""
    registry_bytes = (
        _read_pipeline_input(approval_registry_path) if approval_registry_path is not None else b""
    )
    if pending_mode:
        if policy.evidence_class is not ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ValueError("repository readiness evidence class differs from policy")
        review_series_id = REPOSITORY_REVIEW_SERIES_ID
        review_version = 1
        request = None
        registry = None
    else:
        if policy.evidence_class is ProductionReadinessReviewEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise ValueError("full readiness evidence class differs from policy")
        request = ProductionReadinessReviewRequestV1.model_validate_json(request_bytes)
        registry = TrustedProductionApprovalRegistryV1.model_validate_json(registry_bytes)
        review_series_id = request.review_series_id
        review_version = request.review_version
    request_sha256 = hashlib.sha256(
        policy_bytes + b"\0" + repository_bytes + b"\0" + request_bytes + b"\0" + registry_bytes
    ).hexdigest()
    acceptance_key = _production_readiness_acceptance_key(
        policy.policy_sha256,
        review_series_id,
        review_version,
    )
    previous_acceptance_key = (
        _production_readiness_acceptance_key(
            policy.policy_sha256,
            review_series_id,
            review_version - 1,
        )
        if review_version > 1
        else None
    )
    material = ProductionReadinessMaterialStore(
        private_store,
        max_private_bytes=policy.max_private_bytes,
        max_members=max(16, policy.max_approvals + 8),
    )
    reports = ProductionReadinessReportStore(
        report_store,
        max_report_bytes=policy.max_report_bytes,
    )
    accepted_store = ProductionReadinessAcceptedRunStore(
        material_store=material,
        report_store=reports,
    )
    existing = material.find_acceptance(acceptance_key)
    if existing is not None:
        if existing.request_sha256 != request_sha256:
            raise ProductionReadinessStoreConflictError("production readiness accepted request changed")
        return accepted_store.get_accepted_report(
            acceptance_key,
            previous_acceptance_key=previous_acceptance_key,
        )
    builder = ProductionReadinessReviewBuilder()
    if pending_mode:
        compilation = builder.compile_repository_pending(
            payload=repository_bytes,
            policy=policy,
        )
    else:
        if request is None or registry is None:
            raise ValueError("full readiness review inputs are incomplete")
        compilation = builder.compile_full(
            policy=policy,
            request=request,
            trusted_registry=registry,
        )
    report = ProductionReadinessReviewer().evaluate(
        compilation=compilation,
        policy=policy,
        audit=policy.audit,
    )
    accepted_store.persist(
        acceptance_key=acceptance_key,
        request_sha256=request_sha256,
        compilation=compilation,
        report=report,
        audit=policy.audit,
        previous_acceptance_key=previous_acceptance_key,
    )
    return report


def _production_readiness_acceptance_key(
    policy_sha256: str,
    review_series_id: str,
    review_version: int,
) -> str:
    digest = hashlib.sha256(f"{policy_sha256}:{review_series_id}:{review_version}".encode()).hexdigest()
    return f"production-readiness-acceptance://sha256/{digest}"


def _load_accepted_concurrency_report(
    *,
    run_root: Path,
    policy: ConcurrencyExperimentPolicyV2,
    workload_ref: ObjectRef,
    material_store: ConcurrencyExperimentMaterialStore,
    report_store: ConcurrencyExperimentReportStore,
) -> ConcurrencyExperimentReportV2 | None:
    path = run_root / "accepted-report.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("accepted concurrency report is not a regular file")
    payload = path.read_bytes()
    if len(payload) > policy.max_report_bytes:
        raise ValueError("accepted concurrency report exceeds its byte limit")
    report = ConcurrencyExperimentReportV2.model_validate_json(payload)
    validate_concurrency_experiment_report_v2_identity(report)
    if report.policy_ref != policy.to_ref() or report.private_workload_ref != workload_ref:
        raise ValueError("accepted concurrency report authority differs")
    material_store.get_workload(workload_ref)
    closure = material_store.get_result_set_closure(report.private_result_set_ref)
    level_summaries = tuple(
        result.summary
        for result in sorted(
            closure.level_results,
            key=lambda value: value.worker_count,
        )
    )
    if level_summaries != report.level_summaries:
        raise ValueError("accepted concurrency private levels differ from public report")
    if report.recovery_summary is None:
        if closure.recovery_results:
            raise ValueError("accepted concurrency private recovery differs from public report")
    elif len(closure.recovery_results) != policy.recovery_probe_count or any(
        result.summary != report.recovery_summary for result in closure.recovery_results
    ):
        raise ValueError("accepted concurrency private recovery differs from public report")
    report_store.put(report)
    stored = report_store.get(report.to_ref())
    if stored != report:
        raise ValueError("accepted concurrency report differs from public CAS")
    return stored


def _admit_concurrency_report(
    run_root: Path,
    report: ConcurrencyExperimentReportV2,
) -> None:
    root = run_root.expanduser()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("concurrency run root is not a regular directory")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "accepted-report.json"
    encoded = report.canonical_json() + b"\n"
    if path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError("accepted concurrency report conflicts")
        return
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = root / f".accepted-report.{digest}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError("accepted concurrency report raced with another authority") from None
    finally:
        temporary.unlink(missing_ok=True)


def _cancel(
    *,
    job_store: Path,
    job_id: str,
    reason_code: str,
    idempotency_key: str,
) -> BaseModel:
    service = _service(job_store)
    audit = service.job_store.get_job_spec(job_id).audit
    return service.cancel(
        job_id=job_id,
        reason_code=reason_code,
        audit=audit,
        idempotency_key=idempotency_key,
    )


def _events(
    *,
    job_store: Path,
    job_id: str,
    refresh: bool,
    idempotency_key: str | None,
    offset: int,
    limit: int,
) -> BaseModel:
    service = _service(job_store)
    audit = service.job_store.get_job_spec(job_id).audit if refresh else None
    return service.events(
        job_id=job_id,
        offset=offset,
        limit=limit,
        refresh=refresh,
        audit=audit,
        idempotency_key=idempotency_key,
    )


def _metrics(
    *,
    job_store: Path,
    job_id: str,
    item_id: str | None,
    scope: MetricScopeV2 | None,
    refresh: bool,
    idempotency_key: str | None,
    offset: int,
    limit: int,
) -> BaseModel:
    service = _service(job_store)
    audit = service.job_store.get_job_spec(job_id).audit if refresh else None
    return service.metrics(
        job_id=job_id,
        item_id=item_id,
        scope=scope,
        offset=offset,
        limit=limit,
        refresh=refresh,
        audit=audit,
        idempotency_key=idempotency_key,
    )


def _service(job_store: Path) -> PipelineLifecycleService:
    return PipelineLifecycleService(JobStore(job_store))


def _plan_review_service(
    factory_store: Path,
    *,
    registry_path: Path | None = None,
) -> PlanReviewService:
    compiler = None
    adapter_registry = None
    if registry_path is not None:
        _require_pipeline_file(registry_path, "Agent Registry input")
        registry_input = _load_model(registry_path, _PlanReviewRegistryInput)
        registry = AgentRegistry(
            capabilities=registry_input.capabilities,
            definitions=registry_input.definitions,
        )
        compiler = DatasetBuildPlanCompiler(registry)
        adapter_registry = ReviewablePlanAdapterRegistry(
            (
                erased_adapter(GlobalBuildPlanAdapter(compiler)),
                erased_adapter(AttachmentGenerationPlanAdapter(AttachmentGenerationPlanCompiler(registry))),
                erased_adapter(CriteriaRubricPlanAdapter(CriteriaRubricPlanCompiler(registry))),
                erased_adapter(GradingDesignPlanAdapter(GradingDesignPlanCompiler(registry))),
                erased_adapter(DatasetDeliveryPlanAdapter(DatasetDeliveryPlanCompiler())),
            )
        )
    return PlanReviewService(
        FactoryControlStore(factory_store),
        compiler=compiler,
        adapter_registry=adapter_registry,
    )


def _agent_plan_edit(
    *,
    factory_store: Path,
    review: str,
    submission_path: Path,
    registry_path: Path,
) -> BaseModel:
    _require_pipeline_file(submission_path, "plan edit submission")
    service = _plan_review_service(
        factory_store,
        registry_path=registry_path,
    )
    payload = _load_model(submission_path, PlanReviewEditPayloadV1)
    audit = _plan_review_audit(service, review, payload.decided_by)
    submission = compile_plan_review_edit(
        service.show(review),
        payload,
        audit=audit,
    )
    if isinstance(
        submission,
        (
            AttachmentPlanReviewEditSubmissionV1,
            CriteriaRubricPlanReviewEditSubmissionV1,
            DatasetDeliveryPlanReviewEditSubmissionV1,
            GradingDesignPlanReviewEditSubmissionV1,
        ),
    ):
        return service.edit_plan(
            review,
            submission,
            audit=audit,
        )
    return service.edit_global_plan(
        review,
        submission,
        audit=audit,
    )


def _agent_plan_decide(
    *,
    factory_store: Path,
    review: str,
    expected_plan_version: int,
    decision: PlanDecisionKindV2,
    user: str,
    reason_code: str,
    idempotency_key: str,
) -> BaseModel:
    service = _plan_review_service(factory_store)
    return service.decide(
        review,
        PlanReviewDecisionSubmissionV1(
            expected_plan_version=expected_plan_version,
            decision=decision,
            decided_by=user,
            reason_code=reason_code,
            idempotency_key=idempotency_key,
        ),
        audit=_plan_review_audit(service, review, user),
    )


def _agent_plan_resume(
    *,
    factory_store: Path,
    review: str,
    expected_plan_version: int,
    user: str,
    idempotency_key: str,
) -> BaseModel:
    service = _plan_review_service(factory_store)
    return service.resume(
        review,
        PlanReviewResumeSubmissionV1(
            expected_plan_version=expected_plan_version,
            resumed_by=user,
            idempotency_key=idempotency_key,
        ),
        audit=_plan_review_audit(service, review, user),
    )


def _plan_review_audit(
    service: PlanReviewService,
    review: str,
    actor: str,
) -> ContractAudit:
    current = service.show(review)
    return ContractAudit(
        created_at=datetime.now(UTC),
        created_by=actor,
        governing_versions=current.request.audit.governing_versions,
        input_refs=(current.result.to_ref(),),
    )


def _load_model[T: BaseModel](
    path: Path,
    model_type: type[T],
) -> T:
    return model_type.model_validate_json(_read_pipeline_input(path))


def _read_pipeline_input(path: Path) -> bytes:
    if path.stat().st_size > _MAX_PIPELINE_INPUT_BYTES:
        raise ValueError("pipeline input exceeds the maximum file size")
    with path.open("rb") as handle:
        payload = handle.read(_MAX_PIPELINE_INPUT_BYTES + 1)
    if len(payload) > _MAX_PIPELINE_INPUT_BYTES:
        raise ValueError("pipeline input exceeds the maximum file size")
    return payload


def _require_pipeline_file(path: Path, label: str) -> None:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _require_pipeline_directory(
    path: Path,
    label: str,
    *,
    require_existing: bool = False,
) -> None:
    candidate = path.expanduser()
    if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
        raise ValueError(f"{label} must be a non-symlink directory")
    if require_existing and not candidate.is_dir():
        raise ValueError(f"{label} must exist")


def _execute(operation: Callable[[], BaseModel]) -> None:
    result = _execute_result(operation)
    typer.echo(result.model_dump_json(indent=2))


def _execute_result(operation: Callable[[], BaseModel]) -> BaseModel:
    try:
        return operation()
    except Exception as exc:
        code, message = _error(exc)
        typer.echo(
            json.dumps(
                {
                    "schema_version": ("eval-factory/pipeline-cli-error/v1"),
                    "error_code": code,
                    "message": message,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            err=True,
        )
        raise typer.Exit(code=2) from None


def _error(exc: Exception) -> tuple[str, str]:
    from eval_factory.agent_system.store import (
        FactoryControlConcurrencyError,
    )
    from eval_factory.dataset.production_release import (
        ProductionReleaseAuthorizationError,
        ProductionReleaseConflictError,
        ProductionReleaseError,
        ProductionReleaseIntegrityError,
        ProductionReleasePolicyError,
    )
    from eval_factory.readiness import (
        CanaryRegressionBuilderError,
        CanaryRegressionReportConflictError,
        CanaryRegressionReportIntegrityError,
        CanaryRegressionReportLimitError,
        CanaryRegressionReportTypeError,
        CanaryRegressionRunError,
        ConcurrencyExperimentBuilderError,
        ConcurrencyExperimentRunError,
        ConcurrencyExperimentStoreConflictError,
        ConcurrencyExperimentStoreIntegrityError,
        ConcurrencyExperimentStoreLimitError,
        ConcurrencyExperimentStoreTypeError,
        ProductionAttestationAuthorizationError,
        ProductionAttestationBuilderError,
        ProductionAttestationError,
        ProductionAttestationIntegrityError,
        ProductionAttestationPolicyError,
        ProductionAttestationStoreConflictError,
        ProductionAttestationStoreIntegrityError,
        ProductionAttestationStoreLimitError,
        ProductionAttestationStoreTypeError,
        ProductionReadinessReviewAuthorizationError,
        ProductionReadinessReviewBuilderError,
        ProductionReadinessReviewError,
        ProductionReadinessReviewIntegrityError,
        ProductionReadinessStoreConflictError,
        ProductionReadinessStoreIntegrityError,
        ProductionReadinessStoreLimitError,
        ProductionReadinessStoreTypeError,
        RealTraceStabilityBuilderError,
        RealTraceStabilityRunConflictError,
        RealTraceStabilityRunError,
        RealTraceStabilityStoreConflictError,
        RealTraceStabilityStoreIntegrityError,
        RealTraceStabilityStoreLimitError,
        RealTraceStabilityStoreTypeError,
        SchedulerLoadBuilderError,
        SchedulerLoadRunConflictError,
        SchedulerLoadRunError,
        SchedulerLoadStoreConflictError,
        SchedulerLoadStoreIntegrityError,
        SchedulerLoadStoreLimitError,
        SchedulerLoadStoreTypeError,
    )
    from eval_factory.statistics import (
        LabelQualityEvaluationAuthorizationError,
        LabelQualityEvaluationBuilderError,
        LabelQualityEvaluationError,
        LabelQualityEvaluationIntegrityError,
        LabelQualityStoreConflictError,
        LabelQualityStoreIntegrityError,
        LabelQualityStoreLimitError,
        LabelQualityStoreTypeError,
    )

    if isinstance(exc, FactoryControlConcurrencyError):
        return (
            "CONCURRENCY_CONFLICT",
            "expected Factory run version is stale",
        )
    if isinstance(exc, ProductionReleaseAuthorizationError):
        return (
            "AUTHORIZATION_ERROR",
            "production release authority was not authorized",
        )
    if isinstance(exc, ProductionReleaseIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "production release evidence failed integrity validation",
        )
    if isinstance(exc, ProductionReleaseConflictError):
        return (
            "POLICY_ERROR",
            "production release authority conflicts with this request",
        )
    if isinstance(exc, ProductionReleasePolicyError):
        return (
            "POLICY_ERROR",
            "production release policy does not match this evidence",
        )
    if isinstance(exc, ProductionReleaseError):
        return (
            "INVALID_INPUT",
            "production release input does not satisfy the required contract",
        )
    if isinstance(exc, ProductionAttestationAuthorizationError):
        return (
            "AUTHORIZATION_ERROR",
            "production attestation issuer was not authorized",
        )
    if isinstance(
        exc,
        (
            ProductionAttestationIntegrityError,
            ProductionAttestationStoreIntegrityError,
        ),
    ):
        return (
            "INTEGRITY_ERROR",
            "production attestation evidence failed integrity validation",
        )
    if isinstance(exc, ProductionAttestationStoreConflictError):
        return (
            "POLICY_ERROR",
            "production attestation authority conflicts with this version",
        )
    if isinstance(exc, ProductionAttestationPolicyError):
        return (
            "POLICY_ERROR",
            "production attestation policy does not match this authority",
        )
    if isinstance(
        exc,
        (
            ProductionAttestationBuilderError,
            ProductionAttestationError,
            ProductionAttestationStoreLimitError,
            ProductionAttestationStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "production attestation input does not satisfy the required contract",
        )
    if isinstance(exc, ProductionReadinessReviewAuthorizationError):
        return (
            "AUTHORIZATION_ERROR",
            "production readiness approval was not authorized",
        )
    if isinstance(
        exc,
        (
            ProductionReadinessReviewIntegrityError,
            ProductionReadinessStoreIntegrityError,
        ),
    ):
        return (
            "INTEGRITY_ERROR",
            "production readiness evidence failed integrity validation",
        )
    if isinstance(exc, ProductionReadinessStoreConflictError):
        return (
            "POLICY_ERROR",
            "production readiness authority conflicts with this review",
        )
    if isinstance(
        exc,
        (
            ProductionReadinessReviewBuilderError,
            ProductionReadinessReviewError,
            ProductionReadinessStoreLimitError,
            ProductionReadinessStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "production readiness input does not satisfy the required contract",
        )
    if isinstance(exc, LabelQualityEvaluationAuthorizationError):
        return (
            "AUTHORIZATION_ERROR",
            "label quality access was not authorized",
        )
    if isinstance(
        exc,
        (
            LabelQualityEvaluationIntegrityError,
            LabelQualityStoreIntegrityError,
        ),
    ):
        return (
            "INTEGRITY_ERROR",
            "label quality evidence failed integrity validation",
        )
    if isinstance(exc, LabelQualityStoreConflictError):
        return (
            "POLICY_ERROR",
            "label quality authority conflicts with this evaluation",
        )
    if isinstance(
        exc,
        (
            LabelQualityEvaluationBuilderError,
            LabelQualityEvaluationError,
            LabelQualityStoreLimitError,
            LabelQualityStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "label quality input does not satisfy the required contract",
        )
    if isinstance(exc, ConcurrencyExperimentStoreIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "concurrency experiment material failed integrity validation",
        )
    if isinstance(exc, ConcurrencyExperimentStoreConflictError):
        return (
            "POLICY_ERROR",
            "concurrency experiment authority conflicts with this run",
        )
    if isinstance(
        exc,
        (
            ConcurrencyExperimentBuilderError,
            ConcurrencyExperimentRunError,
            ConcurrencyExperimentStoreLimitError,
            ConcurrencyExperimentStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "concurrency experiment input does not satisfy the required contract",
        )
    if isinstance(exc, SchedulerLoadStoreIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "scheduler load material failed integrity validation",
        )
    if isinstance(
        exc,
        (
            SchedulerLoadRunConflictError,
            SchedulerLoadStoreConflictError,
        ),
    ):
        return (
            "POLICY_ERROR",
            "scheduler load authority conflicts with this run",
        )
    if isinstance(
        exc,
        (
            SchedulerLoadBuilderError,
            SchedulerLoadRunError,
            SchedulerLoadStoreLimitError,
            SchedulerLoadStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "scheduler load input does not satisfy the required contract",
        )
    if isinstance(exc, RealTraceStabilityStoreIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "real-trace stability material failed integrity validation",
        )
    if isinstance(
        exc,
        (
            RealTraceStabilityRunConflictError,
            RealTraceStabilityStoreConflictError,
        ),
    ):
        return (
            "POLICY_ERROR",
            "real-trace stability authority conflicts with this run",
        )
    if isinstance(
        exc,
        (
            RealTraceStabilityBuilderError,
            RealTraceStabilityRunError,
            RealTraceStabilityStoreLimitError,
            RealTraceStabilityStoreTypeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "real-trace stability input does not satisfy the required contract",
        )
    if isinstance(exc, CanaryRegressionReportIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "canary regression report failed integrity validation",
        )
    if isinstance(exc, CanaryRegressionReportConflictError):
        return (
            "POLICY_ERROR",
            "canary regression report authority conflicts with this run",
        )
    if isinstance(
        exc,
        (
            CanaryRegressionBuilderError,
            CanaryRegressionReportLimitError,
            CanaryRegressionReportTypeError,
            CanaryRegressionRunError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "canary regression input does not satisfy the required contract",
        )
    if isinstance(exc, PipelineCheckpointRequiredError):
        return (
            "CHECKPOINT_REQUIRED",
            "Job requires checkpoint-aware interaction",
        )
    if isinstance(exc, CoreOutputConflictError):
        return (
            "OUTPUT_CONFLICT",
            "core output authority conflicts with this request",
        )
    if isinstance(exc, CoreOutputIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "core output failed integrity validation",
        )
    if isinstance(exc, CoreOutputError):
        return "POLICY_ERROR", "core output request violates current policy"
    if isinstance(exc, PlanReviewNotResumableError):
        return (
            "PLAN_NOT_RESUMABLE",
            "plan review is not ready to resume",
        )
    if isinstance(
        exc,
        (
            FactoryControlConflictError,
            PlanReviewConflictError,
        ),
    ):
        return (
            "PLAN_REVIEW_CONFLICT",
            "plan review authority conflicts with this request",
        )
    if isinstance(
        exc,
        (
            FactoryControlIntegrityError,
            PlanReviewIntegrityError,
        ),
    ):
        return (
            "INTEGRITY_ERROR",
            "plan review evidence failed integrity validation",
        )
    if isinstance(exc, FactoryControlNotFoundError):
        return "NOT_FOUND", "requested factory control record was not found"
    if isinstance(exc, PlanReviewError):
        return "POLICY_ERROR", "plan review request violates current policy"
    if isinstance(exc, UserCheckpointInteractionNotResumableError):
        return (
            "CHECKPOINT_NOT_RESUMABLE",
            "checkpoint interaction is not resumable",
        )
    if isinstance(exc, UserCheckpointAuthenticationError):
        return (
            "AUTHENTICATION_MISMATCH",
            "authentication does not match the checkpoint requester",
        )
    if isinstance(
        exc,
        (
            UserDecisionConflictError,
            UserCheckpointInteractionConflictError,
        ),
    ):
        return (
            "DECISION_CONFLICT",
            "checkpoint interaction already has conflicting authority",
        )
    if isinstance(
        exc,
        (
            UserCheckpointInteractionIntegrityError,
            UserCheckpointMaterialIntegrityError,
        ),
    ):
        return (
            "INTEGRITY_ERROR",
            "checkpoint evidence failed integrity validation",
        )
    if isinstance(
        exc,
        PipelineObservabilityRefreshRequiredError,
    ):
        return (
            "REFRESH_REQUIRED",
            "observability projection requires explicit refresh",
        )
    if isinstance(exc, RecordNotFoundError):
        return "NOT_FOUND", "requested pipeline record was not found"
    if isinstance(exc, ConcurrencyConflictError):
        return (
            "CONCURRENCY_CONFLICT",
            "expected aggregate version is stale",
        )
    if isinstance(exc, IdempotencyConflictError):
        return (
            "IDEMPOTENCY_CONFLICT",
            "idempotency key was reused for a different request",
        )
    if isinstance(exc, ImmutableResultError):
        return (
            "INTEGRITY_ERROR",
            "persisted pipeline evidence failed integrity validation",
        )
    if isinstance(
        exc,
        (
            IllegalTransitionError,
            PipelineLifecyclePolicyError,
            JobPlanningPolicyError,
            WorkFanoutPolicyError,
            WorkControlPolicyError,
            ModelControlPolicyError,
            ResourceControlPolicyError,
            BatchObservabilityPolicyError,
            CanaryDriverError,
            CanaryProfileError,
            UserCheckpointInteractionError,
        ),
    ):
        return "POLICY_ERROR", "pipeline request violates current policy"
    if isinstance(exc, BatchObservabilityIntegrityError):
        return (
            "INTEGRITY_ERROR",
            "observability projection failed integrity validation",
        )
    if isinstance(exc, OSError) and exc.errno in {
        errno.EMFILE,
        errno.ENFILE,
    }:
        return "INTERNAL_ERROR", "pipeline operation failed"
    if isinstance(
        exc,
        (
            ValidationError,
            ValueError,
            OSError,
            UnicodeError,
        ),
    ):
        return (
            "INVALID_INPUT",
            "input does not satisfy the required contract",
        )
    if isinstance(exc, JobStoreError):
        return "PIPELINE_ERROR", "pipeline operation failed"
    if isinstance(exc, StageObjectStoreError):
        return (
            "INTEGRITY_ERROR",
            "private stage object evidence failed integrity validation",
        )
    if isinstance(exc, UserCheckpointMaterialError):
        return (
            "INVALID_INPUT",
            "checkpoint material does not satisfy the required contract",
        )
    return "INTERNAL_ERROR", "pipeline operation failed"
