from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from eval_factory.contracts.canary_execution_v2 import (
    R6CanaryExecutionManifestV2,
    R6CanaryExpectedItemOutcomeV2,
    R6CanaryObjectCodecV2,
)
from eval_factory.contracts.canary_regression_v2 import (
    CANARY_REGRESSION_COHORT_SHA256,
    CanaryRegressionExpectationV2,
    CanaryRegressionPolicyV2,
    validate_canary_regression_policy_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import TraceSourceRef
from eval_factory.contracts.orchestration_v2 import DatasetJobSpecV2
from eval_factory.labeling.structured import StructuredFactSet
from eval_factory.orchestration.canary_profile import canary_profile_codec_registry
from eval_factory.orchestration.runner import TraceIndexStageService
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionTemplateV1,
    FrozenCanaryCaseV1,
    FrozenCanaryCohortV1,
    PreparedCanaryRegressionCase,
)
from eval_factory.trace import TraceIndexStore, TraceSourceRegistry

_MAX_COHORT_BYTES = 8 * 1024 * 1024
_TOP_LEVEL_KEYS = frozenset(
    {
        "annotation_candidate_coverage",
        "items",
        "lifecycle_profile",
        "processing_class",
        "schema_version",
        "selected_category_counts",
        "selection_policy",
        "selector_version",
        "source_category_counts",
        "source_count",
        "source_reference",
        "target_count",
        "verified_coverage",
    }
)
_ITEM_KEYS = frozenset(
    {
        "annotation_candidates",
        "instance_id",
        "metadata",
        "metrics",
        "parse",
        "raw_sha256",
        "requires_annotation",
        "signal_quality",
        "size_bytes",
        "source_ref",
        "verified_traits",
    }
)


class CanaryRegressionBuilderError(RuntimeError):
    pass


class CanaryRegressionCohortError(CanaryRegressionBuilderError):
    pass


class CanaryRegressionRawSourceError(CanaryRegressionBuilderError):
    pass


class CanaryRegressionTemplateError(CanaryRegressionBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class CanaryRegressionExecutionIdentity:
    job_id: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if not self.job_id or not self.idempotency_key:
            raise ValueError("canary execution identity values cannot be empty")


class FrozenCanaryCohortLoader:
    def load(
        self,
        path: Path,
        *,
        policy: CanaryRegressionPolicyV2,
    ) -> FrozenCanaryCohortV1:
        try:
            validate_canary_regression_policy_v2_identity(policy)
        except ValueError as exc:
            raise CanaryRegressionCohortError("canary regression policy is stale") from exc
        source = path.expanduser().resolve()
        if not source.is_file() or source.is_symlink():
            raise CanaryRegressionCohortError("canary cohort path is not a regular file")
        payload = source.read_bytes()
        if len(payload) > _MAX_COHORT_BYTES:
            raise CanaryRegressionCohortError("canary cohort exceeds byte limit")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != CANARY_REGRESSION_COHORT_SHA256 or digest != policy.cohort_manifest_ref.object_sha256:
            raise CanaryRegressionCohortError("canary cohort hash differs from approved v4")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise CanaryRegressionCohortError("canary cohort JSON is invalid") from exc
        if not isinstance(decoded, dict) or frozenset(decoded) != _TOP_LEVEL_KEYS:
            raise CanaryRegressionCohortError("canary cohort top-level fields are invalid")
        if (
            decoded.get("schema_version") != "eval-factory-canary-manifest/v4"
            or decoded.get("selector_version") != "4.0.0"
            or decoded.get("source_count") != 91
            or decoded.get("target_count") != 24
        ):
            raise CanaryRegressionCohortError("canary cohort versions or counts are invalid")
        items = decoded.get("items")
        if not isinstance(items, list) or len(items) != policy.required_case_count:
            raise CanaryRegressionCohortError("canary cohort case count is invalid")
        cases = tuple(self._case(item) for item in items)
        try:
            return FrozenCanaryCohortV1.create(
                manifest_ref=policy.cohort_manifest_ref,
                cases=cases,
                audit=policy.audit,
            )
        except ValueError as exc:
            raise CanaryRegressionCohortError("canary cohort cases are invalid") from exc

    @staticmethod
    def _case(value: object) -> FrozenCanaryCaseV1:
        if not isinstance(value, dict) or frozenset(value) != _ITEM_KEYS:
            raise CanaryRegressionCohortError("canary cohort item fields are invalid")
        instance_id = value.get("instance_id")
        source_ref = value.get("source_ref")
        raw_sha256 = value.get("raw_sha256")
        size_bytes = value.get("size_bytes")
        signal_quality = value.get("signal_quality")
        traits = value.get("verified_traits")
        if (
            not isinstance(instance_id, str)
            or not isinstance(source_ref, str)
            or not isinstance(raw_sha256, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or signal_quality not in {"strict_events", "heuristic_requires_annotation"}
            or not isinstance(traits, list)
            or any(not isinstance(item, str) for item in traits)
        ):
            raise CanaryRegressionCohortError("canary cohort item values are invalid")
        expectation = (
            CanaryRegressionExpectationV2.MUST_SUCCEED
            if signal_quality == "strict_events" and "tool.file_read" in traits
            else CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL
        )
        try:
            return FrozenCanaryCaseV1(
                instance_id=instance_id,
                source_ref=source_ref,
                raw_sha256=raw_sha256,
                size_bytes=size_bytes,
                signal_quality=signal_quality,
                verified_traits=tuple(sorted(set(traits))),
                expectation=expectation,
            )
        except ValueError as exc:
            raise CanaryRegressionCohortError("canary cohort expectation is invalid") from exc


class CanaryRegressionBuilder:
    def resolve_sources(
        self,
        cohort: FrozenCanaryCohortV1,
        *,
        raw_root: Path,
    ) -> dict[str, Path]:
        return {case.instance_id: self.resolve_source(case, raw_root=raw_root) for case in cohort.cases}

    def resolve_source(
        self,
        case: FrozenCanaryCaseV1,
        *,
        raw_root: Path,
    ) -> Path:
        root = raw_root.expanduser().resolve()
        if not root.is_dir() or raw_root.is_symlink():
            raise CanaryRegressionRawSourceError("raw root is not an approved regular directory")
        matches = tuple(root.glob(f"{case.instance_id}_*.jsonl"))
        if len(matches) != 1:
            raise CanaryRegressionRawSourceError("canary must resolve to exactly one raw source")
        candidate = matches[0]
        resolved = candidate.resolve()
        if candidate.is_symlink() or not resolved.is_file() or resolved.parent != root:
            raise CanaryRegressionRawSourceError("raw source escapes approved root")
        payload = resolved.read_bytes()
        if len(payload) != case.size_bytes:
            raise CanaryRegressionRawSourceError("raw source size differs from frozen cohort")
        if hashlib.sha256(payload).hexdigest() != case.raw_sha256:
            raise CanaryRegressionRawSourceError("raw source hash differs from frozen cohort")
        return resolved

    def prepare_case(
        self,
        case: FrozenCanaryCaseV1,
        *,
        template: CanaryRegressionTemplateV1,
        raw_root: Path,
        preparation_root: Path,
        policy: CanaryRegressionPolicyV2,
        audit: ContractAudit,
        execution_identity: CanaryRegressionExecutionIdentity | None = None,
    ) -> PreparedCanaryRegressionCase:
        del audit
        try:
            validate_canary_regression_policy_v2_identity(policy)
            template.to_ref()
        except ValueError as exc:
            raise CanaryRegressionTemplateError("regression template or policy is stale") from exc
        raw_path = self.resolve_source(case, raw_root=raw_root)
        trace = TraceSourceRef(
            source_trace_id=f"source-trace://canary-regression/{case.instance_id}",
            source_uri=raw_path.as_uri(),
            raw_sha256=case.raw_sha256,
            adapter_name="raw_traj_v1",
            adapter_version="1.0.0",
            processing_class="RESTRICTED_TRACE_RAW",
        )
        provisional = self._child_manifest(
            case=case,
            trace=trace,
            fact_ref=template.template_manifest.trace_bindings[0].structured_fact_set_ref,
            template=template,
            execution_identity=execution_identity,
        )
        identity_digest = (
            hashlib.sha256(execution_identity.job_id.encode()).hexdigest()
            if execution_identity is not None
            else None
        )
        root = preparation_root.expanduser().resolve() / case.instance_id
        if identity_digest is not None:
            root /= identity_digest
        preparation_job_id = f"job://canary-regression-preparation/{case.instance_id}"
        if identity_digest is not None:
            preparation_job_id = f"{preparation_job_id}/{identity_digest}"
        prep = TraceIndexStageService(
            source_registry=TraceSourceRegistry(root / "source.sqlite3"),
            trace_store=TraceIndexStore(root / "trace-store"),
        ).execute(
            trace=trace,
            audit=provisional.audit,
            job_id=preparation_job_id,
            attempt=1,
        )
        fact_set = StructuredFactSet.from_stored_index(
            prep.stored_index,
            audit=provisional.audit,
        )
        fact_ref = canary_profile_codec_registry().reference(
            R6CanaryObjectCodecV2.STRUCTURED_FACT_SET,
            fact_set,
        )
        child = self._child_manifest(
            case=case,
            trace=trace,
            fact_ref=fact_ref,
            template=template,
            execution_identity=execution_identity,
        )
        return PreparedCanaryRegressionCase(
            cohort_case=case,
            expectation=case.expectation,
            raw_path=raw_path,
            child_manifest=child,
        )

    @staticmethod
    def _child_manifest(
        *,
        case: FrozenCanaryCaseV1,
        trace: TraceSourceRef,
        fact_ref: ObjectRef,
        template: CanaryRegressionTemplateV1,
        execution_identity: CanaryRegressionExecutionIdentity | None,
    ) -> R6CanaryExecutionManifestV2:
        base = template.template_manifest
        job_id = (
            execution_identity.job_id
            if execution_identity is not None
            else f"job://canary-regression/{case.instance_id}"
        )
        idempotency_key = (
            execution_identity.idempotency_key
            if execution_identity is not None
            else f"canary-regression-{case.instance_id}"
        )
        job_spec = DatasetJobSpecV2.model_validate(
            {
                **base.job_spec.model_dump(mode="python"),
                "job_id": job_id,
                "traces": (trace,),
                "idempotency_key": idempotency_key,
            }
        )
        binding = base.trace_bindings[0].model_copy(
            update={
                "source_trace_ref": ObjectRef(
                    object_type="trace-source",
                    object_id=trace.source_trace_id,
                    object_version=trace.adapter_version,
                    object_sha256=trace.raw_sha256,
                ),
                "structured_fact_set_ref": fact_ref,
                "expected_outcome": R6CanaryExpectedItemOutcomeV2.SUCCEEDED,
            }
        )
        return R6CanaryExecutionManifestV2.create(
            job_spec=job_spec,
            control=base.control,
            control_plane=base.control_plane,
            seed_objects=base.seed_objects,
            trace_bindings=(binding,),
            audit=base.audit,
        )
