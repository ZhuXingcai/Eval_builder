from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_canary_pipeline_driver import _manifest

from eval_factory.contracts.canary_regression_v2 import (
    CANARY_REGRESSION_POLICY_VERSION,
    CanaryRegressionExpectationV2,
    CanaryRegressionPolicyV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.readiness.canary_regression_builder import (
    CanaryRegressionBuilder,
    CanaryRegressionCohortError,
    CanaryRegressionExecutionIdentity,
    CanaryRegressionRawSourceError,
    CanaryRegressionTemplateError,
    FrozenCanaryCohortLoader,
)
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionTemplateV1,
)

ROOT = Path(__file__).resolve().parents[3]
CANARY_PATH = ROOT / "evals/golden/eval_factory/canary_manifest.v4.json"
RAW_ROOT = ROOT.parent / "raw_traj"
NOW = datetime(2026, 8, 2, tzinfo=UTC)
requires_private_corpus = pytest.mark.skipif(
    not (RAW_ROOT / "manifest.csv").is_file(),
    reason="private 91-trace corpus is not installed",
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="r8-03-builder-test",
        governing_versions=(
            VersionBinding(
                component="canary-regression",
                version=CANARY_REGRESSION_POLICY_VERSION,
            ),
        ),
    )


def _cohort_ref() -> ObjectRef:
    return ObjectRef(
        object_type="development-canary-manifest",
        object_id="development-canary-manifest://v4",
        object_version="v4",
        object_sha256="1a409bbb92ef0fc075d8dd2144f36cfd5829976ab6718c20db3a8e9f2d61a6a6",
    )


def _policy() -> CanaryRegressionPolicyV2:
    return CanaryRegressionPolicyV2.create(
        cohort_manifest_ref=_cohort_ref(),
        max_case_refs=1_000,
        max_report_bytes=2_000_000,
        audit=_audit(),
    )


def _template(tmp_path: Path) -> CanaryRegressionTemplateV1:
    template_root = tmp_path / "template"
    template_root.mkdir()
    return CanaryRegressionTemplateV1.create(
        template_manifest=_manifest(template_root),
        audit=_audit(),
    )


def test_loader_admits_exact_frozen_24_case_cohort() -> None:
    cohort = FrozenCanaryCohortLoader().load(
        CANARY_PATH,
        policy=_policy(),
    )

    assert len(cohort.cases) == 24
    assert cohort.source_count == 91
    assert cohort.target_count == 24
    assert sum(case.expectation is CanaryRegressionExpectationV2.MUST_SUCCEED for case in cohort.cases) == 15
    assert (
        sum(case.expectation is CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL for case in cohort.cases)
        == 9
    )
    heuristic = next(case for case in cohort.cases if case.instance_id == "LH_011")
    assert heuristic.expectation is CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL


def test_loader_rejects_hash_schema_and_case_drift(tmp_path: Path) -> None:
    payload = json.loads(CANARY_PATH.read_text(encoding="utf-8"))
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CanaryRegressionCohortError, match="hash"):
        FrozenCanaryCohortLoader().load(changed, policy=_policy())

    wrong_policy = _policy().model_copy(
        update={"cohort_manifest_ref": _cohort_ref().model_copy(update={"object_sha256": "f" * 64})}
    )
    with pytest.raises((CanaryRegressionCohortError, ValueError)):
        FrozenCanaryCohortLoader().load(CANARY_PATH, policy=wrong_policy)


@requires_private_corpus
def test_raw_resolver_verifies_all_current_sources(tmp_path: Path) -> None:
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=_policy())
    resolved = CanaryRegressionBuilder().resolve_sources(
        cohort,
        raw_root=RAW_ROOT,
    )

    assert len(resolved) == 24
    assert all(path.is_file() and not path.is_symlink() for path in resolved.values())

    copied_root = tmp_path / "raw"
    copied_root.mkdir()
    first = cohort.cases[0]
    source = resolved[first.instance_id]
    target = copied_root / source.name
    changed = bytearray(source.read_bytes())
    changed[0] ^= 1
    target.write_bytes(changed)
    with pytest.raises(CanaryRegressionRawSourceError, match="hash"):
        CanaryRegressionBuilder().resolve_source(
            first,
            raw_root=copied_root,
        )


@requires_private_corpus
def test_raw_resolver_rejects_root_missing_duplicate_size_and_symlink(
    tmp_path: Path,
) -> None:
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=_policy())
    first = cohort.cases[0]
    builder = CanaryRegressionBuilder()
    with pytest.raises(CanaryRegressionRawSourceError, match="root"):
        builder.resolve_source(first, raw_root=tmp_path / "missing")

    raw = tmp_path / "raw"
    raw.mkdir()
    source = next(RAW_ROOT.glob(f"{first.instance_id}_*.jsonl"))
    (raw / source.name).write_bytes(source.read_bytes()[:-1])
    with pytest.raises(CanaryRegressionRawSourceError, match="size"):
        builder.resolve_source(first, raw_root=raw)

    (raw / source.name).unlink()
    (raw / source.name).symlink_to(source)
    with pytest.raises(CanaryRegressionRawSourceError, match="escapes"):
        builder.resolve_source(first, raw_root=raw)

    (raw / source.name).unlink()
    (raw / source.name).write_bytes(source.read_bytes())
    (raw / f"{first.instance_id}_duplicate.jsonl").write_bytes(source.read_bytes())
    with pytest.raises(CanaryRegressionRawSourceError, match="exactly one"):
        builder.resolve_source(first, raw_root=raw)


def test_template_is_strict_private_and_single_trace(tmp_path: Path) -> None:
    template = _template(tmp_path)

    assert template.template_manifest.claim_scope == "DEVELOPMENT_CANARY_ONLY"
    assert len(template.template_manifest.job_spec.traces) == 1
    assert len(template.template_manifest.trace_bindings) == 1
    with pytest.raises(ValidationError):
        CanaryRegressionTemplateV1.model_validate(
            {
                **template.model_dump(mode="python"),
                "raw_trace": "forbidden",
            }
        )


def test_private_template_seed_payloads_are_stable_across_hash_seeds(
    tmp_path: Path,
) -> None:
    template = _template(tmp_path)
    template_path = tmp_path / "template.json"
    template_path.write_bytes(template.canonical_json())
    script = """
from pathlib import Path
import sys
from eval_factory.readiness import CanaryRegressionTemplateV1
from eval_factory.orchestration.canary_driver import create_canary_stage_store
template = CanaryRegressionTemplateV1.model_validate_json(Path(sys.argv[1]).read_bytes())
store = create_canary_stage_store(Path(sys.argv[2]))
for seed in template.template_manifest.seed_objects:
    store.admit_seed(seed)
print(len(template.template_manifest.seed_objects))
"""

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(template_path),
            str(tmp_path / "cross-process-stage-store"),
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONHASHSEED": "321",
            "PYTHONPATH": "src",
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == "3"


@requires_private_corpus
def test_builder_prepares_one_exact_real_child_manifest(tmp_path: Path) -> None:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    case = next(value for value in cohort.cases if value.instance_id == "LH_005")
    template = _template(tmp_path)

    prepared = CanaryRegressionBuilder().prepare_case(
        case,
        template=template,
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "preparation",
        policy=policy,
        audit=_audit(),
    )

    assert prepared.expectation is CanaryRegressionExpectationV2.MUST_SUCCEED
    assert prepared.child_manifest.job_spec.traces[0].raw_sha256 == case.raw_sha256
    assert prepared.child_manifest.trace_bindings[0].source_trace_ref.object_id.endswith(case.instance_id)
    assert prepared.raw_path.name.startswith(f"{case.instance_id}_")


@requires_private_corpus
def test_builder_uses_final_execution_identity_before_fact_preparation(
    tmp_path: Path,
) -> None:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    case = next(value for value in cohort.cases if value.instance_id == "LH_011")
    template = _template(tmp_path)
    builder = CanaryRegressionBuilder()

    historical = builder.prepare_case(
        case,
        template=template,
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "historical-preparation",
        policy=policy,
        audit=_audit(),
    )
    custom = builder.prepare_case(
        case,
        template=template,
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "trial-preparation",
        policy=policy,
        audit=_audit(),
        execution_identity=CanaryRegressionExecutionIdentity(
            job_id="job://concurrency-experiment/2/0/LH_011",
            idempotency_key="concurrency-experiment-2-0-LH_011",
        ),
    )

    assert historical.child_manifest.job_spec.job_id == ("job://canary-regression/LH_011")
    assert historical.child_manifest.job_spec.idempotency_key == ("canary-regression-LH_011")
    assert custom.child_manifest.job_spec.job_id == ("job://concurrency-experiment/2/0/LH_011")
    assert custom.child_manifest.job_spec.idempotency_key == ("concurrency-experiment-2-0-LH_011")
    assert (
        custom.child_manifest.trace_bindings[0].structured_fact_set_ref
        != historical.child_manifest.trace_bindings[0].structured_fact_set_ref
    )
    assert custom.child_manifest.audit.input_refs != (historical.child_manifest.audit.input_refs)


@requires_private_corpus
def test_builder_rejects_stale_template_before_preparation(
    tmp_path: Path,
) -> None:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    stale = _template(tmp_path).model_copy(update={"template_sha256": "f" * 64})

    with pytest.raises(CanaryRegressionTemplateError, match="stale"):
        CanaryRegressionBuilder().prepare_case(
            cohort.cases[0],
            template=stale,
            raw_root=RAW_ROOT,
            preparation_root=tmp_path / "preparation",
            policy=policy,
            audit=_audit(),
        )
