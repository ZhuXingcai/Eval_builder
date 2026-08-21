from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.canary_execution_v2 import (
    R6_CANARY_EXECUTION_PROFILE,
    R6_CANARY_POLICY_VERSION,
    R6CanaryExecutionManifestV2,
    R6CanaryExpectedItemOutcomeV2,
    R6CanaryObjectCodecV2,
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.canary_regression_v2 import (
    CANARY_REGRESSION_POLICY_VERSION,
    CanaryRegressionExpectationV2,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2


class FrozenCanaryCaseV1(ContractModelV2):
    schema_version: Literal["eval-factory/frozen-canary-case/private-v1"] = (
        "eval-factory/frozen-canary-case/private-v1"
    )
    instance_id: Identifier
    source_ref: Identifier
    raw_sha256: Sha256
    size_bytes: int = Field(ge=1, le=100_000_000)
    signal_quality: Literal["strict_events", "heuristic_requires_annotation"]
    verified_traits: tuple[Identifier, ...]
    expectation: CanaryRegressionExpectationV2

    @field_validator("expectation", mode="before")
    @classmethod
    def parse_expectation(cls, value: object) -> CanaryRegressionExpectationV2:
        if isinstance(value, CanaryRegressionExpectationV2):
            return value
        if isinstance(value, str):
            return CanaryRegressionExpectationV2(value)
        raise TypeError("expectation must be a CanaryRegressionExpectationV2")

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.source_ref != f"raw_traj://{self.instance_id}":
            raise ValueError("frozen canary source ref differs from instance")
        if self.verified_traits != tuple(sorted(set(self.verified_traits))):
            raise ValueError("verified traits must be sorted and unique")
        if self.signal_quality == "strict_events":
            if (
                "tool.file_read" not in self.verified_traits
                or self.expectation is not CanaryRegressionExpectationV2.MUST_SUCCEED
            ):
                raise ValueError("strict canary requires verified file-read success expectation")
        elif self.expectation is not CanaryRegressionExpectationV2.ANY_TYPED_TERMINAL:
            raise ValueError("heuristic canary requires typed-terminal expectation")
        return self


class FrozenCanaryCohortV1(ContractModelV2):
    schema_version: Literal["eval-factory/frozen-canary-cohort/private-v1"] = (
        "eval-factory/frozen-canary-cohort/private-v1"
    )
    cohort_id: Identifier
    manifest_ref: ObjectRef
    selector_version: Literal["4.0.0"]
    source_count: Literal[91]
    target_count: Literal[24]
    cases: tuple[FrozenCanaryCaseV1, ...] = Field(min_length=24, max_length=24)
    policy_version: Literal["canary-regression/r8-03-v1"] = CANARY_REGRESSION_POLICY_VERSION
    cohort_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_cohort(self) -> Self:
        if (
            self.manifest_ref.object_type != "development-canary-manifest"
            or self.manifest_ref.object_version != "v4"
        ):
            raise ValueError("cohort manifest ref must reference development canary v4")
        ids = tuple(item.instance_id for item in self.cases)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("cohort cases must be sorted and unique")
        if sum(item.expectation is CanaryRegressionExpectationV2.MUST_SUCCEED for item in self.cases) != 15:
            raise ValueError("cohort requires exact 15 strict success cases")
        if self.audit.input_refs != (self.manifest_ref,):
            raise ValueError("cohort audit must bind exact manifest")
        observed = frozen_canary_cohort_v1_carried_sha256(self)
        if (
            self.cohort_id != f"frozen-canary-cohort://sha256/{observed}" or self.cohort_sha256 != observed
        ) and not (self.cohort_id == "frozen-canary-cohort://pending" and self.cohort_sha256 == "0" * 64):
            raise ValueError("frozen canary cohort identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        manifest_ref: ObjectRef,
        cases: tuple[FrozenCanaryCaseV1, ...],
        audit: ContractAudit,
    ) -> FrozenCanaryCohortV1:
        value = cls(
            cohort_id="frozen-canary-cohort://pending",
            manifest_ref=manifest_ref,
            selector_version="4.0.0",
            source_count=91,
            target_count=24,
            cases=tuple(sorted(cases, key=lambda item: item.instance_id)),
            cohort_sha256="0" * 64,
            audit=audit.model_copy(update={"input_refs": (manifest_ref,)}),
        )
        digest = frozen_canary_cohort_v1_carried_sha256(value)
        return value.model_copy(
            update={
                "cohort_id": f"frozen-canary-cohort://sha256/{digest}",
                "cohort_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        digest = frozen_canary_cohort_v1_carried_sha256(self)
        if self.cohort_sha256 != digest:
            raise ValueError("frozen canary cohort identity is stale")
        return ObjectRef(
            object_type="frozen-canary-cohort",
            object_id=self.cohort_id,
            object_version="private-v1",
            object_sha256=self.cohort_sha256,
        )


class CanaryRegressionTemplateV1(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-template/private-v1"] = (
        "eval-factory/canary-regression-template/private-v1"
    )
    template_id: Identifier
    template_manifest: R6CanaryExecutionManifestV2
    success_verified_trait: Literal["tool.file_read"] = "tool.file_read"
    policy_version: Literal["canary-regression/r8-03-v1"] = CANARY_REGRESSION_POLICY_VERSION
    template_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_template(self) -> Self:
        manifest = self.template_manifest
        if (
            manifest.execution_profile != R6_CANARY_EXECUTION_PROFILE
            or manifest.policy_version != R6_CANARY_POLICY_VERSION
            or len(manifest.job_spec.traces) != 1
            or len(manifest.trace_bindings) != 1
            or manifest.trace_bindings[0].expected_outcome is not R6CanaryExpectedItemOutcomeV2.SUCCEEDED
        ):
            raise ValueError("regression template requires one successful R6 canary")
        seed_codecs = {seed.codec for seed in manifest.seed_objects}
        required = {
            R6CanaryObjectCodecV2.LABEL_SPEC,
            R6CanaryObjectCodecV2.CANARY_ATTACHMENT_FIXTURE,
            R6CanaryObjectCodecV2.CANARY_REVIEW_FIXTURE,
        }
        if not required.issubset(seed_codecs):
            raise ValueError("regression template is missing required seed codecs")
        manifest_ref = r6_canary_execution_manifest_v2_ref(manifest)
        if self.audit.input_refs != (manifest_ref,):
            raise ValueError("regression template audit must bind exact child manifest")
        observed = canary_regression_template_v1_carried_sha256(self)
        if (
            self.template_id != f"canary-regression-template://sha256/{observed}"
            or self.template_sha256 != observed
        ) and not (
            self.template_id == "canary-regression-template://pending" and self.template_sha256 == "0" * 64
        ):
            raise ValueError("canary regression template identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        template_manifest: R6CanaryExecutionManifestV2,
        audit: ContractAudit,
    ) -> CanaryRegressionTemplateV1:
        value = cls(
            template_id="canary-regression-template://pending",
            template_manifest=template_manifest,
            template_sha256="0" * 64,
            audit=audit.model_copy(
                update={"input_refs": (r6_canary_execution_manifest_v2_ref(template_manifest),)}
            ),
        )
        digest = canary_regression_template_v1_carried_sha256(value)
        return value.model_copy(
            update={
                "template_id": f"canary-regression-template://sha256/{digest}",
                "template_sha256": digest,
            }
        )

    def to_ref(self) -> ObjectRef:
        digest = canary_regression_template_v1_carried_sha256(self)
        if self.template_sha256 != digest:
            raise ValueError("canary regression template identity is stale")
        return ObjectRef(
            object_type="canary-regression-template",
            object_id=self.template_id,
            object_version="private-v1",
            object_sha256=self.template_sha256,
        )


class CanaryRegressionRunLedgerV1(ContractModelV2):
    schema_version: Literal["eval-factory/canary-regression-run-ledger/private-v1"] = (
        "eval-factory/canary-regression-run-ledger/private-v1"
    )
    cohort_ref: ObjectRef
    template_ref: ObjectRef
    policy_ref: ObjectRef
    child_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=24, max_length=24)
    ledger_sha256: Sha256

    @model_validator(mode="after")
    def validate_ledger(self) -> Self:
        expected_types = (
            (self.cohort_ref, "frozen-canary-cohort", "private-v1"),
            (self.template_ref, "canary-regression-template", "private-v1"),
            (self.policy_ref, "canary-regression-policy", "v2"),
        )
        if any(
            ref.object_type != object_type or ref.object_version != version
            for ref, object_type, version in expected_types
        ):
            raise ValueError("regression ledger authority refs are invalid")
        if any(
            ref.object_type != "r6-canary-execution-manifest" or ref.object_version != "v2"
            for ref in self.child_manifest_refs
        ):
            raise ValueError("regression ledger child refs are invalid")
        keys = tuple(_ref_key(value) for value in self.child_manifest_refs)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("child manifest refs must be sorted and unique")
        observed = _payload_sha256(
            self.model_dump(
                mode="python",
                exclude={"ledger_sha256"},
                exclude_none=False,
            )
        )
        if self.ledger_sha256 != observed and self.ledger_sha256 != "0" * 64:
            raise ValueError("regression run ledger identity is stale")
        return self

    @classmethod
    def create(
        cls,
        *,
        cohort_ref: ObjectRef,
        template_ref: ObjectRef,
        policy_ref: ObjectRef,
        child_manifest_refs: tuple[ObjectRef, ...],
    ) -> CanaryRegressionRunLedgerV1:
        value = cls(
            cohort_ref=cohort_ref,
            template_ref=template_ref,
            policy_ref=policy_ref,
            child_manifest_refs=tuple(sorted(child_manifest_refs, key=_ref_key)),
            ledger_sha256="0" * 64,
        )
        digest = _payload_sha256(
            value.model_dump(
                mode="python",
                exclude={"ledger_sha256"},
                exclude_none=False,
            )
        )
        return value.model_copy(update={"ledger_sha256": digest})


@dataclass(frozen=True, slots=True)
class PreparedCanaryRegressionCase:
    cohort_case: FrozenCanaryCaseV1
    expectation: CanaryRegressionExpectationV2
    raw_path: Path
    child_manifest: R6CanaryExecutionManifestV2


def frozen_canary_cohort_v1_carried_sha256(
    value: FrozenCanaryCohortV1,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"cohort_id", "cohort_sha256", "audit"},
            exclude_none=False,
        )
    )


def canary_regression_template_v1_carried_sha256(
    value: CanaryRegressionTemplateV1,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
            exclude={"template_id", "template_sha256", "audit"},
            exclude_none=False,
        )
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


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
