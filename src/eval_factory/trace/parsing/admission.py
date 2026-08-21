from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from eval_factory.contracts.trace import CapabilityStatus, ParseQuality
from eval_factory.trace.parsing.recovery import (
    TraceCapabilityName,
    TraceRecoveryResult,
)


class TraceUse(StrEnum):
    OBSERVED_POSITIVE_FACT = "OBSERVED_POSITIVE_FACT"
    NEGATIVE_PREDICATE = "NEGATIVE_PREDICATE"
    STAGE_REQUIRED = "STAGE_REQUIRED"


class AdmissionMode(StrEnum):
    FULL = "FULL"
    POSITIVE_ONLY = "POSITIVE_ONLY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class TraceAdmissionDecision:
    mode: AdmissionMode
    use: TraceUse
    required_capabilities: tuple[TraceCapabilityName, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.required_capabilities:
            raise ValueError("admission requires at least one capability")
        if len(set(self.required_capabilities)) != len(self.required_capabilities):
            raise ValueError("required capabilities must be unique")
        if self.mode is AdmissionMode.FULL and self.reason_codes:
            raise ValueError("full admission cannot contain reason codes")
        if self.mode is not AdmissionMode.FULL and not self.reason_codes:
            raise ValueError("non-full admission requires reason codes")


def evaluate_admission(
    result: TraceRecoveryResult,
    *,
    use: TraceUse,
    required_capabilities: tuple[TraceCapabilityName, ...],
) -> TraceAdmissionDecision:
    if not required_capabilities:
        raise ValueError("required_capabilities cannot be empty")
    if len(set(required_capabilities)) != len(required_capabilities):
        raise ValueError("required_capabilities must be unique")
    if result.parse_quality is ParseQuality.UNPARSEABLE:
        return TraceAdmissionDecision(
            mode=AdmissionMode.BLOCKED,
            use=use,
            required_capabilities=required_capabilities,
            reason_codes=("trace-unparseable",),
        )

    by_name = {item.capability: item for item in result.capabilities}
    required = tuple(by_name[name] for name in required_capabilities)
    unavailable = tuple(
        item.capability
        for item in required
        if item.status in {CapabilityStatus.MISSING, CapabilityStatus.UNKNOWN}
    )
    if unavailable:
        return TraceAdmissionDecision(
            mode=AdmissionMode.BLOCKED,
            use=use,
            required_capabilities=required_capabilities,
            reason_codes=tuple(f"required-capability-unavailable:{name}" for name in unavailable),
        )

    partial = tuple(item.capability for item in required if item.status is CapabilityStatus.PARTIAL)
    if not partial:
        return TraceAdmissionDecision(
            mode=AdmissionMode.FULL,
            use=use,
            required_capabilities=required_capabilities,
            reason_codes=(),
        )
    if use is TraceUse.OBSERVED_POSITIVE_FACT:
        return TraceAdmissionDecision(
            mode=AdmissionMode.POSITIVE_ONLY,
            use=use,
            required_capabilities=required_capabilities,
            reason_codes=tuple(f"required-capability-partial:{name}" for name in partial),
        )
    return TraceAdmissionDecision(
        mode=AdmissionMode.BLOCKED,
        use=use,
        required_capabilities=required_capabilities,
        reason_codes=tuple(f"required-capability-not-complete:{name}" for name in partial),
    )
