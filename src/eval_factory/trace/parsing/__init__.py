"""Deterministic strict parsing and audited local repair."""

from eval_factory.trace.parsing.admission import (
    AdmissionMode,
    TraceAdmissionDecision,
    TraceUse,
    evaluate_admission,
)
from eval_factory.trace.parsing.curated_trajectory_v1 import (
    CuratedTrajectoryV1Parser,
    CuratedTrajectoryV1Recovery,
)
from eval_factory.trace.parsing.models import (
    NestedFieldStatus,
    ParseDiagnostic,
    ParsedNestedField,
    ParsedRawTrajRecord,
    RawTrajParseOutcome,
    RawTrajParseResult,
)
from eval_factory.trace.parsing.raw_json import OuterCoordinateError
from eval_factory.trace.parsing.raw_traj_v1 import (
    RawTrajV1Parser,
    RegisteredSourceMismatchError,
    RuntimeSnapshotV1Parser,
    TraceParseError,
)
from eval_factory.trace.parsing.recovery import (
    CAPABILITY_ORDER,
    RECOVERY_POLICY_VERSION,
    RawTrajRecovery,
    RecoveredJsonUnit,
    RecoveredNestedField,
    RecoveredRawTrajRecord,
    RecoveryFieldStatus,
    TraceCapabilityName,
    TraceRecoveryResult,
)
from eval_factory.trace.parsing.repair import (
    INVALID_BACKSLASH_RULE_ID,
    INVALID_BACKSLASH_TRANSFORM,
    REPAIR_POLICY_VERSION,
    LocalRepairResult,
    RepairEdit,
    repair_invalid_json_string_backslashes,
    repaired_boundary_to_source,
)
from eval_factory.trace.parsing.streaming import RecoveryUnitKind

__all__ = [
    "CAPABILITY_ORDER",
    "INVALID_BACKSLASH_RULE_ID",
    "INVALID_BACKSLASH_TRANSFORM",
    "RECOVERY_POLICY_VERSION",
    "REPAIR_POLICY_VERSION",
    "AdmissionMode",
    "CuratedTrajectoryV1Parser",
    "CuratedTrajectoryV1Recovery",
    "LocalRepairResult",
    "NestedFieldStatus",
    "OuterCoordinateError",
    "ParseDiagnostic",
    "ParsedNestedField",
    "ParsedRawTrajRecord",
    "RawTrajParseOutcome",
    "RawTrajParseResult",
    "RawTrajRecovery",
    "RawTrajV1Parser",
    "RecoveredJsonUnit",
    "RecoveredNestedField",
    "RecoveredRawTrajRecord",
    "RecoveryFieldStatus",
    "RecoveryUnitKind",
    "RegisteredSourceMismatchError",
    "RepairEdit",
    "RuntimeSnapshotV1Parser",
    "TraceAdmissionDecision",
    "TraceCapabilityName",
    "TraceParseError",
    "TraceRecoveryResult",
    "TraceUse",
    "evaluate_admission",
    "repair_invalid_json_string_backslashes",
    "repaired_boundary_to_source",
]
