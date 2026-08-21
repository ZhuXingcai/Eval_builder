"""Trace source adapters."""

from eval_factory.trace.adapters.base import TraceAdapter
from eval_factory.trace.adapters.curated_trajectory_v1 import (
    CURATED_TRAJECTORY_V1_REQUIRED_FIELDS,
    CuratedTrajectoryV1Adapter,
)
from eval_factory.trace.adapters.raw_traj_v1 import (
    RAW_TRAJ_V1_REQUIRED_FIELDS,
    RawTrajV1Adapter,
)
from eval_factory.trace.adapters.runtime_snapshot_v1 import (
    RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS,
    RuntimeSnapshotV1Adapter,
)

__all__ = [
    "CURATED_TRAJECTORY_V1_REQUIRED_FIELDS",
    "RAW_TRAJ_V1_REQUIRED_FIELDS",
    "RUNTIME_SNAPSHOT_V1_REQUIRED_FIELDS",
    "CuratedTrajectoryV1Adapter",
    "RawTrajV1Adapter",
    "RuntimeSnapshotV1Adapter",
    "TraceAdapter",
]
