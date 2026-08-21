from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, VersionBinding
from eval_factory.readiness.curated_trajectory_container import (
    CURATED_TRAJECTORY_REQUIRED_COLUMNS,
    CuratedTrajectoryContainerBuilder,
    CuratedTrajectoryContainerError,
)


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        created_by="curated-trajectory-container-test",
        governing_versions=(
            VersionBinding(
                component="curated-trajectory-container",
                version="curated-trajectory-container/r8-10-v1",
            ),
        ),
    )


def _trajectory(
    ordinal: int,
) -> str:
    return json.dumps(
        {
            "system": "context",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"request-{ordinal}",
                        }
                    ],
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _write_container(
    path: Path,
    *,
    rows: tuple[dict[str, str], ...],
    columns: tuple[str, ...] = CURATED_TRAJECTORY_REQUIRED_COLUMNS,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(
    ordinal: int,
    *,
    sid: str | None = None,
    trajectory: str | None = None,
) -> dict[str, str]:
    return {
        column: {
            "sid": sid or f"sid-{ordinal}",
            "event_time": "2026-06-29 00:00:00",
            "model": "model-a",
            "trajectory_完整轨迹(JSON)": trajectory or _trajectory(ordinal),
        }.get(column, f"metadata-{ordinal}")
        for column in CURATED_TRAJECTORY_REQUIRED_COLUMNS
    }


def test_container_materializes_exact_logical_member_bytes_and_replays(
    tmp_path: Path,
) -> None:
    container = tmp_path / "curated.csv"
    rows = tuple(_row(index) for index in range(3))
    _write_container(container, rows=rows)
    material_root = tmp_path / "material"
    builder = CuratedTrajectoryContainerBuilder()

    first = builder.compile(
        container_path=container,
        material_root=material_root,
        audit=_audit(),
        max_sources=10,
        max_member_bytes=1_000_000,
        max_total_member_bytes=10_000_000,
        expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
    )
    file_count = sum(path.is_file() for path in material_root.rglob("*"))
    second = builder.compile(
        container_path=container,
        material_root=material_root,
        audit=_audit(),
        max_sources=10,
        max_member_bytes=1_000_000,
        max_total_member_bytes=10_000_000,
        expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
    )

    assert second.inventory == first.inventory
    assert first.inventory.container_sha256 == hashlib.sha256(container.read_bytes()).hexdigest()
    assert first.inventory.member_count == 3
    assert first.inventory.unique_sid_count == 3
    assert first.inventory.unique_member_hash_count == 3
    assert sum(path.is_file() for path in material_root.rglob("*")) == file_count
    for row, member in zip(rows, first.inventory.members, strict=True):
        path = first.path_for(member.source_trace_id)
        expected = row["trajectory_完整轨迹(JSON)"].encode()
        assert path.read_bytes() == expected
        assert member.raw_sha256 == hashlib.sha256(expected).hexdigest()


def test_container_compiles_external_inventory_without_wrapper_hashes(
    tmp_path: Path,
) -> None:
    container = tmp_path / "curated.csv"
    rows = tuple(_row(index) for index in range(100))
    _write_container(container, rows=rows)
    builder = CuratedTrajectoryContainerBuilder()
    logical = builder.compile(
        container_path=container,
        material_root=tmp_path / "material",
        audit=_audit(),
        max_sources=100,
        max_member_bytes=1_000_000,
        max_total_member_bytes=100_000_000,
        expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
    )

    external = builder.compile_external_inventory(
        compilation=logical,
        audit=_audit(),
    )

    assert external.inventory.source_count == 100
    assert external.inventory.adapter_name == "curated_trajectory_v1"
    assert external.inventory.adapter_version == "1.0.0"
    assert {member.runtime.value for member in external.inventory.members} == {"claude_curated"}
    assert all(
        member.adapter_name == "curated_trajectory_v1"
        and member.raw_sha256
        == hashlib.sha256(external.path_for(member.source_trace_id).read_bytes()).hexdigest()
        for member in external.inventory.members
    )
    assert external.inventory.source_authorization_ref.object_sha256 == (
        hashlib.sha256(container.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-column",
        "missing-column",
        "duplicate-sid",
        "duplicate-member",
        "malformed-trajectory",
        "invalid-trajectory-shape",
        "invalid-message-shape",
        "symlink-container",
    ],
)
def test_container_rejects_non_authoritative_inputs(
    tmp_path: Path,
    mutation: str,
) -> None:
    container = tmp_path / "curated.csv"
    rows = [_row(0), _row(1)]
    columns = CURATED_TRAJECTORY_REQUIRED_COLUMNS
    if mutation == "unknown-column":
        columns = (*columns, "unknown")
        rows = [dict(row, unknown="closed") for row in rows]
    elif mutation == "missing-column":
        columns = tuple(column for column in columns if column != "model")
        rows = [{key: value for key, value in row.items() if key != "model"} for row in rows]
    elif mutation == "duplicate-sid":
        rows[1]["sid"] = rows[0]["sid"]
    elif mutation == "duplicate-member":
        rows[1]["trajectory_完整轨迹(JSON)"] = rows[0]["trajectory_完整轨迹(JSON)"]
    elif mutation == "malformed-trajectory":
        rows[1]["trajectory_完整轨迹(JSON)"] = "{"
    elif mutation == "invalid-trajectory-shape":
        rows[1]["trajectory_完整轨迹(JSON)"] = json.dumps({"messages": []})
    elif mutation == "invalid-message-shape":
        rows[1]["trajectory_完整轨迹(JSON)"] = json.dumps(
            {
                "system": "context",
                "messages": [{"role": "user"}],
            }
        )
    _write_container(container, rows=tuple(rows), columns=columns)
    active = container
    if mutation == "symlink-container":
        active = tmp_path / "linked.csv"
        active.symlink_to(container)

    material_root = tmp_path / "material"
    with pytest.raises(CuratedTrajectoryContainerError):
        CuratedTrajectoryContainerBuilder().compile(
            container_path=active,
            material_root=material_root,
            audit=_audit(),
            max_sources=10,
            max_member_bytes=1_000_000,
            max_total_member_bytes=10_000_000,
            expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
        )
    assert not material_root.exists() or not any(path.is_file() for path in material_root.rglob("*"))


def test_container_rejects_changed_expected_hash_and_material_corruption(
    tmp_path: Path,
) -> None:
    container = tmp_path / "curated.csv"
    _write_container(
        container,
        rows=(_row(0), _row(1)),
    )
    builder = CuratedTrajectoryContainerBuilder()
    material_root = tmp_path / "material"

    with pytest.raises(CuratedTrajectoryContainerError, match="hash"):
        builder.compile(
            container_path=container,
            material_root=material_root,
            audit=_audit(),
            max_sources=10,
            max_member_bytes=1_000_000,
            max_total_member_bytes=10_000_000,
            expected_container_sha256="0" * 64,
        )

    compilation = builder.compile(
        container_path=container,
        material_root=material_root,
        audit=_audit(),
        max_sources=10,
        max_member_bytes=1_000_000,
        max_total_member_bytes=10_000_000,
        expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
    )
    compilation.path_for(compilation.inventory.members[0].source_trace_id).write_bytes(b"corrupt")

    with pytest.raises(CuratedTrajectoryContainerError, match="material"):
        builder.compile(
            container_path=container,
            material_root=material_root,
            audit=_audit(),
            max_sources=10,
            max_member_bytes=1_000_000,
            max_total_member_bytes=10_000_000,
            expected_container_sha256=(hashlib.sha256(container.read_bytes()).hexdigest()),
        )
