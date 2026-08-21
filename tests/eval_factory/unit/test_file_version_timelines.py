from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.safety import OriginClass
from eval_factory.contracts.trace import Completeness, FileObservation, FileOperation
from eval_factory.provenance import (
    FileVersionOrigin,
    FileVersionTimelineBuilder,
    FileVersionUncertainty,
)

HASH = "a" * 64
TRACE_ID = "trace-ir://timeline"


def _audit(created_at: datetime = datetime(2026, 7, 23, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="timeline-test",
        governing_versions=(VersionBinding(component="file-version-timeline", version="r2-02"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _observation(
    *,
    path: str,
    operation: FileOperation,
    sequence: int,
    completeness: Completeness = Completeness.COMPLETE,
    truncated: bool | None = False,
    content: bool = True,
) -> FileObservation:
    content_ref = _ref("content-blob", f"content://{path}/{sequence}") if content else None
    return FileObservation(
        observation_id=f"file-observation://{path.replace('/', '-')}/{sequence}",
        trace_ir_version_id=TRACE_ID,
        logical_path=path,
        raw_path_ref=_ref("content-blob", f"raw-path://{path}"),
        operation=operation,
        sequence=sequence,
        observed_start=None,
        observed_end=None,
        completeness=completeness,
        truncated=truncated,
        content_ref=content_ref,
        content_sha256=content_ref.object_sha256 if content_ref is not None else None,
        file_version_id=f"file-version://r1/{path.replace('/', '-')}/{sequence}",
        source_event_refs=(_ref("trace-event", f"trace-event://{path}/{sequence}"),),
    )


def _build(*observations: FileObservation):
    return FileVersionTimelineBuilder().build(
        trace_ir_version_id=TRACE_ID,
        file_observations=observations,
        audit=_audit(),
    )


def test_read_before_write_then_read_after_write_separates_versions() -> None:
    timelines = _build(
        _observation(
            path="a.txt", operation=FileOperation.READ, sequence=1, completeness=Completeness.PARTIAL
        ),
        _observation(path="a.txt", operation=FileOperation.WRITE, sequence=2),
        _observation(
            path="a.txt", operation=FileOperation.READ, sequence=3, completeness=Completeness.PARTIAL
        ),
    )

    assert len(timelines) == 1
    versions = timelines[0].versions
    assert [version.version_index for version in versions] == [0, 1]
    assert versions[0].origin is FileVersionOrigin.PREEXISTING_WORKSPACE_INPUT
    assert versions[0].origin_class is OriginClass.PREEXISTING_WORKSPACE_INPUT
    assert versions[0].creating_observation_ref is None
    assert [ref.object_id for ref in versions[0].read_observation_refs] == ["file-observation://a.txt/1"]
    assert versions[1].origin is FileVersionOrigin.AGENT_GENERATED
    assert versions[1].origin_class is OriginClass.AGENT_GENERATED_INTERMEDIATE
    assert versions[1].creating_observation_ref is not None
    assert [ref.object_id for ref in versions[1].read_observation_refs] == ["file-observation://a.txt/3"]


def test_first_observation_write_keeps_v0_structural_only() -> None:
    timeline = _build(_observation(path="created.txt", operation=FileOperation.WRITE, sequence=1))[0]

    assert [version.version_index for version in timeline.versions] == [0, 1]
    assert timeline.versions[0].origin is FileVersionOrigin.UNKNOWN_PREEXISTING
    assert FileVersionUncertainty.NO_PRE_MUTATION_READ in timeline.versions[0].uncertainties
    assert timeline.versions[0].read_observation_refs == ()
    assert timeline.versions[1].origin is FileVersionOrigin.AGENT_GENERATED


def test_edit_after_read_creates_post_write_agent_version() -> None:
    timeline = _build(
        _observation(path="edit.txt", operation=FileOperation.READ, sequence=1),
        _observation(path="edit.txt", operation=FileOperation.EDIT, sequence=2),
    )[0]

    assert len(timeline.versions) == 2
    assert timeline.versions[0].origin is FileVersionOrigin.PREEXISTING_WORKSPACE_INPUT
    assert timeline.versions[1].operation is FileOperation.EDIT
    assert timeline.versions[1].origin is FileVersionOrigin.AGENT_GENERATED


def test_multiple_writes_create_monotonic_versions() -> None:
    timeline = _build(
        _observation(path="out.txt", operation=FileOperation.WRITE, sequence=1),
        _observation(path="out.txt", operation=FileOperation.WRITE, sequence=2),
        _observation(path="out.txt", operation=FileOperation.EDIT, sequence=3),
    )[0]

    assert [version.version_index for version in timeline.versions] == [0, 1, 2, 3]
    assert [version.operation for version in timeline.versions[1:]] == [
        FileOperation.WRITE,
        FileOperation.WRITE,
        FileOperation.EDIT,
    ]


def test_unknown_mutation_creates_uncertain_boundary() -> None:
    timeline = _build(
        _observation(path="maybe.txt", operation=FileOperation.READ, sequence=1),
        _observation(
            path="maybe.txt",
            operation=FileOperation.UNKNOWN_MUTATION,
            sequence=2,
            completeness=Completeness.UNKNOWN,
            content=False,
        ),
        _observation(path="maybe.txt", operation=FileOperation.READ, sequence=3),
    )[0]

    assert len(timeline.versions) == 2
    assert timeline.versions[1].origin is FileVersionOrigin.UNKNOWN_BOUNDARY
    assert FileVersionUncertainty.UNKNOWN_MUTATION_BOUNDARY in timeline.versions[1].uncertainties
    assert [ref.object_id for ref in timeline.versions[1].read_observation_refs] == [
        "file-observation://maybe.txt/3"
    ]


def test_partial_truncated_and_missing_content_preserve_uncertainty() -> None:
    timeline = _build(
        _observation(
            path="partial.txt",
            operation=FileOperation.READ,
            sequence=1,
            completeness=Completeness.PARTIAL,
            truncated=True,
            content=False,
        )
    )[0]

    version = timeline.versions[0]
    assert FileVersionUncertainty.PARTIAL_OBSERVATION in version.uncertainties
    assert FileVersionUncertainty.TRUNCATED_OBSERVATION in version.uncertainties
    assert FileVersionUncertainty.CONTENT_NOT_OBSERVED in version.uncertainties


def test_groups_paths_independently() -> None:
    timelines = _build(
        _observation(path="b.txt", operation=FileOperation.WRITE, sequence=1),
        _observation(path="a.txt", operation=FileOperation.READ, sequence=2),
    )

    assert [timeline.logical_path for timeline in timelines] == ["a.txt", "b.txt"]
    assert timelines[0].versions[0].origin is FileVersionOrigin.PREEXISTING_WORKSPACE_INPUT
    assert timelines[1].versions[0].origin is FileVersionOrigin.UNKNOWN_PREEXISTING


def test_timeline_ids_ignore_audit_time() -> None:
    observation = _observation(path="stable.txt", operation=FileOperation.READ, sequence=1)
    first = FileVersionTimelineBuilder().build(
        trace_ir_version_id=TRACE_ID,
        file_observations=(observation,),
        audit=_audit(datetime(2026, 7, 23, tzinfo=UTC)),
    )[0]
    second = FileVersionTimelineBuilder().build(
        trace_ir_version_id=TRACE_ID,
        file_observations=(observation,),
        audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)),
    )[0]

    assert first.timeline_id == second.timeline_id
    assert [version.version_id for version in first.versions] == [
        version.version_id for version in second.versions
    ]
    assert first.canonical_sha256() != second.canonical_sha256()


def test_timeline_ids_are_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "timeline_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.trace import Completeness, FileObservation, FileOperation
from eval_factory.provenance import FileVersionTimelineBuilder

def ref(kind, object_id):
    return ObjectRef(object_type=kind, object_id=object_id, object_version='v1', object_sha256='a' * 64)

audit = ContractAudit(
    created_at=datetime(2026, 7, 23, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='file-version-timeline', version='r2-02'),),
)
obs = FileObservation(
    observation_id='file-observation://seed/1',
    trace_ir_version_id='trace-ir://timeline',
    logical_path='seed.txt',
    raw_path_ref=ref('content-blob', 'raw-path://seed'),
    operation=FileOperation.READ,
    sequence=1,
    observed_start=None,
    observed_end=None,
    completeness=Completeness.PARTIAL,
    truncated=False,
    content_ref=ref('content-blob', 'content://seed'),
    content_sha256='a' * 64,
    file_version_id='file-version://r1/seed/1',
    source_event_refs=(ref('trace-event', 'trace-event://seed/1'),),
)
timeline = FileVersionTimelineBuilder().build(
    trace_ir_version_id='trace-ir://timeline',
    file_observations=(obs,),
    audit=audit,
)[0]
print(timeline.timeline_id)
print(timeline.versions[0].version_id)
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        result = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={**dict(PYTHONHASHSEED=seed), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1


def test_timeline_models_reject_raw_text_fields() -> None:
    timeline = _build(_observation(path="strict.txt", operation=FileOperation.READ, sequence=1))[0]
    with pytest.raises(ValidationError):
        timeline.__class__.model_validate(
            {
                **timeline.model_dump(mode="json"),
                "raw_trace_text": "forbidden",
            }
        )
