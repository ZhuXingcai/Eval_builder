from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLIST_PATH = REPO_ROOT / "evals/golden/eval_factory/evaluation_items/v1/gold-worklist.json"
DEFAULT_OUTPUT = REPO_ROOT / "evals/golden/eval_factory/evaluation_items/v1/review-queue.json"

SUBMISSION_SLOTS = {
    "PARSER": (("parser-1", "AUDITOR"),),
    "SAFETY": (
        ("safety-1", "PROVENANCE_REVIEWER"),
        ("safety-2", "PROVENANCE_REVIEWER"),
    ),
    "LABEL": (("label-1", "LABEL_REVIEWER"),),
    "TASK": (("task-1", "TASK_REVIEWER"),),
    "ATTACHMENT": (("attachment-1", "ARTIFACT_REVIEWER"),),
    "RELEASE": (
        ("release-1", "ITEM_REVIEWER"),
        ("release-2", "ITEM_REVIEWER"),
    ),
}
ADJUDICATOR_ROLES = {
    "PARSER": "AUDITOR",
    "SAFETY": "PROVENANCE_REVIEWER",
    "LABEL": "LABEL_REVIEWER",
    "TASK": "TASK_REVIEWER",
    "ATTACHMENT": "ARTIFACT_REVIEWER",
    "RELEASE": "RELEASE_AUTHORITY",
}
PROJECTION_POLICIES = {
    "PARSER": "projection://privileged-parser-audit/v1",
    "SAFETY": "projection://privileged-provenance-audit/v1",
    "LABEL": "projection://structured-labeler/v1",
    "TASK": "projection://task-author-selection-context/v1",
    "ATTACHMENT": "projection://artifact-reviewer/v1",
    "RELEASE": "projection://release-authority-hash-only/v1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(instance_id: str, annotation_type: str, slot: str) -> str:
    return f"review-case://{instance_id}/{annotation_type.casefold()}/{slot}"


def build_queue(worklist: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for item in worklist["items"]:
        instance_id = item["instance_id"]
        source = {
            "source_ref": item["source_ref"],
            "source_sha256": item["source_sha256"],
        }
        adjudication_ids: dict[str, str] = {}
        for annotation_type, slots in SUBMISSION_SLOTS.items():
            submission_ids: list[str] = []
            for slot, role in slots:
                case_id = _case_id(instance_id, annotation_type, slot)
                submission_ids.append(case_id)
                cases.append(
                    {
                        "case_id": case_id,
                        "instance_id": instance_id,
                        "annotation_type": annotation_type,
                        "phase": "SUBMISSION",
                        "slot": slot,
                        "required_role": role,
                        "projection_policy": PROJECTION_POLICIES[annotation_type],
                        "source": source,
                        "depends_on": [],
                        "status": "PENDING",
                        "assignee": None,
                        "synthetic_identity_allowed": False,
                    }
                )
            adjudication_id = _case_id(instance_id, annotation_type, "adjudication")
            adjudication_ids[annotation_type] = adjudication_id
            cases.append(
                {
                    "case_id": adjudication_id,
                    "instance_id": instance_id,
                    "annotation_type": annotation_type,
                    "phase": "ADJUDICATION",
                    "slot": "adjudication",
                    "required_role": ADJUDICATOR_ROLES[annotation_type],
                    "projection_policy": PROJECTION_POLICIES[annotation_type],
                    "source": source,
                    "depends_on": submission_ids,
                    "status": "PENDING",
                    "assignee": None,
                    "synthetic_identity_allowed": False,
                }
            )
        release_case = next(case for case in cases if case["case_id"] == adjudication_ids["RELEASE"])
        release_case["depends_on"] = [
            *release_case["depends_on"],
            *[
                case_id
                for annotation_type, case_id in adjudication_ids.items()
                if annotation_type != "RELEASE"
            ],
        ]
    return {
        "schema_version": "eval-factory-gold-review-queue/v1",
        "status": "AWAITING_HUMAN_ASSIGNMENT",
        "worklist": {
            "path": str(WORKLIST_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(WORKLIST_PATH),
        },
        "annotation_contract": worklist["annotation_contract"],
        "cross_stage_contract": worklist["cross_stage_contract"],
        "case_count": len(cases),
        "synthetic_reviewer_identities_forbidden": True,
        "cases": cases,
    }


def _render(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"refusing to overwrite immutable review queue {path}; use a new version")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklist", type=Path, default=WORKLIST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    worklist = json.loads(args.worklist.read_text(encoding="utf-8"))
    write_immutable(args.output, _render(build_queue(worklist)))
    print(args.output)


if __name__ == "__main__":
    main()
