from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_factory.agent_system.plan_review import PlanReviewService
from eval_factory.console_api.agent_app import create_agent_app
from eval_factory.console_api.agent_service import AgentShellService


def contract_document() -> dict[str, object]:
    plan_reviews = PlanReviewService.__new__(PlanReviewService)
    agent_shell = AgentShellService.__new__(AgentShellService)
    return create_agent_app(plan_reviews, agent_shell).openapi()


def contract_bytes() -> bytes:
    return (
        json.dumps(
            contract_document(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def write_contract(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract_bytes())


def check_contract(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.read_bytes() != contract_bytes():
        raise SystemExit(f"console OpenAPI contract drift: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        write_contract(arguments.path)
    else:
        check_contract(arguments.path)


if __name__ == "__main__":
    main()
