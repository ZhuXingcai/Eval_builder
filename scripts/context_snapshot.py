from __future__ import annotations

import argparse

from env_mock_agent.context.checkpoint import write_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the canonical project-state and task handoff files.")
    parser.add_argument("--task-id")
    parser.add_argument("--validation", action="append")
    parser.add_argument("--failure", action="append")
    parser.add_argument("--next-command")
    args = parser.parse_args()
    state_path, handoff_path = write_checkpoint(
        task_id=args.task_id,
        validation_results=args.validation,
        current_failures=args.failure,
        next_command=args.next_command,
    )
    print(state_path)
    if handoff_path:
        print(handoff_path)


if __name__ == "__main__":
    main()
