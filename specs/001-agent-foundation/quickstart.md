# Quickstart: Agent Foundation

```bash
uv sync --extra dev
uv run envmock context bootstrap
uv run envmock doctor
uv run pytest
```

Normalize an LH package:

```bash
uv run envmock ingest \
  --adapter lh \
  --input <workspace>/private-inputs/LH_067_tfm_outsourcing_employee_survey
```

Live tests are opt-in:

```bash
uv run pytest -m live_claude_cli
uv run pytest -m live_claude_sdk
uv run pytest -m live_pi
uv run pytest -m live_ark
```
