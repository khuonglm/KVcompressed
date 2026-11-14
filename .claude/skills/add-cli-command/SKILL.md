---
name: add-cli-command
description: Use when adding or changing a `kvsel` subcommand in src/kvsel/cli.py. Covers the file's conventions, the required tests and the README update.
---

# Adding a `kvsel` command

## Conventions in `src/kvsel/cli.py`

- One `cmd_<name>(args)` function per command. **Import heavy modules inside the function** (torch,
  transformers, the model) so `kvsel --help` and CPU-only commands stay fast.
- Register with `commands.add_parser("<name>", help="...")` and `sub.set_defaults(handler=cmd_<name>)`.
  The help text is lowercase, short, and says whether it needs the GPU.
- Reuse the helpers: `add_model_dir`, `add_model_options` (`--model-dir`, `--device`,
  `--head-precision`) and `add_item_options` (`--items`, `--images`). Load the model with `_bundle(args)`.
- Argument names must not collide with the dispatcher (`command`, `handler`).
- Commands that run items call `require_development(items)` first.
- Print a JSON summary with `json.dumps(..., indent=2)`; signal failure with `raise SystemExit(...)`
  carrying a message that says what to do next (for example "rerun the command to resume").
- Long jobs write results atomically and skip finished items, so they are safe to rerun.

## Tests (`tests/test_cli.py`)

- Add the name to the parametrized `test_every_command_parses_its_help` list.
- Add a dispatch test: monkeypatch the heavy function and call `run_cli(monkeypatch, "<name>", ...)`,
  asserting the arguments reach it. This catches argument-name bugs like the earlier report command.
- Run `uv run pytest tests/test_cli.py`, then the full suite.

## Docs

If users run the command, add it to `README.md` (GPU runbook step, or its own section) with a
copy-pasteable example using the real paths.
