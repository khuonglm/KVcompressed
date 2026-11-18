# Coding standards

Rules for `src/kvsel`. `uv run pytest` checks the ones marked *checked* (`tests/test_standards.py`, and ruff with the
rule set in `pyproject.toml`).

1. **Library code doesn't exit.** A stop for action raises `kvsel.protocol.Refusal`; only `kvsel.cli` turns it into an
   exit, with code 2. *Checked.*
2. **Records and item files are written atomically**, through `kvsel.files`: to a temporary file in the same folder,
   flushed to disk, then renamed into place. An interrupted write leaves no file.
3. **Pure modules stay pure.** `selectors`, `analysis.statistics` and `models.layout` touch no model and no files, so
   they test in milliseconds. *Checked.*
4. **Each frozen rule cites its PREREG section in its docstring**, for example "PREREG 1.6: the first 1,000 eligible
   items ...". A reviewer checks the code against that section.
5. **No private imports across modules, and no import cycles**, not even through an import inside a function. A name
   another module needs is public. *Checked.*
6. **One name per meaning.** One word for one concept across modules and command options (`GLOSSARY.md`); one default
   per option meaning (the surface rule is `capitalized` wherever a command takes one, `tests/test_cli.py`).
7. **No wrapper that only forwards.** Call the function itself.

Lint (*checked*): unused imports and variables, import order, and modern syntax (ruff's `F`, `I` and `UP` rules).
