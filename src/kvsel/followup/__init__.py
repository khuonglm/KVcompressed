"""Follow-ups to the confirmatory run (PREREG.md, section 2.4). Not confirmatory; no claims.

This is a subpackage on purpose. The stage-3 record hashes `src/kvsel/*.py` (not subfolders), so
adding follow-up code here leaves the confirmatory code hash unchanged and TextVQA can still run under
the same frozen code. Nothing here replaces or changes a module the confirmatory run used: it imports
the frozen engine, selectors, runner and analysis helpers as they are.

Run with `python -m kvsel.followup run ...` and `python -m kvsel.followup report ...`.
"""
