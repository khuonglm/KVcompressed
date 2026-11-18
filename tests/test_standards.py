"""CODING_STANDARDS.md, the parts a machine can check: lint, import cycles, private imports, library exits, pure
modules. Each check also runs once on a planted violation, to show that it can fail."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "src" / "kvsel"
PURE = ("kvsel.selectors", "kvsel.analysis.statistics", "kvsel.models.layout")  # no model, no files
PURE_IMPORTS = {"__future__", "collections", "collections.abc", "dataclasses", "functools", "math", "numpy", "pandas", "scipy"}


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def modules(root: Path) -> dict[str, ast.Module]:
    return {module_name(path, root): ast.parse(path.read_text()) for path in sorted(root.rglob("*.py"))}


def imports(name: str, tree: ast.Module, known: set[str], is_package: bool) -> list[tuple[str, list[str]]]:
    """(module, names) for every import anywhere in the module, function bodies included, relative ones resolved."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = name if is_package else name.rpartition(".")[0]
            for _ in range(node.level - 1):
                base = base.rpartition(".")[0]
            target = f"{base}.{node.module}" if node.level and node.module else (base if node.level else node.module)
            names = [alias.name for alias in node.names]
            submodules = [f"{target}.{n}" for n in names if f"{target}.{n}" in known]
            found += [(sub, []) for sub in submodules]
            if len(submodules) < len(names):
                found.append((target, [n for n in names if f"{target}.{n}" not in known]))
    return found


def graph(root: Path) -> tuple[dict[str, ast.Module], dict[str, list[tuple[str, list[str]]]]]:
    trees = modules(root)
    known = set(trees)
    packages = {module_name(p, root) for p in root.rglob("__init__.py")}
    return trees, {name: imports(name, tree, known, name in packages) for name, tree in trees.items()}


def cycles(edges: dict[str, list[tuple[str, list[str]]]]) -> list[list[str]]:
    found, state = [], {}

    def visit(node, path):
        state[node] = "open"
        for target, _ in edges.get(node, []):
            if target not in edges or target == node:
                continue
            if state.get(target) == "open":
                found.append(path[path.index(target):] + [target])
            elif target not in state:
                visit(target, [*path, target])
        state[node] = "done"

    for node in edges:
        if node not in state:
            visit(node, [node])
    return found


def private_imports(edges) -> list[str]:
    return [f"{name} imports {target}.{n}" for name, targets in edges.items() for target, names in targets
            if target in edges and target != name for n in names if n.startswith("_") and not n.startswith("__")]


def exits(trees: dict[str, ast.Module], allowed: str) -> list[str]:
    """Raising SystemExit or calling sys.exit outside the command line."""
    found = []
    for name, tree in trees.items():
        if name == allowed:
            continue
        for node in ast.walk(tree):
            raised = isinstance(node, ast.Raise) and node.exc is not None and "SystemExit" in ast.unparse(node.exc)
            called = isinstance(node, ast.Call) and ast.unparse(node.func) in ("sys.exit", "exit")
            if raised or called:
                found.append(f"{name}:{node.lineno}")
    return found


def impure(edges, pure: tuple[str, ...]) -> list[str]:
    return [f"{name} imports {target}" for name in pure for target, _ in edges.get(name, [])
            if target not in PURE_IMPORTS and target not in pure]


def plant(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "kvsel"
    for relative, text in {"__init__.py": "", **files}.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    return root


# --- The package -----------------------------------------------------------------------------------------------


def test_lint_passes():
    ruff = shutil.which("ruff")
    assert ruff, "ruff is a dev dependency: uv sync"
    found = subprocess.run([ruff, "check", "src", "tests", "scripts"], cwd=ROOT, capture_output=True, text=True)
    assert found.returncode == 0, found.stdout


def test_no_import_cycles():
    assert cycles(graph(PACKAGE)[1]) == []


def test_no_module_imports_another_modules_private_names():
    assert private_imports(graph(PACKAGE)[1]) == []


def test_only_the_command_line_exits():
    assert exits(graph(PACKAGE)[0], "kvsel.cli") == []


def test_pure_modules_touch_no_model_and_no_files():
    assert impure(graph(PACKAGE)[1], PURE) == []


# --- Each check fails on a planted violation ------------------------------------------------------------------


def test_lint_flags_an_unused_import(tmp_path):
    (tmp_path / "bad.py").write_text("import os\n")
    found = subprocess.run([shutil.which("ruff"), "check", "--config", str(ROOT / "pyproject.toml"), "bad.py"],
                           cwd=tmp_path, capture_output=True, text=True)
    assert found.returncode == 1 and "F401" in found.stdout


@pytest.mark.parametrize("where", ["top", "function"])
def test_a_planted_cycle_is_found_even_inside_a_function(tmp_path, where):
    hidden = "def f():\n    from . import a\n" if where == "function" else "from . import a\n"
    root = plant(tmp_path, {"a.py": "from .b import x\n", "b.py": hidden + "x = 1\n"})
    assert cycles(graph(root)[1]) == [["kvsel.a", "kvsel.b", "kvsel.a"]]


def test_a_planted_private_import_and_exit_and_impurity_are_found(tmp_path):
    root = plant(tmp_path, {"a.py": "def _helper():\n    raise SystemExit(1)\n", "b.py": "from .a import _helper\n",
                            "pure.py": "import torch\n"})
    trees, edges = graph(root)
    assert private_imports(edges) == ["kvsel.b imports kvsel.a._helper"]
    assert exits(trees, "kvsel.cli") == ["kvsel.a:2"]
    assert impure(edges, ("kvsel.pure",)) == ["kvsel.pure imports torch"]
