"""Dependency rules from ARCHITECTURE.md, checked on the source code. Run: pytest tests/test_architecture.py"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEON_ALLOWED = {"leon", "numpy", "regex", "torch"}  # plus the standard library


def imported_modules(path: Path) -> set[str]:
    """Top-level names of every module imported by a Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_leon_depends_only_on_stdlib_numpy_regex_and_torch() -> None:
    """Rule 1: the package never imports phases, apps or tests, nor third-party libraries beyond these."""
    import sys

    for path in (ROOT / "leon").glob("*.py"):
        foreign = imported_modules(path) - LEON_ALLOWED - sys.stdlib_module_names
        assert not foreign, f"{path.name} imports {foreign}"


def test_early_phases_are_self_contained() -> None:
    """Rule 3: phases 0 and 1 are standalone prototypes and must not depend on leon."""
    for phase in ("0_bigram", "1_gpt"):
        for path in (ROOT / "phases" / phase).glob("*.py"):
            assert "leon" not in imported_modules(path), f"{path.relative_to(ROOT)} imports leon"


def test_later_phases_do_not_import_each_other() -> None:
    """Rule 2: shared code goes to leon, never from one phase folder to another."""
    sibling_modules = {
        path.stem for phase in (ROOT / "phases").iterdir() if phase.is_dir() for path in phase.glob("*.py")
    }
    for phase in sorted((ROOT / "phases").iterdir()):
        if not phase.is_dir() or phase.name in ("0_bigram", "1_gpt"):
            continue
        own = {path.stem for path in phase.glob("*.py")}
        for path in phase.glob("*.py"):
            borrowed = (imported_modules(path) & sibling_modules) - own
            assert not borrowed, f"{path.relative_to(ROOT)} imports {borrowed} from another phase"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
