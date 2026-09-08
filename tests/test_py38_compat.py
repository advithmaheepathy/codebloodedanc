"""Python 3.8 compatibility guards.

The Jetson Orin NX we deploy to runs JetPack 5.1.3, which is Ubuntu 20.04 with
**Python 3.8.10**. Two classes of syntax that work fine on the development host fail
there, and neither is caught by importing the package on a newer interpreter:

1. **Builtin generics in pydantic model fields.** ``from __future__ import annotations``
   defers annotation evaluation for ordinary functions, but pydantic evaluates model
   field annotations at runtime to build the schema. ``list[str]`` in a model field
   therefore raises ``TypeError: 'type' object is not subscriptable`` on 3.8 even with
   the future import. ``typing.List[str]`` is required.

2. **PEP 604 unions** (``int | None``) anywhere that gets evaluated at runtime.

These tests scan the source rather than relying on the interpreter, so they fail on the
development machine where the mistake is cheap to fix instead of on the board.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "anc_defence"
BUILTIN_GENERICS = ("list", "dict", "tuple", "set", "frozenset", "type")


def python_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def _pydantic_model_classes(tree: ast.Module) -> list[ast.ClassDef]:
    """Classes that look like pydantic models, including our ``_Base`` subclasses."""
    model_bases = {"BaseModel", "_Base"}
    found: list[ast.ClassDef] = []
    # Two passes: pick up direct subclasses, then subclasses of those.
    for _ in range(3):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node in found:
                continue
            names = {b.id for b in node.bases if isinstance(b, ast.Name)}
            names |= {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
            if names & model_bases:
                found.append(node)
                model_bases.add(node.name)
    return found


def _uses_builtin_generic(annotation: ast.expr) -> list[str]:
    """Builtin generic subscripts anywhere inside an annotation expression."""
    offenders: list[str] = []
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in BUILTIN_GENERICS:
                offenders.append(node.value.id)
    return offenders


def test_pydantic_fields_avoid_builtin_generics():
    """Model fields must use typing.List/Dict/Tuple, not list/dict/tuple."""
    problems: list[str] = []
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover
            problems.append(f"{path.name}: syntax error {exc}")
            continue
        for cls in _pydantic_model_classes(tree):
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
                    for name in _uses_builtin_generic(stmt.annotation):
                        target = getattr(stmt.target, "id", "?")
                        problems.append(
                            f"{path.name}:{stmt.lineno} {cls.name}.{target} uses "
                            f"'{name}[...]' - pydantic evaluates this at runtime and it "
                            f"raises on Python 3.8. Use typing.{name.capitalize()}[...]."
                        )
    assert not problems, "Python 3.8 incompatible pydantic annotations:\n" + "\n".join(problems)


def test_no_pep604_unions_in_runtime_annotations():
    """``X | Y`` in a pydantic field or a class attribute breaks on Python 3.8."""
    problems: list[str] = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in _pydantic_model_classes(tree):
            for stmt in cls.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
                    for node in ast.walk(stmt.annotation):
                        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                            problems.append(
                                f"{path.name}:{stmt.lineno} {cls.name} uses PEP 604 'X | Y'; "
                                "use typing.Optional/Union for Python 3.8."
                            )
    assert not problems, "PEP 604 unions evaluated at runtime:\n" + "\n".join(problems)


def test_modules_with_annotations_have_future_import():
    """Any module using builtin generics in signatures needs the future import."""
    problems: list[str] = []
    pattern = re.compile(r":\s*(list|dict|tuple|set)\[|->\s*(list|dict|tuple|set)\[")
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        if pattern.search(text) and "from __future__ import annotations" not in text:
            problems.append(
                f"{path.name} uses builtin generic annotations without "
                "'from __future__ import annotations'"
            )
    assert not problems, "missing future imports:\n" + "\n".join(problems)


def test_no_walrus_free_syntax_errors_under_38_grammar():
    """Every module must parse. Catches accidental 3.9+/3.10+ only syntax."""
    for path in python_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"{path}: {exc}")


def test_config_loads_and_round_trips():
    """The real schema must build, which is what would fail first on the board."""
    from anc_defence.config import Config, load_config

    cfg = Config()
    assert cfg.pipeline.order == "dfn_then_normalise"
    assert cfg.normalise.mode == "agc"
    assert cfg.audio.sample_rate == 48000

    default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    if default.is_file():
        loaded = load_config([default], ["normalise.target_dbfs=-23", "plain.per_cell=2"])
        assert loaded.normalise.target_dbfs == -23.0
        assert loaded.plain.per_cell == 2
        # Every field must survive a serialise/parse round trip.
        assert Config(**loaded.to_plain()).to_plain() == loaded.to_plain()


def test_unknown_config_keys_are_rejected():
    """A typo must fail loudly rather than being silently ignored."""
    from anc_defence.config import ConfigError, load_config

    with pytest.raises(ConfigError) as exc:
        load_config([], ["normalise.targt_dbfs=-23"])
    assert "targt_dbfs" in str(exc.value) or "Extra inputs" in str(exc.value)
