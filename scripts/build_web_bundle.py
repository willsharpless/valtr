#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "valtr"
DOCS = ROOT / "docs"
PY_ROOT = DOCS / "py"
VALTR_OUT = PY_ROOT / "valtr"

DEFAULT_SPEC = "F target_a && F target_b && G !wall"

VALTR_MODULES = [
    "dag_mermaid.py",
    "dag_passes.py",
    "dag_viz_style.py",
    "ir.py",
    "ir_builder.py",
    "ir_pass.py",
    "ir_rewriter.py",
    "lexer.py",
    "lowering.py",
    "reachability.py",
    "tl_lexer.py",
    "tl_parser.py",
    "valtr.py",
]

ROOT_MODULES = {
    "attrs.py": '''
from dataclasses import MISSING, dataclass, field as _dc_field


def define(_cls=None, *, slots=False, frozen=False, **kwargs):
    def wrap(cls):
        return dataclass(cls, slots=slots, frozen=frozen)
    return wrap if _cls is None else wrap(_cls)


def frozen(_cls=None, **kwargs):
    return define(_cls, frozen=True, **kwargs)


def field(*, eq=True, default=MISSING, default_factory=MISSING, **kwargs):
    params = {"compare": eq}
    if default is not MISSING:
        params["default"] = default
    if default_factory is not MISSING:
        params["default_factory"] = default_factory
    return _dc_field(**params)
'''.lstrip(),
    "ipdb.py": "def set_trace(*args, **kwargs):\n    return None\n",
    "loguru.py": '''
class _Logger:
    def debug(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None


logger = _Logger()
'''.lstrip(),
    "runner.py": '''
from valtr.dag_mermaid import render_dag_mermaid
from valtr.valtr import to_dag


def build_mermaid(spec: str, *, vertical: bool = False) -> str:
    dag, root = to_dag(spec)
    direction = "TD" if vertical else "LR"
    return render_dag_mermaid(dag, root, direction=direction)
'''.lstrip(),
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _copy_valtr_sources() -> dict[str, str]:
    bundle: dict[str, str] = {}
    init_text = '"""Minimal valtr package for the browser bundle."""\n\n__all__ = []\n'
    _write(VALTR_OUT / "__init__.py", init_text)
    bundle["valtr/__init__.py"] = init_text

    for module_name in VALTR_MODULES:
        text = (SRC / module_name).read_text()
        _write(VALTR_OUT / module_name, text)
        bundle[f"valtr/{module_name}"] = text

    return bundle


def _write_root_modules(bundle: dict[str, str]) -> None:
    for filename, text in ROOT_MODULES.items():
        _write(PY_ROOT / filename, text)
        bundle[filename] = text


def _render_default_graphs() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from valtr.dag_mermaid import render_dag_mermaid
    from valtr.valtr import to_dag

    dag, root = to_dag(DEFAULT_SPEC)
    (DOCS / "default-graph.mmd").write_text(render_dag_mermaid(dag, root, direction="LR"))
    (DOCS / "default-graph-vertical.mmd").write_text(render_dag_mermaid(dag, root, direction="TD"))


def main() -> None:
    PY_ROOT.mkdir(parents=True, exist_ok=True)
    VALTR_OUT.mkdir(parents=True, exist_ok=True)

    bundle = _copy_valtr_sources()
    _write_root_modules(bundle)
    (PY_ROOT / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))

    _render_default_graphs()
    print(f"Wrote web bundle to {PY_ROOT / 'bundle.json'}")


if __name__ == "__main__":
    main()
