from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .lexer import Span
from .ltl_ir import (
    And,
    Const,
    ExistsPaths,
    Expr,
    ExprId,
    Finally,
    ForAllPaths,
    Globally,
    Next,
    Not,
    Or,
    Release,
    Until,
    Var,
)


RootKind = Literal["either", "state", "path"]


@dataclass(frozen=True, slots=True)
class CTLStarClassification:
    is_state_formula: bool
    is_path_formula: bool


class CTLStarValidationError(SyntaxError):
    def __init__(self, message: str, *, span: Span | None = None) -> None:
        if span is not None:
            pos = span.start
            message = f"{message} at line {pos.lineno}, col {pos.col}"
        super().__init__(message)
        self.span = span


class CTLStarValidator:
    def __init__(self, nodes: list[Expr]) -> None:
        self.nodes = nodes
        self._memo: dict[int, CTLStarClassification] = {}

    def classify(self, expr_id: ExprId) -> CTLStarClassification:
        idx = int(expr_id)
        if idx in self._memo:
            return self._memo[idx]

        node = self.nodes[idx]
        if isinstance(node, (Const, Var)):
            result = CTLStarClassification(is_state_formula=True, is_path_formula=True)
        elif isinstance(node, Not):
            arg = self.classify(node.arg)
            result = CTLStarClassification(
                is_state_formula=arg.is_state_formula,
                is_path_formula=arg.is_path_formula,
            )
        elif isinstance(node, (And, Or)):
            child_info = [self.classify(child) for child in node.args]
            result = CTLStarClassification(
                is_state_formula=all(info.is_state_formula for info in child_info),
                is_path_formula=all(info.is_path_formula for info in child_info),
            )
        elif isinstance(node, (Next, Finally, Globally)):
            arg = self.classify(node.arg)
            result = CTLStarClassification(
                is_state_formula=False,
                is_path_formula=arg.is_path_formula,
            )
        elif isinstance(node, (Until, Release)):
            left = self.classify(node.left)
            right = self.classify(node.right)
            result = CTLStarClassification(
                is_state_formula=False,
                is_path_formula=left.is_path_formula and right.is_path_formula,
            )
        elif isinstance(node, (ForAllPaths, ExistsPaths)):
            arg = self.classify(node.arg)
            result = CTLStarClassification(
                is_state_formula=arg.is_path_formula,
                is_path_formula=arg.is_path_formula,
            )
        else:
            raise CTLStarValidationError(
                f"Unsupported formula node for CTL* validation: {type(node).__name__}",
                span=node.origin.primary_span,
            )

        self._memo[idx] = result
        return result

    def validate(self, root: ExprId, *, root_kind: RootKind = "either") -> CTLStarClassification:
        result = self.classify(root)
        node = self.nodes[int(root)]
        if root_kind == "state" and not result.is_state_formula:
            raise CTLStarValidationError(
                f"Expected a CTL* state formula at the root, got {type(node).__name__}",
                span=node.origin.primary_span,
            )
        if root_kind == "path" and not result.is_path_formula:
            raise CTLStarValidationError(
                f"Expected a CTL* path formula at the root, got {type(node).__name__}",
                span=node.origin.primary_span,
            )
        if root_kind == "either" and not (result.is_state_formula or result.is_path_formula):
            raise CTLStarValidationError(
                f"Formula is not well-formed CTL*: {type(node).__name__}",
                span=node.origin.primary_span,
            )
        return result


def validate_ctl_star(
    nodes: list[Expr],
    root: ExprId,
    *,
    root_kind: RootKind = "either",
) -> CTLStarClassification:
    return CTLStarValidator(nodes).validate(root, root_kind=root_kind)


def contains_path_quantifier(nodes: list[Expr], root: ExprId) -> bool:
    seen: set[int] = set()
    stack = [int(root)]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        node = nodes[idx]
        if isinstance(node, (ForAllPaths, ExistsPaths)):
            return True
        stack.extend(int(child) for child in node.children())
    return False
