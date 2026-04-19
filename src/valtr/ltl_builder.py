from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .ltl_ir import (
    And,
    Const,
    ExistsPaths,
    Expr,
    ExprId,
    Finally,
    ForAllPaths,
    Globally,
    Interval,
    Next,
    Not,
    Origin,
    Or,
    Release,
    SourceRef,
    Until,
    Var,
)


class LTLBuilder:
    def __init__(self) -> None:
        self.nodes: list[Expr] = []
        self._intern: dict[tuple[Any, ...], ExprId] = {}

    def _intern_node(self, key: tuple[Any, ...], node: Expr) -> ExprId:
        found = self._intern.get(key)
        if found is not None:
            existing = self.nodes[int(found)]
            if existing.origin != node.origin:
                # Preserve all source contributions when structurally identical nodes merge.
                merged_origin = Origin.derived(
                    "intern_merge",
                    existing.origin,
                    node.origin,
                    primary_span=existing.origin.primary_span or node.origin.primary_span,
                )
                self.nodes[int(found)] = replace(existing, origin=merged_origin)
            return found
        node_id = ExprId(len(self.nodes))
        self.nodes.append(node)
        self._intern[key] = node_id
        return node_id

    def _key(self, tag: str, *parts: Any) -> tuple[Any, ...]:
        return (tag, *parts)

    def _combine_origins(self, rule: str, *expr_ids: ExprId, fallback: Origin) -> Origin:
        origins = tuple(self.nodes[int(expr_id)].origin for expr_id in expr_ids)
        if not origins:
            return fallback
        return Origin.derived(rule, *origins, primary_span=fallback.primary_span)

    def _order_nary(self, expr_ids: Iterable[ExprId]) -> tuple[ExprId, ...]:
        return tuple(sorted(expr_ids, key=int))

    def _flatten_and(self, expr_ids: Iterable[ExprId]) -> list[ExprId]:
        out: list[ExprId] = []
        for expr_id in expr_ids:
            node = self.nodes[int(expr_id)]
            if isinstance(node, And):
                out.extend(node.args)
            else:
                out.append(expr_id)
        return out

    def _flatten_or(self, expr_ids: Iterable[ExprId]) -> list[ExprId]:
        out: list[ExprId] = []
        for expr_id in expr_ids:
            node = self.nodes[int(expr_id)]
            if isinstance(node, Or):
                out.extend(node.args)
            else:
                out.append(expr_id)
        return out

    def _dedupe(self, expr_ids: Iterable[ExprId]) -> tuple[ExprId, ...]:
        seen: set[int] = set()
        out: list[ExprId] = []
        for expr_id in expr_ids:
            if int(expr_id) in seen:
                continue
            seen.add(int(expr_id))
            out.append(expr_id)
        return tuple(out)

    def const(self, value: bool, origin: Origin) -> ExprId:
        return self._intern_node(self._key("Const", value), Const(origin=origin, value=value))

    def true(self, origin: Origin) -> ExprId:
        return self.const(True, origin)

    def false(self, origin: Origin) -> ExprId:
        return self.const(False, origin)

    def var(self, name: str, origin: Origin) -> ExprId:
        return self._intern_node(self._key("Var", name), Var(origin=origin, name=name))

    def not_(self, arg: ExprId, origin: Origin) -> ExprId:
        arg_node = self.nodes[int(arg)]
        if isinstance(arg_node, Const):
            return self.const(
                not arg_node.value,
                Origin.derived("not_const_fold", arg_node.origin, primary_span=origin.primary_span),
            )
        if isinstance(arg_node, Not):
            return arg_node.arg
        return self._intern_node(self._key("Not", int(arg)), Not(origin=origin, arg=arg))

    def and_(self, args: Iterable[ExprId], origin: Origin) -> ExprId:
        flat = self._flatten_and(args)
        filtered: list[ExprId] = []
        changed = False
        for expr_id in flat:
            node = self.nodes[int(expr_id)]
            if isinstance(node, Const):
                if not node.value:
                    return self.false(Origin.derived("and_false", origin, node.origin, primary_span=origin.primary_span))
                changed = True
                continue
            filtered.append(expr_id)
        deduped = self._dedupe(self._order_nary(filtered))
        if not deduped:
            return self.true(Origin.derived("and_empty", origin, primary_span=origin.primary_span))
        if len(deduped) == 1:
            return deduped[0]
        node_origin = (
            self._combine_origins("and", *deduped, fallback=origin)
            if changed or len(flat) != len(deduped)
            else origin
        )
        return self._intern_node(
            self._key("And", tuple(int(expr_id) for expr_id in deduped)),
            And(origin=node_origin, args=deduped),
        )

    def or_(self, args: Iterable[ExprId], origin: Origin) -> ExprId:
        flat = self._flatten_or(args)
        filtered: list[ExprId] = []
        changed = False
        for expr_id in flat:
            node = self.nodes[int(expr_id)]
            if isinstance(node, Const):
                if node.value:
                    return self.true(Origin.derived("or_true", origin, node.origin, primary_span=origin.primary_span))
                changed = True
                continue
            filtered.append(expr_id)
        deduped = self._dedupe(self._order_nary(filtered))
        if not deduped:
            return self.false(Origin.derived("or_empty", origin, primary_span=origin.primary_span))
        if len(deduped) == 1:
            return deduped[0]
        node_origin = (
            self._combine_origins("or", *deduped, fallback=origin)
            if changed or len(flat) != len(deduped)
            else origin
        )
        return self._intern_node(
            self._key("Or", tuple(int(expr_id) for expr_id in deduped)),
            Or(origin=node_origin, args=deduped),
        )

    def next(self, arg: ExprId, interval: Interval | None, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("Next", int(arg), interval),
            Next(origin=origin, arg=arg, interval=interval),
        )

    def forall_paths(self, arg: ExprId, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("ForAllPaths", int(arg)),
            ForAllPaths(origin=origin, arg=arg),
        )

    def exists_paths(self, arg: ExprId, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("ExistsPaths", int(arg)),
            ExistsPaths(origin=origin, arg=arg),
        )

    def finally_(self, arg: ExprId, interval: Interval | None, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("Finally", int(arg), interval),
            Finally(origin=origin, arg=arg, interval=interval),
        )

    def globally(self, arg: ExprId, interval: Interval | None, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("Globally", int(arg), interval),
            Globally(origin=origin, arg=arg, interval=interval),
        )

    def until(self, left: ExprId, right: ExprId, interval: Interval | None, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("Until", int(left), int(right), interval),
            Until(origin=origin, left=left, right=right, interval=interval),
        )

    def release(self, left: ExprId, right: ExprId, interval: Interval | None, origin: Origin) -> ExprId:
        return self._intern_node(
            self._key("Release", int(left), int(right), interval),
            Release(origin=origin, left=left, right=right, interval=interval),
        )

    def implies(self, left: ExprId, right: ExprId, origin: Origin) -> ExprId:
        not_left_origin = Origin.derived(
            "implies_not_left",
            origin,
            self.nodes[int(left)].origin,
            primary_span=origin.primary_span,
        )
        return self.or_(
            (self.not_(left, not_left_origin), right),
            Origin.derived(
                "implies_to_or",
                origin,
                self.nodes[int(right)].origin,
                primary_span=origin.primary_span,
            ),
        )


def ast_origin(node_id: int, span) -> Origin:
    return Origin.leaf(span=span, source=SourceRef(phase="ast", node_id=node_id))
