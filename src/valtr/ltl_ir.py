from __future__ import annotations

from dataclasses import dataclass, field

from .lexer import Span


class ExprId(int):
    pass


@dataclass(frozen=True, slots=True)
class Interval:
    lo: int | None
    hi: int | None


@dataclass(frozen=True, slots=True)
class SourceRef:
    phase: str
    node_id: int


@dataclass(frozen=True, slots=True)
class Origin:
    primary_span: Span | None
    spans: tuple[Span, ...] = ()
    sources: frozenset[SourceRef] = field(default_factory=frozenset)
    rule: str | None = None

    @staticmethod
    def leaf(span: Span | None, source: SourceRef) -> Origin:
        spans = () if span is None else (span,)
        return Origin(primary_span=span, spans=spans, sources=frozenset({source}))

    @staticmethod
    def derived(rule: str, *origins: Origin, primary_span: Span | None = None) -> Origin:
        spans = tuple(span for origin in origins for span in origin.spans)
        if primary_span is None:
            primary_span = next((origin.primary_span for origin in origins if origin.primary_span is not None), None)
        return Origin(
            primary_span=primary_span,
            spans=spans,
            sources=frozenset(source for origin in origins for source in origin.sources),
            rule=rule,
        )


@dataclass(frozen=True, slots=True)
class Expr:
    origin: Origin

    def children(self) -> tuple[ExprId, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: bool


@dataclass(frozen=True, slots=True)
class Var(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Not(Expr):
    arg: ExprId

    def children(self) -> tuple[ExprId, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class And(Expr):
    args: tuple[ExprId, ...]

    def children(self) -> tuple[ExprId, ...]:
        return self.args


@dataclass(frozen=True, slots=True)
class Or(Expr):
    args: tuple[ExprId, ...]

    def children(self) -> tuple[ExprId, ...]:
        return self.args


@dataclass(frozen=True, slots=True)
class Next(Expr):
    arg: ExprId
    interval: Interval | None = None

    def children(self) -> tuple[ExprId, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Finally(Expr):
    arg: ExprId
    interval: Interval | None = None

    def children(self) -> tuple[ExprId, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Globally(Expr):
    arg: ExprId
    interval: Interval | None = None

    def children(self) -> tuple[ExprId, ...]:
        return (self.arg,)


@dataclass(frozen=True, slots=True)
class Until(Expr):
    left: ExprId
    right: ExprId
    interval: Interval | None = None

    def children(self) -> tuple[ExprId, ...]:
        return (self.left, self.right)


@dataclass(frozen=True, slots=True)
class Release(Expr):
    left: ExprId
    right: ExprId
    interval: Interval | None = None

    def children(self) -> tuple[ExprId, ...]:
        return (self.left, self.right)


TEMPORAL_EXPR_TYPES = (Next, Finally, Globally, Until, Release)


def is_temporal_expr(node: Expr) -> bool:
    return isinstance(node, TEMPORAL_EXPR_TYPES)
