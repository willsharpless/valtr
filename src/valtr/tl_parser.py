from enum import Enum, auto
from typing import Iterable, List, Optional, Tuple

from attrs import define, frozen

from valtr.lexer import Position, Span, Token
from valtr.tl_lexer import TokenType


@define(slots=True)
class Node:
    """Base class for all AST nodes."""

    span: Span

    def __str__(self):
        return prettyprint(self)


@define(slots=True)
class Interval:
    lo: int | None
    hi: int | None
    span: Span | None = None


@define(slots=True)
class ConstNode(Node):
    value: bool


@define(slots=True)
class Identifier(Node):
    name: str


class UnaryOpKind(Enum):
    NOT = auto()
    FINALLY = auto()
    GLOBALLY = auto()
    NEXT = auto()
    FORALL_PATH = auto()
    EXISTS_PATH = auto()

    @staticmethod
    def from_token_type(ttype: TokenType) -> "UnaryOpKind":
        mapping = {
            TokenType.NOT: UnaryOpKind.NOT,
            TokenType.FINALLY: UnaryOpKind.FINALLY,
            TokenType.GLOBALLY: UnaryOpKind.GLOBALLY,
            TokenType.NEXT: UnaryOpKind.NEXT,
            TokenType.FORALL_PATH: UnaryOpKind.FORALL_PATH,
            TokenType.EXISTS_PATH: UnaryOpKind.EXISTS_PATH,
        }
        if ttype in mapping:
            return mapping[ttype]
        raise ValueError(f"Invalid TokenType for UnaryOpKind: {ttype}")


class BinaryOpKind(Enum):
    AND = auto()
    OR = auto()
    IMPLIES = auto()
    UNTIL = auto()
    RELEASE = auto()

    @staticmethod
    def from_token_type(ttype: TokenType) -> "BinaryOpKind":
        mapping = {
            TokenType.AND: BinaryOpKind.AND,
            TokenType.OR: BinaryOpKind.OR,
            TokenType.IMPLIES: BinaryOpKind.IMPLIES,
            TokenType.UNTIL: BinaryOpKind.UNTIL,
            TokenType.RELEASE: BinaryOpKind.RELEASE,
        }
        if ttype in mapping:
            return mapping[ttype]
        raise ValueError(f"Invalid TokenType for BinaryOpKind: {ttype}")


@define(slots=True)
class UnaryOp(Node):
    op: UnaryOpKind
    operand: Node
    interval: Interval | None  # For timed temporal ops


@define(slots=True)
class BinaryOp(Node):
    op: BinaryOpKind
    left: Node
    right: Node
    interval: Interval | None  # For timed temporal ops


# @define(slots=True)
# class NaryBool(Node):
#     kind: BinaryOpKind  # either and or or.
#     children: List[Node]  # at least 2 terms expected
#
#
# @define(slots=True)
# class MultiFinally(Node):
#     """Represents a conjunction (AND) of multiple Finally operators."""
#
#     operands: List[Node]
#     intervals: List[Interval | None]
#
#
# @define(slots=True)
# class MultiUntil(Node):
#     """Represents a conjunction (AND) of multiple Finally operators."""
#
#     lefts: List[Node]
#     rights: List[Node]
#     intervals: List[Interval | None]


class ParseError(SyntaxError):
    pass


def _join_span(a: Span, b: Span) -> Span:
    return Span(start=a.start, end=b.end)


class TLParser:
    """
    Pratt parser that produces an AST of Node subclasses:
      - Identifier
      - UnaryOp
      - BinaryOp
      - Interval (attached as field)

    Supports:
        Prefix: !, A, E, G, F, X (each optionally with _[lo,hi] where supported)
        Infix:  U, R (each optionally with _[lo,hi]), &&, ||, ->
        Grouping: ( expr )
        Primaries: identifiers
    """

    _INFIX_BP: dict[TokenType, Tuple[int, int, str]] = {
        TokenType.UNTIL: (60, 60, "left"),
        TokenType.RELEASE: (60, 60, "left"),
        TokenType.AND: (50, 50, "left"),
        TokenType.OR: (40, 40, "left"),
        TokenType.IMPLIES: (30, 29, "right"),
    }

    _PREFIX_BP = 70

    def __init__(self, tokens: Iterable[Token]):
        self._tokens: List[Token] = list(tokens)
        if not self._tokens or self._tokens[-1].type is not None:
            self._tokens.append(self._make_eof(self._tokens[-1].span if self._tokens else None))
        self._i = 0

    # ---------- Public API ----------

    def parse(self) -> Node:
        expr = self._expression(0)
        self._expect(None)  # EOF
        return expr

    # ---------- Pratt core ----------

    def _expression(self, rbp: int) -> Node:
        t = self._advance()
        left = self._nud(t)
        while True:
            tok = self._peek()
            if tok.type not in self._INFIX_BP:
                break
            lbp, rbp_infix, _assoc = self._INFIX_BP[tok.type]
            if lbp < rbp:
                break
            op_tok = self._advance()
            interval = self._maybe_interval_suffix() if op_tok.type in (TokenType.UNTIL, TokenType.RELEASE) else None
            right = self._expression(rbp_infix)
            op = BinaryOpKind.from_token_type(op_tok.type)
            left = BinaryOp(op=op, left=left, right=right, interval=interval, span=_join_span(left.span, right.span))
        return left

    # ---------- prefix / primary ----------

    def _nud(self, tok: Token) -> Node:
        tt = tok.type
        if tt is None:
            self._error(tok, "Unexpected end of input")

        if tt == TokenType.LPAREN:
            expr = self._expression(0)
            self._expect(TokenType.RPAREN)
            return expr

        if tt == TokenType.ID:
            return Identifier(name=tok.value, span=tok.span)

        if tt in (
            TokenType.NOT,
            TokenType.GLOBALLY,
            TokenType.FINALLY,
            TokenType.NEXT,
            TokenType.FORALL_PATH,
            TokenType.EXISTS_PATH,
        ):
            interval = (
                self._maybe_interval_suffix() if tt in (TokenType.GLOBALLY, TokenType.FINALLY, TokenType.NEXT) else None
            )
            operand = self._expression(self._PREFIX_BP)
            op = UnaryOpKind.from_token_type(tt)
            return UnaryOp(op=op, operand=operand, interval=interval, span=_join_span(tok.span, operand.span))

        self._error(tok, f"Unexpected token {tt.name if tt else 'EOF'}")

    # ---------- interval suffix ----------

    def _maybe_interval_suffix(self) -> Optional[Interval]:
        if not self._match(TokenType.UNDERSCORE):
            return None
        start_tok = self._previous()

        self._expect(TokenType.LBRACKET)
        lo_tok = self._expect(TokenType.INT)
        self._expect(TokenType.COMMA)
        hi_tok = self._expect(TokenType.INT)
        rbrack = self._expect(TokenType.RBRACKET)

        lo, hi = int(lo_tok.value), int(hi_tok.value)
        if hi < lo:
            self._error(hi_tok, f"Invalid interval: hi ({hi}) < lo ({lo})")

        span = Span(start=start_tok.span.start, end=rbrack.span.end)
        return Interval(lo=lo, hi=hi, span=span)

    # ---------- token utilities ----------

    def _peek(self) -> Token:
        return self._tokens[self._i]

    def _previous(self) -> Token:
        return self._tokens[self._i - 1]

    def _advance(self) -> Token:
        tok = self._tokens[self._i]
        self._i += 1
        return tok

    def _match(self, ttype: Optional[TokenType]) -> bool:
        if self._peek().type == ttype:
            self._advance()
            return True
        return False

    def _expect(self, ttype: Optional[TokenType]) -> Token:
        tok = self._peek()
        if tok.type == ttype:
            return self._advance()
        want = "EOF" if ttype is None else ttype.name
        got = "EOF" if tok.type is None else tok.type.name
        self._error(tok, f"Expected {want}, got {got}")

    def _error(self, tok: Token, msg: str) -> None:
        pos = tok.span.start
        raise ParseError(f"{msg} at line {pos.lineno}, col {pos.col}")

    @staticmethod
    def _make_eof(after_span: Optional[Span]) -> Token:
        if after_span is None:
            zero = Position(lineno=1, col=1, index=0)
            span = Span(start=zero, end=zero)
        else:
            span = Span(start=after_span.end, end=after_span.end)
        return Token(type=None, value="", span=span)  # type: ignore[arg-type]


_INDENT = " " * 4


def _interval_suffix(iv: Optional[Interval]) -> str:
    if iv is None:
        return ""
    lo = "-inf" if iv.lo is None else str(iv.lo)
    hi = "inf" if iv.hi is None else str(iv.hi)
    return f"_[{lo},{hi}]"


def _unary_label(tt: UnaryOpKind, iv: Interval | None) -> str:
    if tt == UnaryOpKind.NOT:
        return "NOT"
    # temporal unary: prefer the letter form
    if tt == UnaryOpKind.FINALLY:
        base = "F"
    elif tt == UnaryOpKind.GLOBALLY:
        base = "G"
    elif tt == UnaryOpKind.NEXT:
        base = "X"
    elif tt == UnaryOpKind.FORALL_PATH:
        base = "A"
    elif tt == UnaryOpKind.EXISTS_PATH:
        base = "E"
    else:
        base = tt.name
    return f"{base}{_interval_suffix(iv)}"


def _binary_label(tt: BinaryOpKind, iv: Interval | None) -> str:
    # boolean infix use enum names
    if tt in (BinaryOpKind.AND, BinaryOpKind.OR, BinaryOpKind.IMPLIES):
        return tt.name
    # temporal infix: show letter + interval if present, else enum name
    if tt == BinaryOpKind.UNTIL:
        return f"U{_interval_suffix(iv) or ''}" if iv else "U"
    if tt == BinaryOpKind.RELEASE:
        return f"R{_interval_suffix(iv) or ''}" if iv else "R"
    return tt.name


def _pp(node: Node, level: int, out: List[str]) -> None:
    indent = _INDENT * level

    if isinstance(node, Identifier):
        out.append(f"{indent}{node.name}")
        return

    if isinstance(node, UnaryOp):
        label = _unary_label(node.op, node.interval)
        # inline if child is identifier
        if isinstance(node.operand, Identifier):
            out.append(f"{indent}{label} {node.operand.name}")
        else:
            out.append(f"{indent}{label}")
            _pp(node.operand, level + 1, out)
        return

    if isinstance(node, BinaryOp):
        label = _binary_label(node.op, node.interval)
        out.append(f"{indent}{label}")
        _pp(node.left, level + 1, out)
        _pp(node.right, level + 1, out)
        return

    # if isinstance(node, NaryBool):
    #     # N-ary AND/OR
    #     label = node.kind.name + "^{}".format(len(node.children))
    #     out.append(f"{indent}{label}")
    #     for c in node.children:
    #         _pp(c, level + 1, out)
    #     return

    if isinstance(node, Interval):
        # Intervals are embedded on ops; printing standalone is unusual.
        out.append(f"{indent}[{node.lo},{node.hi}]")
        return

    # if isinstance(node, MultiFinally):
    #     out.append(f"{indent}FINALLY^{len(node.operands)}")
    #     # Render as a list of F operands (inline for identifiers)
    #     for operand, iv in zip(node.operands, node.intervals):
    #         suffix = "" if iv is None else f"_[{iv.lo},{iv.hi}]"
    #         if isinstance(operand, Identifier):
    #             out.append(f"{indent}{' ' * 4}F{suffix} {operand.name}")
    #         else:
    #             out.append(f"{indent}{' ' * 4}F{suffix}")
    #             _pp(operand, level + 2, out)
    #     return

    # Fallback for unexpected node kinds
    out.append(f"{indent}{type(node).__name__}")


def prettyprint(node: Node) -> str:
    """
    Return a formatted string representation of the AST.
    """
    lines: List[str] = []
    _pp(node, 0, lines)
    return "\n".join(lines)
