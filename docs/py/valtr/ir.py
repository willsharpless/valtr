from enum import Enum, auto
from typing import Dict, List, NewType, Optional, Tuple

from attrs import define, field, frozen

from valtr.lexer import Span
from valtr.tl_parser import BinaryOpKind, UnaryOpKind


class IRId(int):
    pass


@frozen
class IntervalIR:
    lo: Optional[int]  # None => -inf
    hi: Optional[int]  # None => +inf


class UnaryIROpKind(Enum):
    NOT = auto()
    NEXT = auto()
    FINALLY = auto()
    GLOBALLY = auto()

    @staticmethod
    def from_token_type(ttype: UnaryOpKind) -> "UnaryIROpKind":
        mapping = {
            UnaryOpKind.NOT: UnaryIROpKind.NOT,
            UnaryOpKind.NEXT: UnaryIROpKind.NEXT,
            UnaryOpKind.FINALLY: UnaryIROpKind.FINALLY,
            UnaryOpKind.GLOBALLY: UnaryIROpKind.GLOBALLY,
        }
        if ttype in mapping:
            return mapping[ttype]
        raise ValueError(f"Invalid TokenType for UnaryOpKind: {ttype}")


class BinaryIROpKind(Enum):
    IMPLIES = auto()
    UNTIL = auto()
    RELEASE = auto()
    AND = auto()
    OR = auto()

    @staticmethod
    def from_token_type(ttype: BinaryOpKind) -> "BinaryIROpKind":
        mapping = {
            BinaryOpKind.UNTIL: BinaryIROpKind.UNTIL,
            BinaryOpKind.RELEASE: BinaryIROpKind.RELEASE,
            BinaryOpKind.IMPLIES: BinaryIROpKind.IMPLIES,
            BinaryOpKind.AND: BinaryIROpKind.AND,
            BinaryOpKind.OR: BinaryIROpKind.OR,
        }
        if ttype in mapping:
            return mapping[ttype]
        raise ValueError(f"Invalid TokenType for BinaryOpKind: {ttype}")


class NaryKind(Enum):
    AND = auto()
    OR = auto()

    UNTIL = auto()


@frozen
class IRNode:
    span: Span = field(eq=False)  # keep in node, exclude from equality/interning

    def children(self) -> List[IRId]:
        return []


@frozen
class Var(IRNode):
    name: str


@frozen
class ConstBool(IRNode):
    value: bool


@frozen
class Unary(IRNode):
    kind: UnaryIROpKind
    arg: IRId

    def children(self) -> List[IRId]:
        return [self.arg]


@frozen
class TemporalUnary(Unary):
    interval: IntervalIR | None = None


@frozen
class Binary(IRNode):
    kind: BinaryIROpKind
    left: IRId
    right: IRId

    def children(self) -> List[IRId]:
        return [self.left, self.right]


@frozen
class TemporalBinary(Binary):
    interval: Optional[IntervalIR] = None  # for UNTIL/RELEASE (timed)


@frozen
class Nary(IRNode):
    kind: NaryKind
    args: list[IRId]

    def children(self) -> List[IRId]:
        return self.args
