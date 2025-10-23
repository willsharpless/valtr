import re
from enum import Enum, auto
from typing import Iterable, Iterator, Optional, Type, TypeVar

from attrs import define

TokenType = TypeVar("TokenType")


@define(slots=True)
class Position:
    lineno: int
    col: int
    index: int


@define(slots=True)
class Span:
    start: Position
    end: Position


@define(slots=True)
class Token:
    type: TokenType
    value: str
    span: Span

    def __repr__(self):
        return f"Token(type={self.type.name:8}, value={self.value!r}, start={self.span.start})"


class LexError(SyntaxError):
    pass


class Lexer:
    """
    Lexer that:
      * Takes token rules as a list of (TokenType, regex) and compiles a SINGLE master regex
        using named capture groups.
      * Accepts a 'keywords' dict to map identifier lexemes (like 'G','F','X','U','R') to
        token types (e.g., TokenType.GLOBALLY, ...).
      * Yields Token objects with (type, value, lineno, index, end).
    """

    _ID_FOLLOW = r"[A-Za-z0-9_]"  # identifier "following" char set

    def __init__(
        self,
        tok_cls: Type[TokenType],
        rules: Iterable[tuple[TokenType, str]],
        keywords: Iterable[tuple[TokenType, str]],
    ):
        self.tok_cls = tok_cls
        self.rules: list[tuple[TokenType, str]] = list(rules)
        self.keywords: list[tuple[TokenType, str]] = list(keywords or [])

        parts: list[str] = []

        # 1) Add KEYWORDS first, with a lookahead that allows '_' OR a non-ID follow.
        #    This prevents splitting 'Go' into 'G' + 'o', but allows 'G_' to split.
        #    We wrap the user-supplied keyword pattern in a non-capturing group.
        #    Lookahead: (?:(?=_)|(?![A-Za-z0-9_]))  -> either next is '_' OR next isn't an ID char.

        for tok_type, kw_rx in self.keywords:
            guarded_kw = f"(?:{kw_rx})(?:(?=_)|(?!{self._ID_FOLLOW}))"
            parts.append(f"(?P<{tok_type.name}>{guarded_kw})")

            # 2) Add the normal token rules.
            #    IMPORTANT: keep UNDERSCORE and INT before ID to avoid them being eaten by ID.
        for tok_type, rx in self.rules:
            parts.append(f"(?P<{tok_type.name}>{rx})")

            # 3) Whitespace/newline/mismatch channels
        parts.append(r"(?P<SKIP>[ \t]+)")
        parts.append(r"(?P<NEWLINE>\n+)")
        parts.append(r"(?P<MISMATCH>.)")

        self.master_pat = re.compile("|".join(parts))

    def tokenize(self, text: str) -> Iterator[Token]:
        lineno = 1
        line_start = 0

        for m in self.master_pat.finditer(text):
            kind = m.lastgroup
            value = m.group()
            start = m.start()
            end = m.end()

            if kind == "NEWLINE":
                # Advance line/column tracking
                newline_count = value.count("\n")
                lineno += newline_count
                line_start = end
                continue

            elif kind == "SKIP":
                continue
            elif kind == "MISMATCH":
                snippet = text[start : start + 20].replace("\n", "\\n")
                raise LexError(f"Illegal character {value!r} at line {lineno}, index {start}. Next text: {snippet!r}")

            tok_type = self.tok_cls[kind]

            # Compute start/end positions (end is exclusive).
            start_pos = Position(lineno=lineno, col=(start - line_start + 1), index=start)
            # Tokens here never span newlines because NEWLINE is consumed separately.
            end_pos = Position(lineno=lineno, col=(start_pos.col + (end - start)), index=end)
            span = Span(start=start_pos, end=end_pos)

            tok = Token(tok_type, value, span)
            yield tok
