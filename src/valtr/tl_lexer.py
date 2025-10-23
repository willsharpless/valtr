import re
from enum import Enum, auto
from typing import Iterable, Iterator, Optional

from attrs import define

from valtr.lexer import Lexer


class TokenType(Enum):
    # Operators
    AND = auto()  # &&
    OR = auto()  # ||
    NOT = auto()  # !
    IMPLIES = auto()  # ->

    # Keyword-backed temporal/logical ops (matched as keywords, not regex rules)
    GLOBALLY = auto()  # G
    FINALLY = auto()  # F
    NEXT = auto()  # X
    UNTIL = auto()  # U
    RELEASE = auto()  # R

    # Punctuation
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()  # ,
    UNDERSCORE = auto()  # _

    # Literals
    INT = auto()

    # Identifiers
    ID = auto()


RULES = [
    # Operators (multi-char first)
    (TokenType.AND, r"\&\&"),
    (TokenType.OR, r"\|\|"),
    (TokenType.IMPLIES, r"\->"),
    (TokenType.NOT, r"\!"),
    # Punctuation / brackets
    (TokenType.LPAREN, r"\("),
    (TokenType.RPAREN, r"\)"),
    (TokenType.LBRACKET, r"\["),
    (TokenType.RBRACKET, r"\]"),
    (TokenType.COMMA, r","),
    # Single underscore as its own token
    (TokenType.UNDERSCORE, r"_"),
    (TokenType.INT, r"\d+"),
    # Identifiers
    (TokenType.ID, r"[A-Za-z_][A-Za-z0-9_]*"),
]

KEYWORDS = [
    (TokenType.GLOBALLY, r"G"),
    (TokenType.FINALLY, r"F"),
    (TokenType.NEXT, r"X"),
    (TokenType.UNTIL, r"U"),
    (TokenType.RELEASE, r"R"),
]


class TLLexer(Lexer):
    def __init__(self):
        super().__init__(tok_cls=TokenType, rules=RULES, keywords=KEYWORDS)
