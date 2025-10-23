import re
from enum import Enum, auto
from typing import Iterable, Iterator, List, Tuple, Optional


class Token(object):
    '''
    Representation of a single token.
    '''
    __slots__ = ('type', 'value', 'lineno', 'index', 'end')
    def __repr__(self):
        return f'Token(type={self.type!r}, value={self.value!r}, lineno={self.lineno}, index={self.index}, end={self.end})'


class TokenType(Enum):
    # Operators
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()

    # Keywords / temporal logic ops (matched via keywords list)
    GLOBALLY = auto()
    FINALLY = auto()
    NEXT = auto()
    UNTIL = auto()
    RELEASE = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    UNDERSCORE = auto()

    # Literals
    INT = auto()   # <-- new integer token

    # Identifiers
    ID = auto()


class LexError(SyntaxError):
    pass


class Lexer:
    """
    Lexer that:
      * Uses a single master regex with named groups built from (TokenType, regex) pairs.
      * Accepts keyword patterns [(TokenType, regex)] which are tested against ID values
        with fullmatch; on match, the token type is remapped to the keyword’s TokenType.
    """

    def __init__(
        self,
        rules: Iterable[Tuple[TokenType, str]],
        keywords: Optional[Iterable[Tuple[TokenType, str]]] = None
    ):
        self.keywords: List[Tuple[TokenType, re.Pattern]] = [
            (tok, re.compile(rx)) for tok, rx in (keywords or [])
        ]

        parts: List[str] = []
        for tok_type, rx in rules:
            parts.append(f"(?P<{tok_type.name}>{rx})")

        # Whitespace/newline/mismatch channels
        parts.append(r"(?P<SKIP>[ \t]+)")
        parts.append(r"(?P<NEWLINE>\n+)")
        parts.append(r"(?P<MISMATCH>.)")

        self.master_pat = re.compile("|".join(parts))

    def tokenize(self, text: str) -> Iterator[Token]:
        lineno = 1

        for m in self.master_pat.finditer(text):
            kind = m.lastgroup
            value = m.group()
            start = m.start()
            end = m.end()

            if kind == 'NEWLINE':
                lineno += value.count('\n')
                continue
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                snippet = text[start:start+20].replace('\n', '\\n')
                raise LexError(f"Illegal character {value!r} at line {lineno}, index {start}. Next text: {snippet!r}")

            tok_type = TokenType[kind]

            # Apply keyword regexes only for IDs
            if tok_type == TokenType.ID:
                for kw_type, kw_re in self.keywords:
                    if kw_re.fullmatch(value):
                        tok_type = kw_type
                        break

            tok = Token()
            tok.type = tok_type
            tok.value = value
            tok.lineno = lineno
            tok.index = start
            tok.end = end
            yield tok


# ---------------- Example setup ----------------
# NOTE: Order matters. Put UNDERSCORE and INT before ID so '_' and digits are not eaten by ID.
token_specs = [
    # Operators
    (TokenType.AND,       r"\&\&"),
    (TokenType.OR,        r"\|\|"),
    (TokenType.IMPLIES,   r"\->"),
    (TokenType.NOT,       r"\!"),

    # Punctuation
    (TokenType.LPAREN,    r"\("),
    (TokenType.RPAREN,    r"\)"),
    (TokenType.LBRACKET,  r"\["),
    (TokenType.RBRACKET,  r"\]"),
    (TokenType.COMMA,     r","),

    # Singles / literals
    (TokenType.UNDERSCORE, r"_"),
    (TokenType.INT,        r"\d+"),  # integers only

    # Identifiers
    (TokenType.ID,         r"[A-Za-z_][A-Za-z0-9_]*"),
]

# Keywords as (TokenType, regex) applied to ID tokens.
# You can expand these to words like r"(?:Globally|G)" if desired.
keyword_specs = [
    (TokenType.GLOBALLY, r"G"),
    (TokenType.FINALLY,  r"F"),
    (TokenType.NEXT,     r"X"),
    (TokenType.UNTIL,    r"U"),
    (TokenType.RELEASE,  r"R"),
]

lexer = Lexer(token_specs, keywords=keyword_specs)

src = "F_[3, 5] && G(a) || X b -> U_[10,10] R_[2,7] _ id42"
for t in lexer.tokenize(src):
    print(t)