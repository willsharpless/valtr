from valtr.passes import normalize_nary_bool, combine_multi_finally, combine_multi_globally
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser


def main():
    # Assume `lexer` is from your previous implementation
    # source = "(F_[3,5] (p && X q && r) -> G r U_[1,2] s) || (!p && F q)"
    source = "F a && F b && F c && G( F d && F e ) && G h && G j"

    lexer = TLLexer()
    tokens = list(lexer.tokenize(source))
    ast = TLParser(tokens).parse()

    print("Original:")
    print(source)
    print()

    print("AST:")
    print(ast)
    print()

    print("Normalized:")
    ast = normalize_nary_bool(ast)
    print(ast)
    print()

    print("Combined MultiFinally:")
    ast = combine_multi_finally(ast)
    print(ast)
    print()

    print("Combined Globally:")
    ast = combine_multi_globally(ast)
    print(ast)
    print()


if __name__ == "__main__":
    main()
