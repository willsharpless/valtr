# from valtr.passes import normalize_nary_bool, combine_multi_finally, combine_multi_globally
import ipdb

from valtr.ir_builder import IRBuilder
from valtr.ir_graphviz import visualize_ir
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser


def main():
    # Assume `lexer` is from your previous implementation
    # source = "(F_[3,5] (p && X q && r) -> G r U_[1,2] s) || (!p && F q)"
    # source = "F a && F b && F c && G( F d && F e && F a ) && G h && G j"
    # source = "F a && F b && F G c && G d  && G e && G f"
    source = "F a && F b && F c && G d && G e && G f"

    lexer = TLLexer()
    tokens = list(lexer.tokenize(source))
    ast = TLParser(tokens).parse()

    print("Original:")
    print(source)
    print()

    print("AST:")
    print(ast)
    print()

    print("IR:")
    builder = IRBuilder()
    lowerer = Lowerer(builder=builder)
    root_id = lowerer.lower(ast)

    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    # passes = [PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(builder)
        root_id, builder = p.run(root_id)

    dot = visualize_ir(builder, root_id, filename="ir_graph", view=True)

    # print("Normalized:")
    # ast = normalize_nary_bool(ast)
    # print(ast)
    # print()
    #
    # print("Combined MultiFinally:")
    # ast = combine_multi_finally(ast)
    # print(ast)
    # print()
    #
    # print("Combined Globally:")
    # ast = combine_multi_globally(ast)
    # print(ast)
    # print()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
