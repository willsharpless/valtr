from valtr.ltl_lowering import ASTToLTLLowerer
from valtr.ltl_ir_graphviz import visualize_ltl_ir
from valtr.ltl_pass_runner import run_default_ltl_passes
from valtr.ltl_pretty import pretty_ltl
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.path_util import get_plot_dir


def lower_spec(spec: str):
    lexer = TLLexer()
    tokens = list(lexer.tokenize(spec))
    ast = TLParser(tokens).parse()
    lowerer = ASTToLTLLowerer()
    return lowerer.builder, lowerer.lower(ast)


def main():
    builder, root = lower_spec("GF a && F b")

    plot_dir = get_plot_dir()
    dot = visualize_ltl_ir(builder.nodes, root, filename=plot_dir / "smoke_test.pdf")


if __name__ == "__main__":
    main()
