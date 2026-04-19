from valtr.ltl_lowering import ASTToLTLLowerer
from valtr.ltl_ir_graphviz import visualize_ltl_ir
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser
from valtr.util.path_util import get_plot_dir


def main():
    spec = "F a && F b"
    lexer = TLLexer()
    tokens = list(lexer.tokenize(spec))
    ast = TLParser(tokens).parse()
    lowerer = ASTToLTLLowerer()
    root = lowerer.lower(ast)
    assert lowerer.builder.nodes[int(root)].__class__.__name__ == "And"

    plot_dir = get_plot_dir()
    dot = visualize_ltl_ir(lowerer.builder.nodes, root, filename=plot_dir / "smoke_test.pdf")

    assert "And (2)" in dot.source
    assert "Finally" in dot.source


if __name__ == "__main__":
    main()
