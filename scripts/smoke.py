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
    builder, root = lower_spec("F a && F b")

    plot_dir = get_plot_dir()
    dot = visualize_ltl_ir(builder.nodes, root, filename=plot_dir / "smoke_test.pdf")

    assert "And (2)" in dot.source
    assert "Finally" in dot.source

    pre_builder, pre_root = lower_spec("G(q_1 U r_1) && G(q_2 U r_2) && q_3 U r_3 && q_4 U r_4 && G q_5 && G q_6")
    result = run_default_ltl_passes(pre_builder, pre_root)
    rendered = pretty_ltl(result.builder.nodes, result.root)
    assert "U" in rendered
    assert "G (" in rendered or "G(" in rendered
    assert "q_5" in rendered and "q_6" in rendered

    fg_builder, fg_root = lower_spec("G(q_1 U r_1) && G(q_2 U r_2) && q_3 U r_3 && q_4 U r_4 && G q_5 && F G q_6")
    fg_result = run_default_ltl_passes(fg_builder, fg_root)
    fg_rendered = pretty_ltl(fg_result.builder.nodes, fg_result.root)
    assert "q_6" in fg_rendered
    assert "U" in fg_rendered
    assert "G (" in fg_rendered or "G(" in fg_rendered


if __name__ == "__main__":
    main()
