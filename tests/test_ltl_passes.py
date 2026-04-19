from valtr.tl_lexer import TLLexer
from valtr.ltl_pass_runner import run_default_ltl_passes
from valtr.ltl_pretty import pretty_ltl
from tests.ltl_test_utils import assert_ltl_equivalent, canonical_ltl_key, parse_ltl_spec


def rewrite_spec(spec: str):
    builder, root = parse_ltl_spec(spec)
    result = run_default_ltl_passes(builder, root)
    return result.builder, result.root


def test_canonical_ltl_key_is_order_invariant_for_and_or_only():
    lhs_builder, lhs_root = parse_ltl_spec("(a || b) || c")
    rhs_builder, rhs_root = parse_ltl_spec("c || a || b")
    assert canonical_ltl_key(lhs_builder.nodes, lhs_root) == canonical_ltl_key(rhs_builder.nodes, rhs_root)

    lhs_builder, lhs_root = parse_ltl_spec("a && b")
    rhs_builder, rhs_root = parse_ltl_spec("b && a")
    assert canonical_ltl_key(lhs_builder.nodes, lhs_root) == canonical_ltl_key(rhs_builder.nodes, rhs_root)

    lhs_builder, lhs_root = parse_ltl_spec("a U b")
    rhs_builder, rhs_root = parse_ltl_spec("b U a")
    assert canonical_ltl_key(lhs_builder.nodes, lhs_root) != canonical_ltl_key(rhs_builder.nodes, rhs_root)


def test_single_and_double_character_boolean_operators_parse_the_same():
    double_builder, double_root = parse_ltl_spec("(a && b) || (c && d)")
    single_builder, single_root = parse_ltl_spec("(a & b) | (c & d)")
    mixed_builder, mixed_root = parse_ltl_spec("(a && b) | (c & d)")

    double_key = canonical_ltl_key(double_builder.nodes, double_root)
    assert double_key == canonical_ltl_key(single_builder.nodes, single_root)
    assert double_key == canonical_ltl_key(mixed_builder.nodes, mixed_root)


def test_adjacent_temporal_keywords_parse_as_chained_operators():
    gf_builder, gf_root = parse_ltl_spec("GF a")
    spaced_gf_builder, spaced_gf_root = parse_ltl_spec("G F a")
    fg_builder, fg_root = parse_ltl_spec("FG a")
    spaced_fg_builder, spaced_fg_root = parse_ltl_spec("F G a")

    assert canonical_ltl_key(gf_builder.nodes, gf_root) == canonical_ltl_key(spaced_gf_builder.nodes, spaced_gf_root)
    assert canonical_ltl_key(fg_builder.nodes, fg_root) == canonical_ltl_key(spaced_fg_builder.nodes, spaced_fg_root)


def test_identifiers_are_not_split_just_because_they_start_with_temporal_letters():
    tokens = [(token.type.name, token.value) for token in TLLexer().tokenize("Go")]
    assert tokens == [("ID", "Go")]

    tokens = [(token.type.name, token.value) for token in TLLexer().tokenize("Fa")]
    assert tokens == [("ID", "Fa")]


def test_master_formula_stays_in_clean_factored_form():
    spec = "G(q_1 U r_1) && G(q_2 U r_2) && q_3 U r_3 && q_4 U r_4 && G q_5 && G q_6"

    builder, root = rewrite_spec(spec)
    rewritten = pretty_ltl(builder.nodes, root)

    q56 = "q_5 && q_6"
    qr1 = f"(({q56} && q_1) || ({q56} && r_1))"
    qr2 = f"(({q56} && q_2) || ({q56} && r_2))"
    g2 = f"G(({q56} && q_2) U ({q56} && r_2))"
    g1 = f"G(({q56} && q_1) U ({q56} && r_1))"
    branch_34_tail = f"r_4 && {g2} && {g1}"
    branch_43_tail = f"r_3 && {g2} && {g1}"
    left = f"{q56} && q_3 && {qr1} && q_4 && {qr2}"
    inner_left_34 = f"{q56} && {qr1} && q_4 && {qr2}"
    inner_left_43 = f"{q56} && q_3 && {qr1} && {qr2}"
    branch_34 = f"r_3 && (({inner_left_34}) U ({branch_34_tail}))"
    branch_43 = f"r_4 && (({inner_left_43}) U ({branch_43_tail}))"
    expected = f"({left}) U (({branch_34}) || ({branch_43}))"

    assert_ltl_equivalent(builder.nodes, root, expected, actual_rendered=rewritten)


def test_master_formula_with_fg_global_tail_rewrites_as_expected():
    spec = "G(q_1 U r_1) && G(q_2 U r_2) && q_3 U r_3 && q_4 U r_4 && G q_5 && F G q_6"

    builder, root = rewrite_spec(spec)
    rewritten = pretty_ltl(builder.nodes, root)

    q5 = "q_5"
    qr1 = f"(({q5} && q_1) || ({q5} && r_1))"
    qr2 = f"(({q5} && q_2) || ({q5} && r_2))"
    g2 = f"G(({q5} && q_2) U ({q5} && r_2))"
    g1_fg = f"G(({q5} && q_1 && q_6) U ({q5} && r_1 && q_6))"
    g1_fg_nested = f"G({q5} && (({q5} && q_1 && q_6) U ({q5} && r_1 && q_6)))"

    left = f"{q5} && q_4 && {qr1} && q_3 && {qr2}"
    inner_common = f"{q5} && {qr1} && {qr2}"
    inner_left_43 = f"{inner_common} && q_3"
    inner_left_34 = f"{q5} && q_4 && {qr1} && {qr2}"

    inner_43 = f"({inner_common}) U ({g2} && {g1_fg})"
    branch_4_then_3 = f"r_4 && (({inner_left_43}) U (r_3 && ({inner_43})))"

    inner_34 = f"({inner_common}) U ({g2} && {g1_fg_nested})"
    branch_3_then_4 = f"r_3 && (({inner_left_34}) U (r_4 && ({inner_34})))"

    expected = f"({left}) U (({branch_4_then_3}) || ({branch_3_then_4}))"

    assert_ltl_equivalent(builder.nodes, root, expected, actual_rendered=rewritten)
