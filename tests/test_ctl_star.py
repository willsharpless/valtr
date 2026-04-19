import pytest

from valtr import CTLStarValidationError, TLLexer, validate_ctl_star
from valtr.ltl_ir import ExistsPaths, ForAllPaths
from valtr.ltl_pass_runner import run_default_ltl_passes
from valtr.ltl_pretty import pretty_ltl
from tests.ltl_test_utils import canonical_ltl_key, parse_ltl_spec


def test_quantified_formulas_parse_and_lower():
    ag_builder, ag_root = parse_ltl_spec("AG p")
    spaced_builder, spaced_root = parse_ltl_spec("A G p")
    ef_builder, ef_root = parse_ltl_spec("EF p")
    spaced_ef_builder, spaced_ef_root = parse_ltl_spec("E F p")

    assert canonical_ltl_key(ag_builder.nodes, ag_root) == canonical_ltl_key(spaced_builder.nodes, spaced_root)
    assert canonical_ltl_key(ef_builder.nodes, ef_root) == canonical_ltl_key(spaced_ef_builder.nodes, spaced_ef_root)

    assert isinstance(ag_builder.nodes[int(ag_root)], ForAllPaths)
    assert isinstance(ef_builder.nodes[int(ef_root)], ExistsPaths)


def test_a_and_e_remain_identifiers_when_not_standalone_keywords():
    tokens = [(token.type.name, token.value) for token in TLLexer().tokenize("Ax")]
    assert tokens == [("ID", "Ax")]

    tokens = [(token.type.name, token.value) for token in TLLexer().tokenize("Ea")]
    assert tokens == [("ID", "Ea")]


def test_pretty_printer_handles_ctl_star_quantifiers():
    builder, root = parse_ltl_spec("A(G p -> E F q)")
    rendered = pretty_ltl(builder.nodes, root)
    assert rendered.startswith("A ")
    assert "E " in rendered


def test_ctl_star_validator_accepts_path_formulas_at_top_level():
    builder, root = parse_ltl_spec("G p")
    result = validate_ctl_star(builder.nodes, root)
    assert not result.is_state_formula
    assert result.is_path_formula


def test_ctl_star_validator_classifies_state_and_path_formulas():
    builder, root = parse_ltl_spec("A(G p -> F q)")
    result = validate_ctl_star(builder.nodes, root)
    assert result.is_state_formula
    assert result.is_path_formula

    builder, root = parse_ltl_spec("p && G q")
    result = validate_ctl_star(builder.nodes, root)
    assert not result.is_state_formula
    assert result.is_path_formula


def test_ctl_star_validator_can_require_a_state_formula_at_the_root():
    builder, root = parse_ltl_spec("G p")
    with pytest.raises(CTLStarValidationError, match="Expected a CTL\\* state formula at the root"):
        validate_ctl_star(builder.nodes, root, root_kind="state")


def test_ltl_pass_runner_rejects_ctl_star_quantifiers():
    builder, root = parse_ltl_spec("A G p")
    with pytest.raises(ValueError, match="CTL\\* formulas with A/E path quantifiers are not supported"):
        run_default_ltl_passes(builder, root)
