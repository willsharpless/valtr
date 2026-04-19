from __future__ import annotations

from dataclasses import dataclass

from .ltl_builder import LTLBuilder
from .ltl_ir import ExprId
from .ltl_passes import LOOP_PASSES, PRE_PASSES


@dataclass(frozen=True, slots=True)
class LTLPipelineResult:
    root: ExprId
    builder: LTLBuilder
    rounds: int


class LTLPassRunner:
    def __init__(self, max_rounds: int = 50) -> None:
        self.max_rounds = max_rounds

    def run(self, builder: LTLBuilder, root: ExprId) -> LTLPipelineResult:
        current_builder = builder
        current_root = root

        for pass_cls in PRE_PASSES:
            rewrite_pass = pass_cls(current_builder)
            current_root, current_builder = rewrite_pass.run(current_root)

        for round_idx in range(1, self.max_rounds + 1):
            any_change = False
            for pass_cls in LOOP_PASSES:
                rewrite_pass = pass_cls(current_builder)
                current_root, current_builder = rewrite_pass.run(current_root)
                any_change = any_change or rewrite_pass.changed
            if not any_change:
                return LTLPipelineResult(root=current_root, builder=current_builder, rounds=round_idx)

        raise RuntimeError(f"LTL pass pipeline did not converge after {self.max_rounds} rounds")


def run_default_ltl_passes(builder: LTLBuilder, root: ExprId, max_rounds: int = 50) -> LTLPipelineResult:
    return LTLPassRunner(max_rounds=max_rounds).run(builder, root)
