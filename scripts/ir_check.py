import cyclopts
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool
from valtr.ir_builder import IRBuilder
from valtr.ir_graphviz import visualize_ir
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.mintime_rollout import MinTimeRollout
from valtr.reachability import lower_ir_to_dag
from valtr.solve_discrete import solve_discrete
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser


def main():
    spec = "F( a && b && c )"

    lexer = TLLexer()
    tokens = list(lexer.tokenize(spec))
    ast = TLParser(tokens).parse()

    # AST -> IR
    ir = IRBuilder()
    lowerer = Lowerer(builder=ir)
    ir_root_id = lowerer.lower(ast)

    passes = [PassFinallyToUntil, PassCombineGloballySegments]
    for p_cls in passes:
        p = p_cls(ir)
        ir_root_id, ir = p.run(ir_root_id)

    dot_ir = visualize_ir(ir, ir_root_id, filename="ir.pdf", view=False)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
