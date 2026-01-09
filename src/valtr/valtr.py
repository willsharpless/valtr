import cyclopts
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dvi.dynamics.gridworld import GridWorld
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from valtr.dag_graphviz import visualize_dag
from valtr.dag_passes import PassFoldConstBool
from valtr.gridworld_utils import get_drift_fn, parse_rooms
from valtr.ir_builder import IRBuilder
from valtr.ir_graphviz import visualize_ir
from valtr.ir_pass import PassCombineGloballySegments, PassFinallyToUntil
from valtr.lowering import Lowerer
from valtr.mintime_rollout import MinTimeRollout
from valtr.reachability import lower_ir_to_dag
from valtr.solve_discrete import solve_discrete
from valtr.tl_lexer import TLLexer
from valtr.tl_parser import TLParser


def to_dag(spec: str, ir_filename: str | None = None, dag_filename: str | None = None):
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

    if ir_filename is not None:
        dot_ir = visualize_ir(ir, ir_root_id, filename=ir_filename, view=False)

    # IR -> DAG
    value_tree_dag, dag_root = lower_ir_to_dag(ir, ir_root_id)

    n_changes = 0
    # visualize_dag(value_tree_dag, dag_root, filename="rooms_discrete_dag0", view=view_pdf)

    # Perform constant folding.
    passes = [PassFoldConstBool]
    for p_cls in passes:
        changed = True
        while changed:
            p = p_cls(value_tree_dag)
            dag_root, value_tree_dag, changed = p.run(dag_root)
            n_changes += int(changed)
            # if changed:
            #     visualize_dag(value_tree_dag, dag_root, filename=f"rooms_discrete_dag{n_changes}", view=view_pdf)

    if dag_filename is not None:
        dag_dot = visualize_dag(value_tree_dag, dag_root, filename=dag_filename, view=False)

    return value_tree_dag, dag_root
