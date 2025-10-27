from enum import Enum, auto
from typing import Dict, Iterable, List, NewType, Optional, Set, Tuple

import graphviz
from attrs import define, field, frozen

from valtr.ir import (Binary, BinaryIROpKind, ConstBool, IRId, IRNode, Nary, NaryKind, TemporalBinary, TemporalUnary,
                      Unary, UnaryIROpKind, Var)
from valtr.ir_builder import IRBuilder
from valtr.lexer import Position, Span
from valtr.tl_parser import BinaryOpKind, UnaryOpKind


class DAGId(int):
    pass


@frozen
class DAGNode:
    def children(self) -> List[DAGId]:
        return []


@frozen
class DAGConst(DAGNode):
    value: bool


@frozen
class DAGVar(DAGNode):
    name: str


@frozen
class DAGNegate(DAGNode):
    arg: DAGId

    def children(self) -> List[DAGId]:
        return [self.arg]


@frozen
class DAGMinN(DAGNode):
    args: Tuple[DAGId, ...]

    def children(self) -> List[DAGId]:
        return list(self.args)


@frozen
class DAGMaxN(DAGNode):
    args: Tuple[DAGId, ...]

    def children(self) -> List[DAGId]:
        return list(self.args)


@frozen
class DAGReachAvoid(DAGNode):
    reach: DAGId
    avoid: DAGId

    def children(self) -> List[DAGId]:
        return [self.reach, self.avoid]


@frozen
class DAGAvoid(DAGNode):
    avoid: DAGId  # A(arg)

    def children(self) -> List[DAGId]:
        return [self.avoid]


class DagBuilder:
    def __init__(self):
        self.nodes: List[DAGNode] = []
        self._intern: Dict[tuple, DAGId] = {}

    def _get(self, key: tuple, node: DAGNode) -> DAGId:
        if key in self._intern:
            return self._intern[key]
        i = len(self.nodes)
        self._intern[key] = i
        self.nodes.append(node)
        return i

    def const(self, v: bool) -> DAGId:
        return self._get(("Const", v), DAGConst(v))

    def var(self, name: str) -> DAGId:
        return self._get(("Var", name), DAGVar(name))

    def negate(self, arg: DAGId) -> DAGId:
        key = ("Negate", arg)
        return self._get(key, DAGNegate(arg))

    def min_n(self, args: Iterable[DAGId]) -> DAGId:
        s = tuple(sorted(set(args)))
        if len(s) <= 1:
            raise ValueError("min_n requires at least 2 arguments")
        return self._get(("MinN", s), DAGMinN(s))

    def max_n(self, args: Iterable[DAGId]) -> DAGId:
        s = tuple(sorted(set(args)))
        if len(s) <= 1:
            raise ValueError("max_n requires at least 2 arguments")
        return self._get(("MaxN", s), DAGMaxN(s))

    def reachavoid(self, reach: DAGId, stay: DAGId) -> DAGId:
        key = ("ReachAvoid", reach, stay)
        return self._get(key, DAGReachAvoid(reach, stay))

    def avoid(self, arg: DAGId) -> DAGId:
        key = ("Avoid", arg)
        return self._get(key, DAGAvoid(arg))


class LoweringError(Exception):
    pass


def lower_bool_leaf_expr_to_dag(irb: IRBuilder, dag: DagBuilder, rid: IRId) -> DAGId:
    """
    Lower a leaf boolean expression (const/var/AND/OR of those) into DAG.
    Rejects any temporal nodes.

    AND becomes min, OR becomes max.
    """
    n = irb.nodes[int(rid)]
    match n:
        case ConstBool(value=v):
            return dag.const(v)
        case Var(name=s):
            return dag.var(s)
        case Unary(kind=UnaryIROpKind.NOT, arg=arg, span=_):
            kid = lower_bool_leaf_expr_to_dag(irb, dag, arg)
            return dag.negate(kid)
        case Nary(kind=NaryKind.AND, args=args, span=_):
            kids = [lower_bool_leaf_expr_to_dag(irb, dag, a) for a in args]
            return dag.min_n(kids)
        case Nary(kind=NaryKind.OR, args=args, span=_):
            kids = [lower_bool_leaf_expr_to_dag(irb, dag, a) for a in args]
            return dag.max_n(kids)
        case _:
            raise LoweringError(f"Disallowed node inside leaf boolean: {type(n).__name__}")


def lower_ir_to_dag(irb: IRBuilder, root: IRId) -> tuple[DagBuilder, DAGId]:
    """
    Main entry: produce a backend DAG according to the specified ruless.
    """
    dag = DagBuilder()

    # extract terms
    n = irb.nodes[root]
    args: list[IRId]
    if isinstance(n, Nary) and n.kind == NaryKind.AND:
        args = list(n.args)
    else:
        args = [root]

    U_args: list[tuple[IRId, IRId]] = []
    G_args: list[IRId] = []

    for a in args:
        node = irb.nodes[a]
        match node:
            case TemporalBinary(kind=BinaryIROpKind.UNTIL, left=left, right=right, interval=iv, span=_):
                if iv is not None:
                    raise LoweringError("Timed UNTIL not supported")
                U_args.append((left, right))
            case TemporalUnary(kind=UnaryIROpKind.GLOBALLY, arg=arg, interval=iv, span=_):
                if iv is not None:
                    raise LoweringError("Timed GLOBALLY not supported")
                G_args.append(arg)
            case _:
                raise LoweringError(f"Top-level must contain only UNTIL/GLOBALLY, got {type(node).__name__}")

    # Build r_dag: if multiple G, r := AND(r_i)
    if len(G_args) == 0:
        G_dag = dag.const(True)  # if you want to require a G, raise instead
    elif len(G_args) == 1:
        G_dag = lower_bool_leaf_expr_to_dag(irb, dag, G_args[0])
    else:
        raise LoweringError("IR should have been preprocessed to combine multiple G into one")
        # r_dag = dag.and_n(lower_bool_leaf_expr_to_dag(irb, dag, r) for r in G_args)

    # Check/convert each left and right into DAG leaves (or allowed boolean combos)
    U_lefts: list[DAGId] = []
    U_rights: list[DAGId] = []
    for ir_left, ir_right in U_args:
        U_lefts.append(lower_bool_leaf_expr_to_dag(irb, dag, ir_left))
        U_rights.append(lower_bool_leaf_expr_to_dag(irb, dag, ir_right))

    match (len(U_args), len(G_args)):
        case (0, 0):
            raise LoweringError("No UNTIL or GLOBALLY found in formula")
        case (_, n) if n > 1:
            raise LoweringError("IR should have been preprocessed to combine multiple G into one")
        case (0, 1):
            # Only 1 G, lower to avoid(r)
            return dag, dag.avoid(G_dag)
        case (m, 1) if m >= 1:
            # Lower to reach-avoid
            # Handled below.
            pass
        case _:
            raise LoweringError("Unhandled case in lowering")

    # Recursive builder for V on k UNTILs
    def build_V_k(idxs_: list[int]) -> DAGId:
        len_indices = len(idxs_)
        if len_indices == 1:
            # If only one index left, then base case.
            # \max_a ρ( (q_1 U r) ∧ G q2 )   <=>   reachavoid(r_tilde, q_tilde)
            #    r_tilde = min( r, avoid(q2) )
            #    q_tilde = min(q1, q2)
            ii = idxs_[0]
            U_stay = U_lefts[ii]
            U_reach = U_rights[ii]
            r_tilde = dag.min_n([U_reach, dag.avoid(G_dag)])
            q_tilde = dag.min_n([U_stay, G_dag])
            return dag.reachavoid(reach=r_tilde, stay=q_tilde)

        # len(indices_) > 1. Iterate over all indices, popping each and computing the value over the rest.
        r_tildes_childs = []
        for ii in idxs_:
            # \max_a ρ( ⋀_{i ∈ I} (q_i U r_i) ∧ G q )   <=>   reachavoid(r_tilde, q_tilde)
            #     r_tilde = max_{i ∈ I} min( r_i, V_k(I \ {i}) )
            #     q_tilde = min( min_i q_i, q )

            U_reach = U_rights[ii]

            # Form indices without ii
            idxs_rest = [i for i in idxs_ if i != ii]

            r_tilde_child = dag.min_n([U_reach, build_V_k(idxs_rest)])
            r_tildes_childs.append(r_tilde_child)

        r_tilde = dag.max_n(r_tildes_childs)

        U_stays_in_idxs = [U_lefts[i] for i in idxs_]
        q_tilde = dag.min_n(U_stays_in_idxs + [G_dag])

        return dag.reachavoid(reach=r_tilde, stay=q_tilde)

    n_reach = len(U_args)
    indices = list(range(n_reach))

    root_dag = build_V_k(indices)
    return dag, root_dag


SYM_MAX = "max"
SYM_MIN = "min"
SYM_RA = "ReachAvoid"
SYM_A = "Avoid"


def dag_to_str(builder: DagBuilder, rid: DAGId) -> str:
    """
    Convert a DAG node to a Unicode logical expression string.
    Ensures parentheses so ambiguity is avoided.
    Memoizes visited DAG nodes to avoid exponential recomputation.
    """
    cache: dict[int, str] = {}

    def go(i: int, top_level: bool = False) -> str:
        if i in cache:
            return cache[i]

        node = builder.nodes[i]
        match node:

            case DAGConst(value=v):
                s = "⊤" if v else "⊥"  # Unicode True / False

            case DAGVar(name=sym):
                s = sym

            case DAGMinN(args=args):
                items = [go(a) for a in args]
                s = SYM_MIN + "(" + ", ".join(items) + ")"

            case DAGMaxN(args=args):
                items = [go(a) for a in args]
                s = SYM_MAX + "(" + ", ".join(items) + ")"

            case DAGReachAvoid(reach=l, avoid=r):
                if top_level:
                    s = f"ReachAvoid({go(l)}, {go(r)})"
                else:
                    s = f"ReachAvoid%{i})"

            case DAGAvoid(avoid=avoid):
                if top_level:
                    s = f"Avoid({go(avoid)})"
                else:
                    s = f"Avoid%{i})"

            case DAGNegate(arg=a):
                s = f"-{go(a)}"

            case _:
                s = f"{type(node).__name__}"

        cache[i] = s
        return s

    return go(int(rid), top_level=True)
