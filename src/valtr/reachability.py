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


@frozen
class DAGReach(DAGNode):
    reach: DAGId  # A(arg)

    def children(self) -> List[DAGId]:
        return [self.reach]


@frozen
class DAGGU(DAGNode):
    """G( AND_i ( q_i U r_i ) )"""

    args: list[tuple[DAGId, DAGId]]

    def children(self) -> List[DAGId]:
        kids = []
        for q_i, r_i in self.args:
            kids.append(q_i)
            kids.append(r_i)
        return kids


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
        # # Make each const literal is unique.
        # i = len(self.nodes)
        # node = DAGConst(v)
        # self.nodes.append(node)
        # return i
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

    def reach(self, arg: DAGId) -> DAGId:
        key = ("Reach", arg)
        return self._get(key, DAGReach(arg))

    def GU(self, args: list[tuple[DAGId, DAGId]]) -> DAGId:
        args_tup = tuple(args)

        key = ("GU", args_tup)
        return self._get(key, DAGGU(args))


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


def lower_ir_to_dag_old(irb: IRBuilder, root: IRId) -> tuple[DagBuilder, DAGId]:
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
        case (1, 0):
            # Only 1 U, lower to reachavoid(r)
            return dag, dag.reachavoid(reach=U_rights[0], stay=U_lefts[0])
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


def get_and_args_list(node: IRNode, node_id: IRId) -> list[IRId]:
    args: list[IRId]
    match node:
        case Nary(kind=NaryKind.AND, args=args_, span=_):
            return list(args_)
        case _:
            return [node_id]


def lower_ir_to_dag(irb: IRBuilder, root: IRId) -> tuple[DagBuilder, DAGId]:
    """
    AND_i G ( q_i U r_i )  AND  AND_j ( q_j U r_j )  AND  G q_G

    => q_tilde U r_tilde
        q_tilde = q_j AND q_G AND ( q_i OR r_i )
        r_tilde = OR_j (r_j AND V_{without j})

    AND is min, OR is max
    """
    root_node = irb.nodes[root]
    root_arg_ids = get_and_args_list(root_node, root)

    # 1: Extract the top-level AND arguments, separate it into the GU, U and G parts.
    GU_args: list[TemporalBinary] = []
    U_args: list[TemporalBinary] = []
    G_args_id: list[IRId] = []

    for node_id in root_arg_ids:
        node = irb.nodes[node_id]
        match node:
            case TemporalBinary(kind=BinaryIROpKind.UNTIL, left=left, right=right, interval=iv, span=_):
                if iv is not None:
                    raise LoweringError("Timed UNTIL not supported")
                U_args.append(node)
            case TemporalUnary(kind=UnaryIROpKind.GLOBALLY, arg=arg_id, interval=iv, span=_):
                if iv is not None:
                    raise LoweringError("Timed GLOBALLY not supported")

                arg_node = irb.nodes[arg_id]

                # We combined all the G into one previously. Get all the args under the AND (if it exists).
                G_and_arg_ids = get_and_args_list(arg_node, arg_id)

                for g_arg_id in G_and_arg_ids:
                    # Check if it's a G or GU.
                    arg_node = irb.nodes[g_arg_id]
                    match arg_node:
                        case TemporalBinary(kind=BinaryIROpKind.UNTIL, left=left, right=right, interval=iv2, span=_):
                            if iv2 is not None:
                                raise LoweringError("Timed UNTIL not supported")
                            assert isinstance(arg_node, TemporalBinary)
                            GU_args.append(arg_node)
                        case _:
                            G_args_id.append(g_arg_id)
            case _:
                raise LoweringError(f"Top-level must contain only UNTIL/GLOBALLY, got {type(node).__name__}")

    dag = DagBuilder()

    if len(G_args_id) == 0:
        G_dag_arg = dag.const(True)
    elif len(G_args_id) == 1:
        G_dag_arg = lower_bool_leaf_expr_to_dag(irb, dag, G_args_id[0])
    else:
        raise LoweringError("IR should have been preprocessed to combine multiple G into one")

    # if len(GU_args) == 0:
    #     return dag, lower_ir_to_dag_no_GU_(irb, dag, U_args, G_dag_arg)
    # else:
    return dag, lower_ir_to_dag_(irb, dag, U_args, GU_args, G_dag_arg)


def lower_ir_to_dag_(
    irb: IRBuilder,
    dag: DagBuilder,
    U_args: list[TemporalBinary],
    GU_args: list[TemporalBinary],
    G_arg_dag: DAGId,
) -> DAGId:
    """
    AND_i G ( q_i U r_i )  AND  AND_j ( q_j U r_j )  AND  G q_G

    => q_tilde U r_tilde
        q_tilde = q_j AND q_G AND ( q_i OR r_i )
        r_tilde = OR_j (r_j AND V_{without j})

    AND is min, OR is max
    """
    # Base case: U_args is empty.
    if len(U_args) == 0:
        return lower_ir_to_dag_GU(irb, dag, GU_args, G_arg_dag)

    # Construct q_tilde.
    #     AND_i ( q_i OR r_i )
    q_tilde_ands_GU = []
    for node in GU_args:
        left = lower_bool_leaf_expr_to_dag(irb, dag, node.left)
        right = lower_bool_leaf_expr_to_dag(irb, dag, node.right)
        or_left_right = dag.max_n([left, right])
        q_tilde_ands_GU.append(or_left_right)

    #     AND_j q_j
    q_tilde_ands_U = [lower_bool_leaf_expr_to_dag(irb, dag, node.left) for node in U_args]

    #     AND q_G
    q_tilde_ands_G = [G_arg_dag]

    q_tilde = dag.min_n(q_tilde_ands_GU + q_tilde_ands_U + q_tilde_ands_G)

    # Construct r_tilde.
    r_tilde_maxs = []
    for jj, node in enumerate(U_args):
        # r_j AND V_{without j}
        r_j = lower_bool_leaf_expr_to_dag(irb, dag, node.right)

        U_args_without_j = [node for ii, node in enumerate(U_args) if ii != jj]
        V_without_j = lower_ir_to_dag_(irb, dag, U_args_without_j, GU_args, G_arg_dag)
        r_tilde_maxs.append(dag.min_n([r_j, V_without_j]))

    if len(r_tilde_maxs) == 0:
        raise ValueError("Why U_args empty")
    elif len(r_tilde_maxs) == 1:
        r_tilde = r_tilde_maxs[0]
    else:
        r_tilde = dag.max_n(r_tilde_maxs)

    root_id = dag.reachavoid(reach=r_tilde, stay=q_tilde)
    return root_id


def lower_ir_to_dag_GU(irb: IRBuilder, dag: DagBuilder, GU_args: list[TemporalBinary], G_arg_dag: DAGId) -> DAGId:
    """
    Lower AND_i G ( q_i U r_i )  AND  G q_G

    # 1: Merge the G q_G inside the GU.
        (q_i AND q_G) U (r_i AND G q_G)
    # 2: Solve the G( AND U)
    arbitrarily (?) order the GU_args.
    Solve reachavoid( q_tilde_i, r_tilde_i ), where
        q_tilde_i = q_i AND w_{not i}
        r_tilde_i = r_i AND w_{not i} AND X V_{i + 1}
    Have a single DAG node to represent this iteration.
    """
    G_dag = dag.avoid(G_arg_dag)

    if len(GU_args) == 0:
        # Just a normal avoid.
        return G_dag
    else:
        # 1: Merge the G q_G inside the GU.
        GU_args_dag: list[tuple[DAGId, DAGId]] = []
        for node in GU_args:
            q_i_dag = lower_bool_leaf_expr_to_dag(irb, dag, node.left)
            r_i_dag = lower_bool_leaf_expr_to_dag(irb, dag, node.right)

            q_i_new = dag.min_n([q_i_dag, G_arg_dag])
            r_i_new = dag.min_n([r_i_dag, G_dag])
            GU_args_dag.append((q_i_new, r_i_new))

        GU_dag = dag.GU(GU_args_dag)
        return GU_dag


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


# def lower_ir_to_dag_no_GU_(
#     irb: IRBuilder,
#     dag: DagBuilder,
#     U_args: list[TemporalBinary],
#     G_arg_dag: DAGId,
# ) -> DAGId:
#     """
#
#     """
#     ...
#


def collect_predicate_info(
    builder: DagBuilder, root: DAGId
) -> tuple[list[str], dict[str, int], list[int], dict[str, set], dict[str, set]]:
    """
    Collect predicates and track whether they appear negated and in which role (reach/avoid).

    Returns:
        predicates: List of predicate names in order of discovery
        predicate_to_idx: Mapping from predicate name to index
        predicate_ids: List of node IDs for each predicate (first occurrence)
        predicate_negations: Dict mapping predicate name to set of (is_negated, node_id) tuples
        predicate_roles: Dict mapping predicate name to set of (role, node_id) tuples where role is 'reach', 'avoid', or None
    """
    predicates = []
    predicate_to_idx = {}
    predicate_ids = []
    predicate_negations = {}  # predicate_name -> set of (is_negated, node_id)
    predicate_roles = {}  # predicate_name -> set of (role, node_id) where role is 'reach', 'avoid', or None

    def _collect_recursive(node_id: int, is_negated: bool, role: Optional[str], visited: set):
        """Recursive helper that updates the outer scope variables."""
        if node_id in visited:
            return
        visited.add(node_id)

        node = builder.nodes[node_id]

        match node:
            case DAGVar(name=name):
                # Record this occurrence with its negation status and role
                if name not in predicate_negations:
                    predicate_negations[name] = set()
                predicate_negations[name].add((is_negated, node_id))

                if name not in predicate_roles:
                    predicate_roles[name] = set()
                predicate_roles[name].add((role, node_id))

                # Add to predicates list if first occurrence
                if name not in predicate_to_idx:
                    predicate_to_idx[name] = len(predicates)
                    predicates.append(name)
                    predicate_ids.append(node_id)

            case DAGReachAvoid(reach=reach, avoid=avoid):
                # Variables in reach are reach targets
                _collect_recursive(reach, is_negated, "reach", visited.copy())
                # Variables in avoid are avoid constraints
                _collect_recursive(avoid, is_negated, "avoid", visited.copy())

            case DAGAvoid(avoid=avoid):
                # Variables in avoid are avoid constraints
                _collect_recursive(avoid, is_negated, "avoid", visited.copy())

            case DAGNegate(arg=arg):
                # Flip negation context and continue
                _collect_recursive(arg, not is_negated, role, visited)

            case _:
                # For all other nodes, propagate with same context
                for child_id in node.children():
                    _collect_recursive(child_id, is_negated, role, visited.copy())

    _collect_recursive(int(root), is_negated=False, role=None, visited=set())
    return predicates, predicate_to_idx, predicate_ids, predicate_negations, predicate_roles


def extract_trigger_predicate_map(builder: DagBuilder, root: DAGId):
    """
    Extract trigger predicate map from a DAG.

    Returns:
        predicates: List of predicate names (DAGVar names)
        predicate_ids: List of predicate node IDs
        temporal_nodes: List of temporal node IDs in topological order
        trigger_predicate_map: (N, P) array where entry [i, j] is the child node index
                               that we switch to from node i when predicate j is satisfied,
                               or -1 if no such transition exists.
        negated_predicate_mask: (P,) array where entry [i] is True if predicate i is negated

    Example:
        For the task "F reach1 && F reach2 && G !obstacles", yielding a DAG with 4 nodes (RRAA, RAA1, RAA2, A)
        and 3 predicates (reach1, reach2, obstacles), with

        temporal_nodes = [13, 10, 7, 4] # RRAA, RAA1, RAA2, A
        predicates = ["reach1", "reach2", "obstacles"]
        negated_predicate_mask = [False, False, True]

        then the trigger_predicate_map should look like:

        [[ 2,  1, -1],  # RRAA can reach1 (-> RAA2) or reach2 (-> RAA1)
         [ 3, -1, -1],  # RAA1 can reach1 (-> A)
         [-1,  3, -1],  # RAA2 can reach2 (-> A)
         [-1, -1, -1]]  # A is a terminal node
    """

    # Collect predicate information
    predicates, predicate_to_idx, predicate_ids, predicate_negations, predicate_role_sets = collect_predicate_info(
        builder, root
    )

    # Assert and extract unique negation status for each predicate
    # After PassDuplicateMixedPolarity, each predicate should have consistent negation
    negated_predicate_mask = []
    for pred_name in predicates:
        negation_statuses = {is_neg for is_neg, _ in predicate_negations[pred_name]}
        assert len(negation_statuses) == 1, (
            f"Predicate '{pred_name}' has mixed negation contexts: {negation_statuses}. "
            f"Apply PassDuplicateMixedPolarity before extracting trigger map."
        )
        negated_predicate_mask.append(negation_statuses.pop())

    # Assert and extract unique role for each predicate
    # After PassDuplicateMixedRole, each predicate should have consistent role
    predicate_roles = []  # 'reach', 'avoid', or None
    for pred_name in predicates:
        roles = {role for role, _ in predicate_role_sets[pred_name] if role is not None}
        assert len(roles) <= 1, (
            f"Predicate '{pred_name}' has mixed roles: {roles}. "
            f"Apply PassDuplicateMixedRole before extracting trigger map."
        )
        predicate_roles.append(roles.pop() if roles else None)

    # Collect all ReachAvoid and Avoid nodes in post-order (children before parents)
    temporal_nodes_postorder = []

    def collect_temporal_nodes(node_id: int, visited: set):
        if node_id in visited:
            return
        visited.add(node_id)

        node = builder.nodes[node_id]

        # Recursively visit children first (post-order traversal)
        for child_id in node.children():
            collect_temporal_nodes(child_id, visited)

        # Add temporal nodes
        if isinstance(node, (DAGReachAvoid, DAGAvoid)):
            temporal_nodes_postorder.append(node_id)

    collect_temporal_nodes(int(root), set())

    # Reverse to get topological order (root first, leaves last)
    temporal_nodes = list(reversed(temporal_nodes_postorder))

    # Build position map
    node_to_pos = {node_id: pos for pos, node_id in enumerate(temporal_nodes)}

    N = len(temporal_nodes)
    P = len(predicates)

    # Initialize trigger map with -1 (no transition)
    trigger_map = [[-1] * P for _ in range(N)]

    def find_predicate_triggers(node_id: int, parent_pos: int, visited: set):
        """
        Find which predicates trigger transitions from parent_pos to this node.
        A transition occurs when there's a min(predicate, child_node).
        """
        if node_id in visited:
            return
        visited.add(node_id)

        node = builder.nodes[node_id]

        # If this is a temporal node, record its position
        current_pos = node_to_pos.get(node_id, -1)

        match node:
            case DAGReachAvoid(reach=reach_id, avoid=avoid_id):
                # Explore reach and avoid branches
                find_predicate_triggers(reach_id, current_pos, visited.copy())
                find_predicate_triggers(avoid_id, current_pos, visited.copy())

            case DAGAvoid(avoid=avoid_id):
                # Explore avoid branch
                find_predicate_triggers(avoid_id, current_pos, visited.copy())

            case DAGMinN(args=args):
                # Check if this min contains a predicate and a temporal node
                pred_ids = []
                temporal_ids = []

                for arg in args:
                    arg_node = builder.nodes[arg]
                    if isinstance(arg_node, DAGVar):
                        pred_ids.append(arg)
                    elif isinstance(arg_node, (DAGReachAvoid, DAGAvoid)):
                        temporal_ids.append(arg)
                    elif isinstance(arg_node, DAGNegate):
                        # Check if it's negating a variable
                        inner = builder.nodes[arg_node.arg]
                        if isinstance(inner, DAGVar):
                            pred_ids.append(arg_node.arg)

                # If we have both a predicate and a temporal node in the min,
                # this represents a trigger
                if pred_ids and temporal_ids and parent_pos >= 0:
                    for pred_id in pred_ids:
                        pred_node = builder.nodes[pred_id]
                        if isinstance(pred_node, DAGVar):
                            pred_idx = predicate_to_idx[pred_node.name]
                            for temporal_id in temporal_ids:
                                child_pos = node_to_pos.get(temporal_id, -1)
                                if child_pos >= 0:
                                    trigger_map[parent_pos][pred_idx] = child_pos

                # Continue exploring
                for arg in args:
                    find_predicate_triggers(arg, parent_pos, visited.copy())

            case DAGMaxN(args=args):
                # For max, explore all branches
                for arg in args:
                    find_predicate_triggers(arg, parent_pos, visited.copy())

            case DAGNegate(arg=arg_id):
                find_predicate_triggers(arg_id, parent_pos, visited.copy())

            case _:
                pass

    # Start traversal from root
    find_predicate_triggers(int(root), -1, set())

    return predicates, predicate_ids, predicate_roles, negated_predicate_mask, temporal_nodes, trigger_map
