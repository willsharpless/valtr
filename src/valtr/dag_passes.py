from typing import Dict, Iterable, List, Optional, Set, Tuple
from loguru import logger

import graphviz
import ipdb

from valtr.reachability import (DAGAvoid, DagBuilder, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGMinGuard,
                                DAGNegate, DAGNode, DAGReach, DAGReachAvoid, DAGVar)


class DagRewriter:
    """
    Generic DAG rewriter/cloner with memoization. Subclass and override `visit()`
    or provide specific case handlers using match/guards.
    """

    def __init__(self, src: DagBuilder):
        self.src = src
        self.dst = DagBuilder()
        self.memo: Dict[DAGId, DAGId] = {}
        self.changed = False

    def run(self, root: DAGId) -> Tuple[DAGId, DagBuilder, bool]:
        self.changed = False
        out = self.visit(root)
        return out, self.dst, self.changed

    def visit(self, rid: DAGId) -> DAGId:
        i = rid
        # if i in self.memo:
        #     return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case DAGConst(value=v):
                out = self.dst.const(v)

            case DAGVar(name=s):
                out = self.dst.var(s)

            case DAGNegate(arg=arg):
                new_arg = self.visit(arg)
                out = self.dst.negate(new_arg)

            case DAGMinN(args=args):
                new_args = [self.visit(a) for a in args]
                out = self.dst.min_n(new_args)

            case DAGMaxN(args=args):
                new_args = [self.visit(a) for a in args]
                out = self.dst.max_n(new_args)

            case DAGReachAvoid(reach=reach, avoid=stay):
                new_reach = self.visit(reach)
                new_stay = self.visit(stay)
                out = self.dst.reachavoid(reach=new_reach, stay=new_stay)

            case DAGReach(reach=reach):
                new_reach = self.visit(reach)
                out = self.dst.reach(new_reach)

            case DAGAvoid(avoid=avoid):
                new_avoid = self.visit(avoid)
                out = self.dst.avoid(new_avoid)

            case DAGGUSingle(reach=reach_id, avoid=avoid_id):
                new_reach = self.visit(reach_id)
                new_avoid = self.visit(avoid_id)
                out = self.dst.GU_single(new_reach, new_avoid)

            case DAGGUMinN(args=args):
                new_args = [self.visit(a) for a in args]
                out = self.dst.GU_min_n(new_args)

            case DAGMinGuard(temporal_arg=temporal_arg, nontemporal_arg=nontemporal_arg):
                new_temporal_arg = self.visit(temporal_arg)
                new_nontemporal_arg = self.visit(nontemporal_arg)
                out = self.dst.min_guard(temporal_arg=new_temporal_arg, nontemporal_arg=new_nontemporal_arg)

            case _:
                raise AssertionError(f"Unhandled DAG node: {type(n).__name__}")

        self.memo[i] = out
        return out


class PassFoldConstBool(DagRewriter):
    """
    Constant folds:
      NEGATE:
        - child is True (+infty) -> False (-infty)
        - child is False (-infty) -> True (+infty)
      AND (Min):
        - short-circuit: if any child is False (-infty) -> False (-infty)
        - remove True (+infty) children
        - arity collapse: [] -> True, [x] -> x
      OR: (Max)
        - short-circuit: if any child is True (+infty) -> True (+infty)
        - remove False (-infty) children
        - arity collapse: [] -> False, [x] -> x
    Other nodes are cloned recursively without change.
    """

    def visit(self, rid: DAGId) -> DAGId:
        i = rid
        # if i in self.memo:
        #     return self.memo[i]
        n = self.src.nodes[i]

        match n:
            # ---------- NEGATE (-) ---------
            case DAGNegate(arg=arg):
                rebuilt_arg = self.visit(arg)
                cn = self.dst.nodes[rebuilt_arg]
                if isinstance(cn, DAGConst):
                    # Fold
                    self.changed = True
                    out = self.dst.const(not cn.value)
                    self.memo[i] = out
                    return out
                else:
                    out = self.dst.negate(rebuilt_arg)

            # ---------- AND (Min) ----------
            case DAGMinN(args=args):
                # Remove all args that are const.
                removed_true = False
                non_const_args = []
                for a_id in args:
                    a_node = self.src.nodes[a_id]
                    if not isinstance(a_node, DAGConst):
                        non_const_args.append(a_id)
                        continue

                    assert isinstance(a_node, DAGConst)
                    if a_node.value:
                        # Remove True (+infty) operands
                        removed_true = True
                    else:
                        # Any False => False
                        self.changed = True
                        out = self.dst.const(False)
                        self.memo[i] = out
                        return out

                # Remaining args are not const.
                filtered = [self.visit(a) for a in non_const_args]

                self.changed = self.changed or removed_true

                # Collapse
                if len(filtered) == 0:
                    self.changed = True
                    out = self.dst.const(True)
                elif len(filtered) == 1:
                    self.changed = True
                    out = filtered[0]
                else:
                    out = self.dst.min_n(filtered)

            # ---------- OR (Max) ----------
            case DAGMaxN(args=args):
                removed_false = False
                non_const_args = []
                for a_id in args:
                    a_node = self.src.nodes[a_id]
                    if not isinstance(a_node, DAGConst):
                        non_const_args.append(a_id)
                        continue

                    assert isinstance(a_node, DAGConst)
                    if a_node.value:
                        # Any True => True
                        self.changed = True
                        out = self.dst.const(True)
                        self.memo[i] = out
                        return out
                    else:
                        # Remove False (0infty) operands
                        removed_false = True

                # Remaining args are not const.
                filtered = [self.visit(a) for a in non_const_args]
                self.changed = self.changed or removed_false

                # Collapse
                if len(filtered) == 0:
                    self.changed = True
                    out = self.dst.const(False)
                elif len(filtered) == 1:
                    self.changed = True
                    out = filtered[0]
                else:
                    out = self.dst.max_n(filtered)

            # ---------- Everything else: default cloning ----------
            case _:
                out = super().visit(rid)

        self.memo[i] = out
        return out


class PassDuplicateMixedPolarity(DagRewriter):
    """
    Find predicates used in both positive and negative form.
    Create separate DAGVar nodes for each polarity.

    Example: if 'r' appears as both 'r' and '!r', create 'r_pos' and 'r_neg'.
    Eg. "(!r U s) && (F r)".

    This ensures each predicate has a consistent polarity throughout the DAG, which
    is required for proper trigger detection (which could be changed but is complicated).
    """

    def __init__(self, src: DagBuilder):
        super().__init__(src)
        self.var_polarities: Dict[str, Set[Tuple[bool, int]]] = {}
        self.mixed_vars: Set[str] = set()
        self.negate_context = False

    def run(self, root: DAGId) -> Tuple[DAGId, DagBuilder, bool]:
        # First pass: collect variable polarities
        self._collect_polarities(root, is_negated=False, visited=set())

        # Find variables with mixed polarity
        self.mixed_vars = {
            name for name, polarities in self.var_polarities.items() if len({pol for pol, _ in polarities}) > 1
        }

        if not self.mixed_vars:
            # No changes needed
            return root, self.src, False

        # Second pass: rewrite with duplicated variables
        self.changed = True
        out = self.visit(root)
        return out, self.dst, self.changed

    def _collect_polarities(self, node_id: int, is_negated: bool, visited: set):
        """First pass: collect all variable polarities."""
        if node_id in visited:
            return
        visited.add(node_id)

        node = self.src.nodes[node_id]

        match node:
            case DAGVar(name=name):
                if name not in self.var_polarities:
                    self.var_polarities[name] = set()
                self.var_polarities[name].add((is_negated, node_id))

            case DAGNegate(arg=arg):
                self._collect_polarities(arg, not is_negated, visited)

            case DAGMinN(args=args) | DAGMaxN(args=args):
                for child in args:
                    self._collect_polarities(child, is_negated, visited)

            case DAGReachAvoid(reach=reach, avoid=avoid):
                self._collect_polarities(reach, is_negated, visited)
                self._collect_polarities(avoid, is_negated, visited)

            case DAGAvoid(avoid=avoid):
                self._collect_polarities(avoid, is_negated, visited)

            case _:
                pass

    def visit(self, rid: DAGId) -> DAGId:
        """Second pass: rewrite nodes with polarity-specific variable names."""
        i = rid
        if i in self.memo:
            return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case DAGVar(name=name):
                if name in self.mixed_vars:
                    # Create polarity-specific variable
                    suffix = "_neg" if self.negate_context else "_pos"
                    out = self.dst.var(name + suffix)
                else:
                    out = self.dst.var(name)

            case DAGNegate(arg=arg):
                # Toggle negation context
                old_context = self.negate_context
                self.negate_context = not old_context
                new_arg = self.visit(arg)
                self.negate_context = old_context
                out = self.dst.negate(new_arg)

            case _:
                # Use parent's visit for all other nodes
                out = super().visit(rid)

        self.memo[i] = out
        return out


class PassDuplicateMixedRole(DagRewriter):
    """
    Find predicates used in both reach and avoid contexts.
    Create separate DAGVar nodes for each role.

    Example: if 'r' appears as both a reach target (F r) and avoid constraint (G !r),
    create 'r_reach' and 'r_avoid'. Eg. "F r && G !r" or "F (r && FG r)".

    This ensures each predicate has a consistent semantic role throughout the DAG,
    which is required for proper trigger detection (which could be changed but is complicated).
    """

    def __init__(self, src: DagBuilder):
        super().__init__(src)
        self.var_roles: Dict[str, Set[Tuple[str, int]]] = {}  # var_name -> set of (role, node_id)
        self.mixed_role_vars: Set[str] = set()
        self.role_context = None  # 'reach' or 'avoid'

    def run(self, root: DAGId) -> Tuple[DAGId, DagBuilder, bool]:
        # First pass: collect variable roles
        self._collect_roles(root, role=None, visited=set())

        # Find variables with mixed roles
        self.mixed_role_vars = {
            name for name, roles in self.var_roles.items() if len({role for role, _ in roles if role is not None}) > 1
        }

        if not self.mixed_role_vars:
            # No changes needed
            return root, self.src, False

        # Second pass: rewrite with role-specific variables
        self.changed = True
        out = self.visit(root)
        return out, self.dst, self.changed

    def _collect_roles(self, node_id: int, role: Optional[str], visited: set):
        """First pass: collect all variable roles (reach vs avoid)."""
        if node_id in visited:
            return
        visited.add(node_id)

        node = self.src.nodes[node_id]

        match node:
            case DAGVar(name=name):
                if name not in self.var_roles:
                    self.var_roles[name] = set()
                self.var_roles[name].add((role, node_id))

            case DAGReachAvoid(reach=reach, avoid=avoid):
                # Variables in reach are reach targets
                self._collect_roles(reach, "reach", visited.copy())
                # Variables in avoid are avoid constraints
                self._collect_roles(avoid, "avoid", visited.copy())

            case DAGAvoid(avoid=avoid):
                # Variables in avoid are avoid constraints
                self._collect_roles(avoid, "avoid", visited.copy())

            case DAGNegate(arg=arg):
                # Propagate role through negation
                self._collect_roles(arg, role, visited)

            case DAGMinN(args=args) | DAGMaxN(args=args):
                # Propagate role to all children
                for child in args:
                    self._collect_roles(child, role, visited.copy())

            case _:
                pass

    def visit(self, rid: DAGId) -> DAGId:
        """Second pass: rewrite nodes with role-specific variable names."""
        i = rid
        if i in self.memo:
            return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case DAGVar(name=name):
                if name in self.mixed_role_vars and self.role_context is not None:
                    # Create role-specific variable
                    suffix = f"_{self.role_context}"
                    out = self.dst.var(name + suffix)
                else:
                    out = self.dst.var(name)

            case DAGReachAvoid(reach=reach, avoid=avoid):
                # Set reach context
                old_context = self.role_context
                self.role_context = "reach"
                new_reach = self.visit(reach)

                # Set avoid context
                self.role_context = "avoid"
                new_avoid = self.visit(avoid)

                # Restore context
                self.role_context = old_context
                out = self.dst.reachavoid(reach=new_reach, stay=new_avoid)

            case DAGAvoid(avoid=avoid):
                # Set avoid context
                old_context = self.role_context
                self.role_context = "avoid"
                new_avoid = self.visit(avoid)
                self.role_context = old_context
                out = self.dst.avoid(new_avoid)

            case DAGNegate(arg=arg):
                # Propagate role context through negation
                new_arg = self.visit(arg)
                out = self.dst.negate(new_arg)

            case DAGMinN(args=args):
                # Propagate role context to all children
                new_args = [self.visit(a) for a in args]
                out = self.dst.min_n(new_args)

            case DAGMaxN(args=args):
                # Propagate role context to all children
                new_args = [self.visit(a) for a in args]
                out = self.dst.max_n(new_args)

            case _:
                # Use parent's visit for all other nodes
                out = super().visit(rid)

        self.memo[i] = out
        return out


class PassRAToR(DagRewriter):
    """
        ReachAvoid(r, True)  ->  Reach r
    If the avoid argument is a constant true, then change it to a reach DAG node.
    """

    def visit(self, rid: DAGId) -> DAGId:
        i = rid
        if i in self.memo:
            return self.memo[i]
        n = self.src.nodes[i]

        match n:
            case DAGReachAvoid(reach=reach_id, avoid=avoid_id):
                rebuilt_reach_id = self.visit(reach_id)
                avoid_node = self.src.nodes[avoid_id]
                if isinstance(avoid_node, DAGConst) and avoid_node.value is True:
                    self.changed = True
                    out = self.dst.reach(rebuilt_reach_id)
                else:
                    rebuilt_avoid_id = self.visit(avoid_id)
                    out = self.dst.reachavoid(rebuilt_reach_id, rebuilt_avoid_id)
            case _:
                out = super().visit(rid)

        # self.memo[i] = out
        return out

class PassToMinGuard(DagRewriter):
    """
    Orgnaize the mins by splitting the nodes into either temporal nodes and non-temporal nodes.
    """

    def visit(self, rid: DAGId) -> DAGId:
        i = rid
        if i in self.memo:
            return self.memo[i]
        n = self.src.nodes[i]

        match n:
            case DAGMinN(args=args):
                temporal_args = []
                nontemporal_args = []
                for a_id in args:
                    a_node = self.src.nodes[a_id]
                    if a_node.is_temporal():
                        temporal_args.append(self.visit(a_id))
                    else:
                        nontemporal_args.append(self.visit(a_id))

                # Create MinGuard node if there is at exactly one temporal and at least one non-temporal argument.
                if len(temporal_args) == 1 and len(nontemporal_args) >= 1:
                    temporal_arg = temporal_args[0]
                    nontemporal_arg = self.dst.min_n(nontemporal_args)
                    out = self.dst.min_guard(temporal_arg=temporal_arg, nontemporal_arg=nontemporal_arg)
                    self.changed = True
                else:
                    # Rebuild normally.
                    rebuilt_args = temporal_args + nontemporal_args
                    out = self.dst.min_n(rebuilt_args)
            case _:
                out = super().visit(rid)

        self.memo[i] = out
        return out

class PassAbsorbGU(DagRewriter):
    """
    Converts:
        (q1 U r1) AND G( AND_i q_i U r_i )
    to:
        ( (q1 AND AND_i (q_i OR r_i)) U (r1 AND G( AND_i q_i U r_i )) )

    Pattern: a DAGMinN containing exactly one DAGReachAvoid and >=1 DAGGUSingle children
    (no other child types). The DAGGUSingle children collectively represent G(AND_i q_i U r_i).
    """

    def visit(self, rid: DAGId) -> DAGId:
        i = rid
        if i in self.memo:
            return self.memo[i]
        n = self.src.nodes[i]

        match n:
            case DAGMinN(args=args):
                ra_nodes = []    # (id, DAGReachAvoid)
                gu_nodes = []    # (id, DAGGUSingle)
                other_ids = []

                for a_id in args:
                    a_node = self.src.nodes[a_id]
                    match a_node:
                        case DAGReachAvoid():
                            ra_nodes.append((a_id, a_node))
                        case DAGGUMinN(args=args):
                            for gu_node_idx in args:
                                gu_node = self.src.nodes[gu_node_idx]
                                gu_nodes.append((a_id, gu_node))
                        case _:
                            other_ids.append(a_id)

                # logger.debug(f"[Node %{rid}] Found {len(ra_nodes)} RA nodes, {len(gu_nodes)} GU nodes, {len(other_ids)} other nodes")
                if len(ra_nodes) == 1 and len(gu_nodes) >= 1 and len(other_ids) == 0:
                    _, ra = ra_nodes[0]
                    q1 = self.visit(ra.avoid)
                    r1 = self.visit(ra.reach)

                    qi_or_ri_list = []
                    new_gu_single_ids = []
                    for _, gu in gu_nodes:
                        new_qi = self.visit(gu.avoid)
                        new_ri = self.visit(gu.reach)
                        qi_or_ri_list.append(self.dst.max_n([new_qi, new_ri]))
                        new_gu_single_ids.append(self.dst.GU_single(new_ri, new_qi))

                    new_gu = self.dst.GU_min_n(new_gu_single_ids)

                    new_avoid = self.dst.min_n([q1] + qi_or_ri_list)
                    new_reach = self.dst.min_n([r1, new_gu])
                    out = self.dst.reachavoid(reach=new_reach, stay=new_avoid)
                    self.changed = True
                else:
                    out = super().visit(rid)
            case _:
                out = super().visit(rid)

        self.memo[i] = out
        return out


class PassKeepReachable(DagRewriter):
    """
    Remove all nodes that are not reachable from the root.
    """

    def run(self, root: DAGId) -> Tuple[DAGId, DagBuilder, bool]:
        return super().run(root)
