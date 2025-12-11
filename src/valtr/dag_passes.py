from typing import Dict, Iterable, List, Optional, Set, Tuple

import graphviz

from valtr.reachability import (DAGAvoid, DagBuilder, DAGConst, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReachAvoid, DAGReachAvoidLoop, DAGVar, DAGNext)


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
        if i in self.memo:
            return self.memo[i]

        n = self.src.nodes[i]

        match n:
            case DAGConst(value=v):
                out = self.dst.const(v)

            case DAGVar(name=s):
                out = self.dst.var(s)

            case DAGNegate(arg=arg):
                out = self.dst.negate(arg)

            case DAGNext(arg=arg):
                out = self.dst.next(self.visit(arg))

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

            case DAGReachAvoidLoop(reach=reach, avoid=stay):
                new_reach = self.visit(reach)
                new_stay = self.visit(stay)
                out = self.dst.reachavoidloop(reach=new_reach, stay=new_stay)

            case DAGAvoid(avoid=avoid):
                new_avoid = self.visit(avoid)
                out = self.dst.avoid(new_avoid)

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
        if i in self.memo:
            return self.memo[i]
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
                # Recurse first
                rebuilt = [self.visit(a) for a in args]

                # Short-circuit: any False => False
                for cid in rebuilt:
                    cn = self.dst.nodes[cid]
                    if isinstance(cn, DAGConst) and cn.value is False:
                        self.changed = True
                        out = self.dst.const(False)
                        self.memo[i] = out
                        return out

                # Remove True (+infty) operands
                filtered: List[DAGId] = []
                removed_true = False
                for cid in rebuilt:
                    cn = self.dst.nodes[cid]
                    if isinstance(cn, DAGConst) and cn.value is True:
                        removed_true = True
                        continue
                    filtered.append(cid)

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
                # Recurse first
                rebuilt = [self.visit(a) for a in args]

                # Short-circuit: any True => True
                for cid in rebuilt:
                    cn = self.dst.nodes[cid]
                    if isinstance(cn, DAGConst) and cn.value is True:
                        self.changed = True
                        out = self.dst.const(True)
                        self.memo[i] = out
                        return out

                # Remove False (-infty) operands
                filtered: List[DAGId] = []
                removed_false = False
                for cid in rebuilt:
                    cn = self.dst.nodes[cid]
                    if isinstance(cn, DAGConst) and cn.value is False:
                        removed_false = True
                        continue
                    filtered.append(cid)

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
