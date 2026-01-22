from valtr.reachability import (DAGAvoid, DAGConst, DAGGUMinN, DAGGUSingle, DAGId, DAGMaxN, DAGMinN, DAGNegate, DAGNode,
                                DAGReach, DAGReachAvoid, DAGVar)


def dag_node_to_str(dag_nodes: list[DAGNode], dag_root: DAGId) -> str:
    """Print the temporal logic in infix notation."""

    def recurse(node_id: DAGId) -> str:
        node = dag_nodes[node_id]
        match node:
            case DAGConst(value=value):
                return "⊤" if value else "⊥"
            case DAGVar(name=name):
                return name
            case DAGNegate(arg):
                child_node = dag_nodes[arg]
                match child_node:
                    case DAGVar(name=name):
                        return f"¬{name}"
                    case _:
                        return f"¬({recurse(arg)})"
            case DAGMinN(args=args):
                return "(" + " ∧ ".join(recurse(arg) for arg in args) + ")"
            case DAGMaxN(args=args):
                return "(" + " ∨ ".join(recurse(arg) for arg in args) + ")"
            case DAGAvoid(avoid=avoid):
                return f"G({recurse(avoid)})"
            case DAGReach(reach=reach):
                return f"F({recurse(reach)})"
            case DAGReachAvoid(reach=reach, avoid=avoid):
                return f"({recurse(avoid)}) U ({recurse(reach)})"
            case DAGGUSingle(reach=reach, avoid=avoid):
                return f"G( ({recurse(avoid)}) U ({recurse(reach)}) )"
            case DAGGUMinN(args=args):
                return " ∧ ".join(recurse(arg) for arg in args)
            case _:
                raise ValueError(f"Unknown DAG node type: {type(node)}")

    return recurse(dag_root)
