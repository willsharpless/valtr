from valtr.dag_mermaid import render_dag_mermaid
from valtr.valtr import to_dag


def build_mermaid(spec: str, *, vertical: bool = False) -> str:
    dag, root = to_dag(spec)
    direction = "TD" if vertical else "LR"
    return render_dag_mermaid(dag, root, direction=direction)
