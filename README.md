# valtr

`valtr` is a tool to convert temporal logic task formula into a decomposed Value graph (DVG) or "Value tree", which represents a coupled set of Bellman equations. This object serves as a dependency map to ultimately solve the optimal Value associated with the specification (the root node). See [paper](https://willsharpless.github.io/valdec-site/) for more.

The main workflow is:

1. **PARSE** a temporal logic formula, eg. `spec = "Fa && Fb && G!w"`
2. **LOWER** it to an intermediate representation (IR), akin to a temporal logic tree.
3. **PASS** over the IR to convert it via [decomposition rules](https://willsharpless.github.io/valdec-site/) to the DVG, a directed acyclic graph (DAG)
4. **SOLVE** the DVG/DAG to get the specification Value and optimal action/policy 

The final step can be done in multiple ways depending on the problem and desired accuracy:

- *Exactly for small discrete systems* such as gridworlds, via Value iteration using [`dvi`](https://github.com/oswinso/discrete_vi/tree/master)
- *Almost exactly for low-d continuous systems*, via HJ-PDE methods using [`hj_reachability`](https://github.com/StanfordASL/hj_reachability)
- *Approximately for general systems*, via RL using [`vdppo`](https://github.com/willsharpless/vdppo) 

## What Is Here

- `src/valtr/valtr.py`: main logic-to-DAG entry points such as `to_dag(...)`
- `src/valtr/solve_discrete.py`: discrete DAG solver built on top of `dvi`
- `src/valtr/safety_filter.py`: safety filtering over solved discrete tasks
- `src/valtr/control.py`, `src/valtr/solver_utils.py`: HJ-based continuous utilities
- `scripts/dvi/`: discrete gridworld and safety-filter demos
- `scripts/hj_reachability/`: HJ-based examples kept for the continuous workflow

RL examples using `valtr` can be found in [`vdppo`](https://github.com/willsharpless/vdppo).

## Installation

Baisc (no solving):

```bash
git clone https://github.com/willsharpless/valtr.git
cd valtr
pip install -e .
```

with backends (for solving the DVG DAG)

```bash
pip install -e ".[dvi]"
pip install -e ".[hj]"
```

See [`vdppo`](https://github.com/willsharpless/vdppo) for RL usage.

Notes:

- `.[hj]` installs the HJ stack used by the continuous examples.
- `.[dvi]` installs the pinned `dvi` dependency from the `discrete_vi` repository together with the extra packages used by the discrete scripts.
- Base installation now includes shared runtime dependencies such as `attrs`, `cyclopts`, `loguru`, and `tqdm`.

## Core Idea

The package takes formulas in the project temporal-logic syntax and lowers them through several stages:

- lex / parse source text into an AST
- lower the AST into a normalized IR
- apply rewrite passes such as `Finally -> Until` and globally-segment combination
- lower the IR into a DAG whose nodes represent operations like:
  `DAGVar`, `DAGNegate`, `DAGMinN`, `DAGMaxN`, `DAGReach`, `DAGAvoid`, and `DAGReachAvoid`
- solve those nodes in dependency order

The result is a value function or policy-like object that can be used for rollout, analysis, and safety filtering.

## Minimal Usage

The most current high-level entry point on this branch is `to_dag(...)`:

```python
from valtr.valtr import to_dag

spec = "(!d1 U k1) && G(!w)"
dag, root = to_dag(spec)
```

For discrete solving, pass the DAG together with a `dvi` discrete dynamics model and predicate arrays:

```python
from valtr.solve_discrete import solve_discrete

dict_vars, dict_actions, dict_gu_vars, dict_gu_actions = solve_discrete(
    dyn,
    dag.nodes,
    dict_predicates,
)
```

In practice, the best examples are the scripts under:

- `scripts/dvi/rooms_discrete.py`
- `scripts/dvi/rooms_discrete_ma.py`
- `scripts/dvi/run_safety_filter.py`

## Running Examples

Discrete / `dvi` examples:

```bash
python scripts/dvi/rooms_discrete.py
python scripts/dvi/run_safety_filter.py
```

HJ examples:

```bash
python scripts/hj_reachability/rooms.py
python scripts/hj_reachability/rooms_dubins.py
```

Some scripts require extra dependencies or local environment setup beyond what is declared in `pyproject.toml`.

## Parsing Development

Note, some complex logical formulae are difficult to parse and may yield incorrect temporal logic trees, causing `valtr` to error. In such situations, the public tool spot is useful for reducing [formulae](https://slebok.github.io/proverb/spot.html) to standard form. Note, this package is not massively tested and the authors welcome community contributions.
