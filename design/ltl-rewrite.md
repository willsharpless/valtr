# LTL Rewrite Design

## Purpose

This document describes the planned rewrite of the LTL-to-HJ compilation pipeline.
The main goals are:

- separate LTL normalization from HJ-specific lowering
- make the middle representation pleasant to rewrite and inspect
- make the final backend DAG easier to consume downstream
- preserve enough source information for diagnostics and debugging

This design does not aim for backward compatibility with the current implementation.

## Current Problems

The current pipeline in `src/valtr/valtr.py` is:

1. lex source text into tokens
2. parse tokens into an AST
3. lower AST into an IR
4. run IR passes
5. lower IR into a DAG
6. run DAG passes

This has two main issues:

1. `lower_ir_to_dag` is doing too much work.
   It is not just translating one representation into another. It is also recognizing an HJ-solvable fragment and applying LTL equivalence-style transformations on the fly.

2. The final DAG is too close to an execution structure and not rich enough as a semantic representation.
   For example, downstream code often wants to:
   - split a `min` into temporal vs non-temporal children
   - inspect the formula in a DNF-like view
   - reason about guards separately from temporal operators

Those tasks become awkward when the DAG is the first representation that exposes HJ structure.

## Proposed Pipeline

The rewrite should use four conceptual layers:

1. `Tokens`
2. `AST`
3. `LTL IR`
4. `HJ Normal Form`
5. `Backend DAG`

In code, the main pipeline should look like:

1. lex source text
2. parse into a syntax-oriented AST
3. lower AST into an operator-specific LTL IR
4. run LTL simplification and normalization passes
5. lower normalized LTL IR into an HJ-specific normal form
6. optionally run HJ-normal-form passes
7. lower HJ normal form into the backend DAG
8. optionally run backend-only DAG passes

The key design rule is:

- LTL equivalence rules belong in the LTL IR phase
- HJ fragment recognition belongs in the HJ normal form phase
- backend DAG lowering should be mostly structural

## Representation Responsibilities

### AST

The AST is syntax-shaped.
It should preserve:

- source structure
- precedence/grouping
- exact source spans
- operators that are useful to parse directly

The AST is not the place to enforce strong semantic invariants.

### LTL IR

The LTL IR is the main rewrite representation.
It should be:

- semantic rather than syntax-oriented
- easy to pattern-match
- easy to canonicalize
- suitable for repeated simplification
- independent of HJ implementation details

This is the representation where we should implement:

- desugaring
- boolean flattening
- n-ary `and` / `or`
- normalization of intervals
- local equivalence rewrites
- DNF/CNF-style views or transforms where needed
- simplifications such as `F p -> True U p` if that is actually desired for a pass

### HJ Normal Form

This is a new layer that should be added explicitly.

Its job is to encode the fragment that the solver/backend understands, in a way that is still semantic and easy to inspect.
This layer should capture concepts that are currently implicit inside `lower_ir_to_dag`, such as:

- plain propositional guards
- plain `U` terms
- `G` terms
- special grouped cases like `G(q U r)`
- mixed forms that admit the recursive HJ lowering scheme

This layer should be rich enough that:

- the legality of lowering is explicit
- unsupported patterns fail here, not deep in DAG lowering
- downstream consumers can query meaningful structure before the backend DAG exists

### Backend DAG

The backend DAG is the execution/storage representation.
It is the right place for:

- hash-consed backend nodes
- compact structure for evaluation / solving
- HJ operator nodes such as reach, avoid, reach-avoid
- backend-oriented simplification passes

It is not the right place for general-purpose LTL rewriting.

## LTL IR Design

The new `ltl_ir.py` should use one class per operator instead of a generic `Binary` or `TemporalBinary` with a `kind` field.

Recommended node set:

- `Const`
- `Var`
- `Not`
- `And`
- `Or`
- `Next`
- `Finally`
- `Globally`
- `Until`
- `Release`

### Why operator-specific nodes

This design is better for the rewrite-heavy phase because:

- invalid states are harder to represent
- rewrite rules read more like the math
- operator-specific invariants are easier to enforce
- downstream code does less `if kind == ...` dispatch
- printing, debugging, and testing are simpler

The current tagged-union style is compact, but it tends to blur semantic distinctions and encourages giant central functions with many `kind` checks.

## `ltl_ir.py` Layout Sketch

The file should contain the following main pieces.

### 1. Basic source/provenance types

- `Position`
- `Span`
- `SourceRef`
- `Origin`
- `Interval`

### 2. IR node definitions

- `Expr`
- `Const`
- `Var`
- `Not`
- `And`
- `Or`
- `Next`
- `Finally`
- `Globally`
- `Until`
- `Release`

### 3. Arena / builder

- `ExprId`
- `LTLBuilder`

The builder should:

- intern nodes
- flatten nested `And` and `Or`
- dedupe children where appropriate
- sort n-ary children for canonicalization
- collapse one-element `And` / `Or`
- fold empty `And` to `True`
- fold empty `Or` to `False`

### 4. Traversal / rewrite support

- `LTLRewriter`
- helper utilities like `conjuncts`, `disjuncts`, `has_temporal`

### 5. Debug support

- pretty-printer
- optional graphviz or tree printer
- optional normalized-string printer used in tests

## Example Structure

This section is only a sketch, not a final API contract.

```python
from __future__ import annotations

from dataclasses import dataclass, field


class ExprId(int):
    pass


@dataclass(frozen=True, slots=True)
class Interval:
    lo: int | None
    hi: int | None


@dataclass(frozen=True, slots=True)
class SourceRef:
    phase: str
    node_id: int


@dataclass(frozen=True, slots=True)
class Origin:
    primary_span: Span | None
    spans: tuple[Span, ...] = ()
    sources: frozenset[SourceRef] = field(default_factory=frozenset)
    rule: str | None = None


@dataclass(frozen=True, slots=True)
class Expr:
    origin: Origin

    def children(self) -> tuple[ExprId, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class Const(Expr):
    value: bool


@dataclass(frozen=True, slots=True)
class Var(Expr):
    name: str


@dataclass(frozen=True, slots=True)
class Not(Expr):
    arg: ExprId


@dataclass(frozen=True, slots=True)
class And(Expr):
    args: tuple[ExprId, ...]


@dataclass(frozen=True, slots=True)
class Or(Expr):
    args: tuple[ExprId, ...]


@dataclass(frozen=True, slots=True)
class Next(Expr):
    arg: ExprId
    interval: Interval | None = None


@dataclass(frozen=True, slots=True)
class Finally(Expr):
    arg: ExprId
    interval: Interval | None = None


@dataclass(frozen=True, slots=True)
class Globally(Expr):
    arg: ExprId
    interval: Interval | None = None


@dataclass(frozen=True, slots=True)
class Until(Expr):
    left: ExprId
    right: ExprId
    interval: Interval | None = None


@dataclass(frozen=True, slots=True)
class Release(Expr):
    left: ExprId
    right: ExprId
    interval: Interval | None = None
```

## Provenance Design

The AST and lexer should keep exact source spans.
For the LTL IR and later phases, raw `Span` alone is not enough because rewritten nodes often combine information from multiple source locations.

Instead, each semantic node should carry an `Origin`.

Recommended `Origin` fields:

- `primary_span`
- `spans`
- `sources`
- `rule`

### Meaning of each field

- `primary_span`
  A best single span for diagnostics. This is what an error message can point at first.

- `spans`
  All source spans that contributed directly to this node.

- `sources`
  Stable references to source nodes from an earlier phase, such as AST node ids or prior IR node ids.

- `rule`
  The name of the rewrite rule or lowering rule that produced this node, if synthesized.

### Construction rules

Leaf parsed node:

```python
Origin.leaf(span=node.span, source=SourceRef("ast", ast_id))
```

Derived node:

```python
Origin.derived("finally_to_until", old_origin)
```

Merged node:

```python
Origin.derived("merge_globally_conjuncts", origin_a, origin_b)
```

### Why provenance is better than only span

Suppose we rewrite:

```text
G a && G b  ->  G(a && b)
```

The new `G(...)` node does not have a single exact source location in the original program.
If we keep only one `Span`, we either lose information or lie.

With provenance:

- `primary_span` can point to the first `G`
- `spans` can contain both original `G` spans
- `sources` records both contributing source nodes
- `rule="merge_globally_conjuncts"` explains why the node exists

That makes debugging, tracing, and future user-facing diagnostics much cleaner.

## HJ Normal Form Design

The HJ normal form should be a separate module, for example `hj_form.py`.

This layer should not try to preserve arbitrary LTL.
It should model only the fragment that is intended to lower to HJ operators.

Example categories:

- propositional formula
- `Until(prop, formula)`
- `Globally(prop)`
- grouped `Globally(Until(prop, prop))`
- conjunctions/disjunctions of those forms where the recursive HJ scheme applies

This layer should answer questions like:

- is this formula HJ-lowerable?
- which pieces are pure guards?
- which pieces are temporal obligations?
- which recursive lowering rule applies?

In other words, `lower_ir_to_dag` should stop discovering structure and instead consume an already structured HJ-normal-form object.

## DAG Design Guidance

The backend DAG should remain specialized for execution and solver interaction.
However, it should not be the first place where downstream code discovers semantic structure.

That said, it is still reasonable to include convenience nodes or helpers that make backend consumers simpler.
Examples:

- explicit guard-carrying nodes
- helper methods to split temporal vs non-temporal children
- structural printers for backend debugging

The rule is:

- if a distinction is important for LTL reasoning, encode it before the DAG
- if a distinction is only important for backend execution, encode it in the DAG

## Suggested Module Split

One possible layout is:

- `src/valtr/tl_parser.py`
  Syntax-oriented AST

- `src/valtr/ltl_ir.py`
  Operator-specific LTL IR, builder, provenance, and rewriter base

- `src/valtr/ltl_passes.py`
  LTL normalization and simplification passes

- `src/valtr/hj_form.py`
  HJ-specific semantic normal form

- `src/valtr/hj_lowering.py`
  LTL IR to HJ normal form lowering

- `src/valtr/dag.py`
  Backend DAG nodes and builder

- `src/valtr/dag_lowering.py`
  HJ normal form to backend DAG lowering

- `src/valtr/dag_passes.py`
  Backend-only DAG simplification passes

## Pass Placement

Examples of where important transforms should live:

- `Implies(a, b) -> Or(Not(a), b)`
  AST-to-LTL lowering or earliest LTL pass

- `F p -> True U p`
  LTL pass

- merge sibling `G` terms under conjunction
  LTL pass

- DNF view / controlled distribution
  LTL pass or LTL utility

- classify top-level `U`, `G`, `GU`, `UG`
  HJ normal form lowering

- `ReachAvoid(r, True) -> Reach(r)`
  backend DAG pass

- constant folding in solver-oriented nodes
  backend DAG pass

## Invariants

The rewrite should enforce the following invariants.

### LTL IR invariants

- `And` and `Or` are n-ary
- nested same-kind `And` / `Or` are flattened
- n-ary args are deduped and sorted
- `Until` and `Release` are binary
- provenance exists on every node

### HJ normal form invariants

- every node is known to be HJ-lowerable
- guards are explicitly separated from temporal parts when relevant
- unsupported patterns fail during construction

### DAG invariants

- nodes are backend-specific
- lowering is structural
- no hidden LTL normalization happens here

## Migration Plan

Suggested order of implementation:

1. create `ltl_ir.py` with operator-specific nodes, builder, provenance, and printer
2. add AST-to-LTL lowering
3. port current IR-level simplifications into LTL passes
4. define `hj_form.py`
5. move HJ fragment recognition out of DAG lowering into LTL-to-HJ lowering
6. simplify backend DAG lowering so it becomes mostly structural
7. trim DAG passes so they are truly backend-only

## Non-Goals

This design does not try to:

- preserve the old IR API
- preserve the old DAG API exactly
- make the DAG a general-purpose LTL analysis representation
- force full DNF materialization for every formula

## Summary

The rewrite should promote the current IR into a real semantic LTL IR, but with operator-specific node classes instead of tagged `Binary` and `TemporalBinary` nodes.
It should then add one more explicit layer, HJ normal form, before lowering to the backend DAG.

The most important structural change is:

- stop letting backend DAG lowering perform semantic LTL rewriting

The most important metadata change is:

- stop treating `Span` as the only source-tracking mechanism after parsing, and instead carry a richer `Origin` provenance object through the semantic phases
