# LTL Pass Implementation Plan

This document sketches the implementation plan for the new LTL rewrite passes.
The goal is to realize the rewrite system described in [ltl_passes.md](/Users/oswinso/research/me/valtr/design/ltl_passes.md) on top of the live LTL IR, without reintroducing the old monolithic IR-to-DAG lowering logic.

## Goals

- keep the implementation purely LTL-to-LTL during the rewrite stage
- make each pass explicit and testable
- preserve provenance through rewrites
- support repeated local rewriting until the target fragment emerges
- keep the implementation aligned with the rule ordering and structural constraints in `ltl_passes.md`

## Current Foundation

The live rewrite stack already has the basic pieces:

- [src/valtr/ltl_ir.py](/Users/oswinso/research/me/valtr/src/valtr/ltl_ir.py)
  operator-specific LTL IR nodes plus provenance
- [src/valtr/ltl_builder.py](/Users/oswinso/research/me/valtr/src/valtr/ltl_builder.py)
  hash-consing builder with boolean flattening and simple normalization
- [src/valtr/ltl_rewriter.py](/Users/oswinso/research/me/valtr/src/valtr/ltl_rewriter.py)
  recursive clone-and-rebuild traversal

The pass implementation should build on these modules instead of introducing a second IR.

## Proposed Modules

### `src/valtr/ltl_matchers.py`

Shared structural predicates and decomposition helpers.

Planned responsibilities:

- `is_true`, `is_false`
- `is_propositional`
- `is_plain_global_guard`
- `is_gu` for `G(q U r)`
- `is_ug` for `q U G b`
- `is_fg_lowered` for `True U G b`
- `conjuncts` and `disjuncts`
- helpers to remove, replace, and rebuild conjuncts
- helpers to collect mergeable plain `G(...)` guards

This module should centralize matching logic so the passes do not each implement their own shape checks.

### `src/valtr/ltl_pretty.py`

Small pretty-printing helpers for tests and debugging.

Planned responsibilities:

- compact string rendering of LTL formulas
- stable output for assertions in tests
- optional annotation of node ids or origins for debugging

This is mainly for pass verification and regression tests.

### `src/valtr/ltl_passes.py`

Definitions of the concrete pass classes.

This module should hold:

- normalization passes
- local interaction passes
- structural passes
- pass registry or ordered pass lists used by the runner

### `src/valtr/ltl_pass_runner.py`

Orchestration logic for running passes to fixed point.

Planned responsibilities:

- execute passes in the intended schedule
- repeat cycles until no pass changes the formula
- expose a simple API such as `run_ltl_pass_pipeline(nodes, root)` or a builder-based equivalent

## Rewriter Base Changes

[src/valtr/ltl_rewriter.py](/Users/oswinso/research/me/valtr/src/valtr/ltl_rewriter.py) should be upgraded from a pure rebuilding visitor into a hookable base class.

Planned changes:

- recurse into children first
- then dispatch to rewrite hooks such as:
  - `rewrite_and`
  - `rewrite_or`
  - `rewrite_globally`
  - `rewrite_finally`
  - `rewrite_until`
  - `rewrite_release`
- allow passes to report whether they changed anything
- keep the builder-driven rebuild style so we preserve interning and origin handling

This will let individual passes focus on one rule family at a time.

## Pass Categories

### 1. Normalization Passes

These are the earliest passes and should stabilize the formula into an easy-to-match form.

Planned passes:

- `PassNormalizeBoolean`
  - rely on builder canonicalization
  - fold trivial boolean structure
- `PassLowerFinally`
  - `F p -> True U p`
- `PassLowerRelease`
  - `r R q -> (q U (r && q)) || G q`
- `PassSimplifyTemporal`
  - `G G p -> G p`
  - `F F p -> F p`
  - `p U p -> p`
  - `False U p -> p`
  - `p U True -> True`

These should be simple local rewrites, mostly on unary or binary nodes.

### 2. Local Interaction Passes

These are the core LTL-to-LTL interaction rules and should primarily operate on `And(...)` nodes.

Planned passes:

- `PassUntilAndFG`
  - match lowered `FG` as `Until(True, Globally(...))`
- `PassUntilAndGlobally`
- `PassUntilAndUntil`
- `PassGloballyUntilAndUntil`
- `PassUntilGloballyAndGlobally`
- `PassFinallyAndFG`
  - optional direct special case, though it can be derived from `PassUntilAndFG`

Implementation strategy:

- inspect rebuilt conjuncts of an `And`
- find one matching redex
- rewrite only that redex
- rebuild the surrounding conjunction
- rely on repeated pass execution for further simplification

This keeps the implementation predictable and avoids a new hidden mega-lowering step.

### 3. Structural Passes

These enforce the operational constraints from `ltl_passes.md`.

Planned passes:

- `PassRestrictedGlobalGuardMerge`
  - merge only plain propositional `G(...)` guards
  - explicitly avoid consuming `G(q U r)` terms too early
- `PassDistributeAndOverOr`
  - push conjunctive guards into disjunction branches when that exposes new local redexes
- `PassGUBaseCase`
  - match `And(G(q1 U r1), ..., G(qn U rn), G q)` when no plain `U` terms remain
  - rewrite to `And(G((q1 && q) U (r1 && q)), ..., G((qn && q) U (rn && q)))`

The base-case pass should remain separate from generic `GU && G` logic because it has tighter operational preconditions.

## Pass Runner Schedule

The first implementation should use an explicit ordered schedule instead of a generic worklist.

Recommended schedule:

1. `PassNormalizeBoolean`
2. `PassLowerFinally`
3. `PassLowerRelease`
4. `PassSimplifyTemporal`
5. repeat the following block until fixed point:
   - `PassUntilAndFG`
   - `PassGloballyUntilAndUntil`
   - `PassUntilGloballyAndGlobally`
   - `PassUntilAndGlobally`
   - `PassUntilAndUntil`
   - `PassFinallyAndFG`
   - `PassRestrictedGlobalGuardMerge`
   - `PassDistributeAndOverOr`
   - `PassGUBaseCase`
   - `PassNormalizeBoolean`

This ordering is meant to preserve useful redexes long enough for the desired recursive shape to emerge.

## Matching and Provenance

Each rewrite should build new nodes through [src/valtr/ltl_builder.py](/Users/oswinso/research/me/valtr/src/valtr/ltl_builder.py), not by constructing nodes directly.

Rules for provenance:

- preserve child origins when a node is copied unchanged
- use `Origin.derived(rule_name, ...)` when a new node is synthesized
- choose rule names that correspond to the design doc, for example:
  - `lower_finally`
  - `lower_release`
  - `until_and_globally`
  - `until_and_fg`
  - `gu_base_case`
  - `restricted_global_guard_merge`

The matcher layer should return enough structure that each pass can rebuild the result without manually re-parsing child nodes.

## Testing Plan

### Unit Tests for Individual Rules

Add focused tests for each local equivalence and each structural pass.

Examples:

- `F p -> True U p`
- `G G p -> G p`
- `p U p -> p`
- `(q U r) && G b`
- `(q U r) && (True U G b)`
- `(q U G b) && G c`
- `G(q U r) && (q2 U r2)`

These tests should assert normalized formula shape using the pretty-printer rather than raw node ids.

### Regression Tests from the Design Note

Add end-to-end tests for the worked examples in [design/ltl_passes.md](/Users/oswinso/research/me/valtr/design/ltl_passes.md):

- `G(q1 U r1) && G(q2 U r2) && q3 U r3 && q4 U r4 && G q5 && G q6`
- `G(q1 U r1) && G(q2 U r2) && q3 U r3 && q4 U r4 && G q5 && F G q6`

The goal of these tests is not to pin exact interning structure, but to verify that the recursive rewrite shape emerges and that the base case fires when expected.

### Smoke Script

[scripts/smoke.py](/Users/oswinso/research/me/valtr/scripts/smoke.py) should remain a lightweight manual check.

Good smoke-script upgrades:

- parse and lower a few representative formulas
- run the pass pipeline
- render pre- and post-pass graphviz output or pretty-printed formulas

## Implementation Order

Recommended order of work:

1. add matcher/decomposition helpers
2. upgrade the rewriter base to support rewrite hooks and change tracking
3. add pretty-printing for test assertions
4. implement normalization passes
5. implement restricted `G` merge and `And`-over-`Or` distribution
6. implement `U && G`, `U && FG`, and `UG && G`
7. implement `U && U`
8. implement `GU && U`
9. implement the `GU... && G q` base-case pass
10. add the pass runner
11. add focused rule tests
12. add regression tests for the master-equation examples

## Design Constraints to Preserve

- no DAG-specific logic in the LTL rewrite layer
- no eager global merge that hides `GU` redexes
- no special master-equation pass that bypasses local equivalences
- no second semantic IR between the live LTL IR and these passes unless a concrete need appears later

The rewrite system should stay understandable as repeated application of explicit equivalence rules plus a small number of structural pass constraints.
