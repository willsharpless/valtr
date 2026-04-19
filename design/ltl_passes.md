# LTL Rewrite Passes

This document organizes the planned LTL-to-LTL rewrite system into three parts:

1. local equivalence rules
2. the master equation for the conjunctive temporal fragment
3. additional structural pass rules that determine where and how the local rules are applied

The goal is to express the old lowering behavior as repeated LTL rewrites over the live LTL IR, rather than as a single IR-to-DAG transformation.

## Scope

These passes are intended for the fragment built from:

- propositional formulas
- `U`
- `G`
- `F`
- `R`

The intended flow is:

1. normalize boolean structure
2. apply local equivalence rules
3. use structural pass rules to recurse into nested subformulas and preserve non-temporal guards
4. continue applying local equivalence rules until the target fragment emerges

## Assumptions

The rules below are easiest to apply after a small amount of normalization:

- `And` and `Or` are n-ary and flattened
- implication has already been lowered away
- timed operators are either unsupported by these passes or handled separately
- `G(a && b)` may be represented as a single `Globally(And(...))`

## Local Equivalence Rules

These are local rewrites that can be applied directly to matching subterms.

### Finally to Until

$$
F p \equiv \top U p
$$

### Nested Globally

$$
G G p \equiv G p
$$

### Nested Finally

$$
F F p \equiv F p
$$

### Globally Commutes with Conjunction

$$
\bigwedge_i G p_i \equiv G\left( \bigwedge_i p_i \right)
$$

Operational note:
this equivalence should not be applied greedily to every `G(...)` term.
In particular, `G(q U r)` terms should remain exposed long enough for the `GU`-specific interaction rules to fire.
As a rewrite pass, this should therefore be restricted to plain global guards, or at least given lower priority than the `GU`-specific rules.

### Conjunction of Until to Nested Until

For two untils:

$$
q_1 U r_1 \land q_2 U r_2
\equiv
(q_1 \land q_2) U \left( [r_1 \land (q_2 U r_2)] \lor [r_2 \land (q_1 U r_1)] \right)
$$

More generally, for an index set $N$, with $N^{-i} = N \setminus \{i\}$:

$$
\bigwedge_{i \in N} q_i U r_i
\equiv
\left( \bigwedge_{i \in N} q_i \right) U
\left(
\bigvee_{i \in N}
\left(
r_i \land \bigwedge_{j \in N^{-i}} q_j U r_j
\right)
\right)
$$

### Degenerate Until

$$
p U p \equiv p
$$

$$
\bot U p \equiv p
$$

$$
p U \top \equiv \top
$$

### Release as Until or Globally

$$
r R q \equiv q U (r \land q) \lor G q
$$

### Conjunction Distributes Over Disjunction

$$
(a \lor b) \land c \equiv (a \land c) \lor (b \land c)
$$

More generally:

$$
\left(\bigvee_i a_i\right)\land c \equiv \bigvee_i (a_i \land c)
$$

This rule is needed operationally after `Conjunction of Until to Nested Until`, because that rewrite introduces disjunctions whose branches must continue interacting with global guards and other temporal conjuncts.

### Until and Globally

$$
q U r \land G b
\equiv
(q \land b) U (r \land G b)
$$

### Until and Finally Globally

$$
q U r \land F G b
\equiv
q U (r \land F G b)
$$

If `Finally to Until` has already been applied globally, then this rule should be matched in its lowered form:

$$
(q U r) \land (\top U G b)
\equiv
q U \big(r \land (\top U G b)\big)
$$

Operationally, after a global `F -> U` lowering pass, `F G b` should therefore be recognized as an `Until` node whose left side is `\top` and whose right side is a `Globally(...)` term.

### Finally and Finally Globally

$$
F a \land F G b
\equiv
F(a \land F G b)
$$

This can be viewed as the special case of `Until and Finally Globally` with `q = \top` and `r = a`.

### Until and Until Globally

$$
q_1 U r \land q_2 U G b
\equiv
(q_1 \land q_2) U \left( (r \land q_2 U G b) \lor (G b \land q_1 U r) \right)
$$

Then applying `Until and Globally` to the second branch gives:

$$
q_1 U r \land q_2 U G b
\equiv
(q_1 \land q_2) U \left( (r \land q_2 U G b) \lor ((q_1 \land b) U (r \land G b)) \right)
$$

### Base Case: Globally-Until Terms with a Plain Global Guard

For a conjunction of globally-until terms together with a plain global guard:

$$
\bigwedge_{i \in I} G(q_i U r_i) \land G q
\equiv
\bigwedge_{i \in I} G\big((q_i \land q) U (r_i \land q)\big)
$$

This is the intended base-case rewrite for the conjunctive temporal fragment.
Operationally, it should be applied only when no plain `U` terms remain in the same conjunction.

### Globally Until and Until

$$
G(q_1 U r_1) \land q_2 U r_2
\equiv
\big( (q_1 \lor r_1) \land q_2 \big) U \big( r_2 \land G(q_1 U r_1) \big)
$$

This is the local step that pulls a `G(q_1 U r_1)` factor into the left side of an until while preserving the obligation to continue satisfying the globally-until formula on the right.

### Until Globally and Globally

Apply `Until and Globally` and then `Globally Commutes with Conjunction`:

$$
\begin{aligned}
(q U G b) \land G c
&\equiv (q \land c) U (G b \land G c) \\
&\equiv (q \land c) U G(b \land c)
\end{aligned}
$$

## Master Equation

The master equation is not intended to be implemented as its own primitive rewrite rule.
Instead, it records the shape obtained by repeated application of the local equivalence rules above.
It serves as a correctness target and a guide for tests, rather than as a separate operational pass.

Let $I$ and $J$ be index sets, with $I^{-i} = I \setminus \{i\}$ and $J^{-j} = J \setminus \{j\}$.

Define

$$
\psi_I = \bigwedge_{i \in I} G(q_i U r_i)
$$

and

$$
p_{I, J} = \psi_I \land \bigwedge_{j \in J} (q_j U r_j) \land G q
$$

Then, using `Conjunction of Until to Nested Until`, `Until and Globally`, and `Globally Until and Until`:

$$
\begin{aligned}
p_{I, J}
&\equiv
\psi_I \land \bigwedge_{j \in J} (q_j U r_j) \land G q \\
&\equiv
\psi_I \land
\left(
\left(\bigwedge_{j \in J} q_j\right)
U
\left(
\bigvee_{j \in J}
\left(
r_j \land \bigwedge_{k \in J^{-j}} q_k U r_k
\right)
\right)
\right)
\land G q \\
&\equiv
\psi_I \land
\left(
\left(q \land \bigwedge_{j \in J} q_j\right)
U
\left(
\bigvee_{j \in J}
\left(
r_j \land \bigwedge_{k \in J^{-j}} q_k U r_k \land G q
\right)
\right)
\right) \\
&\equiv
\left(
\left(
q \land
\bigwedge_{i \in I} (q_i \lor r_i) \land
\bigwedge_{j \in J} q_j
\right)
U
\left(
\bigvee_{j \in J}
\left(
r_j \land \bigwedge_{k \in J^{-j}} q_k U r_k \land \psi_I \land G q
\right)
\right)
\right) \\
&\equiv
\left(
\left(
q \land
\bigwedge_{i \in I} (q_i \lor r_i) \land
\bigwedge_{j \in J} q_j
\right)
U
\left(
\bigvee_{j \in J}
\left(
r_j \land p_{I, J^{-j}}
\right)
\right)
\right)
\end{aligned}
$$

### Base Case

When $J = \emptyset$,

$$
\begin{aligned}
p_{I, \emptyset}
&=
\bigwedge_{i \in I} G(q_i U r_i) \land G q \\
&\equiv
\bigwedge_{i \in I} G\left((q_i \land q) U (r_i \land q)\right)
\end{aligned}
$$

Operationally, this means the rewrite bottoms out at a conjunction of globally-until terms together with a plain global guard, after which the `Base Case: Globally-Until Terms with a Plain Global Guard` rule applies.

## Additional Structural Pass Rules

These are not themselves semantic equivalences. They are pass rules that explain how the equivalences above should be applied to the IR.

### Boolean Normalization Pass

Before temporal rewriting:

- flatten nested `And` and `Or`
- deduplicate repeated conjuncts/disjuncts
- eliminate trivial constants
- lower implication before temporal passes

This keeps pattern matching stable.

### Restricted Global-Guard Merge

Although `Globally Commutes with Conjunction` is a valid equivalence in full generality, the operational pass should be weaker.

The pass should:

- eagerly merge plain global guards such as `G q_5 && G q_6 -> G(q_5 && q_6)`
- avoid greedily merging `G(q U r)` terms together with plain global guards
- leave `G(q U r)` terms exposed until `GU && U` has had a chance to fire, and until the base-case `GU... && Gq` rewrite can be applied

In other words, the general equivalence remains true semantically, but the pass strategy should not erase useful redexes too early.

### Recursive Local Matching

The rewrite engine should not explicitly partition a conjunction into `$I$`, `$J$`, and `G q` in order to fire a dedicated master-equation rule.
Instead, it should repeatedly search for local redexes and apply the equivalences directly.

In particular, when visiting an `And`, the engine should try to match combinations such as:

- `U && U`
- `U && G`
- `U && FG`
- `GU && U`
- `UG && G`
- `F && FG`
- `GU... && G`
- conjunction pushed into disjunction branches

and then rebuild the enclosing formula. Repeated application of these local rewrites should recover the same recursive shape summarized by the master equation.

### Master Equation as Derived Target

The master equation should be used as:

- a proof sketch that the local rewrite system is sufficient
- a target normal-form description for the conjunctive temporal fragment
- a source of regression tests

It should not be implemented as a separate primitive transformation that bypasses the local equivalences.

### Nested Rewriting Under `Until`

The old lowering handled nested temporal expressions on the right side of an `Until`.
The rewrite system should therefore recurse into:

- `Until.left`
- `Until.right`
- `Globally.arg`
- `Finally.arg`
- `Release.left`
- `Release.right`
- children of `And` and `Or`

This is especially important because the master equation can create nested temporal subformulas on the right-hand side.

### Structural Handling of Nested Propositional Conjuncts

If a recursive rewrite reaches a subformula of the form

$$
\phi \land b
$$

where $b$ is purely propositional and $\phi$ is temporal, then:

- keep `b` as an outside conjunct unless a local equivalence explicitly pulls it inward
- after rewriting `\phi`, rebuild the enclosing `And`

This matches the role of the old lowering's non-temporal `outside_args`.

### Structural Handling of Nested Disjunctions

If a recursive rewrite reaches a subformula whose top-level form is `Or(...)`, then:

- recursively rewrite each branch independently
- if the surrounding context contributes a conjunctive guard, use `Conjunction Distributes Over Disjunction` to push that guard into each branch
- rebuild the `Or`

This is not a new equivalence rule. It is a structural recursion rule needed because the master equation and local rules often introduce disjunctions.

### Pass Ordering

A reasonable pass schedule is:

1. boolean normalization
2. local equivalence passes:
   - `F -> U`
   - `R -> U \/ G`
   - `U && FG`
   - `GU && U`
   - `UG && G`
   - `U && G`
   - `U && U`
    - `F && FG`
   - restricted plain-`G` merge
3. distribute conjunctive guards across `Or` branches when needed
4. recurse into newly created subterms and continue applying the same equivalences
5. when only `GU...` terms and a plain global guard remain, apply the base-case `GU... && Gq` rewrite
6. boolean normalization again

### Termination and Strategy

The local rules should not be fired indiscriminately forever.
In particular:

- some rules should only apply when they move the formula closer to the target fragment
- the master equation should decrease $|J|$ in each recursive branch
- boolean normalization should canonicalize after each major rewrite so equivalent forms are recognized
- plain `G`-merge should not destroy `GU && U` redexes or the final `GU... && Gq` base-case redex before they are used
- distribution should be applied when it exposes new temporal/local redexes in `Or` branches

In implementation terms, this suggests:

- local rewrite passes that run to fixed point
- recursive traversal so newly created subterms are rewritten too
- a restricted operational version of `G`-conjunction merge
- explicit use of conjunction-over-disjunction distribution when branchwise temporal rewriting is needed
- a dedicated base-case rewrite for conjunctions of `G(q_i U r_i)` with a plain global guard
- matching logic that recognizes lowered `F G b` as `(\top U G b)` after the global `F -> U` pass
- then normalization
