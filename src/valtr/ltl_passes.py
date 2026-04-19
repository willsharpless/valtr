from __future__ import annotations

from .ltl_ir import And, ExprId, Finally, Globally, Release, Until
from .ltl_matchers import (
    conjuncts,
    disjuncts,
    get_gu,
    get_node,
    get_ug,
    get_until,
    is_false,
    is_fg_lowered,
    is_plain_global_guard,
    is_propositional,
    is_true,
    replace_expr_ids,
    without_expr_ids,
)
from .ltl_rewriter import LTLRewriter


class PassNormalizeBoolean(LTLRewriter):
    pass


class PassLowerFinally(LTLRewriter):
    def rewrite_finally(self, node: Finally, arg: ExprId) -> ExprId:
        if node.interval is not None:
            return super().rewrite_finally(node, arg)
        self.mark_changed()
        true_expr = self.dst.true(self.origin_for("lower_finally_true", arg, original=node.origin))
        return self.dst.until(true_expr, arg, None, self.origin_for("lower_finally", true_expr, arg, original=node.origin))


class PassLowerRelease(LTLRewriter):
    def rewrite_release(self, node: Release, left: ExprId, right: ExprId) -> ExprId:
        if node.interval is not None:
            return super().rewrite_release(node, left, right)
        self.mark_changed()
        conj = self.dst.and_((left, right), self.origin_for("lower_release_guard", left, right, original=node.origin))
        until_expr = self.dst.until(
            right,
            conj,
            None,
            self.origin_for("lower_release_until", right, conj, original=node.origin),
        )
        globally_expr = self.dst.globally(
            right,
            None,
            self.origin_for("lower_release_globally", right, original=node.origin),
        )
        return self.dst.or_(
            (until_expr, globally_expr),
            self.origin_for("lower_release", until_expr, globally_expr, original=node.origin),
        )


class PassSimplifyTemporal(LTLRewriter):
    def rewrite_globally(self, node: Globally, arg: ExprId) -> ExprId:
        if node.interval is None:
            inner = get_node(self.dst.nodes, arg)
            if isinstance(inner, Globally) and inner.interval is None:
                self.mark_changed()
                return arg
        return super().rewrite_globally(node, arg)

    def rewrite_finally(self, node: Finally, arg: ExprId) -> ExprId:
        if node.interval is None:
            inner = get_node(self.dst.nodes, arg)
            if isinstance(inner, Finally) and inner.interval is None:
                self.mark_changed()
                return arg
        return super().rewrite_finally(node, arg)

    def rewrite_until(self, node: Until, left: ExprId, right: ExprId) -> ExprId:
        if node.interval is not None:
            return super().rewrite_until(node, left, right)
        if int(left) == int(right):
            self.mark_changed()
            return left
        if is_false(left, self.dst.nodes):
            self.mark_changed()
            return right
        if is_true(right, self.dst.nodes):
            self.mark_changed()
            return right
        return super().rewrite_until(node, left, right)


class AndRulePass(LTLRewriter):
    def rewrite_and(self, node: And, args: tuple[ExprId, ...]) -> ExprId:
        rewritten = self.try_rewrite_and(args, node.origin)
        if rewritten is not None:
            self.mark_changed()
            return rewritten
        return super().rewrite_and(node, args)

    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        raise NotImplementedError


class PassUntilAndFG(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        until_id: ExprId | None = None
        fg_id: ExprId | None = None
        for expr_id in args:
            if fg_id is None and is_fg_lowered(expr_id, self.dst.nodes):
                fg_id = expr_id
                continue
            until_match = get_until(expr_id, self.dst.nodes)
            if until_id is None and until_match is not None:
                until_id = expr_id
        if until_id is None or fg_id is None:
            return None
        until_match = get_until(until_id, self.dst.nodes)
        assert until_match is not None
        combined_right = self.dst.and_(
            (until_match.right, fg_id),
            self.origin_for("until_and_fg_right", until_match.right, fg_id, original=origin),
        )
        new_until = self.dst.until(
            until_match.left,
            combined_right,
            None,
            self.origin_for("until_and_fg", until_match.left, combined_right, original=origin),
        )
        replacements = {int(until_id): new_until}
        return self.dst.and_(
            replace_expr_ids(args, replacements, remove={int(fg_id)}),
            self.origin_for("until_and_fg_and", new_until, original=origin),
        )


class PassUntilGloballyAndGlobally(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        ug_id: ExprId | None = None
        g_id: ExprId | None = None
        for expr_id in args:
            if ug_id is None and get_ug(expr_id, self.dst.nodes) is not None:
                ug_id = expr_id
                continue
            if g_id is None and is_plain_global_guard(expr_id, self.dst.nodes):
                g_id = expr_id
        if ug_id is None or g_id is None:
            return None
        ug = get_ug(ug_id, self.dst.nodes)
        guard_global = get_node(self.dst.nodes, g_id)
        assert ug is not None
        assert isinstance(guard_global, Globally)
        merged_guard = self.dst.and_(
            (ug.guard, guard_global.arg),
            self.origin_for("ug_and_g_guard", ug.guard, guard_global.arg, original=origin),
        )
        new_global = self.dst.globally(
            merged_guard,
            None,
            self.origin_for("ug_and_g_global", merged_guard, original=origin),
        )
        new_until = self.dst.until(
            self.dst.and_((ug.left, guard_global.arg), self.origin_for("ug_and_g_left", ug.left, guard_global.arg, original=origin)),
            new_global,
            None,
            self.origin_for("ug_and_g", ug.left, new_global, original=origin),
        )
        replacements = {int(ug_id): new_until}
        return self.dst.and_(
            replace_expr_ids(args, replacements, remove={int(g_id)}),
            self.origin_for("ug_and_g_and", new_until, original=origin),
        )


class PassUntilAndGlobally(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        until_id: ExprId | None = None
        g_id: ExprId | None = None
        for expr_id in args:
            if until_id is None and get_until(expr_id, self.dst.nodes) is not None and get_ug(expr_id, self.dst.nodes) is None:
                until_id = expr_id
                continue
            if g_id is None and is_plain_global_guard(expr_id, self.dst.nodes):
                g_id = expr_id
        if until_id is None or g_id is None:
            return None
        until_match = get_until(until_id, self.dst.nodes)
        guard_global = get_node(self.dst.nodes, g_id)
        assert until_match is not None
        assert isinstance(guard_global, Globally)
        new_left = self.dst.and_(
            (until_match.left, guard_global.arg),
            self.origin_for("until_and_g_left", until_match.left, guard_global.arg, original=origin),
        )
        new_right = self.dst.and_(
            (until_match.right, g_id),
            self.origin_for("until_and_g_right", until_match.right, g_id, original=origin),
        )
        new_until = self.dst.until(
            new_left,
            new_right,
            None,
            self.origin_for("until_and_g", new_left, new_right, original=origin),
        )
        replacements = {int(until_id): new_until}
        return self.dst.and_(
            replace_expr_ids(args, replacements, remove={int(g_id)}),
            self.origin_for("until_and_g_and", new_until, original=origin),
        )


class PassUntilAndUntil(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        first_id: ExprId | None = None
        second_id: ExprId | None = None
        for expr_id in args:
            if get_until(expr_id, self.dst.nodes) is None:
                continue
            if first_id is None:
                first_id = expr_id
            elif second_id is None:
                second_id = expr_id
                break
        if first_id is None or second_id is None:
            return None
        first = get_until(first_id, self.dst.nodes)
        second = get_until(second_id, self.dst.nodes)
        assert first is not None and second is not None
        new_left = self.dst.and_(
            (first.left, second.left),
            self.origin_for("until_and_until_left", first.left, second.left, original=origin),
        )
        branch1 = self.dst.and_(
            (first.right, second_id),
            self.origin_for("until_and_until_branch1", first.right, second_id, original=origin),
        )
        branch2 = self.dst.and_(
            (second.right, first_id),
            self.origin_for("until_and_until_branch2", second.right, first_id, original=origin),
        )
        new_right = self.dst.or_(
            (branch1, branch2),
            self.origin_for("until_and_until_or", branch1, branch2, original=origin),
        )
        new_until = self.dst.until(
            new_left,
            new_right,
            None,
            self.origin_for("until_and_until", new_left, new_right, original=origin),
        )
        replacements = {int(first_id): new_until}
        return self.dst.and_(
            replace_expr_ids(args, replacements, remove={int(second_id)}),
            self.origin_for("until_and_until_and", new_until, original=origin),
        )


class PassGloballyUntilAndUntil(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        gu_id: ExprId | None = None
        until_id: ExprId | None = None
        for expr_id in args:
            if gu_id is None and get_gu(expr_id, self.dst.nodes) is not None:
                gu_id = expr_id
                continue
            if until_id is None and get_until(expr_id, self.dst.nodes) is not None:
                until_id = expr_id
        if gu_id is None or until_id is None:
            return None
        gu = get_gu(gu_id, self.dst.nodes)
        until_match = get_until(until_id, self.dst.nodes)
        assert gu is not None and until_match is not None
        guard = self.dst.or_(
            (gu.left, gu.right),
            self.origin_for("gu_and_u_guard_or", gu.left, gu.right, original=origin),
        )
        new_left = self.dst.and_(
            (guard, until_match.left),
            self.origin_for("gu_and_u_left", guard, until_match.left, original=origin),
        )
        new_right = self.dst.and_(
            (until_match.right, gu_id),
            self.origin_for("gu_and_u_right", until_match.right, gu_id, original=origin),
        )
        new_until = self.dst.until(
            new_left,
            new_right,
            None,
            self.origin_for("gu_and_u", new_left, new_right, original=origin),
        )
        replacements = {int(until_id): new_until}
        return self.dst.and_(
            replace_expr_ids(args, replacements, remove={int(gu_id)}),
            self.origin_for("gu_and_u_and", new_until, original=origin),
        )


class PassCollectPlainGlobalGuards(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        guard_ids = [expr_id for expr_id in args if is_plain_global_guard(expr_id, self.dst.nodes)]
        if len(guard_ids) < 2:
            return None
        inner_args = tuple(get_node(self.dst.nodes, expr_id).arg for expr_id in guard_ids)
        merged_inner = self.dst.and_(
            inner_args,
            self.origin_for("restricted_global_guard_merge_inner", *inner_args, original=origin),
        )
        merged_guard = self.dst.globally(
            merged_inner,
            None,
            self.origin_for("restricted_global_guard_merge", merged_inner, original=origin),
        )
        kept = without_expr_ids(args, {int(expr_id) for expr_id in guard_ids})
        return self.dst.and_(
            (*kept, merged_guard),
            self.origin_for("restricted_global_guard_merge_and", merged_guard, original=origin),
        )


class PassRestrictedGlobalGuardMerge(PassCollectPlainGlobalGuards):
    """Backward-compatible alias for the original pass name."""


class PassApplyCollectedPlainGlobalGuardsToGU(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        guard_ids = [expr_id for expr_id in args if is_plain_global_guard(expr_id, self.dst.nodes)]
        if len(guard_ids) != 1:
            return None

        guard_id = guard_ids[0]
        guard_node = get_node(self.dst.nodes, guard_id)
        assert isinstance(guard_node, Globally)

        replacements: dict[int, ExprId] = {}
        for expr_id in args:
            gu = get_gu(expr_id, self.dst.nodes)
            if gu is None:
                continue
            left_conjunct_ids = {int(child) for child in conjuncts(gu.left, self.dst.nodes)}
            right_conjunct_ids = {int(child) for child in conjuncts(gu.right, self.dst.nodes)}
            guard_inner_id = int(guard_node.arg)
            if guard_inner_id in left_conjunct_ids and guard_inner_id in right_conjunct_ids:
                continue

            left = self.dst.and_(
                (gu.left, guard_node.arg),
                self.origin_for("apply_collected_g_to_gu_left", gu.left, guard_node.arg, original=origin),
            )
            right = self.dst.and_(
                (gu.right, guard_node.arg),
                self.origin_for("apply_collected_g_to_gu_right", gu.right, guard_node.arg, original=origin),
            )
            inner_until = self.dst.until(
                left,
                right,
                None,
                self.origin_for("apply_collected_g_to_gu_until", left, right, original=origin),
            )
            replacements[int(expr_id)] = self.dst.globally(
                inner_until,
                None,
                self.origin_for("apply_collected_g_to_gu", inner_until, original=origin),
            )

        if not replacements:
            return None

        return self.dst.and_(
            replace_expr_ids(args, replacements),
            self.origin_for("apply_collected_g_to_gu_and", *replacements.values(), original=origin),
        )


class PassDistributeAndOverOr(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        or_id: ExprId | None = None
        for expr_id in args:
            if isinstance(get_node(self.dst.nodes, expr_id), And):
                continue
            if get_node(self.dst.nodes, expr_id).__class__.__name__ == "Or":
                or_id = expr_id
                break
        if or_id is None:
            return None
        remaining = without_expr_ids(args, {int(or_id)})
        if not remaining:
            return None
        branches = disjuncts(or_id, self.dst.nodes)
        # Avoid blowing up purely propositional formulas; distribution here is
        # only intended to expose temporal redexes inside disjunctive branches.
        if all(is_propositional(expr_id, self.dst.nodes) for expr_id in remaining) and all(
            is_propositional(branch, self.dst.nodes) for branch in branches
        ):
            return None
        branches = tuple(
            self.dst.and_(
                (*remaining, branch),
                self.origin_for("distribute_and_over_or_branch", *remaining, branch, original=origin),
            )
            for branch in branches
        )
        return self.dst.or_(
            branches,
            self.origin_for("distribute_and_over_or", *branches, original=origin),
        )


class PassGUBaseCase(AndRulePass):
    def try_rewrite_and(self, args: tuple[ExprId, ...], origin) -> ExprId | None:
        if any(get_until(expr_id, self.dst.nodes) is not None for expr_id in args):
            return None
        gu_ids = [expr_id for expr_id in args if get_gu(expr_id, self.dst.nodes) is not None]
        guard_ids = [expr_id for expr_id in args if is_plain_global_guard(expr_id, self.dst.nodes)]
        if not gu_ids or len(guard_ids) != 1:
            return None
        guard_id = guard_ids[0]
        guard_node = get_node(self.dst.nodes, guard_id)
        assert isinstance(guard_node, Globally)
        rewritten_gu_ids: list[ExprId] = []
        for gu_id in gu_ids:
            gu = get_gu(gu_id, self.dst.nodes)
            assert gu is not None
            left = self.dst.and_(
                (gu.left, guard_node.arg),
                self.origin_for("gu_base_case_left", gu.left, guard_node.arg, original=origin),
            )
            right = self.dst.and_(
                (gu.right, guard_node.arg),
                self.origin_for("gu_base_case_right", gu.right, guard_node.arg, original=origin),
            )
            inner_until = self.dst.until(
                left,
                right,
                None,
                self.origin_for("gu_base_case_until", left, right, original=origin),
            )
            rewritten_gu_ids.append(
                self.dst.globally(
                    inner_until,
                    None,
                    self.origin_for("gu_base_case_global", inner_until, original=origin),
                )
            )
        kept = without_expr_ids(args, {int(guard_id), *(int(expr_id) for expr_id in gu_ids)})
        return self.dst.and_(
            (*kept, *rewritten_gu_ids),
            self.origin_for("gu_base_case", *rewritten_gu_ids, original=origin),
        )


PRE_PASSES = (
    PassNormalizeBoolean,
    PassLowerFinally,
    PassLowerRelease,
    PassSimplifyTemporal,
)


LOOP_PASSES = (
    PassCollectPlainGlobalGuards,
    PassApplyCollectedPlainGlobalGuardsToGU,
    PassUntilAndFG,
    PassGloballyUntilAndUntil,
    PassUntilGloballyAndGlobally,
    PassUntilAndGlobally,
    PassUntilAndUntil,
    PassDistributeAndOverOr,
    PassGUBaseCase,
    PassNormalizeBoolean,
)
