"""
Tests for variable assignment inside variant branches, if/else branches, and
switch cases, e.g.: `{cat ${col=orange}|dog ${col=brown}}, ${col} fur`
"""
from __future__ import annotations

from dynamicprompts.enums import SamplingMethod
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(wildcard_manager: WildcardManager, method: SamplingMethod) -> SamplingContext:
    return SamplingContext(
        wildcard_manager=wildcard_manager,
        default_sampling_method=method,
        unknown_variable_value="UNSET",
    )


def _sample_all(ctx: SamplingContext, prompt: str, n: int = 20) -> list[str]:
    return [str(r) for r in ctx.sample_prompts(prompt, n)]


# ---------------------------------------------------------------------------
# Variant branch variable assignment
# ---------------------------------------------------------------------------

class TestVariantBranchAssignment:
    def test_variable_follows_chosen_branch_random(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        results = _sample_all(ctx, "{cat ${col=orange}|dog ${col=brown}}, ${col} fur", 40)
        for r in results:
            if "cat" in r:
                assert "orange fur" in r, r
            else:
                assert "dog" in r
                assert "brown fur" in r, r

    def test_variable_follows_chosen_branch_cyclical(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.CYCLICAL)
        results = _sample_all(ctx, "{cat ${col=orange}|dog ${col=brown}}, ${col} fur", 4)
        for r in results:
            if "cat" in r:
                assert "orange fur" in r, r
            else:
                assert "dog" in r
                assert "brown fur" in r, r

    def test_variable_follows_chosen_branch_combinatorial(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.COMBINATORIAL)
        results = _sample_all(ctx, "{cat ${col=orange}|dog ${col=brown}}, ${col} fur")
        # trailing space before comma because the literals are "cat " / "dog "
        assert sorted(results) == sorted(["cat , orange fur", "dog , brown fur"])

    def test_assignment_strips_from_output(self, wildcard_manager):
        """The assignment token itself should not appear as text in the output."""
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        results = _sample_all(ctx, "{${col=red}|${col=blue}}, ${col} sky", 20)
        for r in results:
            assert "${col=" not in r, r
            assert r in [", red sky", ", blue sky"]

    def test_later_branch_overrides_earlier_assignment(self, wildcard_manager):
        """An assignment in the selected branch replaces a prior top-level assignment."""
        ctx = _ctx(wildcard_manager, SamplingMethod.COMBINATORIAL)
        # Top-level col=green, but the branch always sets it to red or blue.
        results = _sample_all(
            ctx, "${col=green}{${col=red}|${col=blue}}, ${col} sky"
        )
        assert sorted(results) == sorted([", red sky", ", blue sky"])

    def test_no_cross_contamination_between_samples(self, wildcard_manager):
        """Variables set in one sample must not leak into the next sample."""
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        r1 = str(next(ctx.sample_prompts("{cat ${col=orange}|dog ${col=brown}}, ${col} coat")))
        r2 = str(next(ctx.sample_prompts("{cat ${col=orange}|dog ${col=brown}}, ${col} coat")))
        for r in [r1, r2]:
            if "cat" in r:
                assert "orange coat" in r
            else:
                assert "brown coat" in r


# ---------------------------------------------------------------------------
# If/else branch variable assignment
# ---------------------------------------------------------------------------

class TestIfBranchAssignment:
    def test_if_branch_sets_variable(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        # time is always "day", so the if branch is taken.
        prompt = "${time=day}?{${time} == day $$ ${label=sunny} $$ ${label=dark}}, ${label} sky"
        result = str(next(ctx.sample_prompts(prompt)))
        assert "sunny sky" in result, result

    def test_else_branch_sets_variable(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        # time is always "night", so the else branch is taken.
        prompt = "${time=night}?{${time} == day $$ ${label=sunny} $$ ${label=dark}}, ${label} sky"
        result = str(next(ctx.sample_prompts(prompt)))
        assert "dark sky" in result, result


# ---------------------------------------------------------------------------
# Assignment-only variant (no surrounding text)
# ---------------------------------------------------------------------------

class TestAssignmentOnlyVariant:
    def test_pure_assignment_in_variant_combinatorial(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.COMBINATORIAL)
        results = _sample_all(ctx, "{${x=a}|${x=b}}${x}")
        assert sorted(results) == sorted(["a", "b"])

    def test_pure_assignment_in_variant_random(self, wildcard_manager):
        ctx = _ctx(wildcard_manager, SamplingMethod.RANDOM)
        results = _sample_all(ctx, "{${x=a}|${x=b}}${x}", 20)
        for r in results:
            assert r in ("a", "b"), r
