"""
Tests for boolean variable type.

Syntax:
  - ${name=bool}   → random true/false, re-sampled each generation
  - ${name=!bool}  → random true/false, sampled once and reused (immediate)
  - Unary check:   ?{${name} $$ then $$ else}
  - Negation:      ?{!${name} $$ then $$ else}
  - Blank/null resolves as false in boolean context.
"""
from __future__ import annotations

import pytest
from dynamicprompts.commands import LiteralCommand, VariantCommand, VariantOption
from dynamicprompts.commands.conditional_commands import Condition, IfCommand
from dynamicprompts.commands.variable_commands import (
    VariableAccessCommand,
    VariableAssignmentCommand,
)
from dynamicprompts.enums import SamplingMethod
from dynamicprompts.parser.parse import parse
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(method: SamplingMethod = SamplingMethod.RANDOM, unknown: str = "") -> SamplingContext:
    return SamplingContext(
        wildcard_manager=WildcardManager(),
        default_sampling_method=method,
        unknown_variable_value=unknown,
    )


def _sample(prompt: str, method: SamplingMethod = SamplingMethod.RANDOM, unknown: str = "") -> str:
    ctx = _make_ctx(method=method, unknown=unknown)
    return str(next(ctx.sample_prompts(prompt))).strip()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestBooleanAssignmentParsing:
    def test_bool_declaration_is_boolean_flag(self):
        cmd = parse("${isThing=bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.name == "isThing"
        assert cmd.is_boolean is True
        assert isinstance(cmd.value, VariantCommand)
        assert len(cmd.value.variants) == 2

    def test_bool_declaration_immediate_is_false(self):
        """${x=bool} — not immediate, re-sampled each generation."""
        cmd = parse("${isThing=bool}")
        assert cmd.immediate is False

    def test_bool_immediate_declaration(self):
        """${x=!bool} — immediate, sampled once and reused."""
        cmd = parse("${isThing=!bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.is_boolean is True
        assert cmd.immediate is True
        assert isinstance(cmd.value, VariantCommand)
        assert len(cmd.value.variants) == 2

    def test_preserve_bool_declaration(self):
        """${x?=bool} should set is_boolean and preserve_existing."""
        cmd = parse("${x?=bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.is_boolean is True
        assert cmd.overwrite is False

    def test_preserve_bool_immediate_declaration(self):
        """${x?=!bool} should set is_boolean, immediate, and preserve_existing."""
        cmd = parse("${x?=!bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.is_boolean is True
        assert cmd.immediate is True
        assert cmd.overwrite is False


class TestBooleanConditionalParsing:
    def test_unary_bool_check_parses(self):
        cmd = parse("?{${flag} $$ yes $$ no}")
        assert isinstance(cmd, IfCommand)
        assert isinstance(cmd.condition, Condition)
        assert cmd.condition.operator == "bool"
        assert isinstance(cmd.condition.left, VariableAccessCommand)
        assert cmd.condition.left.name == "flag"
        assert cmd.condition.right is None

    def test_negated_bool_check_parses(self):
        cmd = parse("?{!${flag} $$ yes $$ no}")
        assert isinstance(cmd, IfCommand)
        assert cmd.condition.operator == "!bool"

    def test_unary_bool_no_else(self):
        cmd = parse("?{${flag} $$ yes}")
        assert isinstance(cmd, IfCommand)
        assert cmd.condition.operator == "bool"
        assert cmd.else_value is None


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

class TestBooleanEvaluation:
    def test_bool_random_per_generation(self):
        """${flag=bool} produces both true and false across multiple generations."""
        ctx = _make_ctx(method=SamplingMethod.RANDOM)
        results = {str(r).strip() for r in ctx.sample_prompts("${flag=bool}?{${flag} $$ yes $$ no}", 20)}
        assert "yes" in results and "no" in results

    def test_bool_immediate_sampled_once(self):
        """${flag=!bool} picks once and reuses across all results in a batch."""
        ctx = _make_ctx(method=SamplingMethod.CYCLICAL)
        results = [str(r).strip() for r in ctx.sample_prompts("${flag=!bool}?{${flag} $$ yes $$ no}", 4)]
        # All 4 results must be the same value since it's immediate
        assert len(set(results)) == 1

    def test_set_true_then_check(self):
        assert _sample("${flag=true}?{${flag} $$ yes $$ no}") == "yes"

    def test_set_false_then_check(self):
        assert _sample("${flag=false}?{${flag} $$ yes $$ no}") == "no"

    def test_negation_with_false(self):
        assert _sample("${flag=false}?{!${flag} $$ yes $$ no}") == "yes"

    def test_negation_with_true(self):
        assert _sample("${flag=true}?{!${flag} $$ yes $$ no}") == "no"

    def test_blank_variable_is_false(self):
        """An unset variable resolves to '' (unknown_variable_value=''), which is falsy."""
        assert _sample("?{${undefined_flag} $$ yes $$ no}", unknown="") == "no"

    def test_case_insensitive_true(self):
        assert _sample("${flag=True}?{${flag} $$ yes $$ no}") == "yes"

    def test_bool_no_else_when_false(self):
        """When bool picks false and there's no else branch, output is empty."""
        assert _sample("${flag=false}?{${flag} $$ yes}") == ""

    def test_bool_normalization_in_context(self):
        """Setting to TRUE (all caps) should still be truthy after normalization."""
        assert _sample("${flag=TRUE}?{${flag} $$ yes $$ no}") == "yes"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBooleanBackwardCompatibility:
    def test_existing_variant_assignment_still_works(self):
        """${x=!{true|false}} + ?{${x} == true $$ ...} still works."""
        ctx = _make_ctx(method=SamplingMethod.CYCLICAL)
        results = [str(r).strip() for r in ctx.sample_prompts("${x=!{true|false}}?{${x} == true $$ a $$ b}", 4)]
        assert "a" in results or "b" in results

    def test_string_equality_still_works(self):
        assert _sample("${x=hello}?{${x} == hello $$ yes $$ no}") == "yes"
