"""
Tests for boolean variable type.

Syntax:
  - Declaration: ${name=!bool}  (defaults to false)
  - Unary check:  ?{${name} $$ then $$ else}
  - Negation:     ?{!${name} $$ then $$ else}
  - Blank/null resolves as false in boolean context.
"""
from __future__ import annotations

import pytest
from dynamicprompts.commands import LiteralCommand
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
        cmd = parse("${isThing=!bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.name == "isThing"
        assert cmd.is_boolean is True
        assert isinstance(cmd.value, LiteralCommand)
        assert cmd.value.literal == "false"

    def test_bool_declaration_immediate_is_false(self):
        """immediate should be False for bool declarations — it's meaningless since value is already a literal."""
        cmd = parse("${isThing=!bool}")
        assert cmd.immediate is False

    def test_plain_bool_string_is_not_boolean(self):
        """${x=bool} without ! should parse 'bool' as a plain literal string."""
        cmd = parse("${x=bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.is_boolean is False
        assert isinstance(cmd.value, LiteralCommand)
        assert cmd.value.literal == "bool"

    def test_preserve_bool_declaration(self):
        """${x?=!bool} should set is_boolean and preserve_existing."""
        cmd = parse("${x?=!bool}")
        assert isinstance(cmd, VariableAssignmentCommand)
        assert cmd.is_boolean is True
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
    def test_default_false(self):
        assert _sample("${flag=!bool}?{${flag} $$ yes $$ no}") == "no"

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

    def test_bool_declaration_default_false_no_else(self):
        assert _sample("${flag=!bool}?{${flag} $$ yes}") == ""

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

    def test_plain_bool_string_literal_unchanged(self):
        """${x=bool} without ! stores the literal string 'bool', no boolean logic."""
        assert _sample("${x=bool}${x}") == "bool"

    def test_string_equality_still_works(self):
        assert _sample("${x=hello}?{${x} == hello $$ yes $$ no}") == "yes"
