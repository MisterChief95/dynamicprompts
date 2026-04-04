from __future__ import annotations

import dataclasses
import logging

from dynamicprompts.commands import (
    Command,
    IfCommand,
    LiteralCommand,
    SequenceCommand,
    SwitchCommand,
    VariantCommand,
    WildcardCommand,
    WrapCommand,
)
from dynamicprompts.commands.conditional_commands import Condition
from dynamicprompts.commands.variable_commands import (
    VariableAccessCommand,
    VariableAssignmentCommand,
)
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.sampling_result import SamplingResult
from dynamicprompts.types import ResultGen
from dynamicprompts.utils import rotate_and_join

logger = logging.getLogger(__name__)


class Sampler:
    def generator_from_command(
        self,
        command: Command,
        context: SamplingContext,
    ) -> ResultGen:
        # This is purposely not a dict lookup/getattr magic thing, to make
        # it easier for code completion etc. to see what's going on.
        if isinstance(command, LiteralCommand):
            return self._get_literal(command, context)
        if isinstance(command, SequenceCommand):
            return self._get_sequence(command, context)
        if isinstance(command, VariantCommand):
            return self._get_variant(command, context)
        if isinstance(command, WildcardCommand):
            return self._get_wildcard(command, context)
        if isinstance(command, VariableAssignmentCommand):
            return self._get_variable_assignment(command, context)
        if isinstance(command, VariableAccessCommand):
            return self._get_variable(command, context)
        if isinstance(command, WrapCommand):
            return self._get_wrap(command, context)
        if isinstance(command, IfCommand):
            return self._get_if(command, context)
        if isinstance(command, SwitchCommand):
            return self._get_switch(command, context)
        return self._unsupported_command(command)

    def _unsupported_command(self, command: Command) -> ResultGen:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support {command.__class__.__name__}",
        )

    def _get_wildcard(
        self,
        command: WildcardCommand,
        context: SamplingContext,
    ) -> ResultGen:
        return self._unsupported_command(command)

    def _get_variant(
        self,
        command: VariantCommand,
        context: SamplingContext,
    ) -> ResultGen:
        return self._unsupported_command(command)

    def _get_sequence(
        self,
        command: SequenceCommand,
        context: SamplingContext,
    ) -> ResultGen:
        # Pre-process top-level assignment tokens: strip them from the token
        # list and bake their values into context once (before the loop).
        # This preserves "immediate" assignment semantics where the variable is
        # sampled exactly once and reused across all generated results.
        pre_vars = context.variables
        tokens, context = context.process_variable_assignments(command.tokens)
        # Variables newly introduced by the stripped assignments — bubble these
        # up through results so outer sequences (e.g. a variant's parent) see them.
        new_vars: tuple[tuple[str, object], ...] = tuple(
            (name, val)
            for name, val in context.variables.items()
            if name not in pre_vars
        )

        # Create generators once so stateful samplers (e.g. cyclical) maintain
        # their position across successive calls.
        sub_generators = [context.generator_from_command(c) for c in tokens]

        while True:
            results: list[SamplingResult] = []
            current_context = context
            extra_vars: dict[str, object] = {}
            for i, gen in enumerate(sub_generators):
                if extra_vars:
                    # Variables from an earlier token in this cycle need to be
                    # visible when sampling this token.  Create a one-shot
                    # generator with the updated context rather than advancing
                    # the persistent generator (which has the old context).
                    result = next(
                        current_context.generator_from_command(tokens[i])
                    )
                else:
                    result = next(gen)
                # Collect variables from branch assignments that bubbled up.
                if result.variables:
                    extra_vars.update(dict(result.variables))
                    current_context = current_context.with_variables(
                        dict(result.variables),
                    )
                results.append(result)
            joined = SamplingResult.joined_with_affixes(
                results,
                separator=command.separator,
                prefix="",
                suffix="",
            )
            all_new: dict[str, object] = dict(new_vars)
            all_new.update(extra_vars)
            if all_new:
                yield dataclasses.replace(joined, variables=tuple(all_new.items()))
            else:
                yield joined

    def _get_literal(
        self,
        command: LiteralCommand,
        context: SamplingContext,
    ) -> ResultGen:
        while True:
            yield SamplingResult(text=command.literal)

    def _get_variable_assignment(
        self,
        command: VariableAssignmentCommand,
        context: SamplingContext,
    ) -> ResultGen:
        """
        Yield a single empty-text result carrying the variable assignment.
        The surrounding sequence sampler applies the variable to the context
        for subsequent tokens, and the result bubbles up through joined results
        so outer sequences can also see it (e.g. assignments inside a variant branch).
        """
        resolved = context.process_variable_assignment(command)
        yield SamplingResult(text="", variables=((command.name, resolved),))

    def _get_variable(
        self,
        command: VariableAccessCommand,
        context: SamplingContext,
    ) -> ResultGen:
        variable = command.name
        command_to_sample = context.variables.get(variable, command.default)
        if not command_to_sample:
            if context.unknown_variable_value is None:
                raise KeyError(f"Variable {variable} is not defined in this context")
            elif isinstance(context.unknown_variable_value, str):
                command_to_sample = LiteralCommand(context.unknown_variable_value)
            else:
                command_to_sample = context.unknown_variable_value
        return context.for_sampling_variable(variable).generator_from_command(
            command_to_sample,
        )

    def _get_wrap(
        self,
        command: WrapCommand,
        context: SamplingContext,
    ) -> ResultGen:
        return self._unsupported_command(command)

    @staticmethod
    def _evaluate_condition(
        condition: Condition,
        context: SamplingContext,
    ) -> bool:
        """Evaluate a Condition by sampling its operands to string values."""
        _NUMERIC_OPS = (">", "<", ">=", "<=")

        left_text = next(context.generator_from_command(condition.left)).text.strip()

        if condition.operator in ("empty", "!empty"):
            is_empty = left_text == ""
            return is_empty if condition.operator == "empty" else not is_empty

        assert condition.right is not None, "Binary operators require a right operand"
        right_text = next(context.generator_from_command(condition.right)).text.strip()

        if condition.operator in _NUMERIC_OPS:
            try:
                left_val = float(left_text)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot compare non-numeric value {left_text!r} with operator {condition.operator!r}",
                ) from None
            try:
                right_val = float(right_text)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Cannot compare non-numeric value {right_text!r} with operator {condition.operator!r}",
                ) from None
            if condition.operator == ">":
                return left_val > right_val
            elif condition.operator == "<":
                return left_val < right_val
            elif condition.operator == ">=":
                return left_val >= right_val
            else:  # <=
                return left_val <= right_val

        # String comparison
        if condition.operator == "==":
            return left_text == right_text
        elif condition.operator == "!=":
            return left_text != right_text
        else:
            raise ValueError(f"Unknown operator: {condition.operator!r}")

    def _get_if(
        self,
        command: IfCommand,
        context: SamplingContext,
    ) -> ResultGen:
        while True:
            condition_result = self._evaluate_condition(command.condition, context)
            if condition_result:
                result = next(context.generator_from_command(command.if_value))
            elif command.else_value is not None:
                result = next(context.generator_from_command(command.else_value))
            else:
                result = SamplingResult(text="")
            yield result

    def _get_switch(
        self,
        command: SwitchCommand,
        context: SamplingContext,
    ) -> ResultGen:
        while True:
            expr_text = next(context.generator_from_command(command.expr)).text.strip()

            # Find the matching case
            match_idx = None
            default_idx = None
            for i, case in enumerate(command.cases):
                if case.label == "_":
                    default_idx = i
                elif case.label.strip() == expr_text:
                    match_idx = i
                    break

            start_idx = match_idx if match_idx is not None else default_idx

            if start_idx is None:
                yield SamplingResult(text="")
                continue

            # Collect values: from match point, continue through fall-throughs.
            # Thread context through cases so variables assigned in earlier
            # fall-through cases are visible to later ones.
            results = []
            current_context = context
            for i in range(start_idx, len(command.cases)):
                case = command.cases[i]
                val = next(current_context.generator_from_command(case.value))
                if val.variables:
                    current_context = current_context.with_variables(
                        dict(val.variables),
                    )
                results.append(val)
                if not case.fall_through:
                    break

            joined = SamplingResult.joined(results, separator=", ")
            # Bubble up variables assigned inside the matched case(s) so the
            # outer sequence can apply them to subsequent tokens.
            all_vars: dict[str, object] = {}
            for r in results:
                all_vars.update(dict(r.variables))
            if all_vars:
                yield dataclasses.replace(joined, variables=tuple(all_vars.items()))
            else:
                yield joined
