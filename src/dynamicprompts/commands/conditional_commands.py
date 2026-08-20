from __future__ import annotations

import dataclasses

from dynamicprompts.commands.base import Command
from dynamicprompts.enums import SamplingMethod


@dataclasses.dataclass(frozen=True)
class Condition:
    """A comparison condition for if/else commands."""

    left: Command
    operator: str  # ==, !=, >, <, >=, <=, empty, !empty
    right: Command | None = None  # None for unary ops (empty, !empty)


@dataclasses.dataclass(frozen=True)
class IfCommand(Command):
    """Conditional if/else: ?{expr op expr $$ then $$ else}"""

    condition: Condition
    if_value: Command
    else_value: Command | None = None
    sampling_method: SamplingMethod | None = None


@dataclasses.dataclass(frozen=True)
class SwitchCase:
    """A single case in a switch command."""

    label: str  # The match value, or "_" for default
    value: Command
    fall_through: bool = False  # If True, continue to next case


@dataclasses.dataclass(frozen=True)
class SwitchCommand(Command):
    """Switch/case: ?{expr $$ label: value | label: value | _: default}"""

    expr: Command
    cases: list[SwitchCase] = dataclasses.field(default_factory=list)
    sampling_method: SamplingMethod | None = None
