from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Generator, Iterable

from dynamicprompts.commands.base import Command
from dynamicprompts.commands.literal_command import LiteralCommand
from dynamicprompts.enums import SamplingMethod

if TYPE_CHECKING:
    from dynamicprompts.sampling_context import SamplingContext

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class VariantOption:
    value: Command
    weight: float = 1.0


@dataclasses.dataclass(frozen=True)
class VariantCommand(Command):
    variants: list[VariantOption]
    min_bound: int | Command = 1
    max_bound: int | Command = 1
    separator: str = ","
    prefix: str = ""
    suffix: str = ""
    sampling_method: SamplingMethod | None = None

    def __post_init__(self):
        if isinstance(self.min_bound, int) and isinstance(self.max_bound, int):
            min_bound, max_bound = sorted((self.min_bound, self.max_bound))
            min_bound = max(0, min_bound)
            object.__setattr__(self, "min_bound", min_bound)
            object.__setattr__(self, "max_bound", max_bound)

    def resolve_bounds(self, context: SamplingContext) -> tuple[int, int]:
        """Resolve min_bound and max_bound to ints, sampling variable commands if needed."""

        def _resolve(bound: int | Command) -> int:
            if isinstance(bound, int):
                return bound
            text = next(iter(context.generator_from_command(bound))).text.strip()
            try:
                return int(text)
            except ValueError:
                logger.warning(
                    f"Variable bound resolved to non-integer {text!r}, defaulting to 1",
                )
                return 1

        min_b = _resolve(self.min_bound)
        max_b = _resolve(self.max_bound)
        if min_b > max_b:
            min_b, max_b = max_b, min_b
        min_b = max(0, min_b)
        return min_b, max_b

    def __len__(self) -> int:
        return len(self.variants)

    def __getitem__(self, index: int) -> VariantOption:
        return self.variants[index]

    def __iter__(self) -> Iterable[VariantOption]:
        return iter(self.variants)

    @property
    def weights(self) -> list[float]:
        return [p.weight for p in self.variants]

    @property
    def values(self) -> list[Command]:
        return [p.value for p in self.variants]

    def adjust_range(self, context: SamplingContext | None = None) -> VariantCommand:
        if isinstance(self.min_bound, Command) or isinstance(self.max_bound, Command):
            if context is None:
                raise ValueError(
                    "context required to adjust_range when bounds are variable commands",
                )
            min_bound, max_bound = self.resolve_bounds(context)
        else:
            min_bound, max_bound = self.min_bound, self.max_bound
        min_bound = min(min_bound, len(self.values))
        max_bound = min(max_bound, len(self.values))
        return dataclasses.replace(self, min_bound=min_bound, max_bound=max_bound)

    @classmethod
    def from_literals_and_weights(
        cls,
        literals: list[str],
        weights: list[float] | None = None,
        min_bound: int = 1,
        max_bound: int = 1,
        separator: str = ",",
        prefix: str = "",
        suffix: str = "",
        sampling_method: SamplingMethod | None = None,
    ) -> VariantCommand:
        vals = [LiteralCommand(str(v)) for v in literals]
        if weights is None:
            weights = [1.0] * len(vals)
        assert len(vals) == len(weights), "Must have same number of weights as values"
        return VariantCommand(
            variants=[VariantOption(v, w) for v, w in zip(vals, weights)],
            min_bound=min_bound,
            max_bound=max_bound,
            separator=separator,
            prefix=prefix,
            suffix=suffix,
            sampling_method=sampling_method,
        )

    def get_value_combinations(
        self,
        k: int,
        values=None,
    ) -> Generator[list[Command], None, None]:
        if values is None:
            values = self.values

        if k <= 0:
            yield []
        else:
            for value in values:
                other_values = [v for v in values if v != value]
                for item in self.get_value_combinations(k - 1, values=other_values):
                    yield [value] + item
