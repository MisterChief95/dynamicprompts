from __future__ import annotations

import dataclasses
from typing import Iterable

from dynamicprompts.commands.wrap_command import split_wrapper_string


@dataclasses.dataclass(frozen=True)
class SamplingResult:
    """
    An internal result of a sampling. May contain metadata in the future.
    """

    text: str
    # Variables assigned during sampling of this result (e.g. from inside a variant branch).
    # Excluded from hash/eq so that SamplingResult remains hashable even when variables
    # contain Commands with unhashable fields (e.g. lists inside VariantCommand).
    variables: tuple[tuple[str, object], ...] = dataclasses.field(
        default=(),
        compare=False,
        hash=False,
    )

    def __str__(self):
        return self.text

    @property
    def dedupe_key(self) -> tuple:
        # Used by e.g. combinatorial sampling's fragment deduplication.
        # Include variable names+repr so that branches that produce the same
        # text but assign different variables are not collapsed into one result.
        var_key = tuple((name, repr(val)) for name, val in self.variables)
        return (self.text, var_key)

    def whitespace_squashed(self) -> SamplingResult:
        from dynamicprompts.utils import squash_whitespace

        return dataclasses.replace(self, text=squash_whitespace(self.text))

    def commas_squashed(self) -> SamplingResult:
        from dynamicprompts.utils import squash_commas

        return dataclasses.replace(self, text=squash_commas(self.text))

    def text_replaced(self, new_text: str) -> SamplingResult:
        return dataclasses.replace(self, text=new_text)

    def as_wrapper(self):
        """
        Return a function that wraps a SamplingResult with this one,
        partitioning this result's text along the wrap marker.
        """
        prefix, suffix = split_wrapper_string(self.text)
        prefix_res = self.text_replaced(prefix)
        suffix_res = self.text_replaced(suffix)

        def wrapper(inner: SamplingResult) -> SamplingResult:
            return SamplingResult.joined([prefix_res, inner, suffix_res], separator="")

        return wrapper

    @classmethod
    def joined(
        cls,
        results: Iterable[SamplingResult],
        *,
        separator: str,
    ) -> SamplingResult:
        from dynamicprompts.utils import removeprefix, removesuffix

        results_list = list(results)

        if len(results_list) == 1:
            # Special case: when we have a single result,
            # there's no point in joining anything, or doing
            # the special handling to strip separators (since
            # we never added any).  This means that a separator
            # in the input will be preserved; this is intentional.
            return results_list[0]

        joined = separator.join(r.text for r in results_list)

        if separator:
            joined = removeprefix(joined, separator)
            joined = removesuffix(joined, separator)

        # Merge variables from all constituent results (last assignment wins).
        merged_vars: dict[str, object] = {}
        for r in results_list:
            for name, val in r.variables:
                merged_vars[name] = val
        return cls(text=joined, variables=tuple(merged_vars.items()))

    @classmethod
    def joined_with_affixes(
        cls,
        results: Iterable[SamplingResult],
        *,
        separator: str,
        prefix: str = "",
        suffix: str = "",
    ) -> SamplingResult:
        """Join results with separator, then wrap with prefix/suffix if non-empty."""
        result = cls.joined(results, separator=separator)
        if result.text and (prefix or suffix):
            return result.text_replaced(prefix + result.text + suffix)
        return result
