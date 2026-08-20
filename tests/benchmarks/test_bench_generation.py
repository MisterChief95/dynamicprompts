"""
Benchmarks for prompt generation across a range of template complexities.

Templates are grouped into four tiers:
  - simple:       plain text, single variants
  - medium:       nested variants, multiple choices
  - complex:      deeply nested, weighted, multi-pick variants
  - wildcards:    templates that resolve __wildcard__ references
  - many_prompts: bulk generation (N outputs from one template)

Run with:
    pytest tests/benchmarks/ --benchmark-only
    pytest tests/benchmarks/ --benchmark-only -v --benchmark-sort=mean
"""

import pytest
from dynamicprompts.generators import (
    CombinatorialPromptGenerator,
    RandomPromptGenerator,
)
from dynamicprompts.parser.parse import _parse_cached, parse
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.wildcards import WildcardManager

# ---------------------------------------------------------------------------
# Template corpus
# ---------------------------------------------------------------------------

TEMPLATES = {
    # --- simple ----------------------------------------------------------------
    "simple_literal": "a photo of a cat",
    "simple_single_variant": "a {red|blue|green} car",
    "simple_optional": "a [fluffy] cat",
    # --- medium ----------------------------------------------------------------
    "medium_nested_variant": "a {red|{dark|light} blue|green} {sports|family} car",
    "medium_multi_pick": "a {2$$red|blue|green|yellow|purple}",
    "medium_sequence": (
        "{beautiful|stunning|breathtaking} {portrait|landscape|still life} "
        "of a {young|old} {woman|man|child} in {Paris|Tokyo|New York}"
    ),
    "medium_weighted": "a {2::red|1::blue|3::green} car",
    # --- complex ---------------------------------------------------------------
    "complex_deeply_nested": (
        "{a {very|extremely} {beautiful|stunning}|an {average|ordinary}} "
        "{photo|painting|sketch} of a "
        "{happy|sad|{confused|bewildered}} "
        "{dog|cat|{bird|parrot|canary}} "
        "in a {park|forest|{city|town|village}}"
    ),
    "complex_multi_pick_nested": (
        "{"
        "2-3$$"
        "{red|crimson|scarlet} {rose|tulip}|"
        "{blue|azure|navy} {iris|lily}|"
        "{yellow|golden} {sunflower|daisy}"
        "}"
    ),
    "complex_large_variant_set": (
        "{apple|banana|cherry|date|elderberry|fig|grape|honeydew|"
        "kiwi|lemon|mango|nectarine|orange|papaya|quince|raspberry|"
        "strawberry|tangerine|ugli|vanilla|watermelon|ximenia|yuzu|zucchini}"
    ),
    "complex_long_sequence": (
        "{masterpiece|best quality|high quality}, "
        "{1girl|1boy}, "
        "{blonde|brunette|redhead|silver} hair, "
        "{blue|green|brown|red} eyes, "
        "{smiling|serious|thoughtful} expression, "
        "wearing {casual|formal|elegant} {clothes|outfit|attire}, "
        "{indoors|outdoors}, "
        "{day|night|golden hour} lighting, "
        "{photorealistic|anime|oil painting|watercolor} style"
    ),
    # --- wildcards -------------------------------------------------------------
    "wildcard_simple": "__colors-warm__",
    "wildcard_in_template": "a __colors-warm__ car with __colors-cold__ trim",
    "wildcard_nested_in_variant": "a {__colors-warm__|__colors-cold__} background",
    # --- parser stress ---------------------------------------------------------
    "parser_many_options": "|".join(f"option{i}" for i in range(50)).join("{}"),
    "parser_deep_nesting": "{a|{b|{c|{d|{e|{f|{g|h}}}}}}}",
}


# ---------------------------------------------------------------------------
# Parser benchmarks
# ---------------------------------------------------------------------------


class TestParserBenchmarks:
    """Benchmark the parser in isolation (parse string → Command tree)."""

    @pytest.mark.parametrize("name,template", list(TEMPLATES.items()))
    def test_parse(self, benchmark, name, template):
        benchmark.pedantic(
            parse,
            args=(template,),
            setup=_parse_cached.cache_clear,
            rounds=50,
        )


# ---------------------------------------------------------------------------
# Random sampler benchmarks — single prompt
# ---------------------------------------------------------------------------


class TestRandomSamplerSingle:
    """Benchmark random sampling, 1 prompt per call."""

    @pytest.mark.parametrize("name,template", list(TEMPLATES.items()))
    def test_random_single(
        self,
        benchmark,
        random_context: SamplingContext,
        name,
        template,
    ):
        def _run():
            return list(random_context.sample_prompts(template, 1))

        benchmark(_run)


# ---------------------------------------------------------------------------
# Random sampler benchmarks — bulk generation
# ---------------------------------------------------------------------------

BULK_SIZES = [10, 100]


class TestRandomSamplerBulk:
    """Benchmark generating N prompts from representative templates."""

    BULK_TEMPLATES = {
        "simple_single_variant": TEMPLATES["simple_single_variant"],
        "medium_sequence": TEMPLATES["medium_sequence"],
        "complex_deeply_nested": TEMPLATES["complex_deeply_nested"],
        "complex_long_sequence": TEMPLATES["complex_long_sequence"],
    }

    @pytest.mark.parametrize("n", BULK_SIZES)
    @pytest.mark.parametrize("name,template", list(BULK_TEMPLATES.items()))
    def test_random_bulk(
        self,
        benchmark,
        random_context: SamplingContext,
        name,
        template,
        n,
    ):
        def _run():
            return list(random_context.sample_prompts(template, n))

        benchmark(_run)


# ---------------------------------------------------------------------------
# Cyclical sampler benchmarks
# ---------------------------------------------------------------------------


class TestCyclicalSampler:
    TEMPLATES_SUBSET = {
        "simple_single_variant": TEMPLATES["simple_single_variant"],
        "medium_sequence": TEMPLATES["medium_sequence"],
        "complex_deeply_nested": TEMPLATES["complex_deeply_nested"],
    }

    @pytest.mark.parametrize("name,template", list(TEMPLATES_SUBSET.items()))
    def test_cyclical_single(
        self,
        benchmark,
        cyclical_context: SamplingContext,
        name,
        template,
    ):
        def _run():
            return list(cyclical_context.sample_prompts(template, 1))

        benchmark(_run)


# ---------------------------------------------------------------------------
# Combinatorial sampler benchmarks
# ---------------------------------------------------------------------------


class TestCombinatorialSampler:
    """Combinatorial expands all combinations — keep templates small."""

    COMBINATORIAL_TEMPLATES = {
        "simple_single_variant": TEMPLATES["simple_single_variant"],
        "medium_nested_variant": TEMPLATES["medium_nested_variant"],
        "medium_sequence": TEMPLATES["medium_sequence"],
        # cap expansion at 50 to avoid combinatorial explosion in benchmarks
        "complex_large_variant_set_capped": TEMPLATES["complex_large_variant_set"],
    }

    @pytest.mark.parametrize("name,template", list(COMBINATORIAL_TEMPLATES.items()))
    def test_combinatorial(
        self,
        benchmark,
        combinatorial_context: SamplingContext,
        name,
        template,
    ):
        def _run():
            # limit to 50 so combinatorial explosion doesn't dominate timing
            return list(combinatorial_context.sample_prompts(template, 50))

        benchmark(_run)


# ---------------------------------------------------------------------------
# RandomPromptGenerator (public API) benchmarks
# ---------------------------------------------------------------------------


class TestRandomPromptGenerator:
    """Benchmark via the high-level generator API used by end users."""

    @pytest.fixture(scope="class")
    def generator(self, wildcard_manager: WildcardManager) -> RandomPromptGenerator:
        return RandomPromptGenerator(wildcard_manager=wildcard_manager)

    GENERATOR_TEMPLATES = {
        "simple_literal": TEMPLATES["simple_literal"],
        "simple_single_variant": TEMPLATES["simple_single_variant"],
        "medium_sequence": TEMPLATES["medium_sequence"],
        "complex_deeply_nested": TEMPLATES["complex_deeply_nested"],
        "complex_long_sequence": TEMPLATES["complex_long_sequence"],
        "wildcard_in_template": TEMPLATES["wildcard_in_template"],
    }

    @pytest.mark.parametrize("name,template", list(GENERATOR_TEMPLATES.items()))
    def test_generate_single(
        self,
        benchmark,
        generator: RandomPromptGenerator,
        name,
        template,
    ):
        benchmark(generator.generate, template, 1)

    @pytest.mark.parametrize("name,template", list(GENERATOR_TEMPLATES.items()))
    def test_generate_10(
        self,
        benchmark,
        generator: RandomPromptGenerator,
        name,
        template,
    ):
        benchmark(generator.generate, template, 10)


# ---------------------------------------------------------------------------
# CombinatorialPromptGenerator (public API) benchmarks
# ---------------------------------------------------------------------------


class TestCombinatorialPromptGenerator:
    """Benchmark combinatorial generator via the high-level API."""

    @pytest.fixture(scope="class")
    def generator(
        self,
        wildcard_manager: WildcardManager,
    ) -> CombinatorialPromptGenerator:
        return CombinatorialPromptGenerator(wildcard_manager=wildcard_manager)

    GENERATOR_TEMPLATES = {
        "simple_single_variant": TEMPLATES["simple_single_variant"],
        "medium_nested_variant": TEMPLATES["medium_nested_variant"],
        "medium_weighted": TEMPLATES["medium_weighted"],
    }

    @pytest.mark.parametrize("name,template", list(GENERATOR_TEMPLATES.items()))
    def test_generate(
        self,
        benchmark,
        generator: CombinatorialPromptGenerator,
        name,
        template,
    ):
        benchmark(generator.generate, template)


# ---------------------------------------------------------------------------
# "Hell" benchmark — deeply nested conditionals, switches, and variables
# ---------------------------------------------------------------------------
#
# This template exercises every expensive feature at once:
#   - Immediate (${ =! }) and non-immediate (${ = }) variable assignment
#   - if/else conditionals (?{...}) with all comparison operators
#   - switch/case with fall-through (&) and default (_)
#   - Nested variants inside variable values
#   - Wildcards resolved inside conditionals
#   - Multi-pick variants
#   - Long variant sets
#
# The goal is to provide a stable regression target so that new features
# (conditionals, comma squashing, prefix/suffix, etc.) don't silently
# regress overall throughput.
# ---------------------------------------------------------------------------

# Each HELL_TEMPLATE variant is a standalone template string (triple-quoted for readability).
# The parser ignores Python-style comments and insignificant whitespace/newlines.
# They are ordered from least to most expensive so the table reads top-to-bottom.


def _strip(s: str) -> str:
    """Remove leading/trailing whitespace from each line and join.

    NOTE: do NOT use # comments inside template strings passed to this function.
    The parser treats # as a line comment, and since this function removes
    newlines, a # would comment out the rest of the entire template.
    Use Python comments outside the string literal instead.
    """
    return "".join(line.strip() for line in s.splitlines())


# ---- level 1: variables only -----------------------------------------------
# Exercises immediate (=!) and non-immediate (=) variable assignment.
# Non-immediate ${color} is re-evaluated on each reference, so the two
# occurrences may produce different colours.
_HELL_1 = _strip("""
    ${season=!{spring|summer|autumn|winter}}
    ${mood=!{happy|melancholic|serene|tense}}
    ${color={red|blue|green|gold|silver}}
    A ${mood} ${season} scene with ${color} tones and ${color} highlights
""")

# ---- level 2: variables + nested if/else -----------------------------------
# Three-level deep if/else chain on season, plus a two-level chain on temp.
_HELL_2 = _strip("""
    ${season=!{spring|summer|autumn|winter}}
    ${temp=!{hot|cold|mild}}
    ${mood=!{happy|melancholic|serene|tense}}
    ?{${season} == summer $$ blazing sun overhead $$
      ?{${season} == winter $$ frost on every surface $$
        ?{${season} == autumn $$ fallen leaves everywhere $$ fresh blossoms abound}}},
    ?{${temp} == hot $$ heat shimmers in the air $$
      ?{${temp} == cold $$ breath misting in the chill $$ a comfortable breeze}},
    the ${mood} atmosphere is palpable
""")

# ---- level 3: variables + two switch/cases with fall-through ---------------
# Two independent switch blocks: one for render style, one for lighting.
# Fall-through chains (label&:) accumulate multiple outputs joined by ", ".
_HELL_3 = _strip("""
    ${style=!{cinematic|anime|oil painting|watercolor|sketch}}
    ${lighting=!{dramatic|soft|natural|neon|golden hour}}
    ${subject=!{warrior|scholar|merchant|wanderer}}
    ?{${style} $$
        cinematic&: high contrast shadows, lens flare |
        oil painting&: thick impasto brushwork |
        watercolor&: soft washes, bleeding edges |
        anime: cel shading, speed lines |
        sketch: cross-hatching, pencil texture |
        _: clean digital render},
    ?{${lighting} $$
        dramatic&: chiaroscuro, deep shadows |
        neon&: vivid chromatic aberration |
        golden hour: warm rim light, long shadows |
        soft: diffused fill, minimal contrast |
        natural: balanced exposure |
        _: studio lighting},
    a ${subject} rendered in ${style} style
""")

# ---- level 4: nested if/else + switch/case + wildcards in branches ---------
# Wildcards (__colors-warm__, __colors-cold__) inside if/else branches.
# Switch/case on season. Nested if/else for style-specific colour grade.
_HELL_4 = _strip("""
    ${season=!{spring|summer|autumn|winter}}
    ${style=!{cinematic|anime|oil painting|watercolor}}
    ${palette=!{warm|cool|neutral|monochrome}}
    ${subject=!{1girl|1boy|elderly man|child}}
    ?{${palette} == warm $$ __colors-warm__ dominant palette $$
      ?{${palette} == cool $$ __colors-cold__ dominant palette $$
        muted neutral tones}},
    ?{${season} $$
        spring&: cherry blossoms |
        summer&: harsh midday sun |
        autumn: amber and ochre canopy |
        winter: snow-covered stillness |
        _: timeless landscape},
    ?{${style} == cinematic $$
        ?{${season} == summer $$ bleach bypass grade $$ teal-orange grade}
      $$
      ?{${style} == anime $$
        ?{${season} == winter $$ cool blue key light $$ warm rim light}
      $$
      painterly finish}},
    ${subject} as the focal point
""")

# ---- level 5: everything at once (the true hell) ---------------------------
# Combines all features:
#   - 7 variable assignments (5 immediate, 2 non-immediate)
#   - Nested variant inside an immediate variable value (${location})
#   - Switch/case with fall-through on style (6 cases)
#   - 4-deep nested if/else chain on lighting
#   - Wildcards inside nested if/else branches on palette
#   - Switch/case with non-immediate ${modifier} in case bodies
#   - Multi-pick variant {2-3$$...} with 7 options
#   - Non-immediate variables (${accent}, ${modifier}) referenced multiple times
_HELL_5 = _strip("""
    ${season=!{spring|summer|autumn|winter}}
    ${style=!{cinematic|anime|oil|watercolor|sketch}}
    ${lighting=!{dramatic|soft|natural|neon|golden}}
    ${subject=!{warrior|wanderer|merchant|knight|scholar}}
    ${palette=!{warm|cool|neutral|monochrome|vibrant}}
    ${accent={red|blue|gold|silver|violet}}
    ${modifier={lush|desolate|mystical|industrial|serene}}
    ${location=!{
        {dense|ancient|towering} forest |
        {bustling|rain-soaked|neon-lit} city street |
        {crumbling|overgrown|fog-shrouded} ruins |
        {sunlit|wave-crashed} clifftop
    }}
    ?{${style} $$
        cinematic&: high contrast, anamorphic lens |
        oil&: impasto, visible brushwork |
        watercolor&: soft washes, paper texture |
        anime: cel shading, expressive linework |
        sketch: loose pencil strokes, hatching |
        _: clean digital illustration},
    ?{${lighting} == dramatic $$ chiaroscuro shadows, single key light $$
      ?{${lighting} == neon $$ vivid chromatic aberration, bloom glow $$
        ?{${lighting} == golden $$ warm rim light, elongated shadows $$
          ?{${lighting} == soft $$ even diffused fill, minimal contrast $$
            natural balanced exposure}}}},
    ?{${palette} == warm $$ __colors-warm__ accent, ${accent} highlights $$
      ?{${palette} == cool $$ __colors-cold__ wash, ${accent} accents $$
        ?{${palette} == monochrome $$ desaturated, ${accent} single pop $$
          ${accent} tones throughout}}},
    ?{${season} $$
        spring&: blossoms and fresh rain |
        summer&: harsh glare, ${modifier} heat haze |
        autumn&: amber leaves, ${modifier} fog |
        winter: frost, ${modifier} stillness |
        _: timeless season},
    with {2-3$$fine detail|subsurface scattering|ray-traced reflections|volumetric fog|depth of field|motion blur|lens distortion},
    ${subject} standing at ${location},
    ?{${modifier} == mystical $$ ethereal particles swirling around them $$
      ?{${modifier} == industrial $$ steam and machinery in the background $$
        the scene feels ${modifier}}}
""")


HELL_TEMPLATES = {
    "hell_1_variables": _HELL_1,
    "hell_2_variables_if": _HELL_2,
    "hell_3_switch_fallthrough": _HELL_3,
    "hell_4_nested_wildcards": _HELL_4,
    "hell_5_everything": _HELL_5,
}


class TestHellBenchmarks:
    """
    Stress-test the engine with deeply nested conditionals, switches,
    and heavy variable usage (both immediate and non-immediate).

    Run alone with:
        pytest tests/benchmarks/test_bench_generation.py::TestHellBenchmarks --benchmark-only -v
    """

    @pytest.mark.parametrize("name,template", list(HELL_TEMPLATES.items()))
    def test_hell_parse(self, benchmark, name, template):
        """Parser cost in isolation (cache cleared before each round)."""
        benchmark.pedantic(
            parse,
            args=(template,),
            setup=_parse_cached.cache_clear,
            rounds=50,
        )

    @pytest.mark.parametrize("name,template", list(HELL_TEMPLATES.items()))
    def test_hell_random_single(
        self,
        benchmark,
        random_context: SamplingContext,
        name,
        template,
    ):
        """Single-prompt random generation."""

        def _run():
            return list(random_context.sample_prompts(template, 1))

        benchmark(_run)

    @pytest.mark.parametrize("name,template", list(HELL_TEMPLATES.items()))
    def test_hell_random_10(
        self,
        benchmark,
        random_context: SamplingContext,
        name,
        template,
    ):
        """10-prompt batch — shows per-prompt scaling."""

        def _run():
            return list(random_context.sample_prompts(template, 10))

        benchmark(_run)
