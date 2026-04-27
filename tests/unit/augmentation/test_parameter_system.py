"""
Unit tests for augmentation/parameter_system.py.

Coverage approach is the same as item #10 / TODO #6 reference
(`tests/unit/augmentation/test_augmentation_factory.py`): class-per-area,
parametrize variants with descriptive ids, separate happy paths from
error paths, and probe the gap between docstring promises and what the
code actually enforces.

Targets:
- `convert_to_numeric()` — string/numeric coercion + rejection of
  None/non-numeric/other types
- `RangeParameter` — dataclass with __post_init__ validation,
  to_albumentations_format(), sample(), .scalar() classmethod,
  .integer_range() classmethod, .is_scalar()
- `NestedParameter` — dict-of-Any → dict-of-AlbumentationsParameter
  conversion, error-path coverage for each rejected value type

Real bugs / edge cases probed below (each marked with comments + xfail
where the behavior is buggy enough to deserve a fix):
- Bool-as-int silent acceptance (Python's `isinstance(True, int) is True`)
- is_integer=True silently truncating float bounds (e.g., (0.5, 0.9) → (0, 0))
- numpy scalars not recognized as numeric
- float('inf') in integer_range raises OverflowError but isn't caught
- Float-equality in is_scalar() (0.1 + 0.2 != 0.3)
"""

from __future__ import annotations

import random

import pytest

from augmentation.parameter_system import (
    AlbumentationsParameter,
    NestedParameter,
    RangeParameter,
    convert_to_numeric,
)

# ---------------------------------------------------------------------------
# Section 1: convert_to_numeric() — coercion and rejection contract
# ---------------------------------------------------------------------------


class TestConvertToNumericAcceptedValues:
    """Inputs the docstring says are accepted: int, float, numeric strings."""

    @pytest.mark.parametrize(
        "value,expected_type,expected_value",
        [
            (0, int, 0),
            (1, int, 1),
            (-1, int, -1),
            (1_000_000, int, 1_000_000),
            (0.0, float, 0.0),
            (3.14, float, 3.14),
            (-2.5, float, -2.5),
        ],
        ids=lambda v: f"v={v!r}",
    )
    def test_native_numerics_pass_through(self, value, expected_type, expected_value):
        result = convert_to_numeric(value)
        assert type(result) is expected_type, f"type drifted: got {type(result)!r}"
        assert result == expected_value

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0", 0),
            ("10", 10),
            ("-5", -5),
            ("1000", 1000),
        ],
    )
    def test_string_int_coerces_to_int(self, value, expected):
        """Strings without `.` or `e` route to int per source line 31."""
        result = convert_to_numeric(value)
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("0.5", 0.5),
            ("-3.14", -3.14),
            ("1e3", 1e3),
            ("1.5e2", 150.0),
            ("1E5", 1e5),
            ("-1.5e-2", -0.015),
        ],
    )
    def test_string_float_coerces_to_float(self, value, expected):
        """Strings with `.` or `e` route to float per source line 31."""
        result = convert_to_numeric(value)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)


class TestConvertToNumericRejectedValues:
    """None and non-numeric types must raise TypeError per the docstring."""

    def test_none_rejected(self):
        with pytest.raises(TypeError, match="cannot be None"):
            convert_to_numeric(None)

    @pytest.mark.parametrize(
        "value",
        ["abc", "10abc", "abc10", "1.2.3", "10..0", "--5", "+ 10", "10 5"],
        ids=lambda v: f"v={v!r}",
    )
    def test_non_numeric_string_rejected(self, value):
        with pytest.raises(TypeError, match="not a valid number"):
            convert_to_numeric(value)

    @pytest.mark.parametrize(
        "value",
        [[], {}, (1, 2), {1, 2}, object(), b"5"],
        ids=lambda v: type(v).__name__,
    )
    def test_other_types_rejected(self, value):
        with pytest.raises(TypeError, match=r"Cannot convert"):
            convert_to_numeric(value)


class TestConvertToNumericEdgeCases:
    """Behaviors worth pinning so silent regressions are caught."""

    def test_empty_string_rejected(self):
        """Empty string falls into str → int branch → ValueError → wrapped TypeError."""
        with pytest.raises(TypeError, match="not a valid number"):
            convert_to_numeric("")

    def test_whitespace_only_string_rejected(self):
        with pytest.raises(TypeError, match="not a valid number"):
            convert_to_numeric("   ")

    @pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
    def test_inf_nan_strings_currently_rejected(self, value):
        """
        Documenting current behavior: "inf" / "-inf" / "nan" lack `.` and
        `e`, so the routing at parameter_system.py:31 sends them to
        `int(value)` which raises ValueError → wrapped as TypeError. The
        docstring says "string representations of numbers" are accepted;
        Python's `float()` would accept these. The xfail below is the
        opposite assertion — pinning the gap.
        """
        with pytest.raises(TypeError, match="not a valid number"):
            convert_to_numeric(value)

    @pytest.mark.parametrize(
        "value,expected_func",
        [
            ("inf", lambda v: v == float("inf")),
            ("-inf", lambda v: v == float("-inf")),
            ("nan", lambda v: v != v),  # NaN != NaN is the standard NaN check
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="parameter_system.py:31 — routing on `'.' not in value and "
        "'e' not in value.lower()` sends 'inf'/'nan' to int() (which "
        "rejects them) instead of float() (which accepts them). Docstring "
        "says 'string representations of numbers' are accepted but special-"
        "value floats fall through. Either route 'inf'/'-inf'/'nan' to "
        "float() explicitly, or document them as not-accepted.",
    )
    def test_inf_nan_strings_should_be_accepted_as_float(self, value, expected_func):
        result = convert_to_numeric(value)
        assert expected_func(result)


class TestConvertToNumericBoolGap:
    """
    `isinstance(True, int)` is True in Python, so True/False fall into the
    int pass-through branch and `convert_to_numeric(True)` returns the bool
    True (not the int 1). Downstream `RangeParameter(True, False)` then
    raises (True > False), but `RangeParameter(False, True)` silently
    creates a [0.0, 1.0] range from boolean inputs. Whether to treat
    bool-as-int as user error is a design call; pinning current behavior
    here so a fix moves the test off the xfail.
    """

    def test_bool_currently_passes_through_as_int(self):
        """True passes through (because bool is a subclass of int)."""
        assert convert_to_numeric(True) is True
        assert convert_to_numeric(False) is False

    @pytest.mark.xfail(
        strict=True,
        reason="parameter_system.py:24 — `isinstance(value, (int, float))` "
        "matches bool because bool subclasses int. Document by adding an "
        "explicit `if isinstance(value, bool): raise TypeError(...)` BEFORE "
        "the int/float check. RangeParameter shouldn't silently accept "
        "boolean inputs.",
    )
    def test_bool_should_be_rejected(self):
        with pytest.raises(TypeError):
            convert_to_numeric(True)


# ---------------------------------------------------------------------------
# Section 2: RangeParameter — basic construction + validation
# ---------------------------------------------------------------------------


class TestRangeParameterConstruction:
    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (0, 1),
            (0.0, 1.0),
            (-5.0, 5.0),
            ("0", "1"),
            ("0.5", "1.5"),
            (0, 0),  # equal — scalar
        ],
        ids=lambda v: f"v={v!r}",
    )
    def test_valid_construction(self, min_val, max_val):
        p = RangeParameter(min_val, max_val)
        assert p.min_val == float(min_val) if not isinstance(min_val, str) else True
        assert p.max_val == float(max_val) if not isinstance(max_val, str) else True

    def test_min_greater_than_max_rejected(self):
        with pytest.raises(ValueError, match="min_val.*must be <= max_val"):
            RangeParameter(10, 5)

    def test_min_equal_to_max_accepted(self):
        """Scalar case — min == max is a degenerate range, still valid."""
        p = RangeParameter(7, 7)
        assert p.is_scalar()

    @pytest.mark.parametrize(
        "bad_value",
        [None, "abc", [1], object()],
        ids=lambda v: type(v).__name__,
    )
    def test_non_numeric_min_rejected(self, bad_value):
        with pytest.raises(TypeError, match="numeric"):
            RangeParameter(bad_value, 10)

    @pytest.mark.parametrize(
        "bad_value",
        [None, "abc", [1], object()],
        ids=lambda v: type(v).__name__,
    )
    def test_non_numeric_max_rejected(self, bad_value):
        with pytest.raises(TypeError, match="numeric"):
            RangeParameter(0, bad_value)

    def test_post_init_converts_to_float(self):
        """Source line 66-67: bounds are stored as float regardless of input type."""
        p = RangeParameter(1, 5)  # ints in
        assert isinstance(p.min_val, float)
        assert isinstance(p.max_val, float)


class TestRangeParameterToAlbumentationsFormat:
    def test_continuous_returns_float_tuple(self):
        p = RangeParameter(0.5, 1.5)
        assert p.to_albumentations_format() == (0.5, 1.5)

    def test_integer_returns_int_tuple(self):
        p = RangeParameter(0, 10, is_integer=True)
        result = p.to_albumentations_format()
        assert result == (0, 10)
        assert all(isinstance(x, int) for x in result)

    def test_integer_truncates_float_bounds(self):
        """is_integer=True with float bounds: int() truncates toward zero.
        Source line 78. Documenting behavior — could be a footgun."""
        p = RangeParameter(0.9, 5.7, is_integer=True)
        assert p.to_albumentations_format() == (0, 5)

    def test_continuous_preserves_float_bounds(self):
        p = RangeParameter(0.123, 0.456)
        assert p.to_albumentations_format() == (0.123, 0.456)


class TestRangeParameterIntegerTruncationGap:
    """
    `is_integer=True` with float bounds < 1 collapses to a degenerate
    [0, 0] range silently — the user thinks they have a range and gets
    a forced scalar. Real footgun for albumentations params like
    `num_holes` where someone might pass (0.5, 2.0) intending [1, 2]
    but getting [0, 2].
    """

    @pytest.mark.parametrize(
        "min_val,max_val,expected_collapse",
        [
            (0.1, 0.9, (0, 0)),  # both truncate to 0 — silent scalar
            (0.5, 0.99, (0, 0)),  # ditto
            (-0.5, 0.5, (0, 0)),  # bracket zero — both truncate to 0
        ],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="parameter_system.py:78 (to_albumentations_format) and "
        "parameter_system.py:84 (sample) — is_integer=True silently "
        "truncates float bounds via int(). Sub-unit float ranges collapse "
        "to [0, 0]. Either round() instead of int(), or raise on non-integer "
        "bounds when is_integer=True.",
    )
    def test_subunit_float_range_should_not_silently_collapse(self, min_val, max_val, expected_collapse):
        p = RangeParameter(min_val, max_val, is_integer=True)
        # The xfail is asserting this DOES NOT happen — i.e., the code
        # SHOULD reject or round. Today it collapses, so this assertion
        # fails and the xfail "passes".
        assert p.to_albumentations_format() != expected_collapse


class TestRangeParameterSample:
    @pytest.fixture(autouse=True)
    def fixed_seed(self):
        """Deterministic sampling — pinned seed so test failures are
        reproducible. Each test gets a fresh seed before running."""
        random.seed(42)

    def test_continuous_sample_in_range(self):
        p = RangeParameter(0.0, 1.0)
        for _ in range(100):
            assert 0.0 <= p.sample() <= 1.0

    def test_integer_sample_in_range_inclusive(self):
        """random.randint(a, b) is inclusive on both ends."""
        p = RangeParameter(1, 3, is_integer=True)
        seen = set()
        for _ in range(200):
            v = p.sample()
            assert isinstance(v, int)
            assert 1 <= v <= 3
            seen.add(v)
        # With 200 trials, we should see all 3 values
        assert seen == {1, 2, 3}

    def test_continuous_sample_returns_float(self):
        p = RangeParameter(0.0, 1.0)
        assert isinstance(p.sample(), float)

    def test_integer_sample_returns_int(self):
        p = RangeParameter(0, 10, is_integer=True)
        assert isinstance(p.sample(), int)

    def test_scalar_sample_always_returns_same_value(self):
        """min == max → sample() should always return that value."""
        p = RangeParameter(7.5, 7.5)
        for _ in range(20):
            assert p.sample() == 7.5

    def test_negative_range_sample(self):
        p = RangeParameter(-5.0, -1.0)
        for _ in range(50):
            v = p.sample()
            assert -5.0 <= v <= -1.0


class TestRangeParameterScalarClassmethod:
    @pytest.mark.parametrize(
        "value",
        [0, 1, -1, 0.5, "10", "0.5"],
        ids=lambda v: f"v={v!r}",
    )
    def test_scalar_creates_min_equals_max(self, value):
        p = RangeParameter.scalar(value)
        assert p.min_val == p.max_val
        assert p.is_scalar()

    def test_scalar_is_integer_propagates(self):
        p = RangeParameter.scalar(5, is_integer=True)
        assert p.is_integer is True
        assert p.to_albumentations_format() == (5, 5)
        assert all(isinstance(x, int) for x in p.to_albumentations_format())

    def test_scalar_rejects_none(self):
        # Deliberate type violation to verify the runtime guard catches
        # None — static signature excludes it. Cast through Any so both
        # mypy and Pylance accept the call site (the runtime check is
        # the real contract here).
        from typing import Any, cast

        with pytest.raises(TypeError):
            RangeParameter.scalar(cast(Any, None))

    def test_scalar_rejects_non_numeric_string(self):
        with pytest.raises(TypeError):
            RangeParameter.scalar("not_a_number")


class TestRangeParameterIntegerRangeClassmethod:
    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (0, 10),
            (-5, 5),
            ("0", "10"),
            (0.0, 10.0),
        ],
    )
    def test_integer_range_construction(self, min_val, max_val):
        p = RangeParameter.integer_range(min_val, max_val)
        assert p.is_integer is True
        # Bounds are stored as float internally (after int → float conversion
        # at line 113), but to_albumentations_format() returns ints.
        result = p.to_albumentations_format()
        assert all(isinstance(x, int) for x in result)

    def test_integer_range_truncates_floats(self):
        """integer_range(0.9, 5.7) → int(0.9)=0, int(5.7)=5 → range [0, 5].
        Same truncation pattern as is_integer=True."""
        p = RangeParameter.integer_range(0.9, 5.7)
        assert p.to_albumentations_format() == (0, 5)

    @pytest.mark.parametrize("bad_value", [None, "abc", []])
    def test_integer_range_rejects_invalid(self, bad_value):
        with pytest.raises(ValueError):
            RangeParameter.integer_range(bad_value, 10)


class TestRangeParameterIntegerRangeOverflowGap:
    """
    `int(float('inf'))` raises OverflowError, not ValueError or TypeError.
    `integer_range` only catches `(TypeError, ValueError)` — OverflowError
    leaks through.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="parameter_system.py:111 — except clause is "
        "`(TypeError, ValueError)`. `int(float('inf'))` raises "
        "OverflowError; integer_range(float('inf'), 10) leaks the raw "
        "OverflowError instead of the wrapped ValueError. Add OverflowError "
        "to the except tuple.",
    )
    def test_inf_should_raise_value_error_not_overflow(self):
        with pytest.raises(ValueError, match="numeric values"):
            RangeParameter.integer_range(float("inf"), 10)


class TestRangeParameterIsScalar:
    @pytest.mark.parametrize(
        "min_val,max_val,expected",
        [
            (5, 5, True),
            (0.5, 0.5, True),
            (0, 1, False),
            (-1, 1, False),
        ],
    )
    def test_is_scalar(self, min_val, max_val, expected):
        p = RangeParameter(min_val, max_val)
        assert p.is_scalar() is expected


class TestRangeParameterIsScalarFloatEqualityGap:
    """
    Float equality is fragile: 0.1 + 0.2 == 0.3 is False. If a caller
    constructs `RangeParameter(0.1 + 0.2, 0.3)` they intuitively expect
    is_scalar() to be True (same value mathematically). It returns
    False today. Worth a note — math.isclose() would be the fix.
    """

    @pytest.mark.xfail(
        strict=True,
        reason="parameter_system.py:116 — is_scalar() uses `==` on floats. "
        "Use math.isclose() to handle floating-point representation drift "
        "(0.1 + 0.2 != 0.3 exactly).",
    )
    def test_floating_point_drift_should_still_be_scalar(self):
        p = RangeParameter(0.1 + 0.2, 0.3)
        assert p.is_scalar()  # currently False due to fp drift


# ---------------------------------------------------------------------------
# Section 3: NestedParameter — value-type dispatch in __post_init__
# ---------------------------------------------------------------------------


class TestNestedParameterAcceptedValueTypes:
    def test_empty_dict(self):
        n = NestedParameter({})
        assert n.parameters == {}
        assert n.to_albumentations_format() == {}
        assert n.sample() == {}

    def test_already_albumentations_parameter_passes_through(self):
        rp = RangeParameter(0.0, 1.0)
        n = NestedParameter({"x": rp})
        assert n.parameters["x"] is rp  # identity, not just equality

    @pytest.mark.parametrize(
        "raw_value,expected_min,expected_max",
        [
            ((0, 1), 0.0, 1.0),
            ([0, 1], 0.0, 1.0),
            ((0.5, 1.5), 0.5, 1.5),
            (("0", "1"), 0.0, 1.0),  # strings inside tuple
            ((-0.1, 0.1), -0.1, 0.1),
        ],
        ids=lambda v: f"v={v!r}",
    )
    def test_2_element_sequence_becomes_range_parameter(self, raw_value, expected_min, expected_max):
        n = NestedParameter({"k": raw_value})
        assert isinstance(n.parameters["k"], RangeParameter)
        assert n.parameters["k"].min_val == expected_min
        assert n.parameters["k"].max_val == expected_max

    @pytest.mark.parametrize("scalar_value", [0, 1, 0.5, -1.0, "10", "0.5"])
    def test_scalar_becomes_scalar_range_parameter(self, scalar_value):
        n = NestedParameter({"k": scalar_value})
        assert isinstance(n.parameters["k"], RangeParameter)
        assert n.parameters["k"].is_scalar()

    def test_mixed_value_types_in_one_dict(self):
        """All three value-type branches in one construction."""
        existing = RangeParameter(0.0, 1.0)
        n = NestedParameter(
            {
                "as_param": existing,
                "as_range": (0, 10),
                "as_scalar": 5,
            }
        )
        assert n.parameters["as_param"] is existing
        assert n.parameters["as_range"].max_val == 10.0
        assert n.parameters["as_scalar"].is_scalar()


class TestNestedParameterRejectedValueTypes:
    def test_none_value_rejected(self):
        with pytest.raises(TypeError, match="cannot be None"):
            NestedParameter({"x": None})

    def test_dict_value_rejected(self):
        """Nested dict (dict-of-dict) is not supported — only scalars,
        2-tuples, or AlbumentationsParameter."""
        with pytest.raises(TypeError, match="must be"):
            NestedParameter({"x": {"a": 1}})

    @pytest.mark.parametrize(
        "bad_value",
        [(1, 2, 3), (1,), [1, 2, 3, 4], []],
        ids=["3-tuple", "1-tuple", "4-list", "empty-list"],
    )
    def test_wrong_length_sequence_rejected(self, bad_value):
        """Only length-2 sequences are accepted; others fall through to
        the catch-all `must be ... tuple/list of 2 numbers, or scalar`."""
        with pytest.raises(TypeError, match="must be"):
            NestedParameter({"x": bad_value})

    @pytest.mark.parametrize(
        "bad_value",
        [object(), b"bytes"],
        ids=lambda v: type(v).__name__,
    )
    def test_other_object_types_rejected(self, bad_value):
        with pytest.raises(TypeError, match="must be"):
            NestedParameter({"x": bad_value})

    def test_2_element_with_non_numeric_rejected(self):
        """(1, "abc") is structurally a 2-tuple but contents aren't numeric."""
        with pytest.raises(TypeError, match="Invalid range"):
            NestedParameter({"k": (1, "abc")})

    def test_2_element_with_none_rejected(self):
        with pytest.raises(TypeError, match="Invalid range"):
            NestedParameter({"k": (1, None)})


class TestNestedParameterErrorContext:
    """Error messages should include the offending key so the user can find it."""

    def test_none_error_includes_key_name(self):
        with pytest.raises(TypeError, match="'translate_y'"):
            NestedParameter({"translate_x": 0.5, "translate_y": None})

    def test_invalid_range_error_includes_key_name(self):
        with pytest.raises(TypeError, match="'scale'"):
            NestedParameter({"scale": (0.0, "bad")})

    def test_invalid_scalar_error_includes_key_name(self):
        """Scalar branch wraps via `Invalid scalar for key 'X'`."""
        with pytest.raises(TypeError, match="'rotate'"):
            # Empty-string scalar → str branch → ValueError → wrapped TypeError
            NestedParameter({"rotate": ""})


class TestNestedParameterToAlbumentationsFormat:
    def test_format_unwraps_each_value(self):
        """Each parameter's to_albumentations_format() is called and aggregated."""
        n = NestedParameter({"x": (0.0, 1.0), "y": 5})
        result = n.to_albumentations_format()
        assert result == {"x": (0.0, 1.0), "y": (5.0, 5.0)}

    def test_format_with_integer_range_inside(self):
        n = NestedParameter({"holes": RangeParameter(0, 5, is_integer=True)})
        assert n.to_albumentations_format() == {"holes": (0, 5)}


class TestNestedParameterSample:
    @pytest.fixture(autouse=True)
    def fixed_seed(self):
        random.seed(42)

    def test_sample_returns_dict_of_samples(self):
        n = NestedParameter({"x": (0.0, 1.0), "y": 5})
        s = n.sample()
        assert set(s.keys()) == {"x", "y"}
        assert 0.0 <= s["x"] <= 1.0
        assert s["y"] == 5  # scalar always returns the same value

    def test_sample_independence_across_keys(self):
        """Each key is sampled independently — a single bad sample of one
        key shouldn't affect others. Empirical check: vary samples across
        many calls and ensure each key spans its range."""
        n = NestedParameter({"x": (0.0, 1.0), "y": (10.0, 20.0)})
        x_samples = [n.sample()["x"] for _ in range(100)]
        y_samples = [n.sample()["y"] for _ in range(100)]
        assert min(x_samples) < 0.5 < max(x_samples), "x sampling not spread"
        assert min(y_samples) < 15.0 < max(y_samples), "y sampling not spread"


# ---------------------------------------------------------------------------
# Section 4: AlbumentationsParameter abstract base
# ---------------------------------------------------------------------------


class TestAlbumentationsParameterAbstract:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError, match="abstract"):
            AlbumentationsParameter()  # type: ignore[abstract]

    def test_subclass_must_implement_both_methods(self):
        """A subclass that overrides only one of the two abstract methods
        is still abstract."""

        class HalfClass(AlbumentationsParameter):
            def to_albumentations_format(self):
                return None

            # Deliberately missing sample()

        with pytest.raises(TypeError, match="abstract"):
            HalfClass()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        """RangeParameter and NestedParameter are the in-tree concretes —
        cover with one canonical instantiation each."""
        RangeParameter(0.0, 1.0)
        NestedParameter({})
