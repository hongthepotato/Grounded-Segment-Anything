"""
Unit tests for augmentation/parameter_system.py.

Coverage approach is the same as item #10 / TODO #6 reference
(`tests/unit/augmentation/test_augmentation_factory.py`): class-per-area,
parametrize variants with descriptive ids, separate happy paths from
error paths, and probe the gap between docstring promises and what the
code actually enforces.

Targets:
- `convert_to_numeric()` — string/numeric coercion + rejection of
  None/bool/non-finite/non-numeric/other types
- `RangeParameter` — dataclass with __post_init__ validation,
  to_albumentations_format(), sample(), .scalar() classmethod,
  .integer_range() classmethod, .is_scalar()
- `NestedParameter` — dict-of-Any → dict-of-AlbumentationsParameter
  conversion, error-path coverage for each rejected value type

Real bugs surfaced AND FIXED in the same PR (5 categories — see
TODO #19's original draft, now resolved):
- Bool-as-int silent acceptance (Python's `isinstance(True, int) is True`):
  fixed by adding an explicit `isinstance(value, bool)` rejection BEFORE
  the int/float check.
- Non-finite floats ('inf', '-inf', 'nan' as strings or floats): fixed
  by explicit rejection with a clear error message — random.uniform/
  randint with infinite bounds is meaningless.
- is_integer=True silently truncating float bounds via int(): fixed by
  raising ValueError on non-integer bounds in __post_init__. Use
  RangeParameter.integer_range() for explicit float→int conversion.
- OverflowError leak in integer_range: fixed by adding OverflowError to
  the except tuple. Now defensive — convert_to_numeric rejects
  non-finite floats up front so OverflowError is unreachable in normal
  use, but the guard remains for future safety.
- Float-equality in is_scalar(): fixed by switching to math.isclose so
  RangeParameter(0.1 + 0.2, 0.3).is_scalar() correctly returns True.
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

    @pytest.mark.parametrize(
        "value",
        ["inf", "-inf", "+inf", "nan", "INF", "  inf  "],
        ids=lambda v: f"v={v!r}",
    )
    def test_non_finite_strings_rejected_with_clear_message(self, value):
        """Non-finite float strings get an explicit error (not the generic
        'not a valid number'), so the user sees WHY they're rejected.
        Whitespace + case variants are all caught via .strip().lower()."""
        with pytest.raises(TypeError, match="Non-finite float string"):
            convert_to_numeric(value)

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan")],
        ids=lambda v: f"v={v!r}",
    )
    def test_non_finite_floats_rejected(self, value):
        """Direct float('inf')/float('nan') values also rejected — the
        finite-check runs on the int/float pass-through branch too."""
        with pytest.raises(TypeError, match="Non-finite float"):
            convert_to_numeric(value)


class TestConvertToNumericBoolRejection:
    """
    `isinstance(True, int)` is True in Python, so without an explicit
    bool guard, `convert_to_numeric(True)` would return the bool True
    and downstream RangeParameter would happily accept booleans as
    numeric inputs. Source has an explicit `isinstance(value, bool)`
    check BEFORE the int/float branch.
    """

    @pytest.mark.parametrize("value", [True, False])
    def test_bool_rejected(self, value):
        with pytest.raises(TypeError, match="Cannot convert bool"):
            convert_to_numeric(value)


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

    def test_integer_with_integer_valued_floats_accepted(self):
        """is_integer=True accepts float bounds that are integer-valued
        (e.g., 5.0 == int(5.0)). Only non-integer floats are rejected."""
        p = RangeParameter(0.0, 5.0, is_integer=True)
        assert p.to_albumentations_format() == (0, 5)

    def test_continuous_preserves_float_bounds(self):
        p = RangeParameter(0.123, 0.456)
        assert p.to_albumentations_format() == (0.123, 0.456)


class TestRangeParameterIntegerBoundsValidation:
    """
    `is_integer=True` with non-integer float bounds is rejected at
    construction time. Without this check, `int()` later silently
    floors-toward-zero — `RangeParameter(0.5, 0.9, is_integer=True)`
    would collapse to [0, 0] and the user would think they have a
    range but get a forced scalar. Real footgun for albumentations
    params like `num_holes`. Use `.integer_range()` for explicit
    float→int conversion.
    """

    @pytest.mark.parametrize(
        "min_val,max_val",
        [
            (0.1, 0.9),  # both sub-unit — would have collapsed to [0, 0]
            (0.5, 0.99),  # ditto
            (-0.5, 0.5),  # straddles zero
            (0.5, 5.0),  # only min is non-integer
            (0.0, 5.7),  # only max is non-integer
            (-0.5, 5.5),  # both non-integer
        ],
        ids=lambda v: f"v={v}",
    )
    def test_non_integer_bounds_rejected(self, min_val, max_val):
        with pytest.raises(ValueError, match="is_integer=True requires integer bounds"):
            RangeParameter(min_val, max_val, is_integer=True)

    @pytest.mark.parametrize(
        "min_val,max_val",
        [(0, 5), (0.0, 5.0), (-3, 3), ("0", "10")],
        ids=lambda v: f"v={v!r}",
    )
    def test_integer_valued_bounds_accepted(self, min_val, max_val):
        """Pure ints AND float-valued integers (5.0, 0.0) both pass."""
        RangeParameter(min_val, max_val, is_integer=True)

    def test_error_message_suggests_integer_range(self):
        """The error message points at the explicit-truncation API
        (.integer_range) so the user has a clear migration path."""
        with pytest.raises(ValueError, match="integer_range"):
            RangeParameter(0.5, 0.9, is_integer=True)


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


class TestRangeParameterIntegerRangeNonFiniteHandling:
    """
    `int(float('inf'))` raises OverflowError, not ValueError or TypeError.
    Today: convert_to_numeric rejects float('inf') up-front (with a clear
    "Non-finite float" message), so integer_range never gets to call
    int() on inf — but the except clause includes OverflowError as a
    defensive guard for future changes.
    """

    def test_inf_rejected_with_clear_message(self):
        """convert_to_numeric inside integer_range rejects float('inf'),
        and integer_range wraps the TypeError as ValueError per its
        contract."""
        with pytest.raises(ValueError, match="numeric values"):
            RangeParameter.integer_range(float("inf"), 10)

    def test_inf_string_rejected(self):
        """Same path via the string form."""
        with pytest.raises(ValueError, match="numeric values"):
            RangeParameter.integer_range("inf", 10)


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


class TestRangeParameterIsScalarFloatTolerance:
    """
    is_scalar() uses math.isclose so float-representation drift doesn't
    falsely report a scalar as a range. RangeParameter(0.1 + 0.2, 0.3)
    was a real footgun before the fix — `==` reported False because
    0.1 + 0.2 == 0.30000000000000004.
    """

    def test_floating_point_drift_still_counts_as_scalar(self):
        p = RangeParameter(0.1 + 0.2, 0.3)
        assert p.is_scalar()

    def test_meaningfully_different_floats_not_scalar(self):
        """Sanity: math.isclose's default tolerance still distinguishes
        actually-different values."""
        p = RangeParameter(0.3, 0.5)
        assert not p.is_scalar()


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
