"""
Albumentations Parameter System
Provides a unified interface for all albumentations parameters
Supports range, nested, and list parameters
"""

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union


def convert_to_numeric(value: Any) -> Union[int, float]:
    """
    Convert value to numeric type.
    Accepts: int, float, or string representations of finite numbers.
    Rejects: None, bool (subclasses int in Python), non-finite floats
        ('inf'/'-inf'/'nan' and their string forms — meaningless as
        bounds for random.uniform/randint), non-numeric strings, and
        other types.
    """
    # Reject None explicitly
    if value is None:
        raise TypeError("Value cannot be None")

    # Reject bool BEFORE the int/float check — bool is a subclass of int
    # in Python, so without this guard `convert_to_numeric(True)` would
    # silently return the bool True (not the int 1) and downstream
    # RangeParameter would happily build a [0.0, 1.0] range from booleans.
    if isinstance(value, bool):
        raise TypeError(f"Cannot convert bool to numeric (got {value!r})")

    # If already numeric, finite-check floats and pass through
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise TypeError(
                f"Non-finite float not supported (got {value!r}); "
                "parameter ranges must be finite for random sampling"
            )
        return value

    # Try to convert string to numeric. Reject 'inf'/'-inf'/'+inf'/'nan'
    # (case-insensitive) explicitly — Python's float() would accept them
    # but they're nonsense as bounds for random.uniform/randint.
    if isinstance(value, str):
        if value.strip().lower() in {"inf", "+inf", "-inf", "nan", "+nan", "-nan"}:
            raise TypeError(
                f"Non-finite float string not supported (got {value!r}); "
                "parameter ranges must be finite for random sampling"
            )
        try:
            # Try int first (for strings like "10"). Otherwise float.
            if "." not in value and "e" not in value.lower():
                return int(value)
            return float(value)
        except ValueError as exc:
            raise TypeError(f"String '{value}' is not a valid number") from exc

    # Reject all other types
    raise TypeError(f"Cannot convert {type(value)} to numeric")


class AlbumentationsParameter(ABC):
    """Base class for all parameter types"""

    @abstractmethod
    def to_albumentations_format(self) -> Any:
        """Convert to the format expected by albumentations"""

    @abstractmethod
    def sample(self) -> Any:
        """Sample a value (for dynamic parameters)"""


@dataclass
class RangeParameter(AlbumentationsParameter):
    """Handles all range-based parameters - continuous, discrete, and scalar"""

    min_val: float
    max_val: float
    is_integer: bool = False  # Whether to sample/return integers

    def __post_init__(self):
        """Validate and convert inputs after initialization"""
        # Convert to numeric types (handles int, float, and string numeric)
        try:
            self.min_val = float(convert_to_numeric(self.min_val))
            self.max_val = float(convert_to_numeric(self.max_val))
        except TypeError as e:
            raise TypeError(f"RangeParameter requires numeric values: {e}") from e

        # Validate range. Use math.isclose to tolerate float drift —
        # `RangeParameter(0.1 + 0.2, 0.3)` should pass even though
        # 0.1 + 0.2 == 0.30000000000000004 > 0.3 by `==`.
        if self.min_val > self.max_val and not math.isclose(self.min_val, self.max_val):
            raise ValueError(f"min_val ({self.min_val}) must be <= max_val ({self.max_val})")

        # When is_integer=True, bounds must be integer-valued. Without this
        # check, `int()` later silently floors-toward-zero, so e.g.
        # RangeParameter(0.5, 0.9, is_integer=True) collapses to [0, 0]
        # and the user thinks they have a range but get a forced scalar.
        # Use .integer_range() if you want explicit float→int conversion.
        if self.is_integer:
            if self.min_val != int(self.min_val) or self.max_val != int(self.max_val):
                raise ValueError(
                    f"is_integer=True requires integer bounds, got "
                    f"min_val={self.min_val}, max_val={self.max_val}. "
                    f"Use RangeParameter.integer_range(...) to truncate floats explicitly."
                )

    def to_albumentations_format(self) -> Union[Tuple[float, float], Tuple[int, int]]:
        """Return appropriate format for albumentations"""
        if self.is_integer:
            return (int(self.min_val), int(self.max_val))
        return (self.min_val, self.max_val)

    def sample(self) -> Union[float, int]:
        """Sample a value from the range"""
        if self.is_integer:
            return random.randint(int(self.min_val), int(self.max_val))
        return random.uniform(self.min_val, self.max_val)

    @classmethod
    def scalar(cls, value: Union[int, float, str], is_integer: bool = False) -> "RangeParameter":
        """
        Create a scalar parameter (min == max)
        Accepts: int, float, or string representations of numbers
        Rejects: None and non-numeric values
        """
        numeric_value = convert_to_numeric(value)
        return cls(numeric_value, numeric_value, is_integer)

    @classmethod
    def integer_range(
        cls, min_val: Union[int, float, str], max_val: Union[int, float, str]
    ) -> "RangeParameter":
        """
        Create an integer range parameter
        Accepts: int, float, or string representations of numbers
        Values will be converted to integers
        """
        # convert_to_numeric now rejects non-finite floats / 'inf' / 'nan'
        # explicitly, so int() here can't see an infinite value at runtime.
        # OverflowError is still in the except tuple as a defensive guard
        # in case future changes loosen convert_to_numeric.
        try:
            numeric_min = convert_to_numeric(min_val)
            numeric_max = convert_to_numeric(max_val)
            min_int = int(numeric_min)
            max_int = int(numeric_max)
        except (TypeError, ValueError, OverflowError) as e:
            raise ValueError(f"integer_range requires numeric values: {e}") from e
        return cls(float(min_int), float(max_int), is_integer=True)

    def is_scalar(self) -> bool:
        # math.isclose tolerates floating-point representation drift —
        # `RangeParameter(0.1 + 0.2, 0.3).is_scalar()` should be True.
        # Default rel_tol/abs_tol from math.isclose are fine here
        # (rel_tol=1e-09, abs_tol=0.0); parameter ranges are typically
        # in 1e-3..1e3 so relative tolerance is well-defined.
        return math.isclose(self.min_val, self.max_val)


@dataclass
class NestedParameter(AlbumentationsParameter):
    """
    Handles nested dictionary parameters like
    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}

    Automatically converts raw values to appropriate parameter types:
    - Tuples/lists of 2 numbers -> RangeParameter
    - Single numbers -> RangeParameter.scalar
    - Already AlbumentationsParameter -> used as-is
    """

    parameters: Dict[str, Any]  # Accept Any, we'll convert in __post_init__

    def __post_init__(self):
        """Convert raw values to AlbumentationsParameter instances"""
        converted = {}
        for key, value in self.parameters.items():
            if isinstance(value, AlbumentationsParameter):
                # Already a parameter object
                converted[key] = value
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                # Convert tuple/list to RangeParameter
                try:
                    min_val = convert_to_numeric(value[0])
                    max_val = convert_to_numeric(value[1])
                    converted[key] = RangeParameter(min_val, max_val)
                except TypeError as e:
                    raise TypeError(f"Invalid range for key '{key}': {e}") from e
            elif isinstance(value, (int, float, str)):
                # Convert scalar to RangeParameter.scalar
                try:
                    converted[key] = RangeParameter.scalar(value)
                except TypeError as e:
                    raise TypeError(f"Invalid scalar for key '{key}': {e}") from e
            elif value is None:
                raise TypeError(f"Value for key '{key}' cannot be None")
            else:
                raise TypeError(
                    f"Value for key '{key}' must be AlbumentationsParameter, "
                    f"tuple/list of 2 numbers, or scalar. Got {type(value)}"
                )

        self.parameters = converted

    def to_albumentations_format(self) -> Dict[str, Any]:
        return {key: param.to_albumentations_format() for key, param in self.parameters.items()}

    def sample(self) -> Dict[str, Any]:
        return {key: param.sample() for key, param in self.parameters.items()}
