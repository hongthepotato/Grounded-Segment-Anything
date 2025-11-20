"""
Albumentations Parameter System
Provides a unified interface for all albumentations parameters
Supports range, nested, and list parameters
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple
from dataclasses import dataclass
import random

def convert_to_numeric(value: Any) -> Union[int, float]:
    """
    Convert value to numeric type.
    Accepts: int, float, or string representations of numbers
    Rejects: None, non-numeric strings, and other types
    """
    # Reject None explicitly
    if value is None:
        raise TypeError("Value cannot be None")

    # If already numeric, return as-is
    if isinstance(value, (int, float)):
        return value

    # Try to convert string to numeric
    if isinstance(value, str):
        try:
            # Try int first (for strings like "10")
            if '.' not in value and 'e' not in value.lower():
                return int(value)
            # Otherwise convert to float
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

        # Validate range
        if self.min_val > self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be <= max_val ({self.max_val})")

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
    def scalar(cls, value: Union[int, float, str], is_integer: bool = False) -> 'RangeParameter':
        """
        Create a scalar parameter (min == max)
        Accepts: int, float, or string representations of numbers
        Rejects: None and non-numeric values
        """
        numeric_value = convert_to_numeric(value)
        return cls(numeric_value, numeric_value, is_integer)

    @classmethod
    def integer_range(
            cls, min_val: Union[int, float, str],
            max_val: Union[int, float, str]
    ) -> 'RangeParameter':
        """
        Create an integer range parameter
        Accepts: int, float, or string representations of numbers
        Values will be converted to integers
        """
        try:
            numeric_min = convert_to_numeric(min_val)
            numeric_max = convert_to_numeric(max_val)
            min_int = int(numeric_min)
            max_int = int(numeric_max)
        except (TypeError, ValueError) as e:
            raise ValueError(f"integer_range requires numeric values: {e}") from e
        return cls(float(min_int), float(max_int), is_integer=True)

    def is_scalar(self) -> bool:
        return self.min_val == self.max_val

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
        return {key: param.to_albumentations_format()
                for key, param in self.parameters.items()}

    def sample(self) -> Dict[str, Any]:
        return {key: param.sample()
                for key, param in self.parameters.items()}
