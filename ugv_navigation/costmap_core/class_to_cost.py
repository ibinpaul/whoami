"""Canonical semantic class -> costmap cost value mapping.

This is the single place where semantic class IDs (defined by Dev 1's
segmentation contract) are translated into costmap cost values. Keeping the
mapping here -- instead of scattering cost-value literals through other
modules -- means the mapping can be changed later without touching any
consumer of `class_to_cost`.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# Costmap cost range (Nav2 costmap_2d: FREE_SPACE=0 .. NO_INFORMATION=255).
# costmap_ros.occupancy_grid translates exactly this range and rejects the rest.
MIN_COST = 0
MAX_COST = 255


class SemanticClass(IntEnum):
    """Canonical semantic classes published on /segmentation/mask."""

    UNKNOWN = 0
    TRAVERSABLE = 1
    HAZARD = 2


class InvalidSemanticClassError(ValueError):
    """Raised when a class id outside the canonical set is encountered.

    Deliberately not silently coerced to TRAVERSABLE or any other class --
    an invalid class id must fail loudly rather than be treated as safe.
    """


class InvalidCostValueError(ValueError):
    """Raised when a cost value definition is invalid.

    A bad cost definition must fail loudly at construction: e.g. a float
    unknown cost would be truncated by the int64 costmaps, and an unknown
    cost equal to the traversable cost would silently make unknown free.
    """


def require_cost(value: object, *, name: str, error: type[ValueError] = InvalidCostValueError) -> int:
    """Return `value` as an int if it is an integer cost in MIN_COST..MAX_COST.

    Python and numpy integers are accepted; bool (including numpy bool),
    floats (even integral ones), strings and anything else are rejected.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise error(f"{name} must be an integer cost, got {type(value).__name__} {value!r}")
    value = int(value)
    if not MIN_COST <= value <= MAX_COST:
        raise error(f"{name} must be in {MIN_COST}..{MAX_COST}, got {value}")
    return value


@dataclass(frozen=True)
class CostValues:
    """Cost values assigned to each canonical semantic class.

    Defaults follow Nav2's documented costmap_2d cost conventions
    (FREE_SPACE=0, LETHAL_OBSTACLE=254, NO_INFORMATION=255), since that is
    the interface this subsystem must eventually feed. These are still only
    defaults: construct a different CostValues to override them if the
    team's final conventions differ.

    Validation (InvalidCostValueError), from the existing semantics:
    - every cost is an integer in MIN_COST..MAX_COST (require_cost);
    - the three costs are pairwise distinct: traversable == unknown would
      make unknown free (architecture §8.1), traversable == hazard would
      make hazards free, and unknown == hazard would turn every unseen cell
      into a lethal obstacle (the pipeline uses hazard_cost as its lethal
      value for fusion and inflation);
    - traversable_cost < hazard_cost: inflation decays from lethal - 1 down
      to 0 and keeps max(original, inflated), which presumes free < lethal.
    No order is imposed on unknown_cost.
    """

    unknown_cost: int = 255
    traversable_cost: int = 0
    hazard_cost: int = 254

    def __post_init__(self) -> None:
        for name in ("unknown_cost", "traversable_cost", "hazard_cost"):
            object.__setattr__(self, name, require_cost(getattr(self, name), name=name))
        costs = {
            "unknown_cost": self.unknown_cost,
            "traversable_cost": self.traversable_cost,
            "hazard_cost": self.hazard_cost,
        }
        if len(set(costs.values())) != len(costs):
            raise InvalidCostValueError(f"semantic cost values must be pairwise distinct, got {costs}")
        if not self.traversable_cost < self.hazard_cost:
            raise InvalidCostValueError(
                f"traversable_cost ({self.traversable_cost}) must be < hazard_cost ({self.hazard_cost})"
            )

    def as_mapping(self) -> dict:
        return {
            SemanticClass.UNKNOWN: self.unknown_cost,
            SemanticClass.TRAVERSABLE: self.traversable_cost,
            SemanticClass.HAZARD: self.hazard_cost,
        }


DEFAULT_COST_VALUES = CostValues()


def class_to_cost(class_id: int, cost_values: CostValues = DEFAULT_COST_VALUES) -> int:
    """Map a canonical semantic class id to a costmap cost value.

    Raises InvalidSemanticClassError for any class_id outside {0, 1, 2}.
    """
    try:
        semantic_class = SemanticClass(class_id)
    except ValueError as exc:
        raise InvalidSemanticClassError(
            f"Invalid semantic class id: {class_id!r}. "
            f"Expected one of {[c.value for c in SemanticClass]}."
        ) from exc

    return cost_values.as_mapping()[semantic_class]
