"""OptiCarVis preference optimisation search space.

The participant study optimises three continuous rendering parameters and one
nominal palette choice.  Continuous values are represented in the unit cube.
The palette is one hot encoded so the preference model does not invent an
ordinal distance between palette identifiers.

No perceptual step size is imposed yet.  The literature and pilot evidence for
opacity and dimming thresholds must be settled before discretisation can be
claimed to represent a just noticeable difference.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


PREFERENCE_QUESTION = (
    "Which version would you prefer to have while riding in an automated vehicle?"
)

PROTOCOL_VERSION = "pbo_pairwise_eubo_v3"
EXPLORATION_ENV = "OPTICARVIS_PBO_EXPLORATION_COMPARISONS"
EUBO_ENV = "OPTICARVIS_PBO_EUBO_COMPARISONS"

N_EXPLORATION_COMPARISONS = 10
N_EUBO_COMPARISONS = 8
N_TOTAL_COMPARISONS = N_EXPLORATION_COMPARISONS + N_EUBO_COMPARISONS


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if str(value).strip() != str(result) and not isinstance(value, int):
        raise ValueError(f"{name} must not contain a fractional value")
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


@dataclass(frozen=True)
class ComparisonBudget:
    exploration_comparisons: int = N_EXPLORATION_COMPARISONS
    eubo_comparisons: int = N_EUBO_COMPARISONS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exploration_comparisons",
            _positive_integer(
                "exploration_comparisons", self.exploration_comparisons
            ),
        )
        object.__setattr__(
            self,
            "eubo_comparisons",
            _positive_integer("eubo_comparisons", self.eubo_comparisons),
        )

    @property
    def total_comparisons(self) -> int:
        return self.exploration_comparisons + self.eubo_comparisons

    @property
    def protocol_id(self) -> str:
        return (
            f"{PROTOCOL_VERSION}_sobol{self.exploration_comparisons}"
            f"_eubo{self.eubo_comparisons}"
        )

    def to_document(self) -> dict[str, int]:
        return {
            "explorationSobol": self.exploration_comparisons,
            "optimisationEubo": self.eubo_comparisons,
            "total": self.total_comparisons,
        }


def comparison_budget_from_document(raw: Mapping[str, object]) -> ComparisonBudget:
    budget = ComparisonBudget(
        exploration_comparisons=raw["explorationSobol"],  # type: ignore[arg-type]
        eubo_comparisons=raw["optimisationEubo"],  # type: ignore[arg-type]
    )
    if "total" in raw and _positive_integer("total", raw["total"]) != budget.total_comparisons:
        raise ValueError("comparison budget total does not match its components")
    return budget


def comparison_budget_from_environment(
    environ: Mapping[str, str] | None = None,
) -> ComparisonBudget:
    values = os.environ if environ is None else environ
    return ComparisonBudget(
        exploration_comparisons=values.get(
            EXPLORATION_ENV, str(N_EXPLORATION_COMPARISONS)
        ),
        eubo_comparisons=values.get(EUBO_ENV, str(N_EUBO_COMPARISONS)),
    )


def protocol_document(budget: ComparisonBudget) -> dict[str, object]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "protocolId": budget.protocol_id,
        "comparisonBudget": budget.to_document(),
    }


@dataclass(frozen=True)
class ContinuousParameter:
    name: str
    minimum: float
    maximum: float
    default: float
    round_digits: int = 4

    def to_unit(self, value: float) -> float:
        value = _finite_float(self.name, value)
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"{self.name}={value} is outside [{self.minimum}, {self.maximum}]"
            )
        span = self.maximum - self.minimum
        return 0.0 if span == 0.0 else (value - self.minimum) / span

    def from_unit(self, value: float) -> float:
        value = _finite_float(f"normalised {self.name}", value)
        value = min(1.0, max(0.0, value))
        raw = self.minimum + value * (self.maximum - self.minimum)
        return round(raw, self.round_digits)


CONTINUOUS_PARAMETERS = (
    ContinuousParameter("mask_alpha", 0.0, 0.7, 0.14),
    ContinuousParameter("trajectory_alpha", 0.0, 1.0, 0.55),
    ContinuousParameter("background_dim_alpha", 0.0, 0.4, 0.06),
)
PALETTE_VALUES = (0, 1, 2, 3)

PARAMETER_NAMES = tuple(p.name for p in CONTINUOUS_PARAMETERS) + ("palette_id",)
D_CONTINUOUS = len(CONTINUOUS_PARAMETERS)
D_PALETTE_ONE_HOT = len(PALETTE_VALUES)
D_MODEL = D_CONTINUOUS + D_PALETTE_ONE_HOT


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_config(raw: Mapping[str, object]) -> dict[str, float | int]:
    """Return a canonical raw configuration or raise a useful error."""
    config: dict[str, float | int] = {}
    for parameter in CONTINUOUS_PARAMETERS:
        value = _finite_float(parameter.name, raw[parameter.name])
        if not parameter.minimum <= value <= parameter.maximum:
            raise ValueError(
                f"{parameter.name}={value} is outside "
                f"[{parameter.minimum}, {parameter.maximum}]"
            )
        config[parameter.name] = round(value, parameter.round_digits)

    palette_raw = raw["palette_id"]
    if isinstance(palette_raw, bool):
        raise TypeError("palette_id must be an integer category")
    try:
        palette = int(palette_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("palette_id must be an integer category") from exc
    if palette_raw != palette and str(palette_raw) != str(palette):
        raise ValueError("palette_id must not contain a fractional value")
    if palette not in PALETTE_VALUES:
        raise ValueError(f"palette_id must be one of {PALETTE_VALUES}")
    config["palette_id"] = palette
    return config


def default_config() -> dict[str, float | int]:
    return {
        **{parameter.name: parameter.default for parameter in CONTINUOUS_PARAMETERS},
        "palette_id": PALETTE_VALUES[0],
    }


def encode_config(raw: Mapping[str, object]) -> tuple[float, ...]:
    """Encode raw render parameters for PairwiseGP.

    The first three dimensions are normalised continuous values.  The final
    four dimensions form a one hot palette representation.
    """
    config = validate_config(raw)
    continuous = [parameter.to_unit(config[parameter.name]) for parameter in CONTINUOUS_PARAMETERS]
    palette = int(config["palette_id"])
    one_hot = [1.0 if value == palette else 0.0 for value in PALETTE_VALUES]
    return tuple(continuous + one_hot)


def config_from_unit_continuous(
    unit_values: Sequence[float], palette_id: int
) -> dict[str, float | int]:
    if len(unit_values) != D_CONTINUOUS:
        raise ValueError(
            f"expected {D_CONTINUOUS} continuous values, got {len(unit_values)}"
        )
    raw = {
        parameter.name: parameter.from_unit(value)
        for parameter, value in zip(CONTINUOUS_PARAMETERS, unit_values)
    }
    raw["palette_id"] = palette_id
    return validate_config(raw)


def decode_model_row(row: Sequence[float]) -> dict[str, float | int]:
    if len(row) != D_MODEL:
        raise ValueError(f"expected model row length {D_MODEL}, got {len(row)}")
    palette_scores = row[D_CONTINUOUS:]
    palette_index = max(range(len(palette_scores)), key=lambda i: float(palette_scores[i]))
    return config_from_unit_continuous(row[:D_CONTINUOUS], PALETTE_VALUES[palette_index])


def config_key(raw: Mapping[str, object]) -> tuple[float | int, ...]:
    config = validate_config(raw)
    return tuple(config[name] for name in PARAMETER_NAMES)


def unordered_pair_key(
    option_a: Mapping[str, object], option_b: Mapping[str, object]
) -> tuple[tuple[float | int, ...], tuple[float | int, ...]]:
    keys = (config_key(option_a), config_key(option_b))
    return tuple(sorted(keys))  # type: ignore[return-value]


def validate_model_rows(rows: Iterable[Sequence[float]]) -> None:
    for row in rows:
        if len(row) != D_MODEL:
            raise ValueError(f"expected model row length {D_MODEL}, got {len(row)}")
        if any(not math.isfinite(float(value)) for value in row):
            raise ValueError("model rows must contain only finite values")


def describe(budget: ComparisonBudget | None = None) -> str:
    budget = budget or ComparisonBudget()
    lines = [
        "OptiCarVis pairwise preference space",
        f"protocol: {budget.protocol_id}",
        f"comparisons: {budget.exploration_comparisons} Sobol + "
        f"{budget.eubo_comparisons} EUBO = {budget.total_comparisons}",
        f"question: {PREFERENCE_QUESTION}",
    ]
    for parameter in CONTINUOUS_PARAMETERS:
        lines.append(
            f"{parameter.name}: [{parameter.minimum}, {parameter.maximum}] "
            f"default={parameter.default}"
        )
    lines.append(f"palette_id: {list(PALETTE_VALUES)} (one hot model encoding)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
