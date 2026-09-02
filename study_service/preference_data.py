"""Pure data handling for OptiCarVis pairwise preference observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import space


@dataclass(frozen=True)
class TrainingData:
    raw_configs: tuple[dict[str, float | int], ...]
    model_rows: tuple[tuple[float, ...], ...]
    comparisons: tuple[tuple[int, int], ...]
    observed_pair_keys: frozenset[
        tuple[tuple[float | int, ...], tuple[float | int, ...]]
    ]
    completed_steps: tuple[int, ...]


def preference_side(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "a": "A",
        "prefer_a": "A",
        "option_a": "A",
        "b": "B",
        "prefer_b": "B",
        "option_b": "B",
    }
    if text not in aliases:
        raise ValueError("preferredOption must be prefer_a or prefer_b")
    return aliases[text]


def _positive_step(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("comparisonStep must be a positive integer")
    try:
        step = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("comparisonStep must be a positive integer") from exc
    if step < 1:
        raise ValueError("comparisonStep must be a positive integer")
    return step


def build_training_data(
    query_documents: Iterable[Mapping[str, object]],
    result_documents: Iterable[Mapping[str, object]],
) -> TrainingData:
    """Join query and result documents into PairwiseGP inputs.

    Each comparison is stored as ``(winner_index, loser_index)``. Failed
    attention checks and duplicate result deliveries do not create training
    observations.
    """
    queries: dict[int, tuple[dict[str, float | int], dict[str, float | int]]] = {}
    for document in query_documents:
        step = _positive_step(document["comparisonStep"])
        option_a = space.validate_config(document["optionA"])  # type: ignore[arg-type]
        option_b = space.validate_config(document["optionB"])  # type: ignore[arg-type]
        if space.config_key(option_a) == space.config_key(option_b):
            raise ValueError(f"comparison {step} contains identical options")
        queries[step] = (option_a, option_b)

    raw_configs: list[dict[str, float | int]] = []
    model_rows: list[tuple[float, ...]] = []
    config_indices: dict[tuple[float | int, ...], int] = {}
    comparisons: list[tuple[int, int]] = []
    observed_pair_keys = set()
    completed_steps: list[int] = []
    seen_steps = set()

    def config_index(config: dict[str, float | int]) -> int:
        key = space.config_key(config)
        if key not in config_indices:
            config_indices[key] = len(raw_configs)
            raw_configs.append(config)
            model_rows.append(space.encode_config(config))
        return config_indices[key]

    ordered_results = sorted(
        result_documents,
        key=lambda document: _positive_step(document["comparisonStep"]),
    )
    for document in ordered_results:
        if document.get("attentionCheckPassed") is False:
            continue
        if document.get("cityPhase") != "familiar_optimisation":
            continue
        step = _positive_step(document["comparisonStep"])
        if step in seen_steps:
            continue
        if step not in queries:
            raise ValueError(f"result for comparison {step} has no matching query")

        option_a, option_b = queries[step]
        side = preference_side(document["preferredOption"])
        index_a = config_index(option_a)
        index_b = config_index(option_b)
        winner, loser = (index_a, index_b) if side == "A" else (index_b, index_a)

        comparisons.append((winner, loser))
        observed_pair_keys.add(space.unordered_pair_key(option_a, option_b))
        completed_steps.append(step)
        seen_steps.add(step)

    return TrainingData(
        raw_configs=tuple(raw_configs),
        model_rows=tuple(model_rows),
        comparisons=tuple(comparisons),
        observed_pair_keys=frozenset(observed_pair_keys),
        completed_steps=tuple(completed_steps),
    )
