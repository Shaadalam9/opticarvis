"""Offline simulation of the approved 14 comparison preference study."""

from __future__ import annotations

import math
import random

import optimizer_core
import preference_data
import space


FAVOURITE = {
    "mask_alpha": 0.32,
    "trajectory_alpha": 0.72,
    "background_dim_alpha": 0.12,
    "palette_id": 2,
}


def utility(config: dict) -> float:
    row = space.encode_config(config)
    favourite = space.encode_config(FAVOURITE)
    continuous_distance = sum(
        (row[index] - favourite[index]) ** 2
        for index in range(space.D_CONTINUOUS)
    )
    palette_match = config["palette_id"] == FAVOURITE["palette_id"]
    return -continuous_distance + (0.25 if palette_match else 0.0)


def main() -> None:
    random.seed(7)
    queries = []
    results = []

    for step in range(1, space.N_TOTAL_COMPARISONS + 1):
        training = preference_data.build_training_data(queries, results)
        if step <= space.N_EXPLORATION_COMPARISONS:
            option_a, option_b = optimizer_core.sobol_pair(step, seed=7)
            phase = "exploration"
        else:
            model = optimizer_core.fit_preference_model(training)
            option_a, option_b = optimizer_core.propose_eubo_pair(
                model,
                training.observed_pair_keys,
                comparison_step=step,
                seed=7 + step,
            )
            phase = "optimisation"

        query = {
            "comparisonStep": step,
            "optionA": option_a,
            "optionB": option_b,
        }
        noisy_a = utility(option_a) + random.gauss(0.0, 0.03)
        noisy_b = utility(option_b) + random.gauss(0.0, 0.03)
        result = {
            "comparisonStep": step,
            "preferredOption": "prefer_a" if noisy_a >= noisy_b else "prefer_b",
            "cityPhase": "familiar_optimisation",
            "attentionCheckPassed": True,
        }
        queries.append(query)
        results.append(result)
        print(
            f"comparison {step:>2} [{phase:<12}] "
            f"preferred={result['preferredOption'][-1].upper()}"
        )

    training = preference_data.build_training_data(queries, results)
    model = optimizer_core.fit_preference_model(training)
    selected, estimate = optimizer_core.select_best_observed(
        model, training.raw_configs, training.model_rows
    )
    assert len(training.comparisons) == 14
    assert all(math.isfinite(value) for value in training.model_rows[0])
    print(f"\nselected: {selected}")
    print(f"posterior mean utility: {estimate:.4f}")
    print("PASS: 10 Sobol comparisons + 4 EUBO comparisons")


if __name__ == "__main__":
    main()
