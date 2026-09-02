"""Offline simulation of one configurable pairwise preference study."""

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


def run_simulation(
    budget: space.ComparisonBudget,
    seed: int = 7,
    noise_standard_deviation: float = 0.03,
    verbose: bool = True,
) -> dict:
    random_source = random.Random(seed)
    queries = []
    results = []

    for step in range(1, budget.total_comparisons + 1):
        training = preference_data.build_training_data(queries, results)
        if step <= budget.exploration_comparisons:
            option_a, option_b = optimizer_core.sobol_pair(step, seed=seed)
            phase = "exploration"
        else:
            model = optimizer_core.fit_preference_model(training)
            option_a, option_b = optimizer_core.propose_eubo_pair(
                model,
                training.observed_pair_keys,
                comparison_step=step,
                seed=seed + step,
            )
            phase = "optimisation"

        query = {
            "comparisonStep": step,
            "optionA": option_a,
            "optionB": option_b,
        }
        noisy_a = utility(option_a) + random_source.gauss(
            0.0, noise_standard_deviation
        )
        noisy_b = utility(option_b) + random_source.gauss(
            0.0, noise_standard_deviation
        )
        result = {
            "comparisonStep": step,
            "preferredOption": "prefer_a" if noisy_a >= noisy_b else "prefer_b",
            "cityPhase": "familiar_optimisation",
            "attentionCheckPassed": True,
        }
        queries.append(query)
        results.append(result)
        if verbose:
            print(
                f"comparison {step:>2} [{phase:<12}] "
                f"preferred={result['preferredOption'][-1].upper()}"
            )

    training = preference_data.build_training_data(queries, results)
    model = optimizer_core.fit_preference_model(training)
    selected, estimate = optimizer_core.select_best_observed(
        model, training.raw_configs, training.model_rows
    )
    assert len(training.comparisons) == budget.total_comparisons
    assert all(math.isfinite(value) for value in training.model_rows[0])
    selected_utility = utility(selected)
    simple_regret = utility(FAVOURITE) - selected_utility
    result = {
        "protocol_id": budget.protocol_id,
        "seed": seed,
        "exploration_comparisons": budget.exploration_comparisons,
        "eubo_comparisons": budget.eubo_comparisons,
        "total_comparisons": budget.total_comparisons,
        "selected_config": selected,
        "posterior_mean_utility": estimate,
        "synthetic_true_utility": selected_utility,
        "synthetic_simple_regret": simple_regret,
        "favourite_palette_selected": selected["palette_id"] == FAVOURITE["palette_id"],
    }
    if verbose:
        print(f"\nselected: {selected}")
        print(f"posterior mean utility: {estimate:.4f}")
        print(f"synthetic simple regret: {simple_regret:.4f}")
        print(
            f"PASS: {budget.exploration_comparisons} Sobol comparisons + "
            f"{budget.eubo_comparisons} EUBO comparisons"
        )
    return result


def main() -> None:
    budget = space.comparison_budget_from_environment()
    run_simulation(budget)


if __name__ == "__main__":
    main()
