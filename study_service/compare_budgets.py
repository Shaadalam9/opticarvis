"""Compare candidate EUBO budgets with repeatable synthetic participants.

The deployment default remains ten Sobol plus four EUBO comparisons. This
offline study evaluates whether additional EUBO comparisons reduce synthetic
simple regret before the real participant budget is finalised.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import simulate
import space


EUBO_BUDGETS_ENV = "OPTICARVIS_PBO_EUBO_BUDGETS"
SIMULATION_SEEDS_ENV = "OPTICARVIS_PBO_SIMULATION_SEEDS"
RESULT_PATH_ENV = "OPTICARVIS_PBO_BUDGET_RESULTS"


def positive_integer_list(raw: str, name: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{name} must contain comma separated integers") from exc
        if value < 1:
            raise ValueError(f"{name} values must be at least 1")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(dict.fromkeys(values))


def summarise_runs(runs: list[dict]) -> list[dict]:
    summaries = []
    eubo_counts = sorted({int(run["eubo_comparisons"]) for run in runs})
    for eubo_comparisons in eubo_counts:
        matching = [
            run for run in runs if run["eubo_comparisons"] == eubo_comparisons
        ]
        regrets = [float(run["synthetic_simple_regret"]) for run in matching]
        palette_successes = sum(
            bool(run["favourite_palette_selected"]) for run in matching
        )
        summaries.append(
            {
                "exploration_comparisons": matching[0]["exploration_comparisons"],
                "eubo_comparisons": eubo_comparisons,
                "total_comparisons": matching[0]["total_comparisons"],
                "simulation_runs": len(matching),
                "mean_synthetic_simple_regret": statistics.fmean(regrets),
                "median_synthetic_simple_regret": statistics.median(regrets),
                "maximum_synthetic_simple_regret": max(regrets),
                "favourite_palette_selection_rate": palette_successes / len(matching),
            }
        )
    return summaries


def default_result_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "workflow_outputs" / "pbo_budget_study.json"


def main() -> None:
    exploration = space.comparison_budget_from_environment().exploration_comparisons
    eubo_budgets = positive_integer_list(
        os.environ.get(EUBO_BUDGETS_ENV, "4,8,12"), EUBO_BUDGETS_ENV
    )
    seeds = positive_integer_list(
        os.environ.get(SIMULATION_SEEDS_ENV, "7"), SIMULATION_SEEDS_ENV
    )
    output_path = Path(os.environ.get(RESULT_PATH_ENV, str(default_result_path())))

    runs = []
    for eubo_comparisons in eubo_budgets:
        budget = space.ComparisonBudget(exploration, eubo_comparisons)
        for seed in seeds:
            print(f"running {budget.protocol_id} seed={seed}")
            runs.append(
                simulate.run_simulation(budget=budget, seed=seed, verbose=False)
            )

    summaries = summarise_runs(runs)
    output = {
        "purpose": "pilot comparison budget sensitivity analysis",
        "syntheticUtilityOnly": True,
        "warning": "Do not treat synthetic results as participant evidence.",
        "runs": runs,
        "summaries": summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("\nBudget summary")
    print("==============")
    for summary in summaries:
        print(
            f"{summary['exploration_comparisons']} Sobol + "
            f"{summary['eubo_comparisons']} EUBO = "
            f"{summary['total_comparisons']} | "
            f"mean regret={summary['mean_synthetic_simple_regret']:.4f} | "
            f"palette success={summary['favourite_palette_selection_rate']:.2f}"
        )
    print(f"results: {output_path}")


if __name__ == "__main__":
    main()
