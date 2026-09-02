"""Dependency free checks for the pairwise preference study contract."""

from __future__ import annotations

import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVICE = os.path.join(ROOT, "study_service")
sys.path.insert(0, SERVICE)

import preference_data
import space


def test_approved_comparison_budget():
    assert space.N_EXPLORATION_COMPARISONS == 10
    assert space.N_EUBO_COMPARISONS == 4
    assert space.N_TOTAL_COMPARISONS == 14


def test_space_has_three_continuous_parameters_and_one_categorical_palette():
    assert tuple(parameter.name for parameter in space.CONTINUOUS_PARAMETERS) == (
        "mask_alpha",
        "trajectory_alpha",
        "background_dim_alpha",
    )
    assert space.PALETTE_VALUES == (0, 1, 2, 3)
    assert space.D_MODEL == 7


def test_palette_is_one_hot_not_ordinal():
    base = space.default_config()
    rows = []
    for palette in space.PALETTE_VALUES:
        config = dict(base, palette_id=palette)
        row = space.encode_config(config)
        rows.append(row)
        assert sum(row[space.D_CONTINUOUS :]) == 1.0
        assert row[space.D_CONTINUOUS + palette] == 1.0
    assert len(set(rows)) == 4


def test_query_results_become_winner_loser_indices():
    option_a = space.default_config()
    option_b = dict(option_a, mask_alpha=0.4, palette_id=2)
    training = preference_data.build_training_data(
        [
            {
                "comparisonStep": 1,
                "optionA": option_a,
                "optionB": option_b,
            }
        ],
        [
            {
                "comparisonStep": 1,
                "preferredOption": "prefer_b",
                "cityPhase": "familiar_optimisation",
                "attentionCheckPassed": True,
            }
        ],
    )
    assert training.comparisons == ((1, 0),)
    assert training.completed_steps == (1,)
    assert len(training.observed_pair_keys) == 1


def test_failed_attention_check_does_not_train_the_model():
    option_a = space.default_config()
    option_b = dict(option_a, trajectory_alpha=0.9)
    query = {"comparisonStep": 1, "optionA": option_a, "optionB": option_b}
    failed = {
        "comparisonStep": 1,
        "preferredOption": "prefer_a",
        "cityPhase": "familiar_optimisation",
        "attentionCheckPassed": False,
    }
    training = preference_data.build_training_data([query], [failed])
    assert training.comparisons == ()
    assert training.completed_steps == ()


def test_duplicate_trigger_delivery_counts_once():
    option_a = space.default_config()
    option_b = dict(option_a, background_dim_alpha=0.2)
    query = {"comparisonStep": 1, "optionA": option_a, "optionB": option_b}
    result = {
        "comparisonStep": 1,
        "preferredOption": "prefer_a",
        "cityPhase": "familiar_optimisation",
        "attentionCheckPassed": True,
    }
    training = preference_data.build_training_data([query], [result, dict(result)])
    assert len(training.comparisons) == 1


def test_distant_city_result_never_updates_the_model():
    option_a = space.default_config()
    option_b = dict(option_a, palette_id=3)
    query = {"comparisonStep": 1, "optionA": option_a, "optionB": option_b}
    distant = {
        "comparisonStep": 1,
        "preferredOption": "prefer_b",
        "cityPhase": "distant_evaluation",
        "attentionCheckPassed": True,
    }
    training = preference_data.build_training_data([query], [distant])
    assert training.comparisons == ()


def test_protocol_config_matches_the_service():
    path = os.path.join(ROOT, "configs", "pbo_objectives.json")
    with open(path, "r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    assert protocol["rating_mode"] == "forced_choice_pairwise_preference"
    assert protocol["preference_question"] == space.PREFERENCE_QUESTION
    assert protocol["comparison_budget"]["total"] == 14
    assert protocol["response_scale"] == ["prefer_a", "prefer_b"]


def test_render_space_config_matches_the_service():
    path = os.path.join(ROOT, "configs", "bo_render_space.json")
    with open(path, "r", encoding="utf-8") as handle:
        configured = json.load(handle)["parameters"]
    for parameter in space.CONTINUOUS_PARAMETERS:
        definition = configured[parameter.name]
        assert definition["min"] == parameter.minimum
        assert definition["max"] == parameter.maximum
        assert definition["default"] == parameter.default
    assert tuple(configured["palette_id"]["values"]) == space.PALETTE_VALUES


def test_firebase_uses_a_supported_node_runtime():
    path = os.path.join(SERVICE, "firebase", "package.json")
    with open(path, "r", encoding="utf-8") as handle:
        package = json.load(handle)
    assert package["engines"]["node"] == "22"


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS ", test.__name__)


if __name__ == "__main__":
    main()
