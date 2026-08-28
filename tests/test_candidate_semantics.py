"""CPU-only tests for semantic candidate scoring and caching."""

import os
import sys
import tempfile

import numpy as np


SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
PROJECT_ROOT = os.path.abspath(os.path.join(SRC, ".."))

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import candidate_semantics


class FeatureOutput(object):
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


def text_embeddings(event_vector, ordinary_vector):
    embeddings = {
        group: np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        for group in candidate_semantics.SEMANTIC_GROUPS
    }
    embeddings["interaction_region"] = np.asarray(
        [event_vector],
        dtype=np.float32,
    )
    embeddings["negative"] = np.asarray(
        [ordinary_vector],
        dtype=np.float32,
    )
    return embeddings


def candidate_row(start_s, packet_score, packet_rank):
    return {
        "t_start_s": float(start_s),
        "t_end_s": float(start_s + 2.0),
        "candidate_score": float(packet_score),
        "candidate_percentile": 1.0,
        "candidate_rank": int(packet_rank),
    }


def test_policy_prompt_file_has_all_trigger_families_and_negative_baseline():
    path = os.path.join(PROJECT_ROOT, "configs", "candidate_semantic_prompts.json")
    config = candidate_semantics.load_prompt_config(path)

    assert tuple(config["positive_groups"]) == candidate_semantics.SEMANTIC_GROUPS
    assert len(config["negative_prompts"]) >= 2
    assert "ordinary" in " ".join(config["negative_prompts"]).lower()


def test_policy_prompts_treat_obvious_crosswalk_yielding_as_ordinary():
    path = os.path.join(PROJECT_ROOT, "configs", "candidate_semantic_prompts.json")
    config = candidate_semantics.load_prompt_config(path)
    pedestrian_prompts = " ".join(
        config["positive_groups"]["pedestrian_intent"]
    ).lower()
    negative_prompts = " ".join(config["negative_prompts"]).lower()

    assert "uncertain" in pedestrian_prompts
    assert "unexpectedly" in pedestrian_prompts
    assert "marked crosswalk" in negative_prompts
    assert "yields normally" in negative_prompts


def test_transformers_structured_feature_output_is_unwrapped():
    pooled = np.asarray([[1.0, 2.0]], dtype=np.float32)
    output = FeatureOutput(pooled)

    assert candidate_semantics.pooled_feature_tensor(output) is pooled


def test_transformers_tensor_feature_output_is_preserved():
    pooled = np.asarray([[1.0, 2.0]], dtype=np.float32)

    assert candidate_semantics.pooled_feature_tensor(pooled) is pooled


def test_semantic_implementation_version_guards_stale_file_copies():
    assert candidate_semantics.IMPLEMENTATION_VERSION == 4


def test_event_margin_ranks_above_ordinary_driving():
    rows = [
        candidate_row(0.0, packet_score=-1.0, packet_rank=2),
        candidate_row(2.0, packet_score=2.0, packet_rank=1),
    ]
    times = np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    images = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    prompts = text_embeddings([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    candidate_semantics.apply_semantic_ranking(
        rows,
        times,
        images,
        prompts,
        model_id="test/model",
        prompt_hash="test",
    )

    assert rows[0]["candidate_rank"] == 1
    assert rows[1]["candidate_rank"] == 2
    assert rows[0]["candidate_score"] > rows[1]["candidate_score"]
    assert rows[1]["packet_candidate_rank"] == 1


def test_packet_energy_only_breaks_a_semantic_tie():
    rows = [
        candidate_row(0.0, packet_score=0.1, packet_rank=2),
        candidate_row(2.0, packet_score=0.9, packet_rank=1),
    ]
    times = np.asarray([0.5, 2.5], dtype=np.float64)
    images = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    prompts = text_embeddings([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    candidate_semantics.apply_semantic_ranking(
        rows,
        times,
        images,
        prompts,
        model_id="test/model",
        prompt_hash="test",
    )

    assert rows[1]["candidate_rank"] == 1
    assert rows[0]["candidate_rank"] == 2


def test_embedding_cache_is_invalidated_when_source_changes():
    directory = tempfile.mkdtemp()
    source = os.path.join(directory, "source.mp4")
    cache = os.path.join(directory, "cache", "source.npz")

    with open(source, "wb") as handle:
        handle.write(b"first")

    candidate_semantics.write_embedding_cache(
        cache,
        source,
        "test/model",
        [0.0, 1.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    loaded = candidate_semantics.load_embedding_cache(cache, source, "test/model")
    assert loaded is not None

    with open(source, "ab") as handle:
        handle.write(b"changed")

    assert candidate_semantics.load_embedding_cache(
        cache,
        source,
        "test/model",
    ) is None


def test_embedding_cache_is_invalidated_when_mapping_intervals_change():
    directory = tempfile.mkdtemp()
    source = os.path.join(directory, "source.mp4")
    cache = os.path.join(directory, "cache", "source.npz")
    intervals = [{"start_s": 15.0, "end_s": 45.0}]

    with open(source, "wb") as handle:
        handle.write(b"source")

    candidate_semantics.write_embedding_cache(
        cache,
        source,
        "test/model",
        [15.0, 20.0],
        [[1.0, 0.0], [0.0, 1.0]],
        intervals=intervals,
    )

    assert candidate_semantics.load_embedding_cache(
        cache,
        source,
        "test/model",
        intervals=intervals,
    ) is not None
    assert candidate_semantics.load_embedding_cache(
        cache,
        source,
        "test/model",
        intervals=[{"start_s": 20.0, "end_s": 50.0}],
    ) is None


if __name__ == "__main__":
    failures = 0

    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue

        try:
            test()
            print("PASS  %s" % name)
        except AssertionError as error:
            failures += 1
            print("FAIL  %s\n      %s" % (name, error))

    raise SystemExit(1 if failures else 0)
