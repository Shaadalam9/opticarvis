"""Cached keyframe semantics for offline candidate ranking.

The encoder is deliberately separate from job building. Each source video is
decoded once in keyframe-only mode, its image embeddings are cached, and the
fixed candidate windows are scored against auditable policy-derived prompts.
Gemma remains the downstream explanation gate.
"""

import hashlib
import json
import math
import os
import re

import numpy as np

from pipeline_common import CANDIDATE_SEMANTIC_MODEL, HF_LOCAL_FILES_ONLY
from candidate_index import normalise_intervals


CACHE_VERSION = 2
IMPLEMENTATION_VERSION = 4
SEMANTIC_GROUPS = (
    "interaction_region",
    "pedestrian_intent",
    "proximity_risk",
    "critical_conflict",
    "trajectory_uncertainty",
    "unusual_context",
)
SEMANTIC_SCORE_COLUMNS = tuple(
    "semantic_" + group + "_score" for group in SEMANTIC_GROUPS
)


def load_prompt_config(path):
    """Load and validate the policy prompt configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    groups = config.get("positive_groups")
    negatives = config.get("negative_prompts")

    if not isinstance(groups, dict):
        raise ValueError("positive_groups must be an object")
    if tuple(groups) != SEMANTIC_GROUPS:
        raise ValueError(
            "positive_groups must be ordered as: " + ", ".join(SEMANTIC_GROUPS)
        )

    for group, prompts in groups.items():
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("Prompt group %s must contain prompts" % group)
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
            raise ValueError("Prompt group %s contains an empty prompt" % group)

    if not isinstance(negatives, list) or not negatives:
        raise ValueError("negative_prompts must contain prompts")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in negatives):
        raise ValueError("negative_prompts contains an empty prompt")

    return config


def prompt_config_hash(config):
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalise_embeddings(values):
    array = np.asarray(values, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError("Embeddings must be a two dimensional array")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def pooled_feature_tensor(output):
    """Return the embedding tensor across Transformers feature API versions.

    Transformers 5.15 returns BaseModelOutputWithPooling from SigLIP2 feature
    helpers, while earlier releases returned the pooled tensor directly.
    """
    for attribute in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, attribute, None)

        if value is not None:
            return value

    if isinstance(output, (tuple, list)) and output:
        return output[0]

    if hasattr(output, "detach") or isinstance(output, np.ndarray):
        return output

    available = []

    if hasattr(output, "keys"):
        try:
            available = list(output.keys())
        except Exception:
            available = []

    raise TypeError(
        "Unsupported feature output %s with fields %s"
        % (type(output).__name__, available)
    )


def top_k_mean(values, top_k=2):
    array = np.asarray(values, dtype=np.float32).reshape(-1)

    if array.size == 0:
        return float("nan")

    count = min(max(1, int(top_k)), array.size)
    highest = np.partition(array, array.size - count)[-count:]
    return float(np.mean(highest))


def semantic_window_features(
    rows,
    keyframe_times_s,
    image_embeddings,
    text_embeddings,
    top_k=2,
):
    """Return per-window prompt margins from precomputed embeddings.

    A positive group must beat the ordinary-driving baseline on the same
    keyframes. This reduces prompt-independent image/text similarity bias.
    """
    times = np.asarray(keyframe_times_s, dtype=np.float64).reshape(-1)
    images = normalise_embeddings(image_embeddings)

    if len(times) != len(images):
        raise ValueError("Keyframe timestamps and image embeddings differ in length")
    if not len(times):
        raise ValueError("No keyframe embeddings are available")

    negatives = normalise_embeddings(text_embeddings["negative"])
    negative_similarity = np.max(images @ negatives.T, axis=1)
    group_margins = {}

    for group in SEMANTIC_GROUPS:
        prompts = normalise_embeddings(text_embeddings[group])
        group_similarity = np.max(images @ prompts.T, axis=1)
        group_margins[group] = group_similarity - negative_similarity

    features = []

    for row in rows:
        start_s = float(row["t_start_s"])
        end_s = float(row["t_end_s"])
        indices = np.flatnonzero((times >= start_s) & (times < end_s))

        if not len(indices):
            midpoint_s = start_s + ((end_s - start_s) / 2.0)
            indices = np.asarray([int(np.argmin(np.abs(times - midpoint_s)))])

        result = {
            "semantic_keyframe_count": int(len(indices)),
            "semantic_negative_score": round(
                top_k_mean(negative_similarity[indices], top_k),
                8,
            ),
        }
        group_scores = []

        for group in SEMANTIC_GROUPS:
            value = top_k_mean(group_margins[group][indices], top_k)
            result["semantic_" + group + "_score"] = round(value, 8)
            group_scores.append(value)

        result["semantic_score"] = round(max(group_scores), 8)
        features.append(result)

    return features


def apply_semantic_ranking(
    rows,
    keyframe_times_s,
    image_embeddings,
    text_embeddings,
    model_id,
    prompt_hash,
    top_k=2,
):
    """Attach semantic features and rank them with packet energy as tie break."""
    features = semantic_window_features(
        rows,
        keyframe_times_s,
        image_embeddings,
        text_embeddings,
        top_k=top_k,
    )

    for row, feature in zip(rows, features):
        row["packet_candidate_score"] = float(row.get("candidate_score", 0.0))
        row["packet_candidate_percentile"] = float(
            row.get("candidate_percentile", 0.0)
        )
        row["packet_candidate_rank"] = int(row.get("candidate_rank", 0))
        row.update(feature)
        row["semantic_model"] = str(model_id)
        row["semantic_prompt_hash"] = str(prompt_hash)
        row["candidate_score"] = float(feature["semantic_score"])
        row["selection_source"] = "siglip2_keyframe_semantics"

    ranked_indices = sorted(
        range(len(rows)),
        key=lambda index: (
            -float(rows[index]["semantic_score"]),
            -float(rows[index]["packet_candidate_score"]),
            float(rows[index]["t_start_s"]),
        ),
    )
    ranks = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
    denominator = max(1, len(rows) - 1)

    for index, row in enumerate(rows):
        rank = ranks[index]
        row["candidate_rank"] = rank
        row["candidate_percentile"] = round(
            1.0 - ((rank - 1) / denominator),
            8,
        )

    return rows


def cache_path_for(cache_dir, video_id):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id)).strip("._")

    if not safe_id:
        safe_id = "video"

    return os.path.join(os.path.abspath(cache_dir), safe_id + ".npz")


def interval_signature(intervals):
    normalised = normalise_intervals(intervals)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_embedding_cache(cache_path, source_video, model_id, intervals=None):
    if not os.path.isfile(cache_path):
        return None

    source_stat = os.stat(source_video)

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            if int(cached["cache_version"].item()) != CACHE_VERSION:
                return None
            if str(cached["model_id"].item()) != str(model_id):
                return None
            if int(cached["source_size_bytes"].item()) != int(source_stat.st_size):
                return None
            if int(cached["source_mtime_ns"].item()) != int(source_stat.st_mtime_ns):
                return None
            if str(cached["interval_signature"].item()) != interval_signature(
                intervals
            ):
                return None

            times = np.asarray(cached["keyframe_times_s"], dtype=np.float64)
            embeddings = np.asarray(cached["image_embeddings"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        return None

    if embeddings.ndim != 2 or len(times) != len(embeddings) or not len(times):
        return None

    return times, embeddings


def write_embedding_cache(
    cache_path,
    source_video,
    model_id,
    keyframe_times_s,
    image_embeddings,
    intervals=None,
):
    source_stat = os.stat(source_video)
    parent = os.path.dirname(cache_path)
    os.makedirs(parent, exist_ok=True)
    temporary_path = cache_path + ".tmp.npz"

    try:
        np.savez_compressed(
            temporary_path,
            cache_version=np.asarray(CACHE_VERSION, dtype=np.int64),
            model_id=np.asarray(str(model_id)),
            source_size_bytes=np.asarray(source_stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
            interval_signature=np.asarray(interval_signature(intervals)),
            keyframe_times_s=np.asarray(keyframe_times_s, dtype=np.float64),
            image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        )
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.isfile(temporary_path):
            os.remove(temporary_path)

    return cache_path


class SiglipKeyframeEncoder(object):
    """Lazy SigLIP2 encoder shared across every video in one indexing run."""

    def __init__(self, model_id=None, local_files_only=None, batch_size=32):
        self.model_id = str(model_id or CANDIDATE_SEMANTIC_MODEL)
        self.local_files_only = (
            HF_LOCAL_FILES_ONLY
            if local_files_only is None
            else bool(local_files_only)
        )
        self.batch_size = max(1, int(batch_size))
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )
        model = AutoModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            local_files_only=self.local_files_only,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device

    def _device_batch(self, inputs):
        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def encode_texts(self, prompts):
        self._ensure_loaded()
        inputs = self._processor(
            text=list(prompts),
            padding="max_length",
            return_tensors="pt",
        )

        with self._torch.inference_mode():
            output = self._model.get_text_features(**self._device_batch(inputs))

        features = pooled_feature_tensor(output)

        return normalise_embeddings(features.detach().float().cpu().numpy())

    def encode_images(self, images):
        self._ensure_loaded()
        inputs = self._processor(images=list(images), return_tensors="pt")

        with self._torch.inference_mode():
            output = self._model.get_image_features(**self._device_batch(inputs))

        features = pooled_feature_tensor(output)

        return normalise_embeddings(features.detach().float().cpu().numpy())

    def encode_video_keyframes(self, source_video, intervals=None):
        import av

        times = []
        batches = []
        pending_images = []
        pending_times = []

        allowed_intervals = normalise_intervals(intervals)

        if intervals is not None and not allowed_intervals:
            raise ValueError("No valid mapping intervals for %s" % source_video)

        with av.open(source_video) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            decode_intervals = allowed_intervals or [None]

            for interval in decode_intervals:
                if interval is not None:
                    offset = int(interval["start_s"] / float(stream.time_base))
                    container.seek(
                        offset,
                        backward=True,
                        any_frame=False,
                        stream=stream,
                    )

                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue

                    timestamp_s = float(frame.pts * stream.time_base)

                    if not math.isfinite(timestamp_s) or timestamp_s < 0:
                        continue
                    if interval is not None and timestamp_s < interval["start_s"]:
                        continue
                    if interval is not None and timestamp_s >= interval["end_s"]:
                        break

                    pending_times.append(timestamp_s)
                    pending_images.append(frame.to_image())

                    if len(pending_images) >= self.batch_size:
                        batches.append(self.encode_images(pending_images))
                        times.extend(pending_times)
                        pending_images = []
                        pending_times = []

        if pending_images:
            batches.append(self.encode_images(pending_images))
            times.extend(pending_times)

        if not batches:
            raise ValueError("No decodable keyframes in %s" % source_video)

        return np.asarray(times, dtype=np.float64), np.concatenate(batches, axis=0)


class SemanticCandidateScorer(object):
    """Own prompt embeddings and reuse one encoder throughout an index build."""

    def __init__(
        self,
        prompt_path,
        cache_dir,
        model_id=None,
        local_files_only=None,
        batch_size=32,
        top_k=2,
    ):
        self.prompt_path = os.path.abspath(prompt_path)
        self.cache_dir = os.path.abspath(cache_dir)
        self.prompt_config = load_prompt_config(self.prompt_path)
        self.prompt_hash = prompt_config_hash(self.prompt_config)
        self.top_k = max(1, int(top_k))
        self.encoder = SiglipKeyframeEncoder(
            model_id=model_id,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )
        self._text_embeddings = None
        self.last_cache_hit = False

    @property
    def model_id(self):
        return self.encoder.model_id

    def text_embeddings(self):
        if self._text_embeddings is not None:
            return self._text_embeddings

        embeddings = {}

        for group, prompts in self.prompt_config["positive_groups"].items():
            embeddings[group] = self.encoder.encode_texts(prompts)

        embeddings["negative"] = self.encoder.encode_texts(
            self.prompt_config["negative_prompts"]
        )
        self._text_embeddings = embeddings
        return embeddings

    def video_embeddings(self, source_video, video_id, intervals=None):
        path = cache_path_for(self.cache_dir, video_id)
        cached = load_embedding_cache(
            path,
            source_video,
            self.model_id,
            intervals=intervals,
        )

        if cached is not None:
            self.last_cache_hit = True
            return cached

        self.last_cache_hit = False
        times, embeddings = self.encoder.encode_video_keyframes(
            source_video,
            intervals=intervals,
        )
        write_embedding_cache(
            path,
            source_video,
            self.model_id,
            times,
            embeddings,
            intervals=intervals,
        )
        return times, embeddings

    def score_rows(self, rows, source_video, video_id, intervals=None):
        times, images = self.video_embeddings(
            source_video,
            video_id,
            intervals=intervals,
        )
        return apply_semantic_ranking(
            rows,
            times,
            images,
            self.text_embeddings(),
            model_id=self.model_id,
            prompt_hash=self.prompt_hash,
            top_k=self.top_k,
        )
