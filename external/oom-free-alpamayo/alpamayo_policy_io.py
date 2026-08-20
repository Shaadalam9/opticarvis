"""Shared helpers for turning Alpamayo-R1 inference output into policy CSVs.

Used by scripts/infer_crowd_clip.py, scripts/infer_inspect_output.py and
convert_alpamayo_json_to_policy_csv.py, which previously carried
copy-pasted versions of these functions that had already started to drift.
"""

from __future__ import annotations

import csv

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────
# JSON serialization of inference results
# ─────────────────────────────────────────────────────────────────────

def tensor_to_json(value: torch.Tensor) -> dict:
    value_cpu = value.detach().float().cpu()
    return {
        "type": "torch.Tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "values": value_cpu.tolist(),
    }


def object_to_json(value, depth: int = 0, max_depth: int = 5):
    if depth > max_depth:
        return {
            "type": str(type(value)),
            "value": str(value)[:1000],
        }

    if isinstance(value, torch.Tensor):
        return tensor_to_json(value)

    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
            "values": {
                str(key): object_to_json(item, depth + 1, max_depth)
                for key, item in value.items()
            },
        }

    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "length": len(value),
            "items": [object_to_json(item, depth + 1, max_depth) for item in value],
        }

    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "items": [object_to_json(item, depth + 1, max_depth) for item in value],
        }

    if isinstance(value, (str, bool, int, float)):
        return {
            "type": type(value).__name__,
            "value": value,
        }

    if value is None:
        return {
            "type": "NoneType",
            "value": None,
        }

    return {
        "type": str(type(value)),
        "value": str(value)[:5000],
    }


def add_object_summary(lines: list, value, name: str, depth: int = 0, max_depth: int = 4) -> None:
    indent = "  " * depth
    lines.append(indent + name + "_type: " + str(type(value)))

    if isinstance(value, torch.Tensor):
        lines.append(indent + name + "_shape: " + str(tuple(value.shape)))
        lines.append(indent + name + "_dtype: " + str(value.dtype))
        lines.append(indent + name + "_device: " + str(value.device))

        # Slice before moving to CPU so a large tensor is not copied whole.
        preview = value.detach().flatten()[:10].float().cpu().tolist()
        lines.append(indent + name + "_preview: " + str(preview))
        return

    if isinstance(value, np.ndarray):
        lines.append(indent + name + "_shape: " + str(value.shape))
        lines.append(indent + name + "_dtype: " + str(value.dtype))
        lines.append(indent + name + "_value: " + str(value)[:1000])
        return

    if isinstance(value, dict):
        lines.append(indent + name + "_keys: " + str(list(value.keys())))
        if depth >= max_depth:
            return
        for key, item in value.items():
            add_object_summary(lines, item, str(key), depth + 1, max_depth)
        return

    if isinstance(value, tuple):
        lines.append(indent + name + "_length: " + str(len(value)))
        if depth >= max_depth:
            return
        for index, item in enumerate(value):
            add_object_summary(lines, item, name + "_" + str(index), depth + 1, max_depth)
        return

    if isinstance(value, list):
        lines.append(indent + name + "_length: " + str(len(value)))
        if depth >= max_depth:
            return
        preview_count = min(10, len(value))
        for index in range(preview_count):
            add_object_summary(lines, value[index], name + "_" + str(index), depth + 1, max_depth)
        return

    lines.append(indent + name + "_value: " + str(value)[:1000])


# ─────────────────────────────────────────────────────────────────────
# Extraction of text fields from the saved JSON payload
# ─────────────────────────────────────────────────────────────────────

def clean_text(value) -> str:
    """Strip repr-style brackets/quotes from single-element numpy string reprs (legacy JSONs)."""
    if value is None:
        return ""

    text = str(value).strip()

    while text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    while len(text) >= 2 and text[0] in ["'", '"'] and text[-1] == text[0]:
        text = text[1:-1].strip()

    return text


def _flatten_strings(value) -> list:
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            flat.extend(_flatten_strings(item))
        return flat
    text = str(value).strip()
    return [text] if text else []


def get_nested_text_from_json_field(field) -> str:
    if isinstance(field, dict):
        for key in ("value", "values", "items"):
            if key in field:
                inner = field[key]
                # ndarray/list payloads store real nested lists: join their
                # elements directly instead of reverse-parsing a repr string.
                if isinstance(inner, (list, tuple)):
                    return " | ".join(_flatten_strings(inner))
                return clean_text(inner)
        return ""

    return clean_text(field)


def get_extra_text(json_payload: dict, key: str) -> str:
    extra = json_payload.get("extra", {})
    values = extra.get("values", {})
    field = values.get(key, "")
    return get_nested_text_from_json_field(field)


# ─────────────────────────────────────────────────────────────────────
# Keyword heuristics for the offline WHEN policy CSV
# ─────────────────────────────────────────────────────────────────────

def infer_meta_action(reasoning_trace: str) -> str:
    text = reasoning_trace.lower()

    if "nudge" in text:
        return "nudge"
    if "slow" in text or "yield" in text or "brake" in text:
        return "slow_or_yield"
    if "stop" in text:
        return "stop"
    if "turn" in text:
        return "turn"
    if "clearance" in text:
        return "nudge"

    return "continue"


def infer_reason(reasoning_trace: str) -> str:
    text = reasoning_trace.lower()

    if "construction" in text or "cone" in text or "clearance" in text:
        return "trajectory adjustment for clearance"
    if "pedestrian" in text or "cross" in text:
        return "ambiguous pedestrian interaction"
    if "vehicle" in text or "traffic" in text:
        return "traffic interaction"
    if "turn" in text or "nudge" in text:
        return "planned manoeuvre explanation"

    return "alpamayo reasoning based manoeuvre"


def infer_scores(reasoning_trace: str):
    text = reasoning_trace.lower()

    confidence_score = 0.70
    uncertainty_score = 0.50
    ambiguity_score = 0.60
    trajectory_conflict_score = 0.50

    if "pedestrian" in text or "cross" in text:
        uncertainty_score = 0.65
        ambiguity_score = 0.75
        trajectory_conflict_score = 0.65

    if "construction" in text or "cone" in text or "clearance" in text:
        uncertainty_score = 0.50
        ambiguity_score = 0.55
        trajectory_conflict_score = 0.70

    if "nudge" in text or "slow" in text or "yield" in text or "brake" in text or "stop" in text:
        trajectory_conflict_score = max(trajectory_conflict_score, 0.65)

    return confidence_score, uncertainty_score, ambiguity_score, trajectory_conflict_score


# ─────────────────────────────────────────────────────────────────────
# Policy CSV writer (schema consumed by opticarvis/policy_demo.py)
# ─────────────────────────────────────────────────────────────────────

POLICY_CSV_COLUMNS = [
    "video_id",
    "segment_start_time_s",
    "frame_count",
    "frame_local",
    "when_start_local_s",
    "when_end_local_s",
    "reasoning_trace",
    "meta_action",
    "confidence_score",
    "uncertainty_score",
    "ambiguity_score",
    "trajectory_conflict_score",
    "explanation_needed",
    "explanation_reason",
    "model_source",
    "answer",
]


def build_policy_row(
    json_payload: dict,
    video_id: str,
    segment_start_time_s: float,
    when_start_local_s: float,
    when_end_local_s: float,
    model_source: str,
) -> dict:
    """Derive one offline WHEN row from a saved inference JSON payload."""
    reasoning_trace = get_extra_text(json_payload, "cot")
    meta_action = get_extra_text(json_payload, "meta_action")
    answer = get_extra_text(json_payload, "answer")

    if not meta_action:
        meta_action = infer_meta_action(reasoning_trace)

    confidence_score, uncertainty_score, ambiguity_score, trajectory_conflict_score = infer_scores(reasoning_trace)

    return {
        "video_id": video_id,
        "segment_start_time_s": segment_start_time_s,
        "frame_count": "",
        "frame_local": "",
        "when_start_local_s": when_start_local_s,
        "when_end_local_s": when_end_local_s,
        "reasoning_trace": reasoning_trace,
        "meta_action": meta_action,
        "confidence_score": confidence_score,
        "uncertainty_score": uncertainty_score,
        "ambiguity_score": ambiguity_score,
        "trajectory_conflict_score": trajectory_conflict_score,
        "explanation_needed": 1,
        "explanation_reason": infer_reason(reasoning_trace),
        "model_source": model_source,
        "answer": answer,
    }


def write_policy_csv(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=POLICY_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
