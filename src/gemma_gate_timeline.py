r"""Sliding window Gemma 4 gate for a time varying explain or clean decision.

Runs the multimodal gate on one representative frame per short window across the
whole clip, producing a timeline. The renderer uses this timeline to show the
visualisation only while an explanation is warranted and remove it again once
the situation is over.

This is GPU bound and sequential because it uses one Gemma 4 instance. It makes
roughly one VLM call per window. Increase STRIDE_S to trade temporal resolution
for speed.

Run with explicit paths:
    python gemma_gate_timeline.py <clip.mp4> <timeline_out.json> [stride_s] [alpamayo_stream.json]

Run for the current pipeline_common job:
    python gemma_gate_timeline.py
"""

import json
import os
import sys

import cv2

from pipeline_common import (
    CLIP_VIDEO,
    STATE_JSON,
    ensure_dir,
    normalise_path,
    read_json,
    segment_tag,
    workflow_path,
    write_json,
)

import gemma_reasoning_module as G


WINDOW_S = 6.0
STRIDE_S = 6.0

WINDOW_USER_PROMPT = (
    "Judge only this frame and the current planner context for this moment. "
    "Default to do_not_explain. Usually no explanation is required. "
    "Return proper_time_to_explain=true only if the vehicle behaviour at this "
    "moment would be noticeable or potentially confusing to the passenger AND "
    "visual grounding would clarify a hidden, partially occluded, ambiguous, "
    "non local, or unusual safety margin reason. "
    "Return false for standard or self evident behaviour, including ordinary "
    "car following, visible merging or cut in traffic, normal traffic light "
    "behaviour, normal lane keeping or lane adjustment, normal yielding, and "
    "normal stopping behind traffic. "
    "Do not explain just because there is a pedestrian, cyclist, traffic light, "
    "lead vehicle, merging vehicle, or cut in vehicle. "
    "If uncertain, return false. Output the JSON decision object only."
)


def window_decision(processor, model, frame_bgr, alpamayo_context):
    import torch
    from PIL import Image

    alpamayo = {
        "alpamayo_action": alpamayo_context.get("alpamayo_action"),
        "alpamayo_reasoning_trace": alpamayo_context.get("alpamayo_reasoning_trace"),
        "scene_cause": alpamayo_context.get("scene_cause"),
        "uncertainty_score": alpamayo_context.get("uncertainty_score"),
        "trajectory_metrics": alpamayo_context.get("trajectory_metrics"),
    }

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": G.GEMMA_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Current Alpamayo planner output:\n" + json.dumps(alpamayo, indent=2),
                },
                {"type": "text", "text": "Current dashcam frame at this moment:"},
                {"type": "image", "image": Image.fromarray(rgb)},
                {"type": "text", "text": WINDOW_USER_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=G.GEMMA4_MAX_NEW_TOKENS,
            do_sample=False,
        )

    text = processor.decode(output[0][input_len:], skip_special_tokens=True)

    return G.parse_gate_json(text)


def plan_from_stream(stream):
    """Return per window planner context from an Alpamayo stream JSON."""
    plan = []

    for window in stream.get("windows", []):
        context = {
            "alpamayo_action": window.get("alpamayo_action"),
            "alpamayo_reasoning_trace": window.get("alpamayo_reasoning_trace"),
            "scene_cause": window.get("scene_cause"),
            "uncertainty_score": window.get("uncertainty_score"),
            "trajectory_metrics": window.get("trajectory_metrics"),
        }

        plan.append(
            (
                float(window["t_center"]),
                float(window["t_start"]),
                float(window["t_end"]),
                int(window["frame_index"]),
                context,
            )
        )

    return plan


def plan_static(fps, duration, window_s, stride_s):
    """Return per window plan reusing the single current Alpamayo context."""
    state = read_json(STATE_JSON, "workflow state")
    context = dict(state.get("alpamayo_context", {}))

    override = os.environ.get("OPTICARVIS_GATE_REASONING")

    if override:
        context["alpamayo_reasoning_trace"] = override
        context["scene_cause"] = os.environ.get(
            "OPTICARVIS_GATE_SCENE_CAUSE",
            "manual_gate_reasoning_override",
        )
        print("Alpamayo reasoning override:", override)

    plan = []
    t = window_s / 2.0

    while t < duration:
        frame_index = int(round(t * fps))
        plan.append(
            (
                round(t, 2),
                round(t - window_s / 2.0, 2),
                round(t + window_s / 2.0, 2),
                frame_index,
                context,
            )
        )
        t += stride_s

    return plan


def normalise_model_decision(parsed):
    proper = bool(parsed.get("proper_time_to_explain", False))

    target = parsed.get("display_target", "none")
    if target not in G.VALID_DISPLAY_TARGETS:
        target = "none"

    if not proper:
        target = "none"

    decision = parsed.get("decision", "do_not_explain")
    if proper and decision == "do_not_explain":
        decision = "explain_now"
    if not proper:
        decision = "do_not_explain"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "proper_time_to_explain": proper,
        "decision": decision,
        "display_target": target,
        "passenger_facing_text": parsed.get("passenger_facing_text", "") if proper else "",
        "confidence": round(confidence, 3),
        "decision_reason": parsed.get("decision_reason", ""),
    }


def run_timeline(clip_video, window_s=WINDOW_S, stride_s=STRIDE_S, stream=None):
    if not os.path.isfile(clip_video):
        print("Missing clip video:")
        print(clip_video)
        raise SystemExit(1)

    capture = cv2.VideoCapture(clip_video)

    if not capture.isOpened():
        print("Could not open clip video:")
        print(clip_video)
        raise SystemExit(1)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0.0 or frame_count <= 0:
        print("Invalid video metadata:")
        print(clip_video)
        capture.release()
        raise SystemExit(1)

    duration = frame_count / fps

    if stream is not None:
        plan = plan_from_stream(stream)
        print("Using Alpamayo stream: %d per timestep planner outputs." % len(plan))
    else:
        plan = plan_static(fps, duration, window_s, stride_s)

    processor, model = G.load_gemma4()

    windows = []

    for t_center, t_start, t_end, frame_index, context in plan:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()

        if not ok:
            continue

        parsed = window_decision(processor, model, frame, context) or {}
        decision = normalise_model_decision(parsed)

        window = {
            "index": len(windows),
            "t_center": t_center,
            "t_start": t_start,
            "t_end": t_end,
            "frame_index": frame_index,
            "planner_action": context.get("alpamayo_action"),
            "planner_scene_cause": context.get("scene_cause"),
            "proper_time_to_explain": decision["proper_time_to_explain"],
            "decision": decision["decision"],
            "display_target": decision["display_target"],
            "passenger_facing_text": decision["passenger_facing_text"],
            "confidence": decision["confidence"],
            "decision_reason": decision["decision_reason"],
        }

        windows.append(window)

        print(
            "t=%6.1fs planner=%-13s -> %-7s (%s)"
            % (
                t_center,
                str(context.get("alpamayo_action")),
                "EXPLAIN" if decision["proper_time_to_explain"] else "clean",
                decision["display_target"],
            )
        )

    capture.release()

    return {
        "clip_video": clip_video,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": round(duration, 3),
        "window_s": window_s,
        "stride_s": stride_s,
        "used_alpamayo_stream": stream is not None,
        "windows": windows,
    }


def default_timeline_path():
    return workflow_path("gemma_reasoning", segment_tag() + "_gate_timeline.json")


def parse_args(argv):
    if len(argv) == 1:
        return {
            "clip": CLIP_VIDEO,
            "output": default_timeline_path(),
            "stride": STRIDE_S,
            "stream": None,
        }

    if len(argv) in [3, 4, 5]:
        return {
            "clip": normalise_path(argv[1]),
            "output": normalise_path(argv[2]),
            "stride": float(argv[3]) if len(argv) >= 4 else STRIDE_S,
            "stream": normalise_path(argv[4]) if len(argv) == 5 and argv[4] else None,
        }

    print(__doc__)
    raise SystemExit(2)


def main():
    args = parse_args(sys.argv)

    stream = read_json(args["stream"], "Alpamayo stream") if args["stream"] else None

    timeline = run_timeline(
        args["clip"],
        window_s=max(WINDOW_S, args["stride"]),
        stride_s=args["stride"],
        stream=stream,
    )

    ensure_dir(os.path.dirname(args["output"]))
    write_json(args["output"], timeline)

    total = len(timeline["windows"])
    explain = sum(
        bool(window["proper_time_to_explain"])
        for window in timeline["windows"]
    )

    print("")
    print("Gemma gate timeline complete")
    print("============================")
    print("windows:", total)
    print("explain:", explain)
    print("clean:", total - explain)
    print("saved:", args["output"])


if __name__ == "__main__":
    main()