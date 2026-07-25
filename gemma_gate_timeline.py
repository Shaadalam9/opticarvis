r"""Sliding-window Gemma 4 gate: a time-varying explain/clean decision over a clip.

Runs the multimodal gate on one representative frame per short window across the
whole clip, producing a timeline. The renderer uses this timeline to show the
visualization only while an explanation is warranted and remove it again once
the situation is over.

This is GPU-bound and sequential (a single Gemma 4 instance), so it is slow:
~one VLM call per window. Coarsen STRIDE_S to trade resolution for speed.

Run:
    python gemma_gate_timeline.py <clip.mp4> <timeline_out.json> [stride_s]
"""

import os
import sys

import cv2

from pipeline_common import read_json, write_json
import gemma_reasoning_module as G


import json

WINDOW_S = 6.0
STRIDE_S = 6.0

WINDOW_USER_PROMPT = (
    "Judge ONLY this frame, for the current moment. Given Alpamayo's output "
    "above, decide whether a relevant road user (pedestrian or cyclist) is RIGHT "
    "NOW in the vehicle's own driving path, or stepping into it directly ahead, "
    "such that the vehicle must actively yield at this instant. Return "
    "proper_time_to_explain=true ONLY while such a road user is actually present "
    "in the path in THIS frame. If the road directly ahead is clear of such road "
    "users (an empty road, or people only on the sidewalks and not in the path), "
    "return false. Output the JSON decision object only."
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
        {"role": "system", "content": [{"type": "text", "text": G.GEMMA_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Current Alpamayo planner output:\n" + json.dumps(alpamayo, indent=2)},
            {"type": "text", "text": "Current dashcam frame at this moment:"},
            {"type": "image", "image": Image.fromarray(rgb)},
            {"type": "text", "text": WINDOW_USER_PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=G.GEMMA4_MAX_NEW_TOKENS, do_sample=False)
    text = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return G.parse_gate_json(text)


def _plan_from_stream(stream):
    """Per-window (t_center, t_start, t_end, frame_index, alpamayo_context) from a stream."""
    plan = []
    for w in stream.get("windows", []):
        ctx = {
            "alpamayo_action": w.get("alpamayo_action"),
            "alpamayo_reasoning_trace": w.get("alpamayo_reasoning_trace"),
            "scene_cause": w.get("scene_cause"),
            "uncertainty_score": w.get("uncertainty_score"),
        }
        plan.append((w["t_center"], w["t_start"], w["t_end"], int(w["frame_index"]), ctx))
    return plan


def _plan_static(fps, duration, window_s, stride_s):
    """Per-window plan reusing a single static Alpamayo context (with optional override)."""
    state = G.read_json(G.STATE_JSON)
    ctx = dict(state.get("alpamayo_context", {}))
    override = os.environ.get("OPTICARVIS_GATE_REASONING")
    if override:
        ctx["alpamayo_reasoning_trace"] = override
        ctx["scene_cause"] = "pedestrian_jaywalk_interaction"
        print("Alpamayo reasoning override:", override)
    plan = []
    t = window_s / 2.0
    while t < duration:
        fi = int(round(t * fps))
        plan.append((round(t, 2), round(t - window_s / 2.0, 2), round(t + window_s / 2.0, 2), fi, ctx))
        t += stride_s
    return plan


def run_timeline(clip_video, window_s=WINDOW_S, stride_s=STRIDE_S, stream=None):
    capture = cv2.VideoCapture(clip_video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0

    if stream is not None:
        plan = _plan_from_stream(stream)
        print("Using live Alpamayo stream: %d per-timestep planner outputs." % len(plan))
    else:
        plan = _plan_static(fps, duration, window_s, stride_s)

    processor, model = G.load_gemma4()

    windows = []
    for (t_center, t_start, t_end, frame_index, ctx) in plan:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        parsed = window_decision(processor, model, frame, ctx) or {}
        proper = bool(parsed.get("proper_time_to_explain", False))
        target = parsed.get("display_target", "none")
        if target not in G.VALID_DISPLAY_TARGETS:
            target = "none"

        windows.append({
            "index": len(windows),
            "t_center": t_center,
            "t_start": t_start,
            "t_end": t_end,
            "frame_index": frame_index,
            "planner_action": ctx.get("alpamayo_action"),
            "proper_time_to_explain": proper,
            "decision": parsed.get("decision", "do_not_explain"),
            "display_target": target if proper else "none",
            "passenger_facing_text": parsed.get("passenger_facing_text", "") if proper else "",
            "confidence": parsed.get("confidence", 0.0),
            "decision_reason": parsed.get("decision_reason", ""),
        })
        print("t=%6.1fs planner=%-11s -> %-7s (%s)" % (
            t_center, ctx.get("alpamayo_action"), "EXPLAIN" if proper else "clean",
            target if proper else "none"))

    capture.release()
    return {
        "clip_video": clip_video,
        "fps": fps,
        "frame_count": frame_count,
        "window_s": window_s,
        "stride_s": stride_s,
        "windows": windows,
    }


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else G.CLIP_VIDEO
    out = sys.argv[2] if len(sys.argv) > 2 else "gate_timeline.json"
    stride = float(sys.argv[3]) if len(sys.argv) > 3 else STRIDE_S
    stream = read_json(sys.argv[4]) if len(sys.argv) > 4 else None

    timeline = run_timeline(clip, window_s=max(WINDOW_S, stride), stride_s=stride, stream=stream)
    write_json(out, timeline)

    n = len(timeline["windows"])
    e = sum(w["proper_time_to_explain"] for w in timeline["windows"])
    print("\nwindows: %d | explain: %d | clean: %d" % (n, e, n - e))
    print("saved:", out)


if __name__ == "__main__":
    main()
