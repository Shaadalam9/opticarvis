r"""Stage 2: Gemma4 explanation timing gate.

Correct workflow:
Alpamayo output plus selected scene frames are fed to Gemma4.
Gemma4 decides whether this is a proper time to explain.
Only if Gemma4 says yes do segmentation, depth, MIRAGE and rendering run.

Current version is a dry run placeholder with the final JSON contract.
"""

import json
import os

import cv2

from pipeline_common import (
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    CLIP_VIDEO,
    STATE_JSON,
    GEMMA4_MODEL,
    HF_LOCAL_FILES_ONLY,
    read_json,
    write_json,
    ensure_dir,
    segment_tag,
    workflow_path,
)

OUTPUT_DIR = workflow_path("gemma_reasoning")
KEY_FRAME_DIR = workflow_path("gemma_reasoning", segment_tag() + "_key_frames")
GEMMA_GATE_JSON = workflow_path("gemma_reasoning", segment_tag() + "_gemma_gate.json")
GEMMA_PROMPT_JSON = workflow_path("gemma_reasoning", segment_tag() + "_gemma_prompt.json")
GEMMA_COMPAT_JSON = workflow_path("gemma_reasoning", segment_tag() + "_gemma_reasoning.json")

KEY_FRAME_POSITIONS = [0.25, 0.50, 0.75]
GEMMA_MODE = "dry_run_placeholder"
GEMMA_MODEL_ID = "Gemma4_gate_placeholder"

# Real Gemma 4 multimodal gate. Falls back to the heuristic dry_run_gate if the
# model is unavailable or its output cannot be parsed. Override the checkpoint
# with OPTICARVIS_GEMMA4_MODEL (e.g. google/gemma-4-E2B-it for a faster gate).
USE_REAL_GEMMA = True
GEMMA4_MODEL_ID = GEMMA4_MODEL

# Set for a batch run: turns the silent downgrade to the heuristic into an error.
# Left off by default so a machine without the gate model can still be exercised.
REQUIRE_GEMMA_GATE = os.environ.get("OPTICARVIS_REQUIRE_GEMMA_GATE", "0") == "1"
GEMMA4_MAX_NEW_TOKENS = 320


def make_dirs():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(KEY_FRAME_DIR)


def extract_key_frames():
    if not os.path.isfile(CLIP_VIDEO):
        print("Missing clip video:", CLIP_VIDEO)
        raise SystemExit(1)

    cap = cv2.VideoCapture(CLIP_VIDEO)
    if not cap.isOpened():
        print("Could not open clip video:", CLIP_VIDEO)
        raise SystemExit(1)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        print("Clip has no frames:", CLIP_VIDEO)
        raise SystemExit(1)

    frames = []
    for position in KEY_FRAME_POSITIONS:
        frame_index = int((frame_count - 1) * position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame:", frame_index)
            raise SystemExit(1)

        frame_file = os.path.join(KEY_FRAME_DIR, "frame_" + str(frame_index).zfill(6) + ".jpg")
        cv2.imwrite(frame_file, frame)
        frames.append({
            "frame_index": frame_index,
            "frame_time_s": round(frame_index / fps, 3),
            "frame_file": frame_file,
        })

    cap.release()
    return {"clip_video": CLIP_VIDEO, "frame_count": frame_count, "fps": fps, "key_frames": frames}


def display_target_for(scene_cause, action):
    if scene_cause == "pedestrian_crosswalk_interaction":
        return "pedestrians_and_crosswalk"
    if scene_cause == "traffic_light_interaction":
        return "traffic_light_and_stop_line"
    if scene_cause == "vehicle_or_traffic_interaction":
        return "relevant_vehicle_or_traffic_region"
    if scene_cause == "construction_or_clearance_interaction":
        return "construction_or_clearance_region"
    if action == "turn":
        return "planned_turn_region"
    return "ego_future_path"


def passenger_text_for(display_target, action):
    if display_target == "pedestrians_and_crosswalk":
        return "Yielding for pedestrians in the crosswalk"
    if display_target == "traffic_light_and_stop_line":
        return "Stopping for the traffic light"
    if display_target == "relevant_vehicle_or_traffic_region":
        return "Adjusting for nearby traffic"
    if display_target == "construction_or_clearance_region":
        return "Adjusting path for clearance"
    if action == "turn":
        return "Preparing to turn"
    return "Vehicle behaviour adjusted for the scene"


GEMMA_SYSTEM_PROMPT = (
    "You are the explanation timing gate for OptiCarVis, an autonomous vehicle "
    "passenger interface. You receive Alpamayo planner context and selected scene "
    "frames. Your task is to decide whether THIS moment is a proper time to "
    "trigger a brief passenger facing visual explanation.\\n\\n"
    "DEFAULT DECISION: do_not_explain. Usually no explanation is required. The "
    "vehicle should not explain standard driving behaviour just because Alpamayo "
    "detects an action, a traffic interaction, or a nearby road user.\\n\\n"
    "Say proper_time_to_explain=true only when ALL of these are true:\\n"
    "1. The vehicle behaviour may be noticeable or potentially confusing to the "
    "passenger.\\n"
    "2. The reason for the behaviour is not self evident from normal traffic "
    "rules or obvious visible motion.\\n"
    "3. A visual explanation would clarify a specific hidden, ambiguous, distant, "
    "or safety relevant cause, object, area, or risk in the scene.\\n"
    "4. The explanation would help the passenger understand why the vehicle is "
    "behaving this way now.\\n\\n"
    "Valid reasons for explaining are limited to:\\n"
    "- a hidden or partially occluded cause, such as a pedestrian, cyclist, "
    "vehicle, queue, obstacle, or hazard that is not immediately obvious;\\n"
    "- ambiguous intent of another road user, for example someone may enter the "
    "ego path but has not clearly done so yet;\\n"
    "- a non local cause, such as a downstream queue, blocked lane, roadworks, or "
    "hazard farther ahead;\\n"
    "- unusual safety margin behaviour, for example slowing early, stopping "
    "early, hesitating, or keeping extra distance because of uncertainty;\\n"
    "- behaviour that may feel surprising from inside the vehicle, such as not "
    "proceeding when the path appears open.\\n\\n"
    "Say do_not_explain for standard or self evident driving behaviour, "
    "including:\\n"
    "- ordinary car following;\\n"
    "- slowing because the lead vehicle is slow;\\n"
    "- normal merging or cut in interactions when the cause is visible;\\n"
    "- normal traffic light behaviour;\\n"
    "- normal lane keeping or lane adjustment;\\n"
    "- normal yielding when the reason is obvious;\\n"
    "- normal stopping behind traffic;\\n"
    "- explanations that would merely restate what the passenger can already "
    "see.\\n\\n"
    "Important constraints:\\n"
    "- Do not explain just because the action is stop, slow, yield, nudge, turn, "
    "or continue.\\n"
    "- Do not explain just because there is a pedestrian, cyclist, traffic light, "
    "merging vehicle, cut in vehicle, or lead vehicle.\\n"
    "- Only explain when visual grounding adds useful information beyond the "
    "obvious scene interpretation.\\n"
    "- If uncertain, choose do_not_explain.\\n\\n"
    "Respond with ONLY a single minified JSON object, no prose and no markdown, "
    "with exactly these keys:\\n"
    '{"proper_time_to_explain": true|false, '
    '"decision": "explain_now"|"do_not_explain", '
    '"decision_reason": "<short technical reason>", '
    '"passenger_facing_text": "<short on-screen text if explaining, else empty>", '
    '"display_target": "pedestrians_and_crosswalk|traffic_light_and_stop_line|'
    "relevant_vehicle_or_traffic_region|construction_or_clearance_region|"
    'planned_turn_region|ego_future_path|none", '
    '"confidence": <number between 0 and 1>}'
)

VALID_DISPLAY_TARGETS = {
    "pedestrians_and_crosswalk",
    "traffic_light_and_stop_line",
    "relevant_vehicle_or_traffic_region",
    "construction_or_clearance_region",
    "planned_turn_region",
    "ego_future_path",
    "none",
}

def load_gemma4():
    """Load the Gemma 4 multimodal model + processor, cached process-wide.

    The cache lives in gemma_model_cache, not here: gemma_gate_batch.py reloads
    this module per job (its paths and job identity are fixed at import time),
    and a module-level cache would be dropped on every reload.
    """
    from gemma_model_cache import get_gemma4

    return get_gemma4(GEMMA4_MODEL_ID, HF_LOCAL_FILES_ONLY)


def build_gemma_messages(state, frames):
    """System + multimodal user message: Alpamayo context text + key frames."""
    from PIL import Image

    context = state.get("alpamayo_context", {})
    alpamayo = {
        "alpamayo_action": context.get("alpamayo_action"),
        "alpamayo_reasoning_trace": context.get("alpamayo_reasoning_trace"),
        "scene_cause": context.get("scene_cause"),
        "uncertainty_score": context.get("uncertainty_score"),
        "trajectory_metrics": context.get("trajectory_metrics"),
    }

    user_content = [
        {
            "type": "text",
            "text": "Alpamayo planner analysis of the clip:\n" + json.dumps(alpamayo, indent=2),
        }
    ]

    key_frames = frames.get("key_frames", [])
    if key_frames:
        user_content.append({"type": "text", "text": "Key frames sampled across the clip:"})
        for key_frame in key_frames:
            path = key_frame.get("frame_file")
            if path and os.path.isfile(path):
                user_content.append({"type": "image", "image": Image.open(path).convert("RGB")})

    user_content.append({"type": "text", "text": "Now output the JSON decision object only."})

    return [
        {"role": "system", "content": [{"type": "text", "text": GEMMA_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]


def parse_gate_json(text):
    """Return the last complete top-level JSON object in text, or None.

    Robust to any 'thinking' preamble the model may emit before the answer.
    """
    objects = []
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:index + 1])
                    start = None
    for chunk in reversed(objects):
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def gate_from_model_output(parsed, state, raw_text):
    """Map the model's JSON to the gate contract used by the rest of the pipeline."""
    context = state.get("alpamayo_context", {})
    action = context.get("alpamayo_action", "continue")
    scene_cause = context.get("scene_cause", "unclassified_driving_context")

    proper = bool(parsed.get("proper_time_to_explain", False))

    display_target = parsed.get("display_target") or "none"
    if display_target not in VALID_DISPLAY_TARGETS:
        display_target = "none"
    if proper and display_target == "none":
        display_target = display_target_for(scene_cause, action)
    if not proper:
        display_target = "none"

    text = parsed.get("passenger_facing_text") or ""
    if proper and not text:
        text = passenger_text_for(display_target, action)
    if not proper:
        text = ""

    try:
        confidence = float(parsed.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))

    decision = parsed.get("decision") or ("explain_now" if proper else "do_not_explain")
    reason = parsed.get("decision_reason") or ""

    return {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "gemma4_explanation_timing_gate",
        "mode": "gemma4_live",
        "model_id": GEMMA4_MODEL_ID,
        "model_called": True,
        "proper_time_to_explain": proper,
        "decision": decision,
        "decision_reason": reason,
        "passenger_facing_text": text,
        "display_target": display_target,
        "confidence": round(confidence, 3),
        "alpamayo_context_used": {
            "action": action,
            "reasoning_trace": context.get("alpamayo_reasoning_trace"),
            "scene_cause": scene_cause,
            "uncertainty_score": context.get("uncertainty_score"),
            "trajectory_metrics": context.get("trajectory_metrics"),
        },
        "raw_model_output": raw_text.strip(),
        "important_note": "Live Gemma 4 multimodal gate over the Alpamayo context and key frames.",
    }



def flatten_gemma_messages_for_processor(messages):
    """Convert role/content messages to plain text plus image list for processors without chat templates."""
    text_blocks = []
    images = []

    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = message.get("content", [])
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type == "text":
                value = str(item.get("text", "")).strip()
                if value:
                    parts.append(value)

            elif item_type == "image":
                image = item.get("image")
                if image is not None:
                    images.append(image)
                    parts.append("<|image|>")

        if parts:
            text_blocks.append(role + ":\n" + "\n".join(parts))

    text_blocks.append(
        "ASSISTANT:\n"
        "Return exactly one minified JSON object. "
        "Do not include markdown, explanation text, or code fences."
    )

    return "\n\n".join(text_blocks), images


def run_gemma4_gate(state, frames):
    """Run the real Gemma 4 gate; return the gate dict, or None to fall back."""
    if not USE_REAL_GEMMA:
        return None
    try:
        import torch

        processor, model = load_gemma4()
        messages = build_gemma_messages(state, frames)
        if getattr(processor, "chat_template", None):
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            prompt_text, prompt_images = flatten_gemma_messages_for_processor(messages)
            processor_kwargs = {
                "text": prompt_text,
                "return_tensors": "pt",
            }
            if prompt_images:
                processor_kwargs["images"] = prompt_images

            inputs = processor(**processor_kwargs).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=GEMMA4_MAX_NEW_TOKENS, do_sample=False
            )
        text = processor.decode(output[0][input_len:], skip_special_tokens=True)
        parsed = parse_gate_json(text)
        if parsed is None:
            print("Gemma 4 output was not parseable JSON; using heuristic gate.")
            print("Raw output:", text[:300])
            return None
        return gate_from_model_output(parsed, state, text)
    except Exception as error:
        print("Gemma 4 gate unavailable (%s); using heuristic gate." % error)

        # The fallback is a reasonable default for a smoke run and a bad one for
        # a batch: the gate is the thing under study, and a missing compile
        # toolchain is enough to replace it with a heuristic for every city
        # without failing anything.
        if REQUIRE_GEMMA_GATE:
            raise RuntimeError(
                "Gemma 4 gate could not run and OPTICARVIS_REQUIRE_GEMMA_GATE is "
                "set: %s" % error
            )

        return None


def build_prompt(state, frame_payload):
    context = state.get("alpamayo_context", {})
    prompt = {
        "task": "Decide whether this is a proper time for an AV to explain its behaviour to a passenger.",
        "instruction": "Default to do_not_explain. Explain only when the vehicle behaviour is noticeable or potentially confusing and visual grounding would clarify a hidden, ambiguous, non local, or unusual safety margin reason beyond what the passenger can already see.",
        "alpamayo_context": {
            "action": context.get("alpamayo_action"),
            "reasoning_trace": context.get("alpamayo_reasoning_trace"),
            "scene_cause": context.get("scene_cause"),
            "uncertainty_score": context.get("uncertainty_score"),
            "trajectory_metrics": context.get("trajectory_metrics"),
        },
        "selected_scene_frames": frame_payload.get("key_frames", []),
        "required_output_schema": {
            "proper_time_to_explain": "boolean",
            "decision": "explain_now or do_not_explain",
            "decision_reason": "short technical reason",
            "passenger_facing_text": "short display text if explaining",
            "display_target": "visual target if explaining",
            "confidence": "0 to 1",
        },
    }
    return prompt


def dry_run_gate(state, prompt):
    context = state.get("alpamayo_context", {})
    action = context.get("alpamayo_action", "continue")
    scene_cause = context.get("scene_cause", "unclassified_driving_context")
    uncertainty = float(context.get("uncertainty_score", 0.0))
    reasoning = str(context.get("alpamayo_reasoning_trace") or "").lower()
    candidate = bool(context.get("candidate_context", False))

    hidden_terms = [
        "occluded",
        "hidden",
        "blocked view",
        "behind",
        "emerging",
        "sudden",
        "not visible",
        "partially visible",
    ]
    ambiguity_terms = [
        "ambiguous",
        "uncertain",
        "may enter",
        "might enter",
        "hesitate",
        "hesitation",
        "unclear",
        "unpredictable",
    ]
    non_local_terms = [
        "downstream",
        "queue ahead",
        "blocked lane",
        "roadworks",
        "construction",
        "hazard ahead",
        "farther ahead",
    ]
    unusual_margin_terms = [
        "early braking",
        "stopping early",
        "extra distance",
        "large margin",
        "safety margin",
        "path appears open",
    ]

    has_hidden_cause = any(term in reasoning for term in hidden_terms)
    has_ambiguity = any(term in reasoning for term in ambiguity_terms)
    has_non_local_cause = any(term in reasoning for term in non_local_terms)
    has_unusual_margin = any(term in reasoning for term in unusual_margin_terms)

    noticeable_action = action in ["stop", "slow", "slow_or_yield", "yield", "nudge", "turn"]

    proper = False
    reason = "No explanation needed because the behaviour appears standard or self evident."
    confidence = 0.72

    if candidate and noticeable_action and (
        has_hidden_cause
        or has_ambiguity
        or has_non_local_cause
        or has_unusual_margin
        or uncertainty >= 0.82
    ):
        proper = True
        confidence = 0.74
        if has_hidden_cause:
            reason = "Explanation may help because the relevant cause appears hidden or partially occluded."
            confidence = 0.78
        elif has_ambiguity:
            reason = "Explanation may help because another road user's intent is ambiguous."
            confidence = 0.75
        elif has_non_local_cause:
            reason = "Explanation may help because the cause is downstream or not local to the immediate scene."
            confidence = 0.75
        elif has_unusual_margin:
            reason = "Explanation may help because the vehicle is using an unusual safety margin."
            confidence = 0.74
        else:
            reason = "Explanation may help because the planner reports high uncertainty in a noticeable action."
            confidence = 0.72

    if proper:
        target = display_target_for(scene_cause, action)
        text = passenger_text_for(target, action)
        decision = "explain_now"
    else:
        target = "none"
        text = ""
        decision = "do_not_explain"

    return {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "gemma4_explanation_timing_gate",
        "mode": GEMMA_MODE,
        "model_id": GEMMA_MODEL_ID,
        "model_called": False,
        "proper_time_to_explain": proper,
        "decision": decision,
        "decision_reason": reason,
        "passenger_facing_text": text,
        "display_target": target,
        "confidence": confidence,
        "alpamayo_context_used": prompt.get("alpamayo_context", {}),
        "selected_scene_frames": prompt.get("selected_scene_frames", []),
        "important_note": "Dry run fallback following the same conservative gate policy expected from real Gemma4.",
    }


def display_plan_from_gate(gate):
    if not gate.get("proper_time_to_explain", False):
        return {"display_target": "none", "display_mode": "none", "display_intensity": "none", "label_text": ""}

    if float(gate.get("confidence", 0.0)) >= 0.75:
        intensity = "high"
        mode = "mirage_augmented_highlight_plus_short_label"
    else:
        intensity = "medium"
        mode = "mirage_subtle_highlight_plus_short_label"

    return {
        "display_target": gate.get("display_target"),
        "display_mode": mode,
        "display_intensity": intensity,
        "label_text": gate.get("passenger_facing_text"),
    }


def write_compat_gemma_json(gate):
    payload = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "status": gate.get("mode"),
        "model_called": gate.get("model_called"),
        "model_id": gate.get("model_id"),
        "proper_time_to_explain": gate.get("proper_time_to_explain"),
        "semantic_reason": gate.get("decision_reason"),
        "causal_objects": ["pedestrians"] if gate.get("display_target") == "pedestrians_and_crosswalk" else [],
        "causal_regions": ["crosswalk", "ego future path"] if gate.get("display_target") == "pedestrians_and_crosswalk" else [],
        "recommended_display_target": gate.get("display_target"),
        "recommended_display_text": gate.get("passenger_facing_text"),
        "confidence": gate.get("confidence"),
    }
    write_json(GEMMA_COMPAT_JSON, payload)


def update_state(state, gate):
    proper = bool(gate.get("proper_time_to_explain", False))
    context = state.get("alpamayo_context", {})
    display_plan = display_plan_from_gate(gate)

    state["outputs"]["gemma_gate_json"] = GEMMA_GATE_JSON
    state["outputs"]["gemma_prompt_json"] = GEMMA_PROMPT_JSON
    state["outputs"]["gemma_reasoning_json"] = GEMMA_COMPAT_JSON

    state["gemma_gate"] = {
        "required": context.get("gemma_gate_required", False),
        "status": "complete",
        "proper_time_to_explain": proper,
        "model_called": gate.get("model_called"),
        "mode": gate.get("mode"),
        "confidence": gate.get("confidence"),
        "decision_reason": gate.get("decision_reason"),
    }

    # Derived, never hardcoded. call_gemma4_gate() falls back to dry_run_gate()
    # whenever the model cannot be loaded or its output will not parse, and that
    # happens quietly -- a missing CUDA compile toolchain is enough. Claiming
    # gemma4_gate regardless made every state file assert a decision the model
    # never made, which is precisely the field an analysis would trust.
    model_called = bool(gate.get("model_called"))

    state["explanation"] = {
        "needed": proper,
        "status": "explain_now" if proper else "do_not_explain",
        "decided_by": "gemma4_gate" if model_called else "heuristic_gate",
        "gate_mode": gate.get("mode"),
        "model_called": model_called,
        "decision_reason": gate.get("decision_reason"),
        "passenger_facing_text": gate.get("passenger_facing_text"),
    }

    state["decision"] = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "gemma4_gate_decision",
        "alpamayo_action": context.get("alpamayo_action"),
        "scene_cause": context.get("scene_cause"),
        "alpamayo_reasoning_trace": context.get("alpamayo_reasoning_trace"),
        "uncertainty_score": context.get("uncertainty_score"),
        "proper_time_to_explain": proper,
        "explanation_needed": proper,
        "decision_reason": gate.get("decision_reason"),
        "display_plan": display_plan,
    }

    if proper:
        state["current_stage"] = "gemma_gate_yes"
        state["next_modules"]["semantic_segmentation"] = {"needed": True, "status": "pending"}
        state["next_modules"]["depth_estimation"] = {"needed": True, "status": "pending"}
        state["next_modules"]["mirage"] = {"needed": True, "status": "pending"}
        state["next_modules"]["final_render"] = {"needed": True, "status": "pending"}
    else:
        state["current_stage"] = "gemma_gate_no"
        state["next_modules"]["semantic_segmentation"] = {"needed": False, "status": "skipped_gemma_said_no"}
        state["next_modules"]["depth_estimation"] = {"needed": False, "status": "skipped_gemma_said_no"}
        state["next_modules"]["mirage"] = {"needed": False, "status": "skipped_gemma_said_no"}
        state["next_modules"]["final_render"] = {"needed": False, "status": "skipped_gemma_said_no"}

    write_json(STATE_JSON, state)


def main():
    make_dirs()
    state = read_json(STATE_JSON)
    frames = extract_key_frames()
    prompt = build_prompt(state, frames)
    write_json(GEMMA_PROMPT_JSON, prompt)

    gate = run_gemma4_gate(state, frames)
    if gate is None:
        gate = dry_run_gate(state, prompt)
    write_json(GEMMA_GATE_JSON, gate)
    write_compat_gemma_json(gate)
    update_state(state, gate)

    print("\nStage 2: Gemma4 explanation timing gate")
    print("=======================================")
    print("mode:", gate["mode"])
    print("model_called:", gate["model_called"])
    print("proper_time_to_explain:", gate["proper_time_to_explain"])
    print("decision:", gate["decision"])
    print("decision_reason:", gate["decision_reason"])
    print("display_target:", gate["display_target"])
    print("passenger_facing_text:", gate["passenger_facing_text"])
    print("confidence:", gate["confidence"])
    print("\nGate JSON:", GEMMA_GATE_JSON)
    print("Prompt JSON:", GEMMA_PROMPT_JSON)
    print("Updated state:", STATE_JSON)


if __name__ == "__main__":
    main()
