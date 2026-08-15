r"""Drive nvidia/Alpamayo2-Super from the OptiCarVis batch pipeline.

batch_corrected_pipeline.py invokes a planner backend with a fixed CLI:

    <python> <script> --clips <manifest.csv> --output-dir <dir> --config <file>
                      [--model <hf-id>] [extra args...]

Alpamayo2-Super does not expose that CLI, so this file adapts it:

    export OPTICARVIS_ALPAMAYO_PYTHON=/opt/alpamayo2/.venv/bin/python
    export OPTICARVIS_OOM_FREE_ALPAMAYO_REPO=/opt/alpamayo2
    export OPTICARVIS_ALPAMAYO_SCRIPT=<opticarvis>/scripts/alpamayo2_super_wrapper.py
    export OPTICARVIS_ALPAMAYO_MODEL=nvidia/Alpamayo2-Super

It runs under the *planner's* interpreter, not the OptiCarVis venv, so it
imports nothing from this repo. The batch runner sets cwd to the planner repo,
so every path comes from argv and is made absolute.

Output, one file per manifest row, matching what src/workflow_runner.py parses:

    <output-dir>/alpamayo_inference_output_<video_id>.json
    -> video_id, segment_start_time_s, clip_video, run_meta
    -> result   : serialised 3-tuple
    -> pred_xyz : [1, 1, 1, 64, 3]     64 waypoints, ego-frame XYZ, metres
    -> pred_rot : [1, 1, 1, 64, 3, 3]
    -> extra    : dict whose values.cot holds the Chain-of-Causation text

THREE THINGS TO UNDERSTAND BEFORE TRUSTING THE OUTPUT
-----------------------------------------------------
1. One camera, not six. Every profile NVIDIA ships is six cameras; the model
   was trained only on those. The model core is camera-agnostic (cameras enter
   as text labels plus a variable number of image tokens), so a single forward
   view runs end to end -- but it is out of distribution, and accuracy degrades
   exactly where the missing views carry the information: cross-traffic,
   cut-ins, merges, anything needing rear awareness.

   Do NOT "fill" the missing views by repeating the forward frame. There is no
   validity mask and no missing-view embedding, so a repeated frame is consumed
   as genuine imagery labelled "Rear camera" -- it asserts something false.
   Omitting the views is tolerated; fabricating them is actively worse.

   This also means select_task_input() must be bypassed: it hard-raises on
   anything that is not the exact ordered seven-camera ring.

2. Ego history is required, and faking it is a silent trap. The model consumes
   exactly 16 past waypoints at 0.1 s (t0-1.5 s .. t0, metres, ego frame at t0,
   last entry pinned to the origin). The tokenizer turns 16 waypoints into 15
   deltas into exactly 45 tokens and raises on any count mismatch.

   An all-zeros history is NOT a neutral default: zero delta quantises to the
   "stationary" bin, so the model is told the car is stopped and will predict
   accordingly no matter what the video shows. A dashcam mp4 carries no
   egomotion, so this wrapper estimates it by planar visual odometry using the
   same flat-ground calibration as src/ego_trajectory.py (metric because depth
   comes from d = f*H/(v - horizon)). If estimation fails, the clip FAILS --
   it does not fall back to zeros unless you explicitly ask for that.

3. Operational gates. nvidia/Alpamayo2-Super is a GATED repo: provision HF_TOKEN
   or the batch 401s partway through. Linux + CUDA only, and the 34B BF16
   weights alone are ~68 GB.

Validate the plumbing without a GPU or the checkpoint:

    python scripts/alpamayo2_super_wrapper.py --clips m.csv --output-dir out \
        --config cfg.json --self-test
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time


# Canonical camera ids (alpamayo2_super/common/constants.py). A forward dashcam
# is id 1, camera_front_wide_120fov, which both shipped profiles include.
FRONT_WIDE_CAMERA_ID = 1

# Fixed by the model: 4 frames per camera at 10 Hz, ending exactly at t0.
NUM_CONTEXT_FRAMES = 4
CONTEXT_FRAME_DT_S = 0.1

# Fixed by the tokenizer: 16 history waypoints -> 15 deltas -> 45 tokens.
NUM_HISTORY_WAYPOINTS = 16
HISTORY_DT_S = 0.1
HISTORY_SPAN_S = (NUM_HISTORY_WAYPOINTS - 1) * HISTORY_DT_S  # 1.5 s

# Trajectory head: 64 waypoints, 0.1 .. 6.4 s.
NUM_WAYPOINTS = 64

# Flat-ground pinhole calibration, kept in step with src/ego_trajectory.py and
# src/final_preview_renderer.py. Retune all three together, never just one.
#
# These numbers are expressed in 1280x720 pixels (VANISH_U 636 ~ 1280/2), so
# every VO frame is resized to the reference first. Applying them to a 4K frame
# puts the ground band in the sky and the depth term collapses -- which reads
# as a stationary vehicle rather than as an error.
CALIB_REF_WIDTH = 1280
CALIB_REF_HEIGHT = 720

FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
HORIZON_V = 448.0
VANISH_U = 636.0
GROUND_BAND = (520, 690)
GROUND_MIN_ROWS = 60.0
MAX_STEP_M = 2.0

# A moving vehicle covers well over this in 1.5 s (0.25 m is ~0.6 km/h). Below
# it, the estimate is indistinguishable from a stopped car -- and a stopped-car
# history is precisely what corrupts the trajectory head. Treat it as a failed
# estimate rather than a measurement, and make the caller decide.
MIN_HISTORY_DISTANCE_M = 0.25


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Alpamayo2-Super over an OptiCarVis clip manifest.",
    )
    parser.add_argument("--clips", required=True, help="Manifest CSV from the batch runner.")
    parser.add_argument("--output-dir", "-o", required=True, help="Where per-clip JSON is written.")
    parser.add_argument("--config", "-c", default="", help="Backend config file (unused by this backend).")
    parser.add_argument("--model", default="nvidia/Alpamayo2-Super", help="Hugging Face id or local path.")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--t0-local-s",
        type=float,
        default=None,
        help="Clip-local time to condition on. Defaults to the manifest's "
             "when_start_local_s, clamped so the history window fits.",
    )
    parser.add_argument(
        "--allow-stationary-ego-history",
        action="store_true",
        help="Fall back to an all-zero ego history when VO fails. This tells "
             "the model the vehicle is STOPPED and biases every trajectory; "
             "only for smoke tests.",
    )

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Emit a correctly shaped synthetic payload without loading the model.",
    )

    # OPTICARVIS_ALPAMAYO_EXTRA_ARGS is forwarded verbatim, so tolerate unknown
    # flags rather than dying on them.
    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print("Ignoring unrecognised arguments:", " ".join(unknown))

    return args


def absolute(path):
    return os.path.abspath(path).replace("\\", "/")


def read_manifest(path):
    if not os.path.isfile(path):
        print("Missing clip manifest:", path)
        raise SystemExit(1)

    clips = []

    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("video_id"):
                continue

            clips.append(
                {
                    "video_id": row["video_id"],
                    "segment_start_time_s": float(row.get("segment_start_time_s") or 0.0),
                    "clip_video": absolute(row["clip_video"]),
                    "when_start_local_s": float(row.get("when_start_local_s") or 0.0),
                    "when_end_local_s": float(row.get("when_end_local_s") or 0.0),
                }
            )

    if not clips:
        print("Manifest contained no usable rows:", path)
        raise SystemExit(1)

    return clips


def output_json_path(output_dir, video_id):
    """The exact filename copy_alpamayo_json_to_expected_path() looks for."""
    return absolute(os.path.join(output_dir, "alpamayo_inference_output_" + video_id + ".json"))


# ---------------------------------------------------------------------------
# Serialisation -- mirrors alpamayo_policy_io.object_to_json so the payload is
# byte-compatible with what the Alpamayo-R1 wrapper produced.
# ---------------------------------------------------------------------------


def object_to_json(value, depth=0, max_depth=5):
    if depth > max_depth:
        return {"type": str(type(value)), "value": str(value)[:1000]}

    torch = sys.modules.get("torch")
    numpy = sys.modules.get("numpy")

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "values": value.detach().float().cpu().tolist(),
        }

    if numpy is not None and isinstance(value, numpy.ndarray):
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

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return {"type": str(type(value)), "value": str(value)[:1000]}


def text_block(text):
    """Wrap text the way workflow_runner.py expects to unwrap it.

    It walks extra -> values -> <key> -> values and strips nesting, so the text
    must sit inside a numpy-style [1, 1, 1] block.
    """
    text = "" if text is None else str(text)

    return {
        "type": "numpy.ndarray",
        "shape": [1, 1, 1],
        "dtype": "<U" + str(max(len(text), 1)),
        "values": [[[text]]],
    }


def build_extra(cot_text, meta_action="", answer=""):
    return {
        "type": "dict",
        "keys": ["cot", "meta_action", "answer"],
        "values": {
            "cot": text_block(cot_text),
            "meta_action": text_block(meta_action),
            "answer": text_block(answer),
        },
    }


# ---------------------------------------------------------------------------
# Ego history by planar visual odometry
#
# Same flat-ground model as src/ego_trajectory.py: a ground point at image row v
# lies at d = f*H/(v - horizon), so its vertical motion between frames gives a
# metric forward step, and its horizontal motion minus the predicted
# translational component gives yaw. Metric scale is what matters here, because
# the model's tokenizer quantises absolute metres.
# ---------------------------------------------------------------------------


def calibration_gray(frame_bgr):
    """Greyscale at the resolution the calibration constants assume."""
    import cv2

    resized = cv2.resize(
        frame_bgr,
        (CALIB_REF_WIDTH, CALIB_REF_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def estimate_frame_motion(prev_gray, cur_gray):
    """Return (d_yaw_rad, d_forward_m, ok) between two consecutive frames."""
    import cv2
    import numpy as np

    mask = np.zeros_like(prev_gray)
    mask[GROUND_BAND[0]:GROUND_BAND[1], :] = 255
    points = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 7, mask=mask)

    if points is None or len(points) <= 8:
        return 0.0, 0.0, False

    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    good = status.ravel() == 1

    if good.sum() <= 6:
        return 0.0, 0.0, False

    u0 = points[good][:, 0, 0].astype(np.float64)
    v0 = points[good][:, 0, 1].astype(np.float64)
    u1 = nxt[good][:, 0, 0].astype(np.float64)
    v1 = nxt[good][:, 0, 1].astype(np.float64)

    usable = (v0 > HORIZON_V + GROUND_MIN_ROWS) & (v1 > HORIZON_V + GROUND_MIN_ROWS)

    if usable.sum() <= 6:
        return 0.0, 0.0, False

    u0, v0, u1, v1 = u0[usable], v0[usable], u1[usable], v1[usable]

    depth0 = FOCAL_PX * CAM_HEIGHT_M / (v0 - HORIZON_V)
    depth1 = FOCAL_PX * CAM_HEIGHT_M / (v1 - HORIZON_V)

    steps = depth0 - depth1
    plausible = np.abs(steps) < MAX_STEP_M

    if plausible.sum() <= 6:
        return 0.0, 0.0, False

    forward = float(np.median(steps[plausible]))

    # Remove the translational part of horizontal flow before reading yaw;
    # using raw horizontal flow fabricates turns on straight roads.
    predicted = (u0 - VANISH_U) * forward / np.maximum(depth0, 1e-6)
    rotational = (u1 - u0) - predicted
    d_yaw = float(np.median(rotational[plausible])) / FOCAL_PX

    return d_yaw, forward, True


def ego_history_from_clip(clip_video, t0_s):
    """Return 16 past waypoints (metres) and rotations in the ego frame at t0.

    Raises RuntimeError when the motion cannot be estimated, so the caller can
    fail the clip rather than silently feeding the model a stationary history.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(clip_video)

    if not capture.isOpened():
        raise RuntimeError("could not open clip: " + clip_video)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if fps <= 0.0:
            raise RuntimeError("clip reports no usable fps")

        start_s = t0_s - HISTORY_SPAN_S

        if start_s < 0.0:
            raise RuntimeError(
                "t0=%.2fs leaves less than %.1fs of history" % (t0_s, HISTORY_SPAN_S)
            )

        first_index = int(round(start_s * fps))
        last_index = int(round(t0_s * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)

        ok, frame = capture.read()

        if not ok:
            raise RuntimeError("could not read the first history frame")

        prev = calibration_gray(frame)

        # Integrate (yaw, forward) into a path in the frame of the FIRST sample.
        times = [first_index / fps]
        xs, ys, psis = [0.0], [0.0], [0.0]
        x = y = psi = 0.0
        usable_frames = 0

        for index in range(first_index + 1, last_index + 1):
            ok, frame = capture.read()

            if not ok:
                break

            cur = calibration_gray(frame)
            d_yaw, d_forward, valid = estimate_frame_motion(prev, cur)
            prev = cur

            if valid:
                usable_frames += 1

            psi += d_yaw
            x += d_forward * math.cos(psi)
            y += d_forward * math.sin(psi)

            times.append(index / fps)
            xs.append(x)
            ys.append(y)
            psis.append(psi)

        if usable_frames < NUM_HISTORY_WAYPOINTS // 2:
            raise RuntimeError(
                "only %d frames yielded usable ground flow" % usable_frames
            )

        # Resample onto the exact 0.1 s grid the tokenizer expects.
        grid = [t0_s - HISTORY_SPAN_S + i * HISTORY_DT_S for i in range(NUM_HISTORY_WAYPOINTS)]
        times = np.asarray(times)
        rx = np.interp(grid, times, np.asarray(xs))
        ry = np.interp(grid, times, np.asarray(ys))
        rpsi = np.interp(grid, times, np.asarray(psis))

        # Express in the ego frame at t0: translate so the last sample is the
        # origin, then rotate so its heading is identity.
        cos_t0 = math.cos(-rpsi[-1])
        sin_t0 = math.sin(-rpsi[-1])
        dx = rx - rx[-1]
        dy = ry - ry[-1]

        xyz = []
        rot = []

        for i in range(NUM_HISTORY_WAYPOINTS):
            fx = dx[i] * cos_t0 - dy[i] * sin_t0
            fy = dx[i] * sin_t0 + dy[i] * cos_t0
            xyz.append([float(fx), float(fy), 0.0])

            heading = float(rpsi[i] - rpsi[-1])
            c, s = math.cos(heading), math.sin(heading)
            rot.append([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        # The contract pins the final entry to the origin with identity rotation.
        xyz[-1] = [0.0, 0.0, 0.0]
        rot[-1] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        travelled = math.hypot(xyz[0][0], xyz[0][1])

        if travelled < MIN_HISTORY_DISTANCE_M:
            raise RuntimeError(
                "estimated only %.3f m of travel over %.1fs (%.1f km/h). Ground "
                "flow gave no usable forward step, so this is a failed estimate "
                "rather than a stopped vehicle -- check that the flat-ground "
                "calibration (HORIZON_V/GROUND_BAND) matches this clip"
                % (travelled, HISTORY_SPAN_S, travelled / HISTORY_SPAN_S * 3.6)
            )

        return xyz, rot, usable_frames
    finally:
        capture.release()


def stationary_ego_history():
    xyz = [[0.0, 0.0, 0.0] for _ in range(NUM_HISTORY_WAYPOINTS)]
    rot = [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for _ in range(NUM_HISTORY_WAYPOINTS)
    ]

    return xyz, rot, 0


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def read_context_frames(clip_video, t0_s):
    """Return uint8 RGB CHW frames at t0-0.3, t0-0.2, t0-0.1, t0.

    Shape is (1, 4, 3, H, W): one camera, four frames. Resolution is left
    untouched -- the packaged Qwen processor resizes and normalises.
    """
    import cv2
    import numpy as np
    import torch

    capture = cv2.VideoCapture(clip_video)

    if not capture.isOpened():
        raise RuntimeError("could not open clip: " + clip_video)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))

        if fps <= 0.0:
            raise RuntimeError("clip reports no usable fps")

        frames = []

        for step in range(NUM_CONTEXT_FRAMES - 1, -1, -1):
            timestamp = t0_s - step * CONTEXT_FRAME_DT_S
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
            ok, bgr = capture.read()

            if not ok:
                raise RuntimeError("could not read frame at %.2fs" % timestamp)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(torch.uint8))
    finally:
        capture.release()

    return torch.stack(frames, dim=0).unsqueeze(0)


def resolve_t0(clip, args):
    """Clip-local t0, clamped so the history window fits inside the clip."""
    t0 = args.t0_local_s if args.t0_local_s is not None else clip["when_start_local_s"]

    return max(float(t0), HISTORY_SPAN_S)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def torch_dtype(name):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def patch_expert_attention_mask_dtype(dtype):
    """Make the expert's additive attention mask match the compute dtype.

    build_expert_pos_ids_and_attn_mask() hardcodes float32 for its 4-D additive
    mask (alpamayo2_super/models/expert_utils.py). Torch's memory-efficient SDPA
    kernel requires attn_mask.dtype == query.dtype, so with bf16 weights the
    expert denoiser dies with

        RuntimeError: invalid dtype for bias - should match query's dtype

    Upstream does not see this on H100, where these shapes select the flash
    backend instead -- and flash takes no bias at all, so the dtype never gets
    checked. Anywhere flash is unavailable the memory-efficient kernel runs and
    the mismatch surfaces; on GB10/sm_121 that is every time, because the shipped
    wheels compile only to sm_120.

    The clamp matters: float32's finfo.min is larger in magnitude than
    bfloat16's, so a bare .to(bfloat16) rounds the masked entries to -inf. That
    is harmless while every row keeps one unmasked key and silently produces NaN
    the moment one does not.

    Patched here rather than in external/alpamayo2 because that checkout is
    gitignored and re-cloned; a fix living there would vanish without warning.
    """
    import torch
    from alpamayo2_super.models import alpamayo2_super as module

    original = module.build_expert_pos_ids_and_attn_mask

    if getattr(original, "_opticarvis_dtype_patched", False):
        return

    def build_in_compute_dtype(*args, **kwargs):
        position_ids, attention_mask = original(*args, **kwargs)

        if attention_mask is not None and attention_mask.dtype != dtype:
            attention_mask = attention_mask.clamp(min=torch.finfo(dtype).min).to(dtype)

        return position_ids, attention_mask

    build_in_compute_dtype._opticarvis_dtype_patched = True
    module.build_expert_pos_ids_and_attn_mask = build_in_compute_dtype


def load_model(args):
    import torch  # noqa: F401  (registers the module for object_to_json)
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    patch_expert_attention_mask_dtype(torch_dtype(args.dtype))

    print("Loading model:", args.model)
    started = time.time()

    model = Alpamayo2Super.from_pretrained(
        args.model,
        dtype=torch_dtype(args.dtype),
        device_map=args.device,
    )

    print("Loaded in %.1f s" % (time.time() - started))

    return model


def run_model(model, frames, ego_xyz, ego_rot, args):
    """Return (pred_xyz, pred_rot, extra) for one clip.

    Deliberately bypasses load_physical_aiavdataset() and select_task_input():
    the latter hard-raises on anything that is not the exact ordered
    seven-camera ring, while prepare_model_inputs() itself needs only
    image_frames, camera_indices and the two ego-history tensors.
    """
    import torch
    from alpamayo2_super import helper

    data = {
        "image_frames": frames,
        "camera_indices": torch.tensor([FRONT_WIDE_CAMERA_ID], dtype=torch.int64),
        "ego_history_xyz": torch.tensor(ego_xyz, dtype=torch.float32).view(
            1, 1, NUM_HISTORY_WAYPOINTS, 3
        ),
        "ego_history_rot": torch.tensor(ego_rot, dtype=torch.float32).view(
            1, 1, NUM_HISTORY_WAYPOINTS, 3, 3
        ),
    }

    model_inputs = helper.to_device(
        helper.prepare_model_inputs(data, model.config, model.tokenizer),
        args.device,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        pred_xyz, pred_rot, _logprob, extra = model.sample_trajectories_from_data(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            diffusion_kwargs={"inference_step": args.diffusion_steps},
            return_extra=True,
        )

    cot = extra.get("cot") if isinstance(extra, dict) else None
    cot_text = cot[0] if isinstance(cot, (list, tuple)) and cot else (cot or "")

    if not str(cot_text).strip():
        print("    WARNING: empty Chain-of-Causation text; the gate will have no reasoning")

    meta = extra.get("meta_action") if isinstance(extra, dict) else ""
    meta_text = meta[0] if isinstance(meta, (list, tuple)) and meta else (meta or "")

    return (
        pred_xyz.reshape(1, 1, 1, NUM_WAYPOINTS, 3),
        pred_rot.reshape(1, 1, 1, NUM_WAYPOINTS, 3, 3),
        build_extra(cot_text, meta_text),
    )


def synthetic_result():
    """Correctly shaped stand-in so --self-test exercises the whole path."""
    xyz = [[[[[round(0.85 * (i + 1), 4), 0.0, 0.0] for i in range(NUM_WAYPOINTS)]]]]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rot = [[[[identity for _ in range(NUM_WAYPOINTS)]]]]

    pred_xyz = {
        "type": "torch.Tensor",
        "shape": [1, 1, 1, NUM_WAYPOINTS, 3],
        "dtype": "torch.float32",
        "device": "cpu",
        "values": xyz,
    }
    pred_rot = {
        "type": "torch.Tensor",
        "shape": [1, 1, 1, NUM_WAYPOINTS, 3, 3],
        "dtype": "torch.float32",
        "device": "cpu",
        "values": rot,
    }
    extra = build_extra(
        "SELF TEST: synthetic reasoning trace, no model was loaded.",
        "keep_lane",
    )

    return pred_xyz, pred_rot, extra


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_clip_json(path, clip, run_meta, pred_xyz, pred_rot, extra, serialise):
    if serialise:
        pred_xyz = object_to_json(pred_xyz)
        pred_rot = object_to_json(pred_rot)

    payload = {
        "video_id": clip["video_id"],
        "segment_start_time_s": clip["segment_start_time_s"],
        "clip_video": clip["clip_video"],
        "run_meta": run_meta,
        "result": {"type": "tuple", "length": 3, "items": [pred_xyz, pred_rot, extra]},
        "pred_xyz": pred_xyz,
        "pred_rot": pred_rot,
        "extra": extra,
    }

    directory = os.path.dirname(path)

    if directory and not os.path.isdir(directory):
        os.makedirs(directory)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_run_meta(args, clip, t0_s, ego_frames, elapsed_s):
    return {
        "created_at": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "config_file": absolute(args.config) if args.config else "",
        "model_id": "self_test" if args.self_test else args.model,
        "wrapper": "alpamayo2_super_wrapper.py",
        "sampling": {
            "seed": args.seed,
            "dtype": args.dtype,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "num_traj_samples": args.num_traj_samples,
            "num_diffusion_steps": args.diffusion_steps,
        },
        "num_time_steps": NUM_CONTEXT_FRAMES,
        "num_camera_slots": 1,
        "camera_indices": [FRONT_WIDE_CAMERA_ID],
        "single_forward_camera": True,
        "t0_local_s": round(t0_s, 3),
        "ego_history_source": "stationary_override" if ego_frames == 0 else "planar_vo",
        "ego_history_usable_frames": ego_frames,
        "inference_time_s": round(elapsed_s, 3),
        "when_start_local_s": clip["when_start_local_s"],
        "when_end_local_s": clip["when_end_local_s"],
    }


def process_clip(clip, args, model):
    t0_s = resolve_t0(clip, args)

    if args.self_test:
        pred_xyz, pred_rot, extra = synthetic_result()
        return pred_xyz, pred_rot, extra, t0_s, 0, False

    try:
        ego_xyz, ego_rot, ego_frames = ego_history_from_clip(clip["clip_video"], t0_s)
    except RuntimeError as error:
        if not args.allow_stationary_ego_history:
            raise RuntimeError(
                "ego history unavailable (%s). Refusing to substitute a zero "
                "history: it encodes a STOPPED vehicle and biases every "
                "trajectory. Pass --allow-stationary-ego-history to override."
                % error
            )

        print("    WARNING: VO failed (%s); using a STATIONARY ego history" % error)
        ego_xyz, ego_rot, ego_frames = stationary_ego_history()

    frames = read_context_frames(clip["clip_video"], t0_s)
    pred_xyz, pred_rot, extra = run_model(model, frames, ego_xyz, ego_rot, args)

    return pred_xyz, pred_rot, extra, t0_s, ego_frames, True


def main():
    args = parse_args()

    clips = read_manifest(absolute(args.clips))
    output_dir = absolute(args.output_dir)

    print("")
    print("Alpamayo2-Super wrapper")
    print("=======================")
    print("clips:      ", len(clips))
    print("output_dir: ", output_dir)
    print("model:      ", "SELF TEST (model not loaded)" if args.self_test else args.model)
    print("cameras:     1 (front wide, id %d) x %d frames" % (FRONT_WIDE_CAMERA_ID, NUM_CONTEXT_FRAMES))
    print("")

    model = None if args.self_test else load_model(args)

    written = skipped = failed = 0

    for index, clip in enumerate(clips, start=1):
        path = output_json_path(output_dir, clip["video_id"])

        if args.skip_existing and os.path.isfile(path):
            print("[%d/%d] %s -- exists, skipping" % (index, len(clips), clip["video_id"]))
            skipped += 1
            continue

        print("[%d/%d] %s" % (index, len(clips), clip["video_id"]))
        started = time.time()

        try:
            pred_xyz, pred_rot, extra, t0_s, ego_frames, serialise = process_clip(
                clip, args, model
            )
            elapsed = time.time() - started

            write_clip_json(
                path,
                clip,
                build_run_meta(args, clip, t0_s, ego_frames, elapsed),
                pred_xyz,
                pred_rot,
                extra,
                serialise,
            )
            print("    wrote %s (%.1f s)" % (path, elapsed))
            written += 1

        except Exception as error:
            # One bad clip must not abandon a 100-city batch; the batch runner
            # reports any clip whose JSON never appeared.
            print("    FAILED:", type(error).__name__, str(error)[:400])
            failed += 1

    print("")
    print("written:", written, "| skipped:", skipped, "| failed:", failed)

    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
