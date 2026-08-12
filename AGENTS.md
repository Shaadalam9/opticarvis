# AGENTS.md — operating manual for coding agents

This repo renders explainable-AV overlays (a future-path ribbon + road-user
highlights, gated by a VLM) onto dashcam footage. [README.md](README.md) is the
user-facing description; **this file is for you, the agent working on the code.**

## Before you change anything

1. Read the ribbon contract in
   [README → "What shapes the ribbon"](README.md#what-shapes-the-ribbon-the-contract).
2. Read [docs/ENGINEERING.md](docs/ENGINEERING.md) — every rule in it was paid
   for with a shipped defect, and it defines the verification protocols you are
   expected to run after touching geometry, tracking, or VO. The figures in
   `docs/images/` are the evidence; regenerate them if your change alters what
   they show.
3. A path overlay that points where the car is **not** going is worse than no
   overlay. When in doubt, prefer the conservative behaviour (straight in-lane,
   fall back, decay to zero).

## Environment (non-negotiable facts)

- **Windows**, Python 3.12, **`uv` for all package operations — never bare pip**:
  `uv pip install --python .venv/Scripts/python.exe <pkg>`.
- GPU is an **RTX 5080 (Blackwell, sm_120)** → PyTorch must be a **CUDA 12.8**
  build. There are **no prebuilt mmcv wheels for cu128** — anything requiring
  mmcv/mmdet (e.g. CLRerNet) is effectively unusable here without a source
  build; that is why UFLDv2 (pure PyTorch) was chosen for lanes. Do not
  re-attempt mmcv-based models without a source-build plan.
- **Upstream checkouts live in `external/`** (gitignored, cloned per install):
  `external/alpamayo`, `external/oom-free-alpamayo`, `external/UFLDv2`. UFLDv2
  needs weights `culane_res34.pth` (~865 MB, gdown id
  `1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj`); override via `OPTICARVIS_UFLD_REPO` /
  `OPTICARVIS_UFLD_WEIGHTS`. Its training-only imports (nvidia-dali,
  tensorboard) are stubbed in `src/scene_models.py: load_lane_instance_model` —
  keep it that way.
- Model weights (`*.pt`, `*.pth`) are **gitignored**; never commit them.
- `ffmpeg` must be on PATH for the H.264 delivery encode (the render degrades to
  the mp4v master with a warning if missing).
- The Gemma gate model is `google/gemma-4-E2B-it` — **E2B, not E4B**: E4B does
  not fit the 16 GB GPU.
- Renders are GPU-heavy (~10 min per 90 s clip): run them via Bash with
  `run_in_background`, never block on them.

## Repo layout

```
src/        the AV visualisation pipeline (run scripts from here: `python src/<script>.py`,
            which puts src/ on sys.path so the flat intra-pipeline imports resolve)
docs/       README figures + ENGINEERING.md
*.py (root) the separate mobility-study code (policy_demo, common, logmod, ...) - it
            resolves `config`/`secret` next to itself, so it must stay at the root

external/           gitignored upstream checkouts (alpamayo, oom-free-alpamayo, UFLDv2)
videos/             gitignored source dashcam videos
alpamayo_outputs/   gitignored: extracted clips + planner JSON (created by the pipeline)
workflow_outputs/   gitignored: renders, timelines, state (created by the pipeline)
```

Every path resolves from `PROJECT_ROOT`, which `src/pipeline_common.py` derives
from its own `__file__` — **never reintroduce an absolute local path**. Each
directory above has an `OPTICARVIS_*` override; see the README Configuration table.

## Repo map

| File | Role |
|---|---|
| `src/final_preview_renderer.py` | The renderer: ribbon geometry, lane-anchor tracking, VO blend, chevrons, compositing, both render loops |
| `src/render_timeline_clip.py` | **The CLI entry point for renders** (derives output names, transcodes, records workflow state) |
| `src/scene_models.py` | Lazy models: SegFormer road seg, Depth Anything V2, UFLDv2 lane instances, YOLOP (legacy) |
| `src/ego_trajectory.py` | Future-frame visual odometry → per-frame future path JSON (the ONLY curvature source for real turns) |
| `src/ego_motion.py` | Legacy phase-correlation pan track; feeds the disabled look-ahead only |
| `src/gemma_gate_timeline.py`, `src/gemma_reasoning_module.py` | Sliding-window VLM gate → timeline JSON |
| `src/alpamayo_stream.py` | Simulated per-timestep planner output feeding the gate |
| `src/pipeline_common.py` | Paths, env-overridable clip selection, `transcode_h264`, `clip_stem` |
| `docs/ENGINEERING.md` | Measured evidence behind every geometry/tracking decision |

Outputs land in `<PROJECT_ROOT>/workflow_outputs/final_renders/` — inside the repo
but **gitignored**; videos are never committed. Filenames derive from the rendered
clip (`clip_stem`), so clips cannot overwrite each other.

## How to render

```bash
# gate timeline (slow, VLM per window)
python src/gemma_gate_timeline.py <clip.mp4> gate_timeline.json 6.0

# standard render (lane centering + curve fit, no VO)
python src/render_timeline_clip.py <clip.mp4> gate_timeline.json <tag>

# with turn following: build the VO track, then enable it
python src/ego_trajectory.py <clip.mp4> vo_traj.json
OPTICARVIS_VO_TRAJECTORY=1 python src/render_timeline_clip.py <clip.mp4> gate_timeline.json <tag> "" vo_traj.json
```

`""` skips an optional argv slot. Supplying a track without its env flag warns
and ignores it. Every render writes two MP4s (pedestrians / +vehicles) and
records its effective config into the clip's workflow-state JSON. Env flags are
read **at module import** — set them before Python starts.

## Invariants — do not break these

Geometry (see ENGINEERING.md §1):
- The default ribbon is the **exact image of a straight ground line**: offset
  from `VANISH_U` scales with `(v − HORIZON_V)`, reaching zero **at the horizon
  only**. Never converge at the ribbon's far end; never interpolate the
  centreline with a smoothstep; never aim the far end from the road-mask
  centroid or the lane detector's far column. All three shipped a ribbon that
  pointed where the car was not driving.
- After ANY geometry change, assert the straight-case equivalence check
  (zero heading/curvature → ≤ 0.01 px from the analytic line).

Tracking (§2):
- A detection dropout is **no new information** — coast, never blend the target
  toward `VANISH_U`/any prior. Trust scales *gains*, not values.

Visual odometry (§3):
- Emitted lateral is **right-positive** (`lateral_convention` in the track
  JSON). Verify turn direction with the phase-correlate protocol — a sign error
  once survived visual inspection and mirrored every turn.
- Look-ahead is **distance-based** (`LOOKAHEAD_M`), never time-based — a time
  window is blind to slow turns (a real ~78° turn at 8 km/h was missed this way).
- Yaw must subtract translation parallax; VO rows must span the ribbon rows
  before blending; thin paths by arc length.

Compositing (§6):
- Layer alpha comes from coverage masks, never from colour luminance.

Process:
- **Measure, don't eyeball.** Record per-frame measurements once, replay
  offline to tune (ENGINEERING.md §7). Turn directions are verified against
  phase-correlation ground truth. Jitter is second-difference std.
- When your analysis says the footage lacks something a human says is there,
  **suspect your metric first**. This happened; the human was right.
- Keep the module docstrings and the README contract in sync with behaviour —
  stale comments here have directly caused regressions to be "restored".

## Committing

- Conventional, explanatory commit messages (this repo's history documents the
  *why* — keep that standard). Push only when the user asks.
- Before pushing: no `*.pt`/`*.pth` staged, README image links resolve, and if
  you changed the overlay's look, regenerate the affected `docs/images/` figures
  from real renders (they must show what the shipped code produces).
