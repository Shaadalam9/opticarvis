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
- **VO forward speed is unrecoverable on the dense-traffic clips, and this is a
  scene limitation, not a bug.** Measured at frame 800 of
  `Esyp2P0uJu4_1056_30s.mp4`: median vertical optical flow in GROUND_BAND is
  **-0.3 px** where 30 km/h needs +3-4 px, p90 |dv| < 1 px. In a queue moving
  with the ego car the lead vehicle has near-zero relative motion, and the road
  that would carry the signal is both occluded and texture-poor. Masking
  features to segmented road was **tried and made it worse** (total path 6.0 m
  -> 3.2 m, heading -1.2 deg -> -14 deg) because asphalt yields fewer and weaker
  corners than vehicle edges. Do not re-attempt road masking or recalibration
  for this; calibration converts flow to metres, it cannot create flow. Speed
  has to come from another cue (GPS/metadata, or lane/crosswalk markings) or be
  accepted as unavailable. Note `median_speed_kmh` is a MEDIAN, so 0.0 is also
  the honest answer for a car stopped more than half the clip.
- `tests/test_calibration_scaling.py` guards the resolution scaling; run it
  after touching any pixel constant in the renderer. No GPU or clip needed.
- `scripts/alpamayo2_preflight.py` checks a host before the 68 GB download.
- **Editing `pyproject.toml` dependencies means running `uv lock` in the same
  commit.** `uv sync --frozen` does *not* fail on a stale lock — it silently
  under-installs, so a forgotten re-lock ships an environment missing packages
  with no error anywhere. Verified against uv 0.11.7.
- **PyPI's Windows `torch` wheel is CPU-only** (its CUDA deps are gated on
  `sys_platform == 'linux'`). `pyproject.toml` routes Windows to the cu128 index
  for that reason; do not "simplify" it back to a plain PyPI dependency, and do
  not extend the cu128 pin to Linux `aarch64` — those wheels exist, so it would
  resolve cleanly while giving DGX Spark (sm_121) a CUDA 12.8 build.
- `ffmpeg` must be on PATH for the H.264 delivery encode (the render degrades to
  the mp4v master with a warning if missing).
- The Gemma gate model is `google/gemma-4-E2B-it` — **E2B, not E4B**: E4B does
  not fit the 16 GB GPU.
- **Every model id lives in the `Models` block of `src/pipeline_common.py`** and
  is env-overridable. Do not hardcode a checkpoint in the module that loads it;
  add it there instead. The planner is a *subprocess*, so its interpreter
  (`OPTICARVIS_ALPAMAYO_PYTHON`), checkpoint (`OPTICARVIS_ALPAMAYO_MODEL`) and
  flags (`OPTICARVIS_ALPAMAYO_EXTRA_ARGS`) are swappable without touching this
  repo — a bigger planner needs its own torch, so it will not share this venv.
  `ALPAMAYO_PYTHON` is deliberately **not** run through `normalise_path`: it may
  be a POSIX path while the runner is on Windows.
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
| `src/ego_trajectory.py` | Future-frame visual odometry → per-frame future path JSON (the metric curvature source; also supplies the anchors' arclength) |
| `src/future_anchor.py` | Chains ground-plane homographies to the future frames → per-frame polylines of the street pixels the car will drive over. Highest-precedence ribbon source; the only one that survives real turns |
| `src/ego_motion.py` | Legacy phase-correlation pan track; feeds the disabled look-ahead only |
| `src/gemma_gate_timeline.py`, `src/gemma_reasoning_module.py` | Sliding-window VLM gate → timeline JSON |
| `src/alpamayo_stream.py` | Simulated per-timestep planner output feeding the gate |
| `src/pipeline_common.py` | Paths, env-overridable clip selection, `transcode_h264`, `clip_stem` |
| `src/city_sampler.py`, `src/lpm.py` | Which cities to film: spatially balanced probability sampling (Local Pivotal Method), design weights included |
| `docs/ENGINEERING.md` | Measured evidence behind every geometry/tracking decision |
| `docs/CITY_SAMPLING.md` | Why the cities are drawn rather than picked, with references |

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

# anchored on the road (what the batch does): add the homography-chain pass
python src/future_anchor.py <clip.mp4> vo_traj.json anchors.json
OPTICARVIS_VO_TRAJECTORY=1 OPTICARVIS_FUTURE_ANCHOR_JSON=anchors.json \
  python src/render_timeline_clip.py <clip.mp4> gate_timeline.json <tag> "" vo_traj.json
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

Future anchoring (`src/future_anchor.py`):
- The anchors' placement must stay **model-free**. Their whole value is that they
  are found by image registration, so heading, calibration and slope errors cancel.
  Ground metres may parameterise the *drawing* (resampling, smoothing, rail
  offsets); they must never re-derive an anchor's position.
- Rails offset perpendicular to the **ground** tangent, then project. Offsetting
  perpendicular in pixel space fans the band into a wedge wherever the path runs
  across the image — on screen, "perpendicular" is partly depth and must
  foreshorten.
- A band ends where its data ends. Never extrapolate a measured path to fill a
  fixed forward window: constant-extending 1.26 m of measured lateral out to 21 m
  is exactly how the ribbon came to point at a median mid-turn.
- A hop below the RANSAC inlier floor truncates the chain. Truncating is honest;
  fabricating anchors is not.
- Path geometry is arc-parameterised, never `y(x_forward)`: a real turn folds back
  (max ~12 m forward on a 96° turn) and a single-valued lateral function cannot
  express it.

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
