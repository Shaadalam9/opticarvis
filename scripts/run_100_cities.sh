#!/usr/bin/env bash
# Canonical launcher for the 100 city batch on a machine with the Alpamayo2
# Super planner at external/alpamayo2 (see README "Swapping the planner model").
#
#   bash scripts/run_100_cities.sh [max_jobs] [start_index]
#
# Encodes every setting the batch needs and a bring-up discovered the hard way;
# override any of them by exporting the variable before running.
set -euo pipefail

cd "$(dirname "$0")/.."

# The planner and gate checkpoints total ~77 GB; pointing HF_HOME somewhere
# without them means a full re-download, so refuse to guess.
if [ -z "${HF_HOME:-}" ]; then
    echo "Set HF_HOME to the hub cache that holds nvidia/Alpamayo2-Super" >&2
    echo "and google/gemma-4-E2B-it (e.g. export HF_HOME=\$HOME/hf-cache)." >&2
    exit 1
fi

# The gate is the thing under study: refuse the silent heuristic fallback.
export OPTICARVIS_REQUIRE_GEMMA_GATE="${OPTICARVIS_REQUIRE_GEMMA_GATE:-1}"

# One rendered video per city, scanning up to 5 candidate windows for one the
# gate approves. The gate's design default is do_not_explain, so first windows
# frequently yield nothing.
export OPTICARVIS_CLIPS_PER_CITY="${OPTICARVIS_CLIPS_PER_CITY:-1}"
export OPTICARVIS_WINDOWS_PER_CITY="${OPTICARVIS_WINDOWS_PER_CITY:-5}"

# The ribbon shows the PLANNER's intended path -- the thing the overlay is
# explaining -- projected from Alpamayo's 64 waypoints at the planned moment.
# Perception (lane centering + validated VO) remains the fallback when a clip
# has no planner context.
export OPTICARVIS_RIBBON_SOURCE="${OPTICARVIS_RIBBON_SOURCE:-planner}"

# The ribbon should bend into real turns: reconstruct the ego path per clip
# (ego_trajectory.py stage) and let the renderer blend it in through its
# guards. Falls back to straight-in-lane on scenes where VO cannot recover
# motion. Ego-lane centering and gentle curves additionally need the UFLDv2
# checkout + culane_res34.pth (see README "External repositories").
export OPTICARVIS_VO_TRAJECTORY="${OPTICARVIS_VO_TRAJECTORY:-1}"

# The in-code defaults for these two are not valid Hugging Face repo ids; the
# render fails out of the box without them (see README "Models used").
export OPTICARVIS_ROAD_SEG_MODEL="${OPTICARVIS_ROAD_SEG_MODEL:-nvidia/segformer-b0-finetuned-cityscapes-1024-1024}"
export OPTICARVIS_DEPTH_MODEL="${OPTICARVIS_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Small-hf}"

# First run on a fresh cache must be allowed to hit the Hub.
export OPTICARVIS_HF_LOCAL_FILES_ONLY="${OPTICARVIS_HF_LOCAL_FILES_ONLY:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Refuse to start with assets missing. The failure mode this prevents is not a
# crash but a plausible-looking wrong result: without the UFLDv2 lane model the
# renderer silently falls back to a straight ribbon, and a whole batch can
# complete that way. OPTICARVIS_SKIP_ASSET_CHECK=1 overrides (at your own risk).
if [ "${OPTICARVIS_SKIP_ASSET_CHECK:-0}" != "1" ]; then
    if ! .venv/bin/python scripts/setup_assets.py --check-only; then
        echo "" >&2
        echo "Assets missing -- run: .venv/bin/python scripts/setup_assets.py" >&2
        echo "(add --with-planner for the 67 GB checkpoint)" >&2
        exit 1
    fi
fi

# Rebuild the job list every run: it is cheap (<1 s), and a stale list built
# under different WINDOWS_PER_CITY/CLIPS_PER_CITY silently changes what the
# batch does. OPTICARVIS_REUSE_CLIP_JOBS=1 keeps a hand-crafted list.
if [ "${OPTICARVIS_REUSE_CLIP_JOBS:-0}" != "1" ] || [ ! -f workflow_outputs/clip_jobs.jsonl ]; then
    echo "Building clip jobs from mapping.csv..."
    .venv/bin/python src/clip_job_builder.py
fi

# An empty argv[1] would break int(sys.argv[1]) in the batch runner, so only
# forward the arguments that were actually given.
exec .venv/bin/python src/batch_corrected_pipeline.py "$@"
