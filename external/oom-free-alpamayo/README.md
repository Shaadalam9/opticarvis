# oom-free-alpamayo

Memory-efficient inference framework for **NVIDIA Alpamayo-R1** Vision-Language-Action (VLA) model on resource-constrained GPU platforms.

> ✓ Accepted at **IEEE RTCSA 2026**

Alpamayo-R1-10B requires 21.52 GB of VRAM, exceeding the 12–16 GB capacity of consumer-grade GPUs. This framework enables memory-efficient inference on such GPUs through **system-level optimization alone — no model modification, no quantization, no pruning**.

## Performance

<!-- Fill in OOM or measured inference time for each platform. -->


| Platform | VRAM | Baseline | Ours |
|---|---:|---:|---:|
| RTX 5070 Ti | 16 GB | OOM | **4.09 s** |
| RTX 3080 Ti | 12 GB | OOM | **15.46 s** |

## How it works

Three coordinated optimizations:

1. **Sequential Demand Layering** — reduces VRAM usage from model-level (21.5 GB) to layer-level granularity by loading layer parameters on demand.
2. **Pipelined Demand Layering** — two-slot GPU buffer + dedicated prefetch CUDA stream hides H2D parameter transfer behind layer execution time.
3. **GPU-Resident Layer Decision Policy** — selectively keeps the most beneficial VLM layers permanently on the GPU using **interleaved residency placement** (paper Eq. 8). Eliminates the residual transfer overhead that pipelining cannot hide for DMA-intensive modules (e.g., VLM Decode, r ≈ 12).

A linear residency-benefit prediction model selects the optimal resident-layer count from a single profiling run, validated within **1.3 % prediction error**.

## Requirements

- NVIDIA GPU with ≥ 12 GB VRAM (12 GB tested on RTX 3080 Ti, 16 GB on RTX 5070 Ti)
- CUDA 12.x
- Python ≥ 3.10
- PyTorch ≥ 2.0
- CPU DRAM large enough to hold the full model weights (≥ 22 GB recommended)
- **NVIDIA Alpamayo-R1 source** (installed separately, Apache 2.0)
- **Alpamayo-R1-10B model weights** (subject to NVIDIA's separate model license)

## Installation

### 1. Install Alpamayo-R1 (separate dependency)

Follow NVIDIA's official instructions to install the Alpamayo-R1 Python package and download the `nvidia/Alpamayo-R1-10B` weights.

### 2. Install this framework

```bash
git clone https://github.com/aveeslab/oom-free-alpamayo.git
cd oom-free-alpamayo
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

The framework operates in two stages: **profile** (one-time per machine) → **infer** (repeatable).

### Step 1: Profile your system

```bash
python scripts/profile.py --output config.json
```

The profiler will:

1. Detect CPU DRAM total and verify it can hold all model weights.
2. Detect GPU VRAM total. By default, the budget is `total − 1 GB` (override with `--vram-budget`).
3. Load the model, run **Sequential Demand Layering** once with all VLM layers offloaded.
4. Compute the maximum number of resident VLM layers that fit in VRAM, apply a conservative margin of **−2** layers, and select indices via interleaved placement.
5. Predict the inference time and save the configuration.

Example output:

```
[1] Detecting system specifications...
    CPU DRAM total : 32.00 GB
    GPU            : NVIDIA GeForce RTX 5070 Ti
    VRAM total     : 16.30 GB
    VRAM budget    : 15.30 GB

[2] Loading Alpamayo-R1-10B (CPU)...
    Model weights total : 21.52 GB
    CPU DRAM check       : OK

...

[5] Residency planning...
    Max possible         : 31
    Conservative (-2)    : 29
    Resident indices     : [0, 1, 3, 4, ...]

[6] Saving config...
    Config saved to      : config.json
```

#### CLI options

| Option | Default | Description |
|---|---|---|
| `--output / -o` | `config.json` | Output config path |
| `--vram-budget` | `total − 1 GB` | Allowed VRAM in GB |
| `--baseline-time` | `14.52` | Reference baseline (s) for speedup reporting |
| `--decode-tokens` | `21` | Expected VLM decode token count |
| `--margin` | `2` | Conservative resident-count margin |
| `--device` | `0` | CUDA device index |

### Step 2: Run optimal inference

```bash
python scripts/infer.py --config config.json
```

Multiple iterations with timing statistics:

```bash
python scripts/infer.py --config config.json --num-iterations 5
```

Save predicted trajectories:

```bash
python scripts/infer.py --config config.json --output trajectory.json
```

## Project structure

```
oom-free-alpamayo/
├── alpamayo_memopt/        # Python package
│   ├── __init__.py
│   ├── hook.py             # DoubleBufHook (DFB + pipelined prefetch)
│   ├── profiler.py         # System / model / bandwidth profiling
│   ├── predictor.py        # Linear residency-benefit prediction model
│   └── config.py           # Config dataclass + JSON I/O
├── scripts/
│   ├── profile.py          # CLI: system profiling → config.json
│   └── infer.py            # CLI: optimal inference using config.json
├── pyproject.toml          # Package metadata (PEP 621)
├── LICENSE                 # MIT
├── NOTICE                  # Third-party attribution (Alpamayo Apache 2.0)
└── README.md
```

### Programmatic use

```python
from alpamayo_memopt import DoubleBufHook, load_config

config = load_config("config.json")
hook = DoubleBufHook(auto_restart=True)
hook.pin(vlm_layers, offload_indices)
hook.allocate(hook.max_elements())
hook.register(vlm_layers, offload_indices)
hook.start()
# model.inference(...)
hook.remove()
```

## Citation

```bibtex
@inproceedings{roh2026alpamayo,
  title     = {Memory-Efficient Deployment of Vision-Language-Action Models
               on Resource-Constrained GPU Platforms},
  author    = {Roh, Seungwoo and Kim, Huiyeong and Kim, Jong-Chan},
  booktitle = {Proc. IEEE Real-Time Computing Systems and Applications (RTCSA)},
  year      = {2026}
}
```

## License

This repository is released under the **MIT License** (see `LICENSE`).

This work depends on, but does not redistribute, NVIDIA Alpamayo-R1 source
(Apache 2.0) and the Alpamayo-R1-10B model weights (subject to NVIDIA's
separate model license). See `NOTICE` for details.

## Acknowledgements

- AVEES Lab, Graduate School of Automobile and Mobility, Kookmin University
- NVIDIA NVlabs for the Alpamayo-R1 model
