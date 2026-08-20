"""Config JSON serialization.

Schema:
{
  "system": {
    "gpu_name": str,
    "vram_total_gb": float,
    "vram_budget_gb": float,
    "cpu_dram_total_gb": float
  },
  "model": {
    "name": str,
    "weights_total_gb": float,
    "vlm_layers": int,
    "vlm_layer_size_mb": float
  },
  "profiling": {
    "vlm_layer_dma_ms": float,
    "vlm_layer_exe_ms": float,
    "vit_layer_dma_ms": float,
    "vit_layer_exe_ms": float,
    "non_vlm_overhead_gb": float
  },
  "residency": {
    "max_possible": int,
    "num_resident": int,
    "resident_indices": [int, ...],
    "placement_rule": "interleaved (paper Eq. interleaved_placement)"
  },
  "predicted_performance": {
    "inference_time_s": float,
    "vit_resident": bool,
    "expert_resident": bool
  },
  "_meta": {
    "alpamayo_memopt_version": str,
    "created_at": str  # ISO 8601
  }
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass
class SystemConfig:
    gpu_name: str
    vram_total_gb: float
    vram_budget_gb: float
    cpu_dram_total_gb: float


@dataclass
class ModelConfig:
    name: str
    weights_total_gb: float
    vlm_layers: int
    vlm_layer_size_mb: float


@dataclass
class ProfilingConfig:
    vlm_layer_dma_ms: float
    vlm_layer_exe_ms: float
    vit_layer_dma_ms: float
    vit_layer_exe_ms: float
    non_vlm_overhead_gb: float


@dataclass
class ResidencyConfig:
    max_possible: int
    num_resident: int
    resident_indices: List[int]
    placement_rule: str = "interleaved (paper Eq. interleaved_placement)"


@dataclass
class PredictedPerformance:
    inference_time_s: float
    vit_resident: bool = False
    expert_resident: bool = False


@dataclass
class Config:
    system: SystemConfig
    model: ModelConfig
    profiling: ProfilingConfig
    residency: ResidencyConfig
    predicted_performance: PredictedPerformance
    _meta: dict = field(default_factory=dict)


def save_config(config: Config, path: str | Path) -> None:
    """Serialize config to JSON file."""
    from alpamayo_memopt import __version__

    data = asdict(config)
    data["_meta"] = {
        "alpamayo_memopt_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_config(path: str | Path) -> Config:
    """Load config from JSON file."""
    data = json.loads(Path(path).read_text())
    return Config(
        system=SystemConfig(**data["system"]),
        model=ModelConfig(**data["model"]),
        profiling=ProfilingConfig(**data["profiling"]),
        residency=ResidencyConfig(**data["residency"]),
        predicted_performance=PredictedPerformance(**data["predicted_performance"]),
        _meta=data.get("_meta", {}),
    )
