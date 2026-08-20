"""oom-free-alpamayo

Memory-efficient inference framework for NVIDIA Alpamayo-R1 Vision-Language-Action
model on resource-constrained GPU platforms.

Main components:
    - DoubleBufHook: Pipelined Demand Layering with Double Flat Buffer
    - profiler:     System spec detection + per-layer profiling + config generation
    - predictor:    Performance prediction model (interleaved residency placement)
    - config:       Config JSON serialization
"""

__version__ = "0.1.0"

from alpamayo_memopt.hook import DoubleBufHook
from alpamayo_memopt.config import Config, load_config, save_config

__all__ = [
    "DoubleBufHook",
    "Config",
    "load_config",
    "save_config",
]
