"""Performance prediction model.

Linear residency benefit model from the paper (validated to within 1.3 %
prediction error across the resident sweep range):

    C_total(k) = C_total(0) - k * slope_per_layer

where:
    k                = number of resident VLM layers
    C_total(0)       = full-offload inference time (Nr = 0), measured
    slope_per_layer  = per-resident-layer time saving (ms)

For VLM Decode (DMA-intensive, r >> 1), the per-layer saving is:
    slope_per_layer ≈ R_decode * C_DMA_per_layer
where R_decode is the number of decode tokens generated per inference (≈ 21
for Alpamayo on the standard benchmark).
"""

from __future__ import annotations

from typing import Optional

# Default decode-token count for Alpamayo on the standard benchmark.
# Override via predict_inference_time(num_decode_tokens=...) when known.
_DEFAULT_DECODE_TOKENS = 21


def per_layer_benefit_ms(
    vlm_layer_dma_ms: float,
    vlm_layer_exe_ms: float = 0.0,
    num_decode_tokens: int = _DEFAULT_DECODE_TOKENS,
) -> float:
    """Per-resident-layer inference-time saving (ms).

    Paper formula:
        slope = R_decode * (C_DMA - C_EXE)
    where C_DMA - C_EXE is the per-call time saved by making a VLM Decode
    layer resident (DMA-intensive regime, EXE was previously hidden under DMA).
    """
    delta_c = max(0.0, vlm_layer_dma_ms - vlm_layer_exe_ms)
    return num_decode_tokens * delta_c


def predict_inference_time(
    num_resident: int,
    full_offload_time_s: float,
    vlm_layer_dma_ms: float,
    vlm_layer_exe_ms: float = 0.0,
    num_decode_tokens: int = _DEFAULT_DECODE_TOKENS,
) -> float:
    """Predict E2E inference time (s) for `num_resident` VLM layers."""
    slope_ms = per_layer_benefit_ms(
        vlm_layer_dma_ms, vlm_layer_exe_ms, num_decode_tokens
    )
    saved_s = num_resident * slope_ms / 1000
    return max(0.0, full_offload_time_s - saved_s)


def predict_speedup(
    num_resident: int,
    full_offload_time_s: float,
    baseline_time_s: float,
    vlm_layer_dma_ms: float,
    num_decode_tokens: int = _DEFAULT_DECODE_TOKENS,
) -> float:
    """Speedup vs an external baseline (e.g., Accelerate offloading)."""
    predicted = predict_inference_time(
        num_resident, full_offload_time_s, vlm_layer_dma_ms, num_decode_tokens
    )
    if predicted <= 0:
        return float("inf")
    return baseline_time_s / predicted


def estimate_pcie_bw_used(vlm_layer_size_mb: float, vlm_layer_dma_ms: float) -> float:
    """Effective PCIe bandwidth (GB/s) inferred from layer DMA time."""
    return (vlm_layer_size_mb / 1024) / (vlm_layer_dma_ms / 1000)
