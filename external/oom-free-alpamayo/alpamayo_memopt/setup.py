"""Alpamayo-R1 model loading + GPU placement helpers.

This is a near-verbatim port of the RF research code's `common.py` setup
helpers (dropping the hook class, which lives in alpamayo_memopt.hook).
The behavior is preserved as-is to retain the bit-exact paths validated
in the paper.
"""

import copy
import subprocess

import torch


# ─────────────────────────────────────────────────────────────────────
# Constants (keep in sync with the paper)
# ─────────────────────────────────────────────────────────────────────

MODEL_ID = "nvidia/Alpamayo-R1-10B"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US = 5_100_000
HF_CACHED_REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
N_VLM = 36
DECODE_TOKENS = 21


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────

def set_max_clock(clock: int) -> None:
    """Lock the GPU graphics clock (requires sudo). No-op on failure."""
    subprocess.run(
        f"sudo nvidia-smi -lgc {clock}",
        shell=True, capture_output=True, input=b"nvidia",
    )


def deep_copy_inputs(mi):
    """Deep-copy a model-input dict. Tensors are .clone()-d."""
    result = {}
    for k, v in mi.items():
        if isinstance(v, dict):
            result[k] = {
                kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        elif isinstance(v, torch.Tensor):
            result[k] = v.clone()
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_data():
    """Load the standard Alpamayo benchmark sample.

    Returns ``{"data": ..., "messages": ...}``.
    """
    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    avdi = None
    for rev in [HF_CACHED_REVISION, "37a7cc2c868d684d0456b5412a7ec5d18597a96a"]:
        try:
            avdi = physical_ai_av.PhysicalAIAVDatasetInterface(revision=rev)
            break
        except Exception:
            continue
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    data = load_physical_aiavdataset(CLIP_ID, t0_us=T0_US, avdi=avdi)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    return {"data": data, "messages": messages}


def prepare_inputs(model, data_cache):
    """Prepare model inputs (tokenize + move to GPU)."""
    from alpamayo_r1 import helper

    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        data_cache["messages"], tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    return helper.to_device({
        "tokenized_data": inputs,
        "ego_history_xyz": data_cache["data"]["ego_history_xyz"],
        "ego_history_rot": data_cache["data"]["ego_history_rot"],
    }, "cuda")


# ─────────────────────────────────────────────────────────────────────
# Model loading + placement
# ─────────────────────────────────────────────────────────────────────

def load_model():
    """Load Alpamayo-R1-10B onto CPU."""
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    # Force SDPA attention so the model runs without flash-attn (e.g. on
    # Windows, where flash-attn has no prebuilt wheels). The repo reads the
    # custom `config.attn_implementation` attribute, so it must be set on the
    # config object itself (passing attn_implementation= to from_pretrained
    # only sets transformers' internal _attn_implementation and is ignored).
    config = AlpamayoR1.config_class.from_pretrained(MODEL_ID)
    config.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained(
        MODEL_ID, config=config, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def setup_gpu_essentials(model, vit_gpu: bool = False, expert_gpu: bool = False):
    """Move non-layer essentials to GPU.

    Args:
        vit_gpu:    If True, ViT blocks also move to GPU (bypass hook).
        expert_gpu: If True, Expert layers also move to GPU.

    Returns:
        Tuple ``(vlm_layers, vit_blocks, expert_layers)`` of layer lists.
    """
    lm = model.vlm.model.language_model
    visual = model.vlm.model.visual

    # Always-resident: embed, norm, rotary, lm_head, action projs, diffusion
    lm.embed_tokens.to("cuda")
    lm.norm.to("cuda")
    lm.rotary_emb.to("cuda")
    model.vlm.lm_head.to("cuda")
    if hasattr(model, "action_in_proj"):
        model.action_in_proj.to("cuda")
    if hasattr(model, "action_out_proj"):
        model.action_out_proj.to("cuda")
    if hasattr(model, "diffusion"):
        model.diffusion.to("cuda")

    # Vision: non-blocks always on GPU
    for n, m in visual.named_children():
        if n != "blocks":
            m.to("cuda")

    vblocks = list(visual.blocks)
    if vit_gpu:
        for b in vblocks:
            b.to("cuda")

    # Expert: non-layers always on GPU
    for n, child in model.expert.named_children():
        if n != "layers":
            child.to("cuda")
    for n, p in model.expert.named_parameters():
        if "layers." not in n and p.device.type == "cpu":
            p.data = p.data.to("cuda")
    for n, b in model.expert.named_buffers():
        if "layers." not in n and b.device.type == "cpu":
            b.data = b.data.to("cuda")

    expert_layers = list(model.expert.layers)
    if expert_gpu:
        for ly in expert_layers:
            ly.to("cuda")

    return lm.layers, vblocks, expert_layers


# ─────────────────────────────────────────────────────────────────────
# Residency placement (paper Eq. 8)
# ─────────────────────────────────────────────────────────────────────

def interleaved_placement(k: int, N: int = 35) -> list:
    """Interleaved residency placement. Layer 0 always included, layer N excluded."""
    if k <= 0:
        return []
    if k >= N:
        return list(range(N))
    return sorted(set(int(i * N / k) for i in range(k)))


def contiguous_placement(k: int) -> list:
    """Contiguous residency placement: [0, 1, ..., k-1]."""
    return list(range(k))
