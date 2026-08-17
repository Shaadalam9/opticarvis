"""Process-wide cache for the Gemma 4 gate model.

Deliberately separate from gemma_reasoning_module: that module derives its
paths and job identity from environment variables at import time, so a driver
that gates many jobs in one process must importlib.reload() it per job -- and a
reload resets module globals, which would drop a cached 9.6 GB model on every
job. This module is never reloaded, so the model survives.

Keyed by (model_id, local_files_only) so an override mid-process cannot hand
back the wrong checkpoint.
"""

_CACHE = {}


def get_gemma4(model_id, local_files_only):
    key = (str(model_id), bool(local_files_only))

    if key not in _CACHE:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # local_files_only avoids slow/hanging Hub metadata calls; the weights
        # are expected to be cached already. A machine that has not cached them
        # needs OPTICARVIS_HF_LOCAL_FILES_ONLY=0 for the first run.
        processor = AutoProcessor.from_pretrained(
            model_id,
            padding_side="left",
            local_files_only=local_files_only,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            local_files_only=local_files_only,
        ).to(device).eval()

        _CACHE[key] = (processor, model)

    return _CACHE[key]
