r"""Lazy-loaded scene-understanding models for the OptiCarVis workflow.

Three optional models, loaded on first use so lightweight tools (e.g. the
renderer's --calibrate path) never pay for them:

  * road segmentation  - SegFormer trained on Cityscapes, gives a drivable-road
    mask used to clip the path ribbon to the real road surface (and thereby
    occlude it behind anything that is not road: people, vehicles, curbs).
  * monocular depth     - Depth Anything V2, a relative inverse-depth map
    (higher = nearer) used for occlusion ordering and depth ranking.
  * lane lines          - YOLOP, a multitask driving model whose lane-line head
    gives a lane-marking mask used to center the ribbon in the ego's own lane
    (Solution B: ego-lane detection instead of assuming image center).

All run on CUDA when available. Requires torch (+ transformers for the first
two; YOLOP is pulled from torch.hub).
"""

import os

import cv2
import numpy as np


ROAD_SEG_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
# Cityscapes label ids that count as drivable / walkable ground for the ribbon.
# 0 = road (includes painted crosswalks). Sidewalk (1) is intentionally excluded.
ROAD_LABEL_IDS = (0,)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# YOLOP lane-line head. Input is 640x384 RGB; ll_seg is a 2-class (bg / lane)
# segmentation upsampled back to the frame.
LANE_MODEL_REPO = "hustvl/yolop"
LANE_MODEL_NAME = "yolop"
LANE_INPUT_W = 640
LANE_INPUT_H = 384

# UFLDv2 (Ultra-Fast-Lane-Detection-v2): a row-anchor lane detector that returns
# lane *instances* (per-lane point sequences) rather than a mask, which the
# renderer straddle-selects into clean ego-lane boundaries. Pure PyTorch, so it
# runs on the same torch/CUDA as everything else (no mmcv/mmdet). The repo is
# vendored beside the project; only inference code is touched (DALI / tensorboard
# imports in its util chain are stubbed since they are training-only).
UFLD_REPO = os.environ.get(
    "OPTICARVIS_UFLD_REPO", "C:/Users/localadmin/Desktop/Shadab/UFLDv2")
UFLD_WEIGHTS = os.environ.get("OPTICARVIS_UFLD_WEIGHTS", UFLD_REPO + "/culane_res34.pth")
UFLD_CFG = {
    "backbone": "34", "num_row": 72, "num_col": 81, "num_cell_row": 200,
    "num_cell_col": 100, "num_lanes": 4, "train_height": 320, "train_width": 1600,
    "crop_ratio": 0.6,
}
UFLD_MEAN = (0.485, 0.456, 0.406)
UFLD_STD = (0.229, 0.224, 0.225)

_road_model = None
_depth_model = None
_lane_model = None
_lane_instance_model = None


def _torch():
    import torch

    return torch


def load_road_model():
    """Load and cache the SegFormer road-segmentation model."""
    global _road_model
    if _road_model is None:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        torch = _torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = SegformerImageProcessor.from_pretrained(ROAD_SEG_MODEL)
        model = SegformerForSemanticSegmentation.from_pretrained(ROAD_SEG_MODEL)
        model = model.to(device).eval()
        _road_model = (processor, model, device)
    return _road_model


def road_mask(frame_bgr, label_ids=ROAD_LABEL_IDS):
    """Return a uint8 {0,1} drivable-road mask at the frame's resolution."""
    torch = _torch()
    processor, model, device = load_road_model()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits,
        size=frame_bgr.shape[:2],
        mode="bilinear",
        align_corners=False,
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()
    return np.isin(labels, label_ids).astype(np.uint8)


def load_depth_model():
    """Load and cache the Depth Anything V2 pipeline."""
    global _depth_model
    if _depth_model is None:
        from transformers import pipeline

        torch = _torch()
        device = 0 if torch.cuda.is_available() else -1
        _depth_model = pipeline("depth-estimation", model=DEPTH_MODEL, device=device)
    return _depth_model


def depth_map(frame_bgr):
    """Return a float32 relative inverse-depth map (higher = nearer)."""
    from PIL import Image

    pipe = load_depth_model()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pipe(Image.fromarray(rgb))
    return np.array(result["depth"], dtype=np.float32)


def load_lane_model():
    """Load and cache the YOLOP lane-line model (from torch.hub)."""
    global _lane_model
    if _lane_model is None:
        torch = _torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = torch.hub.load(
            LANE_MODEL_REPO, LANE_MODEL_NAME, pretrained=True, trust_repo=True
        )
        model = model.to(device).eval()
        _lane_model = (model, device)
    return _lane_model


def lane_line_mask(frame_bgr):
    """Return a uint8 {0,1} lane-line mask at the frame's resolution.

    Runs YOLOP's lane-line head on the frame and upsamples the mask back to the
    input resolution. Marks the painted longitudinal lane lines (and, at
    intersections, crosswalk stripes); the renderer's ego-lane detector then
    keeps only the converging longitudinal pair straddling the car.
    """
    torch = _torch()
    model, device = load_lane_model()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (LANE_INPUT_W, LANE_INPUT_H))
    tensor = (
        torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    )
    with torch.no_grad():
        _det, _da_seg, ll_seg = model(tensor)
    mask = (ll_seg.argmax(dim=1)[0].cpu().numpy() > 0).astype(np.uint8)
    return cv2.resize(
        mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def load_lane_instance_model():
    """Load and cache the UFLDv2 lane-instance model from the vendored repo."""
    global _lane_instance_model
    if _lane_instance_model is None:
        import sys
        import types

        if UFLD_REPO not in sys.path:
            sys.path.insert(0, UFLD_REPO)
        # Stub the repo's training-only imports (DALI data loader, tensorboard)
        # so the inference model class imports without those heavy deps.
        for name, attr in (("data.dali_data", "TrainCollect"),
                           ("torch.utils.tensorboard", "SummaryWriter")):
            if name not in sys.modules:
                stub = types.ModuleType(name)
                setattr(stub, attr, object)
                sys.modules[name] = stub

        import torchvision.transforms as transforms
        from model.model_culane import parsingNet

        torch = _torch()
        cfg = UFLD_CFG
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = parsingNet(
            pretrained=False, backbone=cfg["backbone"], num_grid_row=cfg["num_cell_row"],
            num_cls_row=cfg["num_row"], num_grid_col=cfg["num_cell_col"],
            num_cls_col=cfg["num_col"], num_lane_on_row=cfg["num_lanes"],
            num_lane_on_col=cfg["num_lanes"], use_aux=False,
            input_height=cfg["train_height"], input_width=cfg["train_width"], fc_norm=True,
        ).to(device).eval()

        state = torch.load(UFLD_WEIGHTS, map_location="cpu")["model"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        net.load_state_dict(state, strict=False)

        transform = transforms.Compose([
            transforms.Resize((int(cfg["train_height"] / cfg["crop_ratio"]), cfg["train_width"])),
            transforms.ToTensor(),
            transforms.Normalize(UFLD_MEAN, UFLD_STD),
        ])
        row_anchor = np.linspace(0.42, 1.0, cfg["num_row"])
        _lane_instance_model = (net, device, transform, row_anchor)
    return _lane_instance_model


LANE_MIN_ANCHOR_FRAC = 0.35   # a lane must be present on this fraction of the row anchors


def lane_instances(frame_bgr, min_points=8, local_width=1):
    """Return detected lane-line polylines from UFLDv2.

    Each lane is an Nx2 float array of (x, y) points in the frame's pixel coords,
    ordered by the model's row anchors. The renderer straddle-selects the pair
    around the ego column to form the ego-lane centre.

    A lane must be detected on at least LANE_MIN_ANCHOR_FRAC of the row anchors
    (as well as min_points absolute). The reference decoder demands >50%; a bare
    8-of-72 (11%) threshold admitted sparse, hallucinated polylines, and because
    the renderer picks the *innermost* line on each side, such a fragment wins the
    selection and drags the ribbon out of the real lane.
    """
    from PIL import Image

    torch = _torch()
    net, device, transform, row_anchor = load_lane_instance_model()
    cfg = UFLD_CFG
    height, width = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb))
    tensor = tensor[:, -cfg["train_height"]:, :].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(tensor)

    loc = pred["loc_row"]
    exist = pred["exist_row"]
    num_grid, num_cls, num_lane = loc.shape[1], loc.shape[2], loc.shape[3]
    max_idx = loc.argmax(1)[0].cpu()
    valid = exist.argmax(1)[0].cpu()
    loc_cpu = loc[0].cpu()

    lanes = []
    for i in range(num_lane):
        xs = []
        ys = []
        for k in range(num_cls):
            if valid[k, i]:
                center = int(max_idx[k, i])
                lo = max(0, center - local_width)
                hi = min(num_grid - 1, center + local_width)
                ind = torch.arange(lo, hi + 1)
                x = (loc_cpu[ind, k, i].softmax(0) * ind.float()).sum().item() + 0.5
                xs.append(x / (num_grid - 1) * width)
                ys.append(float(row_anchor[k]) * height)
        if len(xs) >= max(min_points, int(LANE_MIN_ANCHOR_FRAC * num_cls)):
            lanes.append(np.stack([np.asarray(xs), np.asarray(ys)], axis=1).astype(np.float32))
    return lanes
