"""Per-frame overlay geometry sink for post-hoc restyling.

The renderer burns its overlay into the pixels, so changing a colour costs a
full re-render -- and the cost of a render is the four models per frame, not
the drawing. This module lets final_preview_renderer.py dump, per frame, the
geometry those models produced: the ribbon centreline, the selected detections
(mask polygon, smoothed box, score, distance), the animation ramp and label,
and the occlusion map. src/restyle_render.py then re-composites any style from
that in seconds, with no GPU and no models.

Artifacts, per clip:
    workflow_outputs/overlay_geometry/<tag>_geometry.jsonl.gz
        line 0: header (video meta, effective camera constants, fps)
        then one line per frame
    workflow_outputs/overlay_geometry/<tag>_occlusion.mp4
        grayscale occlusion map (255 = fully occluding). Soft by nature, so a
        lossy codec is acceptable.

Only the ribbon CENTRELINE is stored: the edges derive from it row by row via
half = half_width_m * (v - HORIZON_V) / CAM_HEIGHT_M, which makes the ribbon
width itself restylable. Chevron positions are a pure function of the
centreline, the frame index and the chevron speed, so they are recomputed at
composite time -- speed and spacing are restylable too.
"""

import gzip
import json
import os

import cv2
import numpy as np


DUMP_VERSION = 1


def _round_points(points, decimals=2):
    return [[round(float(u), decimals), round(float(v), decimals)] for u, v in points]


def _detection_record(detection, include_class):
    mask = detection.get("mask")

    record = {
        "poly": mask.tolist() if mask is not None else None,
        "box": [int(value) for value in detection.get("draw_box", detection["box"])],
        "score": round(float(detection.get("spatial_score", 0.0)), 4),
        "dist": (
            round(float(detection["distance_m"]), 2)
            if detection.get("distance_m") is not None
            else None
        ),
        "tid": int(detection.get("track_id", -1)),
    }

    if include_class:
        record["cls"] = detection.get("class_name", "")

    return record


class GeometryDump(object):
    """Writes the per-frame geometry stream beside the renders.

    Failure to dump must never fail a render: every write is guarded, and the
    first error disables the sink for the rest of the clip with one warning.
    """

    def __init__(self, output_dir, tag, meta):
        self.enabled = True
        self._handle = None
        self._occlusion_writer = None
        self.geometry_path = os.path.join(output_dir, tag + "_geometry.jsonl.gz")
        self.occlusion_path = os.path.join(output_dir, tag + "_occlusion.mp4")

        try:
            os.makedirs(output_dir, exist_ok=True)
            self._handle = gzip.open(self.geometry_path, "wt", encoding="utf-8")

            header = dict(meta)
            header["type"] = "header"
            header["version"] = DUMP_VERSION
            header["occlusion_video"] = self.occlusion_path
            self._write(header)

            self._occlusion_writer = cv2.VideoWriter(
                self.occlusion_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(meta.get("fps") or 30.0),
                (int(meta["width"]), int(meta["height"])),
            )
            self._size = (int(meta["height"]), int(meta["width"]))
        except Exception as error:
            self._disable(error)

    def _write(self, payload):
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _disable(self, error):
        print("Geometry dump disabled (%s: %s); the render continues without it."
              % (type(error).__name__, str(error)[:200]))
        self.enabled = False
        self.close()

    def frame(self, index, ramp_value, label_text, ribbon_geometry,
              persons, vehicles, occlusion):
        if not self.enabled:
            return

        try:
            record = {
                "type": "frame",
                "i": int(index),
                "ramp": round(float(ramp_value), 4),
                "label": label_text or "",
                "ribbon": None,
                "persons": [_detection_record(d, include_class=False) for d in persons],
                "vehicles": [_detection_record(d, include_class=True) for d in vehicles],
            }

            if ribbon_geometry is not None:
                record["ribbon"] = {
                    "centre": _round_points(ribbon_geometry["centre"]),
                    "near_v": round(float(ribbon_geometry["near_v"]), 2),
                    "far_v": round(float(ribbon_geometry["far_v"]), 2),
                }

            self._write(record)

            if self._occlusion_writer is not None:
                if occlusion is None:
                    gray = np.zeros(self._size, dtype=np.uint8)
                else:
                    gray = np.clip(occlusion * 255.0, 0, 255).astype(np.uint8)

                self._occlusion_writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        except Exception as error:
            self._disable(error)

    def close(self):
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

        if self._occlusion_writer is not None:
            try:
                self._occlusion_writer.release()
            except Exception:
                pass
            self._occlusion_writer = None
