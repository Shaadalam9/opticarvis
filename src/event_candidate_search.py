"""
Event based candidate clip search for OptiCarVis.

video -> cheap event scoring -> candidate windows -> Alpamayo -> Gemma
"""

import cv2
import os


def score_frame_change(previous_gray, current_gray):
    if previous_gray is None:
        return 0.0

    diff = cv2.absdiff(previous_gray, current_gray)
    return float(diff.mean()) / 255.0


def sample_video_events(video_path, sample_step_s=5.0):
    if not os.path.isfile(video_path):
        return []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        cap.release()
        return []

    frame_step = max(1, int(fps * sample_step_s))

    previous = None
    events = []
    frame_index = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_index % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            events.append(
                {
                    "time_s": frame_index / fps,
                    "score": score_frame_change(previous, gray),
                }
            )

            previous = gray

        frame_index += 1

    cap.release()

    return events


def generate_refined_windows(event_time, clip_length_s=30.0):
    offsets = [-15, -10, -5, 0, 5, 10]

    windows = []

    for offset in offsets:
        start = max(0, event_time + offset)

        windows.append(
            {
                "start_s": round(start, 2),
                "end_s": round(start + clip_length_s, 2),
            }
        )

    return windows


def select_candidate_windows(video_path, clip_length_s=30.0, max_events=10):
    events = sample_video_events(video_path)

    events = sorted(
        events,
        key=lambda x: x["score"],
        reverse=True,
    )

    candidates = []

    for event in events[:max_events]:
        candidates.extend(
            generate_refined_windows(
                event["time_s"],
                clip_length_s,
            )
        )

    unique = {}

    for candidate in candidates:
        key = (
            candidate["start_s"],
            candidate["end_s"],
        )
        unique[key] = candidate

    return list(unique.values())