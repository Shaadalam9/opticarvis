r"""Fetch every missing source video before a batch run.

The batch downloads sources inline, one at a time, as each city reaches its
prepare step -- measured at 16-19 MB/s that is ~1.7 hours of wall clock spent
serially inside the run. Prefetching moves it in front (or overnight):

    .venv/bin/python scripts/prefetch_source_videos.py [--workers 3]

Reads the job list the batch will use (clip_jobs.jsonl, built by
clip_job_builder.py), fetches each missing video with the batch runner's own
download function -- same aliases, same atomic .part rename -- and reports a
tally. Videos already on disk are skipped, so re-running is free.

Run it from the repository root; it needs the `config`/`secret` files there.
"""

import argparse
import os
import sys
import threading

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    import batch_corrected_pipeline as batch  # noqa: E402
except SystemExit:
    # common.py exits when the `config` file is absent; say what to do about it
    # instead of dying with a bare exit code.
    print("")
    print("This script needs the `config` and `secret` files in the repository")
    print("root (cp default.config config && cp default.secret secret, then")
    print("fill in the credentials).")
    raise


def missing_video_ids(jobs_path):
    import json

    seen = []

    with open(jobs_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()

            if not text:
                continue

            job = json.loads(text)
            video_id = str(job.get("video_id", ""))

            if not video_id or video_id in seen:
                continue

            seen.append(video_id)

    present = []
    missing = []

    for video_id in seen:
        for ext in batch.VIDEO_EXTENSIONS:
            if os.path.isfile(os.path.join(batch.SOURCE_VIDEO_DIR, video_id + ext)):
                present.append(video_id)
                break
        else:
            missing.append(video_id)

    return present, missing


def main():
    parser = argparse.ArgumentParser(description="Prefetch missing source videos.")
    parser.add_argument(
        "--jobs-jsonl",
        default=os.path.join(batch.WORKFLOW_OUTPUTS, "clip_jobs.jsonl"),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isfile(args.jobs_jsonl):
        print("No job list at %s -- run src/clip_job_builder.py first." % args.jobs_jsonl)
        return 1

    present, missing = missing_video_ids(args.jobs_jsonl)

    print("")
    print("Source video prefetch")
    print("=====================")
    print("videos referenced:", len(present) + len(missing))
    print("already on disk:  ", len(present))
    print("to download:      ", len(missing))

    if not missing:
        return 0

    lock = threading.Lock()
    queue = list(missing)
    results = {"ok": 0, "failed": []}

    def worker():
        while True:
            with lock:
                if not queue:
                    return

                video_id = queue.pop(0)

            # A transient network error must cost one video, not the thread --
            # a dead worker would silently strand its share of the queue.
            try:
                path = batch.download_source_video_from_ftp(video_id)
            except Exception as error:
                print("Download error for %s: %s %s"
                      % (video_id, type(error).__name__, str(error)[:200]))
                path = None

            with lock:
                if path:
                    results["ok"] += 1
                else:
                    results["failed"].append(video_id)

    threads = [
        threading.Thread(target=worker)
        for _ in range(max(1, min(args.workers, len(missing))))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("")
    print("downloaded:", results["ok"], "| failed:", len(results["failed"]))

    for video_id in results["failed"]:
        print("  not found on the file server:", video_id)

    return 1 if results["failed"] and not results["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
