import os
import sys

import policy_demo


VIDEO_ID = "TuCsyBF3nHU"


def main():
    print("VIDEO_BASE_URL:", policy_demo.VIDEO_BASE_URL)
    print("SOURCE_VIDEO_DIR:", policy_demo.SOURCE_VIDEO_DIR)

    os.makedirs(policy_demo.SOURCE_VIDEO_DIR, exist_ok=True)

    print("Looking for or downloading video:", VIDEO_ID)

    local_path = policy_demo.find_source_video(VIDEO_ID)

    if local_path:
        print("Video ready:", local_path)
        return

    print("Download failed or video not found.")
    sys.exit(1)


if __name__ == "__main__":
    main()
