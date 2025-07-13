import cv2
import subprocess
import json
from pathlib import Path

# List of video files
video_files = [
    Path("data/abby_road_view.mp4"),
    Path("data/parking_view.mp4"),
    Path("data/street_view.mp4"),
    Path("data/street_view2.mp4"),
    Path("data/superland_parking_view.mp4")
]

for video_path in video_files:
    print(f"===== {video_path} =====")

    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        continue

    # Get basic metadata from OpenCV
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    print(f"Resolution      : {width} x {height}")
    print(f"Frame rate      : {fps:.2f} FPS")
    print(f"Frame count     : {frame_count}")
    print(f"Duration        : {duration:.2f} seconds")

    # Try to get codec and pixel format using ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,codec_long_name,pix_fmt",
            "-of", "json", str(video_path)
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        codec_name = stream.get("codec_name", "N/A")
        codec_long = stream.get("codec_long_name", "N/A")
        pix_fmt = stream.get("pix_fmt", "N/A")
        print(f"Codec           : {codec_name} ({codec_long})")
        print(f"Pixel format    : {pix_fmt}")

        # Container format
        cmd_fmt = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=format_name",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        fmt = subprocess.check_output(cmd_fmt, text=True).strip()
        print(f"Container format: {fmt}")
    except Exception as e:
        print(f"ffprobe not available or error: {e}")

    cap.release()
    print()
