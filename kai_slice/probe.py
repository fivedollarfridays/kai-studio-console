"""kai_slice.probe — ffprobe reads (duration, fps, frame count, recorded creation time).

ffprobe comes from PATH. ``creation_time`` is the clip's RECORDED creation
time (format or stream tag), the decider's tie-break key — never file mtime,
which a verified copy rewrites.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kai_slice.media import local_path


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags=creation_time:stream=r_frame_rate,width,height,codec_type,nb_frames,duration",
            "-of",
            "json",
            local_path(str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout
    return json.loads(out)


def duration(path: Path) -> float:
    """Clip length in seconds."""
    return float(_ffprobe(path)["format"]["duration"])


def creation_time(path: Path) -> str:
    """The clip's recorded creation time, or '' when it carries none."""
    data = _ffprobe(path)
    tags = data.get("format", {}).get("tags", {})
    if "creation_time" in tags:
        return str(tags["creation_time"])
    for stream in data.get("streams", []):
        if "creation_time" in stream.get("tags", {}):
            return str(stream["tags"]["creation_time"])
    return ""


def video_fps(path: Path) -> float:
    """The clip's video frame rate, or 30.0 when there is no video stream."""
    for stream in _ffprobe(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return _rate(stream)
    return 30.0


def _rate(stream: dict) -> float:
    num, _, den = stream.get("r_frame_rate", "30/1").partition("/")
    return float(num) / float(den or 1)


def video_frame_count(path: Path) -> int:
    """Whole frames in the clip's FIRST video stream: its ``nb_frames``, else
    its stream duration (else the format's) times its rate, floored -- the
    video can end before the audio, so the format duration overstates it.
    0 when there is no video stream."""
    data = _ffprobe(path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        if str(stream.get("nb_frames", "")).isdigit():
            return int(stream["nb_frames"])
        seconds = stream.get("duration") or data.get("format", {}).get("duration", 0)
        return int(float(seconds) * _rate(stream) + 1e-6)
    return 0


def video_dimensions(path: Path) -> tuple[int, int]:
    """The clip's video (width, height) in pixels, or (0, 0) when there is
    no video stream."""
    for stream in _ffprobe(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    return 0, 0


__all__ = ["creation_time", "duration", "video_dimensions", "video_fps", "video_frame_count"]
