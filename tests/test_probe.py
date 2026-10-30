"""Red-first: probe.video_dimensions reads the media's real width/height --
pinned against a NON-default resolution (320x240) so a hard-coded 160x120
fallback in a caller can never pass this test by accident.
"""

from __future__ import annotations

import subprocess

from conftest import requires_ffmpeg

from kai_slice import probe


@requires_ffmpeg
def test_video_dimensions_reads_a_non_default_resolution(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )

    assert probe.video_dimensions(clip) == (320, 240)
