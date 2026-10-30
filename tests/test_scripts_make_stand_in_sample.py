"""Red-first: scripts/make-stand-in-sample — the committed, free, no-download
generator (espeak-ng speech over an ffmpeg lavfi picture) that stands in for
the owner's not-yet-delivered clip.
"""

from __future__ import annotations

import json
import subprocess

from conftest import load_script, requires_espeak, requires_ffmpeg


def _ffprobe(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=width,height,codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


@requires_espeak
@requires_ffmpeg
def test_generate_stand_in_produces_a_clip_of_the_requested_length_and_size(tmp_path):
    module = load_script("make-stand-in-sample")
    out = tmp_path / "stand-in.mp4"

    result = module.generate_stand_in(out, seconds=3.0, resolution="640x360")

    assert result == out
    assert out.is_file()
    probe = _ffprobe(out)
    assert abs(float(probe["format"]["duration"]) - 3.0) < 0.5
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (video["width"], video["height"]) == (640, 360)
    assert audio["codec_type"] == "audio"


@requires_espeak
@requires_ffmpeg
def test_generate_stand_in_pads_short_speech_to_the_target_duration(tmp_path):
    module = load_script("make-stand-in-sample")
    out = tmp_path / "stand-in.mp4"

    module.generate_stand_in(out, seconds=5.0, resolution="320x240", text="hi")

    probe = _ffprobe(out)
    assert abs(float(probe["format"]["duration"]) - 5.0) < 0.5


def test_default_text_is_non_empty_and_has_no_private_paths():
    # Needles built from fragments on purpose (see test_scrub.py) so this
    # file does not itself trip the private-path scrub it is checking for.
    module = load_script("make-stand-in-sample")
    assert module.DEFAULT_TEXT.strip()
    assert "E:" + "/" not in module.DEFAULT_TEXT
    assert "/" + "home/" not in module.DEFAULT_TEXT


@requires_espeak
@requires_ffmpeg
def test_cli_main_writes_to_the_given_out_path(tmp_path):
    module = load_script("make-stand-in-sample")
    out = tmp_path / "cli-stand-in.mp4"

    rc = module.main(["--out", str(out), "--seconds", "2", "--resolution", "320x240"])

    assert rc == 0
    assert out.is_file()
