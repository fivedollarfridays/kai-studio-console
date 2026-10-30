"""Red-first: scripts/make-sample — raw clip -> shipped card (720p, <=25MB,
sha256 manifest, samples/README.md provenance line).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import load_script, requires_espeak, requires_ffmpeg


@pytest.fixture
def stand_in_clip(tmp_path):
    make_stand_in = load_script("make-stand-in-sample")
    clip = tmp_path / "raw.mp4"
    make_stand_in.generate_stand_in(clip, seconds=3.0, resolution="1920x1080")
    return clip


@requires_espeak
@requires_ffmpeg
def test_transcode_under_budget_hits_the_byte_cap_even_when_it_must_retry(tmp_path, stand_in_clip):
    module = load_script("make-sample")
    dest = tmp_path / "CLIP_0001.mp4"

    size = module.transcode_under_budget(
        stand_in_clip, dest, max_bytes=60_000, resolution="640x360", attempts=6
    )

    assert dest.is_file()
    assert size <= 60_000
    assert dest.stat().st_size == size


@requires_espeak
@requires_ffmpeg
def test_transcode_under_budget_raises_after_attempts_exhausted(tmp_path, stand_in_clip):
    module = load_script("make-sample")
    dest = tmp_path / "CLIP_0001.mp4"

    with pytest.raises(RuntimeError):
        module.transcode_under_budget(
            stand_in_clip, dest, max_bytes=1, resolution="640x360", attempts=2
        )


@requires_espeak
@requires_ffmpeg
def test_make_sample_produces_a_card_with_a_matching_manifest(tmp_path, stand_in_clip):
    module = load_script("make-sample")
    samples_dir = tmp_path / "samples"

    result = module.make_sample(
        stand_in_clip,
        samples_dir,
        card_name="card-01",
        provenance="Synthetic stand-in, no real footage.",
        max_bytes=25 * 1024 * 1024,
        resolution="1280x720",
    )

    card_dir = samples_dir / "card-01"
    assert result["card_dir"] == card_dir
    clip = card_dir / "CLIP_0001.mp4"
    assert clip.is_file()
    assert clip.stat().st_size <= 25 * 1024 * 1024

    manifest = json.loads((card_dir / "manifest.json").read_text())
    rec = manifest["files"][0]
    assert rec["name"] == "CLIP_0001.mp4"
    assert rec["sha256"] == hashlib.sha256(clip.read_bytes()).hexdigest()
    assert rec["bytes"] == clip.stat().st_size


def test_update_samples_readme_writes_and_updates_the_provenance_section(tmp_path):
    module = load_script("make-sample")
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()

    module.update_samples_readme(samples_dir, card_name="card-01", provenance="first version")
    text = (samples_dir / "README.md").read_text()
    assert "## card-01" in text
    assert "first version" in text

    module.update_samples_readme(samples_dir, card_name="card-01", provenance="second version")
    text = (samples_dir / "README.md").read_text()
    assert "second version" in text
    assert "first version" not in text
    assert text.count("## card-01") == 1


def test_update_samples_readme_keeps_the_sections_prose_after_the_provenance(tmp_path):
    """The provenance is the section's FIRST paragraph -- the only one a
    re-run replaces. Prose after it (how ./demo finds the card, how to swap
    it) survives, and every other card's section is untouched."""
    module = load_script("make-sample")
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "README.md").write_text(
        "# Samples\n\nProvenance for every sample card in this directory.\n\n"
        "## card-01\n\nold provenance\n\n"
        "`./demo` needs no path argument.\n\nSwapping it needs no code change.\n\n"
        "## card-02\n\ncard-02 provenance\n\ncard-02 notes\n"
    )

    module.update_samples_readme(samples_dir, card_name="card-01", provenance="new provenance")

    text = (samples_dir / "README.md").read_text()
    assert "old provenance" not in text
    assert text.index("## card-01\n\nnew provenance\n\n") < text.index("`./demo` needs no path")
    assert "Swapping it needs no code change.\n\n## card-02" in text
    assert text.endswith("## card-02\n\ncard-02 provenance\n\ncard-02 notes\n")


def _tagged_camera_clip(path) -> None:
    """A clip shaped like a camera original: creation_time, a GPS location tag,
    a tmcd timecode track."""
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=30:d=1",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=1",
         "-metadata", "creation_time=2000-09-16T14:16:45.000000Z",
         "-metadata", "location=+32.7555-097.3308/", "-timecode", "14:03:09;07",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip


def _probe_all(path) -> dict:
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return json.loads(out)


def _tags(probe: dict) -> dict:
    tags = dict(probe["format"].get("tags", {}))
    for stream in probe["streams"]:
        tags.update(stream.get("tags", {}))
    return {k.lower(): v for k, v in tags.items()}


@requires_ffmpeg
def test_transcode_strips_creation_time_location_and_the_timecode_track(tmp_path):
    module = load_script("make-sample")
    source = tmp_path / "camera_original.mp4"
    _tagged_camera_clip(source)
    before = _probe_all(source)
    # the guard can fail: the source really carries every tag we strip
    assert {"creation_time", "location", "timecode"} <= set(_tags(before))
    assert any(s.get("codec_tag_string") == "tmcd" for s in before["streams"])

    dest = tmp_path / "CLIP_0001.mp4"
    module.transcode_under_budget(source, dest, max_bytes=5_000_000, resolution="320x240")

    after = _probe_all(dest)
    tags = _tags(after)
    assert "creation_time" not in tags
    assert not any("location" in k or "gps" in k or "xyz" in k for k in tags)
    assert "timecode" not in tags
    assert sorted(s["codec_type"] for s in after["streams"]) == ["audio", "video"]


def test_update_samples_readme_refuses_a_provenance_with_a_paragraph_break(tmp_path):
    """The provenance is ONE paragraph -- a ``\\n\\n`` in it would be read
    back as the section's prose and survive the next re-run."""
    module = load_script("make-sample")
    with pytest.raises(ValueError, match="one paragraph"):
        module.update_samples_readme(tmp_path, card_name="card-01", provenance="a\n\nb")
    assert not (tmp_path / "README.md").exists()


def test_update_samples_readme_matches_a_card_header_at_eof_without_a_newline(tmp_path):
    module = load_script("make-sample")
    (tmp_path / "README.md").write_text("# Samples\n\n## card-00\n\nzero\n\n## card-01")

    module.update_samples_readme(tmp_path, card_name="card-01", provenance="new provenance")

    text = (tmp_path / "README.md").read_text()
    assert text.count("## card-01") == 1
    assert text.endswith("## card-01\n\nnew provenance\n")
    assert "## card-00\n\nzero\n\n" in text


def test_make_sample_refuses_a_bad_provenance_before_writing_anything(tmp_path, monkeypatch):
    module = load_script("make-sample")
    monkeypatch.setattr(
        module, "transcode_under_budget", lambda *a, **k: pytest.fail("must not transcode")
    )
    with pytest.raises(ValueError, match="one paragraph"):
        module.make_sample(tmp_path / "raw.mp4", tmp_path / "samples", provenance="a\n\nb")
    assert not (tmp_path / "samples").exists()
