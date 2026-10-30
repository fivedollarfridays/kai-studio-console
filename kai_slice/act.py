"""kai_slice.act — render the decided spans (stage 4).

One ffmpeg filter graph trims every span and concatenates them into
``<out>/cut.mp4`` (a single graph, so the cut's duration is the exact sum of
the span lengths, not a drifting concat of re-encoded parts), plus an FCP7
XMEML edit list ``<out>/edit.xml``. ffmpeg comes from PATH.

Each span is first snapped OUTWARD to its own clip's frame grid (start floors,
end ceils): a video trim can only stop on a frame, and rounding inward would
eat the padded air after the last word. The snapped end is then clamped to
that clip's last whole VIDEO frame -- the transcript's duration is the
format/audio length, which can run past the last frame -- and every span
keeps at least one frame.

Snapping outward can make two adjacent spans of one clip overlap by under a
frame (the decider meets two back-to-back spans on one instant, and the snap
then ceils the first's end and floors the second's start past it): the overlap repeats at most one frame of source, never
drops one, so it is left as is.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from pathlib import Path

from kai_slice import probe, xmeml
from kai_slice.decide import DecideError

_GRID_EPS = 1e-6


def _snap_to_frames(
    spans: list[dict],
    fps: float | Mapping[str, float],
    frame_limits: dict[str, int] | None = None,
) -> list[dict]:
    """Every span widened to whole frames (start floors, end ceils; a bound
    already on the grid is unchanged), its end clamped to
    ``frame_limits[clip]`` -- the clip's video frame count -- when given.
    ``fps`` is one rate for every span, or each clip's own rate by name."""
    snapped = []
    for s in spans:
        rate = fps[s["clip"]] if isinstance(fps, Mapping) else fps
        first = max(math.floor(s["start"] * rate + _GRID_EPS), 0)
        last = math.ceil(s["end"] * rate - _GRID_EPS)
        limit = (frame_limits or {}).get(s["clip"])
        if limit:
            first = min(first, limit - 1)
            last = max(min(last, limit), first + 1)
        snapped.append({**s, "start": round(first / rate, 6), "end": round(last / rate, 6)})
    return snapped


def rendered_bounds(spans: list[dict], name_to_path: dict[str, str]) -> list[dict]:
    """``spans`` as act renders them: validated, each snapped to ITS OWN
    clip's frame grid and clamped to that clip's last whole video frame."""
    if not spans:
        raise DecideError("act: no spans to render")
    _validate_span_timings(spans)
    clips = {s["clip"] for s in spans}
    rates = {name: probe.video_fps(Path(name_to_path[name])) for name in clips}
    limits = {name: probe.video_frame_count(Path(name_to_path[name])) for name in clips}
    return _snap_to_frames(spans, rates, limits)


def _unique_inputs(
    spans: list[dict], name_to_path: dict[str, str]
) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    index: dict[str, int] = {}
    for s in spans:
        if s["clip"] not in index:
            index[s["clip"]] = len(order)
            order.append(name_to_path[s["clip"]])
    return order, index


def _filter_graph(spans: list[dict], index: dict[str, int]) -> str:
    parts = []
    for k, s in enumerate(spans):
        c = index[s["clip"]]
        parts.append(
            f"[{c}:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS[v{k}];"
            f"[{c}:a]atrim=start={s['start']}:end={s['end']},asetpts=PTS-STARTPTS[a{k}];"
        )
    labels = "".join(f"[v{k}][a{k}]" for k in range(len(spans)))
    parts.append(f"{labels}concat=n={len(spans)}:v=1:a=1[v][a]")
    return "".join(parts)


def _ffmpeg_concat(inputs: list[str], graph: str, out: Path) -> None:
    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += [
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _edit_xml(
    spans: list[dict], name_to_path: dict[str, str], fps: float, width: int, height: int
) -> str:
    timebase = max(round(fps), 1)
    files = {}
    for i, (name, path) in enumerate(sorted(name_to_path.items())):
        file_width, file_height = probe.video_dimensions(Path(path))
        files[name] = xmeml.MediaFile(
            file_id=f"file-{i}",
            name=name,
            path_url=Path(path).resolve().as_uri(),
            duration_frames=max(probe.video_frame_count(Path(path)), 1),
            has_video=True,
            has_audio=True,
            width=file_width,
            height=file_height,
        )
    v_track: list[xmeml.Clip] = []
    a_track: list[xmeml.Clip] = []
    cursor = 0
    for k, s in enumerate(spans):
        length = max(round((s["end"] - s["start"]) * fps), 1)
        in_frame = round(s["start"] * fps)
        mf = files[s["clip"]]
        common = {
            "file": mf,
            "start": cursor,
            "end": cursor + length,
            "in_frame": in_frame,
            "out_frame": in_frame + length,
            "link_id": f"span-{k}",
        }
        v_track.append(xmeml.Clip(name=f"{s['clip']}#{k}", **common))
        a_track.append(xmeml.Clip(name=f"{s['clip']}#{k}", **common))
        cursor += length
    tl = xmeml.Timeline(
        name="kai_slice-cut",
        timebase=timebase,
        width=width,
        height=height,
        duration_frames=cursor,
        video_tracks=[v_track],
        audio_tracks=[a_track],
    )
    return xmeml.emit_xmeml(tl)


def _validate_span_timings(spans: list[dict]) -> None:
    """Every span's ``start``/``end`` must be finite, non-negative floats
    with ``end >= start`` -- caught here, named, before either is formatted
    into the ffmpeg filter graph string."""
    for s in spans:
        for key in ("start", "end"):
            value = s[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DecideError(f"invalid span timing: clip={s['clip']!r} {key}={value!r}")
            if not math.isfinite(value) or value < 0:
                raise DecideError(f"invalid span timing: clip={s['clip']!r} {key}={value!r}")
        if s["end"] < s["start"]:
            raise DecideError(
                f"invalid span timing: clip={s['clip']!r} end {s['end']} < start {s['start']}"
            )


def act_stage(spans: list[dict], name_to_path: dict[str, str], out_dir: Path) -> dict:
    """Render ``cut.mp4`` and ``edit.xml`` from decided spans."""
    spans = rendered_bounds(spans, name_to_path)
    out_dir = Path(out_dir)
    cut = out_dir / "cut.mp4"
    edit = out_dir / "edit.xml"
    inputs, index = _unique_inputs(spans, name_to_path)
    fps = probe.video_fps(Path(inputs[0]))
    _ffmpeg_concat(inputs, _filter_graph(spans, index), cut)
    width, height = probe.video_dimensions(Path(inputs[0]))
    edit.write_text(_edit_xml(spans, name_to_path, fps, width, height))
    return {
        "cut": str(cut),
        "edit": str(edit),
        "spans_total_s": sum(s["end"] - s["start"] for s in spans),
    }


__all__ = ["act_stage", "rendered_bounds"]
