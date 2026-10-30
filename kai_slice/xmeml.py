"""kai_slice.xmeml — FCP7 XMEML (.xml) emitter.

Re-homed verbatim from the source project (only imports and docstrings changed).
Premiere Pro imports the legacy FCP7 XMEML interchange, not FCP-X .fcpxml;
this module turns a frame-based :class:`Timeline` into that XML so an
auto-assembled edit opens natively. Timing is integer frames at the sequence
timebase (e.g. 30 = 30fps non-NTSC).
"""

from __future__ import annotations

import itertools
import uuid as _uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from xml.sax.saxutils import escape, quoteattr


@dataclass(frozen=True)
class MediaFile:
    """A source media file Premiere will relink by ``path_url``."""

    file_id: str
    name: str
    path_url: str
    duration_frames: int
    has_video: bool = True
    has_audio: bool = False
    width: int = 0
    height: int = 0
    audio_rate: int = 48000
    audio_depth: int = 16


@dataclass(frozen=True)
class Clip:
    """One clip placed on a track: source in/out mapped to a sequence window."""

    name: str
    file: MediaFile
    start: int
    end: int
    in_frame: int = 0
    out_frame: int = 0
    link_id: str | None = None


@dataclass(frozen=True)
class Marker:
    """A sequence marker at a frame position."""

    name: str
    frame: int
    comment: str = ""


@dataclass(frozen=True)
class Timeline:
    """A frame-based timeline: ordered video/audio tracks plus markers."""

    name: str
    timebase: int
    width: int
    height: int
    duration_frames: int
    video_tracks: list[list[Clip]] = field(default_factory=list)
    audio_tracks: list[list[Clip]] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)


def _rate(timebase: int) -> str:
    return f"<rate><timebase>{timebase}</timebase><ntsc>FALSE</ntsc></rate>"


def _video_format(tl: Timeline) -> str:
    return (
        "<format><samplecharacteristics>"
        f"{_rate(tl.timebase)}<width>{tl.width}</width><height>{tl.height}</height>"
        "<anamorphic>FALSE</anamorphic><pixelaspectratio>square</pixelaspectratio>"
        "<fielddominance>none</fielddominance><colordepth>24</colordepth>"
        "</samplecharacteristics></format>"
    )


def _file_xml(mf: MediaFile, timebase: int) -> str:
    media = ""
    if mf.has_video:
        dims = f"<width>{mf.width}</width><height>{mf.height}</height>" if mf.width else ""
        media += (
            f"<video><samplecharacteristics>{_rate(timebase)}{dims}</samplecharacteristics></video>"
        )
    if mf.has_audio:
        # channelcount 1: mono source assumption (see _clip_xml sourcetrack note)
        media += (
            f"<audio><samplecharacteristics><depth>{mf.audio_depth}</depth>"
            f"<samplerate>{mf.audio_rate}</samplerate></samplecharacteristics>"
            "<channelcount>1</channelcount></audio>"
        )
    # file_id lands in an attribute, so quoteattr() (quotes + &<>); text uses escape()
    return (
        f"<file id={quoteattr(mf.file_id)}><name>{escape(mf.name)}</name>"
        f"<pathurl>{escape(mf.path_url)}</pathurl>{_rate(timebase)}"
        f"<duration>{mf.duration_frames}</duration><media>{media}</media></file>"
    )


@dataclass(frozen=True)
class _Placed:
    """A clip with its assigned clipitem id + position (for cross-track links)."""

    clip: Clip
    clipid: str
    mediatype: str
    trackindex: int
    clipindex: int


def _place_tracks(
    tracks: list[list[Clip]], mediatype: str, counter: Iterator[int]
) -> list[list[_Placed]]:
    """Assign each clip its clipitem id + position, preserving track structure."""
    placed: list[list[_Placed]] = []
    for ti, track in enumerate(tracks, start=1):
        row = [
            _Placed(clip, f"clipitem-{next(counter)}", mediatype, ti, ci)
            for ci, clip in enumerate(track, start=1)
        ]
        placed.append(row)
    return placed


def _link_groups(placed: list[_Placed]) -> dict[str, list[_Placed]]:
    """Bucket placed clips by ``link_id`` (clips that move together)."""
    groups: dict[str, list[_Placed]] = {}
    for p in placed:
        if p.clip.link_id is not None:
            groups.setdefault(p.clip.link_id, []).append(p)
    return groups


def _links_xml(members: list[_Placed]) -> str:
    """One ``<link>`` per group member (each clipitem lists the whole group)."""
    return "".join(
        f"<link><linkclipref>{m.clipid}</linkclipref>"
        f"<mediatype>{m.mediatype}</mediatype>"
        f"<trackindex>{m.trackindex}</trackindex>"
        f"<clipindex>{m.clipindex}</clipindex></link>"
        for m in members
    )


def _clip_xml(p: _Placed, timebase: int, seen: set[str], links: str) -> str:
    clip = p.clip
    out = clip.out_frame or clip.in_frame + (clip.end - clip.start)
    if clip.file.file_id in seen:
        file_xml = f"<file id={quoteattr(clip.file.file_id)}/>"
    else:
        file_xml = _file_xml(clip.file, timebase)
        seen.add(clip.file.file_id)
    source = (
        "<sourcetrack><mediatype>audio</mediatype><trackindex>1</trackindex></sourcetrack>"
        if p.mediatype == "audio"
        else ""
    )
    return (
        f'<clipitem id="{p.clipid}"><name>{escape(clip.name)}</name>'
        f"<enabled>TRUE</enabled><duration>{clip.end - clip.start}</duration>"
        f"{_rate(timebase)}<start>{clip.start}</start><end>{clip.end}</end>"
        f"<in>{clip.in_frame}</in><out>{out}</out>{file_xml}{source}{links}</clipitem>"
    )


def _tracks_xml(
    placed_tracks: list[list[_Placed]],
    timebase: int,
    seen: set[str],
    groups: dict[str, list[_Placed]],
) -> str:
    out = []
    for track in placed_tracks:
        items = "".join(
            _clip_xml(
                p,
                timebase,
                seen,
                _links_xml(groups.get(p.clip.link_id, []) if p.clip.link_id is not None else []),
            )
            for p in track
        )
        out.append(f"<track>{items}</track>")
    return "".join(out)


def _markers_xml(markers: list[Marker]) -> str:
    return "".join(
        f"<marker><name>{escape(m.name)}</name>"
        f"<comment>{escape(m.comment)}</comment>"
        f"<in>{m.frame}</in><out>-1</out></marker>"
        for m in markers
    )


def emit_xmeml(tl: Timeline) -> str:
    """Render ``tl`` as an FCP7 XMEML document string."""
    counter = itertools.count(1)
    seen: set[str] = set()
    v_placed = _place_tracks(tl.video_tracks, "video", counter)
    a_placed = _place_tracks(tl.audio_tracks, "audio", counter)
    groups = _link_groups([p for track in (*v_placed, *a_placed) for p in track])
    video = _tracks_xml(v_placed, tl.timebase, seen, groups)
    audio = _tracks_xml(a_placed, tl.timebase, seen, groups)
    seq_uuid = _uuid.uuid5(_uuid.NAMESPACE_DNS, tl.name)
    body = (
        f'<sequence id="sequence-1"><uuid>{seq_uuid}</uuid>'
        f"<name>{escape(tl.name)}</name>"
        f"<duration>{tl.duration_frames}</duration>{_rate(tl.timebase)}"
        f"<timecode>{_rate(tl.timebase)}<string>00:00:00:00</string>"
        "<frame>0</frame><displayformat>NDF</displayformat></timecode>"
        f"<media><video>{_video_format(tl)}{video}</video>"
        f"<audio>{audio}</audio></media>"
        f"{_markers_xml(tl.markers)}"
        "</sequence>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE xmeml>\n"
        f'<xmeml version="5">{body}</xmeml>\n'
    )


__all__ = ["Clip", "Marker", "MediaFile", "Timeline", "emit_xmeml"]
