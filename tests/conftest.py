"""Make the public tree's ``kai_slice`` package importable regardless of the
pytest rootdir (this tree carries no pytest config of its own, so it can be
run standalone or collected from the kai-studio root), plus ffmpeg-gated
fixtures for the lavfi card the end-to-end tests build.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)
requires_espeak = pytest.mark.skipif(
    shutil.which("espeak-ng") is None, reason="espeak-ng not on PATH"
)


def load_script(name: str) -> ModuleType:
    """Load an extensionless ``scripts/<name>`` (or top-level ``demo``) entry
    point as a module, for direct unit testing of its ``main``/helpers."""
    path = _PKG_ROOT / "scripts" / name if name != "demo" else _PKG_ROOT / "demo"
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_") + "_script", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


#: The bytes a fake downloader writes, and the pinned-hash table that matches
#: them -- the ``pinned_sha256=`` seam keeps fetch_model tests network-free.
FAKE_MODEL_FILES = {"model.bin": b"fake-weights", "config.json": b"{}"}


def fake_pins(revision: str, files: dict[str, bytes] = FAKE_MODEL_FILES) -> dict:
    import hashlib

    return {revision: {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}


def write_fake_model(output_dir, files: dict[str, bytes] = FAKE_MODEL_FILES) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return out


def _lavfi_clip(path: Path, *, seconds: float, color: str, freq: int, creation: str) -> None:
    """A tiny solid-colour clip with a sine tone and a stamped creation_time."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=160x120:r=30:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={seconds}",
            "-metadata",
            f"creation_time={creation}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _stt_doc(sentences: list[str], source: str) -> dict:
    segments = []
    t = 0.0
    for sentence in sentences:
        words = []
        seg_start = t
        for surface in sentence.split():
            words.append({"word": surface, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t = round(t + 0.5, 3)
        segments.append(
            {"start": seg_start, "end": round(t - 0.1, 3), "text": sentence, "words": words}
        )
        t = round(t + 1.0, 3)
    return {"file": source, "duration": t, "model": "fake", "segments": segments}


@pytest.fixture
def lavfi_card(tmp_path):
    """A card dir of two lavfi clips plus a replay dir of fake transcripts.

    Returns ``(card_dir, replay_dir)``. The replay dir is handed to the CLI
    through ``KAI_SLICE_STT_REPLAY_DIR`` so no model is needed.
    """
    card = tmp_path / "card"
    replay = tmp_path / "replay"
    card.mkdir()
    replay.mkdir()
    clips = {
        "CLIP_0001.mp4": (
            ["hello there how are you doing", "i think the honest answer is yes"],
            "red",
            220,
        ),
        "CLIP_0002.mp4": (
            ["i am doing great thanks for asking", "no i strongly disagree with that"],
            "blue",
            440,
        ),
    }
    for i, (name, (sentences, color, freq)) in enumerate(clips.items()):
        _lavfi_clip(
            card / name,
            seconds=12.0,
            color=color,
            freq=freq,
            creation=f"2026-10-26T10:0{i}:00.000000Z",
        )
        stem = name.split(".")[0]
        (replay / f"{stem}.stt.json").write_text(json.dumps(_stt_doc(sentences, name)))
    return card, replay
