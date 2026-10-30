"""Red-first: the real STT decode path, no replay, no model.

``tests/test_run_e2e.py`` and friends all set ``KAI_SLICE_STT_REPLAY_DIR``,
so ``faster_whisper.audio.decode_audio`` -- the function that actually calls
``av.open`` -- never runs in the suite. That gap is how an incompatible
``av`` resolved by an unpinned ``faster-whisper>=1.0,<2`` (``av`` 19 dropped
the ``metadata_errors`` kwarg ``decode_audio`` passes) shipped to every judge
running a fresh ``pip install -e .``: the suite stayed green while ``./demo``
crashed. This test needs no model -- decoding audio into samples is a
separate step from transcribing them -- so it is fast and runs every time,
unlike a real transcription.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "card-01" / "CLIP_0001.mp4"


def test_decode_audio_runs_on_the_committed_sample_with_no_replay_and_no_model():
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(_SAMPLE), sampling_rate=16000)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] > 0
