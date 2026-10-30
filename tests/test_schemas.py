"""The console payloads the run emits validate against the vendored schemas."""

from __future__ import annotations

from kai_slice import console


def _manifest():
    return {
        "source": "card",
        "dest": "ingest",
        "count": 2,
        "bytes": 2_000_000_000,
        "files": [
            {"name": "CLIP_0001.mp4", "bytes": 1_000_000_000, "sha256": "a" * 64},
            {"name": "CLIP_0002.mp4", "bytes": 1_000_000_000, "sha256": "b" * 64},
        ],
    }


def _rows():
    return [
        {
            "stage": "capture",
            "wall_seconds": 0.1,
            "rc": 0,
            "inputs": [],
            "outputs": [],
            "error": None,
        },
        {
            "stage": "transcribe",
            "wall_seconds": 1.2,
            "rc": 0,
            "inputs": [{"path": "a", "sha256": "a" * 64}],
            "outputs": [{"path": "b", "sha256": "b" * 64}],
            "error": None,
        },
        {
            "stage": "decide",
            "wall_seconds": 0.05,
            "rc": 0,
            "inputs": [{"path": "b", "sha256": "b" * 64}],
            "outputs": [{"path": "c", "sha256": "c" * 64}],
            "error": None,
        },
        {
            "stage": "act",
            "wall_seconds": 2.0,
            "rc": 0,
            "inputs": [{"path": "c", "sha256": "c" * 64}],
            "outputs": [{"path": "d", "sha256": "d" * 64}],
            "error": None,
        },
    ]


def test_ingest_payload_validates():
    payload = console.build_ingest(_manifest(), slug="demo", verified_at="2026-10-26T10:00:00Z")
    console.validate(payload, "ingest")
    assert [c["camera"] for c in payload["cards"]] == ["A", "B"]
    # No NAS archive exists in this pipeline, so these are honestly unknown --
    # never hard-coded True or copied from sha256_card.
    for card in payload["cards"]:
        assert card["sha256_match"] is None
        assert card["sha256_nas"] is None
        assert card["audio"]["present"] is None
        assert card["sha256_card"]  # the one hash this pipeline actually has


def test_build_payload_validates_done():
    payload = console.build_build(_rows(), slug="demo")
    console.validate(payload, "build")
    assert payload["status"] == "done"
    assert [s["name"] for s in payload["steps"]] == ["capture", "transcribe", "decide", "act"]


def test_build_payload_validates_refused():
    rows = _rows()[:3]
    rows[-1] = {
        "stage": "decide",
        "wall_seconds": 0.05,
        "rc": 1,
        "inputs": [],
        "outputs": [],
        "error": "line 5 not found in any take, best 0.40 in CLIP_0002.mp4",
    }
    payload = console.build_build(rows, slug="demo")
    console.validate(payload, "build")
    assert payload["status"] == "refused"
    assert payload["stopped_at"] == "decide"
    assert payload["steps"][2]["refusal_reason"].startswith("line 5 not found")


def test_a_capture_refusal_is_its_own_step_not_a_relabeled_transcribe():
    rows = [
        {
            "stage": "capture",
            "wall_seconds": 0.02,
            "rc": 1,
            "inputs": [],
            "outputs": [],
            "error": "at most 3 cameras (A-C); the card produced 4 media files",
        }
    ]
    payload = console.build_build(rows, slug="demo")
    console.validate(payload, "build")
    assert payload["stopped_at"] == "capture"
    assert payload["steps"][0]["name"] == "capture"
    assert payload["steps"][0]["state"] == "refused"
    assert payload["steps"][0]["refusal_reason"].startswith("at most 3 cameras")
    assert [s["state"] for s in payload["steps"][1:]] == ["not run yet"] * 3


def test_too_many_cards_refuses():
    manifest = _manifest()
    manifest["files"] = manifest["files"] + [
        {"name": f"C{i}.mp4", "bytes": 1, "sha256": "e" * 64} for i in range(3)
    ]
    import pytest

    with pytest.raises(console.TooManyCamerasError, match="at most 3 cameras"):
        console.build_ingest(manifest, slug="demo", verified_at="x")
