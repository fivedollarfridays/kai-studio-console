"""Red-first: a refusal text is scrubbed of this machine's ABSOLUTE paths
only, at a path boundary -- never by raw substring.

The reproduced defect: ``--out o`` turned ``matched no take: "good to go"``
into ``matched n<out> take: "g<out><out>d t<out> g<out>"``.
"""

from __future__ import annotations

from pathlib import Path

from kai_slice import scrub


def _labels(tmp_path: Path, monkeypatch, out: Path, card: Path | None = None) -> dict:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return scrub.scrub_labels(out, card or tmp_path / "card")


def test_a_relative_out_never_mangles_ordinary_words(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = 'decide: matched no take: "good to go"'

    assert scrub.scrub(text, _labels(tmp_path, monkeypatch, Path("o"))) == text


def test_an_absolute_out_path_inside_a_message_is_replaced(tmp_path, monkeypatch):
    out = tmp_path / "o"
    labels = _labels(tmp_path, monkeypatch, out)
    text = f'act: cannot write {out.resolve()}/cut.mp4 (see "{out.resolve()}")'

    assert scrub.scrub(text, labels) == 'act: cannot write <out>/cut.mp4 (see "<out>")'


def test_a_path_only_sharing_a_prefix_is_not_replaced(tmp_path, monkeypatch):
    out = tmp_path / "o"
    text = f"cannot read {tmp_path.resolve()}/other/x and {out.resolve()}x"

    assert scrub.scrub(text, _labels(tmp_path, monkeypatch, out)) == text


def test_a_path_outside_home_under_the_repo_root_is_replaced(tmp_path, monkeypatch):
    # A clone at /srv/... (or on a Windows drive under WSL) is not under $HOME.
    labels = _labels(tmp_path, monkeypatch, tmp_path / "o")
    assert not str(scrub.ROOT).startswith(str(tmp_path / "home"))
    text = f"capture: cannot read {scrub.ROOT}/samples/card-01/CLIP_0001.mp4"

    assert scrub.scrub(text, labels) == "capture: cannot read <root>/samples/card-01/CLIP_0001.mp4"


def test_the_models_root_is_replaced(tmp_path, monkeypatch):
    labels = _labels(tmp_path, monkeypatch, tmp_path / "o")
    text = f"transcribe: no model at {scrub.MODELS_ROOT}/small"

    assert scrub.scrub(text, labels) == "transcribe: no model at <models>/small"


def test_a_relative_out_is_never_a_key_even_as_a_whole_word(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "decide: cue o matched nothing (o)"

    assert scrub.scrub(text, _labels(tmp_path, monkeypatch, Path("o"))) == text


def test_a_match_glued_to_a_preceding_path_is_not_replaced(tmp_path, monkeypatch):
    out = tmp_path / "o"
    text = f"cannot copy to /backup{out.resolve()}/cut.mp4"

    assert scrub.scrub(text, _labels(tmp_path, monkeypatch, out)) == text


def test_an_absolute_out_path_followed_by_punctuation_is_replaced(tmp_path, monkeypatch):
    out = tmp_path / "o"
    labels = _labels(tmp_path, monkeypatch, out)
    o = out.resolve()

    assert scrub.scrub(f"cannot read {o}: gone", labels) == "cannot read <out>: gone"
    assert scrub.scrub(f"wrote {o}, then", labels) == "wrote <out>, then"
    assert scrub.scrub(f"(see {o}) and [{o}];", labels) == "(see <out>) and [<out>];"


def test_a_relative_or_bare_word_before_a_colon_is_never_replaced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "decide: o: no take for o, (o) [o];"

    assert scrub.scrub(text, _labels(tmp_path, monkeypatch, Path("o"))) == text
