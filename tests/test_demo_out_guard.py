"""Red-first: ./demo's recursive delete of a previous ``--out`` is guarded.

Each refusal test plants a VALID marker (identifying content) plus a sentinel
file, so the only thing standing between the sentinel and ``rmtree`` is the
guard under test -- remove that guard and the sentinel is deleted (or, for
paths the test cannot write into, the refusal message stops matching).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import load_script


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.delenv("KAI_SLICE_STT_REPLAY_DIR", raising=False)
    return load_script("demo")


def _marked(module, path: Path, name: str = "keep.txt") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / module.MARKER_NAME).write_text(module.MARKER_TOKEN)
    sentinel = path / name
    sentinel.write_text("keep me")
    return sentinel


def _replay(tmp_path: Path, monkeypatch) -> None:
    # Any replay dir gets run_demo past its model check and onto the out dir.
    monkeypatch.setenv("KAI_SLICE_STT_REPLAY_DIR", str(tmp_path / "replay"))


def test_missing_model_refusal_leaves_previous_marked_output_intact(demo, tmp_path):
    # A wholly-owned previous output (marker + a demo entry), so only the
    # model check stands between it and the replace.
    sentinel = _marked(demo, tmp_path / "demo-out", "decision.json")

    with pytest.raises(RuntimeError, match="model not found"):
        demo.run_demo(out=tmp_path / "demo-out", models_root=tmp_path / "models")

    assert sentinel.read_text() == "keep me"


def test_refuses_a_symlinked_out(demo, tmp_path, monkeypatch):
    sentinel = _marked(demo, tmp_path / "real")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real", target_is_directory=True)
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="symlink"):
        demo.run_demo(out=link, models_root=tmp_path / "models")

    assert sentinel.read_text() == "keep me"


def test_refuses_a_symlinked_marker(demo, tmp_path, monkeypatch):
    out = tmp_path / "demo-out"
    out.mkdir()
    real_marker = tmp_path / "elsewhere"
    real_marker.write_text(demo.MARKER_TOKEN)
    (out / demo.MARKER_NAME).symlink_to(real_marker)
    (out / "keep.txt").write_text("keep me")
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="symlink"):
        demo.run_demo(out=out, models_root=tmp_path / "models")

    assert (out / "keep.txt").read_text() == "keep me"


def test_refuses_a_marker_whose_content_does_not_identify_this_script(demo, tmp_path, monkeypatch):
    out = tmp_path / "demo-out"
    out.mkdir()
    (out / demo.MARKER_NAME).touch()  # empty: e.g. a copied tree, or hand-made
    (out / "keep.txt").write_text("keep me")
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="does not identify"):
        demo.run_demo(out=out, models_root=tmp_path / "models")

    assert (out / "keep.txt").read_text() == "keep me"


def test_refuses_cwd(demo, tmp_path, monkeypatch):
    sentinel = _marked(demo, tmp_path / "here")
    monkeypatch.chdir(tmp_path / "here")
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="protected"):
        demo.run_demo(out=Path("."), models_root=tmp_path / "models")

    assert sentinel.read_text() == "keep me"


def test_refuses_home(demo, tmp_path, monkeypatch):
    sentinel = _marked(demo, tmp_path / "home")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="protected"):
        demo.run_demo(out=Path("~").expanduser(), models_root=tmp_path / "models")

    assert sentinel.read_text() == "keep me"


def test_refuses_an_ancestor_of_home(demo, tmp_path, monkeypatch):
    sentinel = _marked(demo, tmp_path / "users")
    (tmp_path / "users" / "me").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "users" / "me"))
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="protected"):
        demo.run_demo(out=tmp_path / "users", models_root=tmp_path / "models")

    assert sentinel.read_text() == "keep me"


@pytest.mark.parametrize("which", ["root", "parent-of-root", "filesystem-root"])
def test_refuses_root_its_ancestors_and_slash(demo, tmp_path, monkeypatch, which):
    """These paths cannot carry a planted marker, so the guard is proven by
    its own message: without it they would refuse only as "non-empty"."""
    target = {"root": demo.ROOT, "parent-of-root": demo.ROOT.parent, "filesystem-root": Path("/")}
    _replay(tmp_path, monkeypatch)

    with pytest.raises(demo.DemoOutputError, match="protected"):
        demo.run_demo(out=target[which], models_root=tmp_path / "models")


def test_writes_an_identifying_marker(demo, tmp_path):
    out = tmp_path / "fresh"
    demo._prepare_out_dir(out, replace=False)

    assert (out / demo.MARKER_NAME).read_text() == demo.MARKER_TOKEN
    assert "demo" in demo.MARKER_TOKEN


def test_a_directory_marker_refuses_cleanly_and_deletes_nothing(demo, tmp_path, capsys):
    out = tmp_path / "demo-out"
    (out / demo.MARKER_NAME).mkdir(parents=True)
    (out / "keep.txt").write_text("keep me")

    rc = demo.main(["--out", str(out), "--models-root", str(tmp_path / "models")])

    assert rc != 0
    err = capsys.readouterr().err
    assert "Traceback" not in err and "not a regular file" in err
    assert (out / "keep.txt").read_text() == "keep me"
    assert (out / demo.MARKER_NAME).is_dir()


def test_an_unreadable_marker_is_a_clean_refusal(demo, tmp_path, monkeypatch):
    out = tmp_path / "demo-out"
    _marked(demo, out)

    def unreadable(self, *a, **k):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(demo.DemoOutputError, match="cannot read"):
        demo._check_out_dir(out)


def test_a_marked_dir_holding_a_user_file_refuses_by_name(demo, tmp_path):
    out = tmp_path / "demo-out"
    sentinel = _marked(demo, out)
    (out / "cut.mp4").write_bytes(b"a previous demo output")

    with pytest.raises(demo.DemoOutputError, match="keep.txt") as exc:
        demo.run_demo(out=out, models_root=tmp_path / "models")

    assert "cut.mp4" not in str(exc.value)  # only the FOREIGN entries are named
    assert sentinel.read_text() == "keep me"
    assert (out / "cut.mp4").exists()


def test_replace_rechecks_after_renaming_aside_and_restores_on_refusal(demo, tmp_path):
    # A file landing between the check and the replace is caught on the
    # renamed-aside copy, which is moved back untouched.
    out = tmp_path / "demo-out"
    sentinel = _marked(demo, out)

    with pytest.raises(demo.DemoOutputError, match="keep.txt"):
        demo._prepare_out_dir(out, replace=True)

    assert sentinel.read_text() == "keep me"
    assert [p.name for p in tmp_path.iterdir()] == ["demo-out"]


def test_replace_deletes_the_renamed_aside_path_never_out_itself(demo, tmp_path, monkeypatch):
    out = tmp_path / "demo-out"
    out.mkdir()
    (out / demo.MARKER_NAME).write_text(demo.MARKER_TOKEN)
    (out / "decision.json").write_text("{}")
    removed = []
    real_rmtree = demo.shutil.rmtree
    monkeypatch.setattr(demo.shutil, "rmtree", lambda p: (removed.append(Path(p)), real_rmtree(p)))

    demo._prepare_out_dir(out, replace=True)

    assert removed == [tmp_path / f"demo-out.kai-demo-old-{demo.os.getpid()}"]
    assert sorted(p.name for p in out.iterdir()) == [demo.MARKER_NAME]
    assert not removed[0].exists()
