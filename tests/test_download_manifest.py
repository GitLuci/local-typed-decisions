"""download_model.py: the default manifest path is never a committed file."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import download_model as dm  # noqa: E402


def test_default_manifest_is_outside_committed_files():
    target = dm.manifest_path(ROOT / "qwen.lock.json", None)
    rel = target.relative_to(ROOT).as_posix()
    tracked = subprocess.check_output(["git", "ls-files", "--", rel], cwd=ROOT).decode().strip()
    assert tracked == "", f"the default manifest would overwrite a committed file: {rel}"
    assert rel.startswith(".cache/") and rel.endswith("model-manifest-qwen.json")


def test_explicit_manifest_path_is_honoured(tmp_path):
    explicit = tmp_path / "m.json"
    assert dm.manifest_path(ROOT / "qwen-4b.lock.json", explicit) == explicit


def test_build_manifest_hashes_every_file_and_skips_cache(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "ignored").write_text("no", encoding="utf-8")
    manifest = dm.build_manifest(tmp_path)
    assert set(manifest) == {"a.bin", "sub/b.txt"}
    assert manifest["a.bin"] == {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
