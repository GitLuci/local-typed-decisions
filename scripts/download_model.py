"""Download only the pinned inference artifacts, never datasets or sibling models.

`--manifest <path>` sets where the SHA-256 manifest is written; by default it goes to `.cache/manifests/` (git-ignored),
never over a committed `reports/manifests/model-manifest*.json`. Compare it afterwards with the committed manifest.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

QWEN_PATTERNS = ["README.md", "LICENSE", "config.json", "generation_config.json", "model*.safetensors",
                 "model.safetensors.index.json", "tokenizer*", "merges.txt", "vocab.json"]
LAYA_PATTERNS = ["README.md", "rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]


def manifest_path(lock: Path, explicit: "Path | None") -> Path:
    """Explicit path wins; the default lives in the git-ignored `.cache/`, so a literal run never dirties a committed file."""
    if explicit is not None:
        return explicit
    return ROOT / ".cache" / "manifests" / f"model-manifest-{lock.stem.replace('.lock', '')}.json"


def build_manifest(path: Path) -> dict:
    manifest = {}
    for file in sorted(path.rglob("*")):
        if file.is_file() and ".cache" not in file.parts:
            with file.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest[file.relative_to(path).as_posix()] = {"bytes": file.stat().st_size, "sha256": digest}
    return manifest


def main(argv=None):
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=ROOT / "model.lock.json")
    parser.add_argument("--manifest", type=Path, default=None, help="where to write the manifest; default: .cache/manifests/")
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    patterns = QWEN_PATTERNS if lock["repo_id"].startswith("Qwen/") else LAYA_PATTERNS
    path = Path(snapshot_download(
        repo_id=lock["repo_id"], revision=lock["revision"],
        local_dir=ROOT / lock["path"], max_workers=2,
        allow_patterns=patterns,
        token=False,
    ))
    manifest = build_manifest(path)
    target = manifest_path(args.lock, args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"model": lock, "files": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": str(path), "files": len(manifest), "manifest": str(target)}))


if __name__ == "__main__":
    main()
