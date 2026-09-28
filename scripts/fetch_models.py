"""Fetch the release GGUFs and pinned public tokenizers; verify size and SHA-256 before installing.

The GGUF repository may be public or private. For a private one, credentials come only from the
standard Hugging Face login (`hf auth login` or the HF_TOKEN environment variable), never from this JSON config.
No inference, weight conversion, upload or training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify(path, item):
    if path.stat().st_size != item["bytes"]:
        raise ValueError(f"{item['path']}: unexpected size; file was not installed")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != item["sha256"]:
        raise ValueError(f"{item['path']}: SHA256 mismatch; file was not installed")


def plan(manifest, root):
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("invalid release manifest")
    result, destinations = [], set()
    for item in manifest["files"]:
        if (item.get("kind") not in ("gguf", "tokenizer") or type(item.get("bytes")) is not int
                or item["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))):
            raise ValueError("invalid artifact identity")
        relative = PurePosixPath(item["path"])
        filename = PurePosixPath(item["filename"])
        if (relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("models",)
                or filename.is_absolute() or ".." in filename.parts
                or any(c in item["path"] + item["filename"] for c in ("\\", ":"))):
            raise ValueError("artifact paths must stay inside models/")
        target = (root / item["path"]).resolve()
        if not target.is_relative_to((root / "models").resolve()) or target in destinations:
            raise ValueError("duplicate or escaped destination")
        if item["kind"] == "tokenizer" and not re.fullmatch(r"[0-9a-f]{40}", item.get("revision", "")):
            raise ValueError("tokenizer revision must be pinned")
        destinations.add(target)
        result.append((item, target))
    return result


def obtain(config, manifest, *, root=ROOT, verify_only=False, download=None, repo_info=None):
    if not isinstance(config, dict) or set(config) - {"repo_id", "revision"}:
        raise ValueError("config accepts only repo_id and revision, never credentials")
    files = plan(manifest, Path(root))
    pending = []
    for item, target in files:
        if target.exists():
            verify(target, item)  # never overwrite an existing mismatched file
        else:
            pending.append((item, target))
    if verify_only:
        if pending:
            raise ValueError("missing release artifacts: " + ", ".join(i["path"] for i, _ in pending))
        return {"verified": len(files), "downloaded": 0}
    if not pending:
        return {"verified": len(files), "downloaded": 0}
    hub = any(i["kind"] == "gguf" for i, _ in pending)
    repo = config.get("repo_id") or manifest.get("repo_id")
    revision = config.get("revision", manifest.get("revision", "main"))
    if hub and (not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo)
                or not isinstance(revision, str) or not revision):
        raise ValueError("set the Hugging Face repo_id and revision in the download config")
    if download is None or (hub and repo_info is None):
        from huggingface_hub import HfApi, hf_hub_download
        download = download or hf_hub_download
        repo_info = repo_info or HfApi().model_info
    if hub:
        try:
            info = repo_info(repo_id=repo, revision=revision, token=None)
        except Exception:
            raise RuntimeError("Cannot access the model repository; check connectivity, revision and, "
                               "for a private repository, your HF login") from None
        if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
            raise ValueError("GGUF repository must resolve to a commit")
        revision = info.sha  # pin all files in this run to the same snapshot
    for item, target in pending:
        is_gguf = item["kind"] == "gguf"
        try:
            cached = Path(download(repo_id=repo if is_gguf else item["repo_id"],
                revision=revision if is_gguf else item["revision"], filename=item["filename"],
                token=None if is_gguf else False, cache_dir=str(Path(root) / ".cache" / "huggingface")))
        except Exception:
            raise RuntimeError(f"Download failed for {item['path']}; check access and connectivity") from None
        target.parent.mkdir(parents=True, exist_ok=True)
        # Temporary file on the destination volume: only a verified copy is installed.
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
        try:
            shutil.copyfile(cached, temporary)
            verify(temporary, item)
            if target.exists():
                verify(target, item)  # another downloader may have completed meanwhile
            else:
                temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {"verified": len(files), "downloaded": len(pending)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "models-hf.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "release-models.json")
    parser.add_argument("--verify-only", action="store_true", help="Verify local files, without network or HF credentials")
    args = parser.parse_args(argv)
    try:
        result = obtain(read(args.config), read(args.manifest), verify_only=args.verify_only)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
