"""Write a float32 copy of a pinned bf16 Qwen checkpoint, one shard at a time.

Why: loading the bf16 4B as float32 converts in memory and can exceed the free RAM of a
16 GB machine; a float32 checkpoint on disk loads without that conversion. bf16 -> fp32 is
exact, so results must be identical; `--check` verifies logits against recorded runs.
Peak memory here is about one shard in float32 (~8 GB for the 4B), not the whole model.
The source directory is never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def convert(src, dst):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    dst.mkdir(parents=True, exist_ok=False)
    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            metadata = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                tensors[name] = t.to(torch.float32) if t.dtype == torch.bfloat16 else t
        total += sum(t.numel() * t.element_size() for t in tensors.values())
        save_file(tensors, str(dst / shard), metadata=metadata)
        del tensors
        print(json.dumps({"shard": shard, "done": True}), flush=True)
    index["metadata"] = {**index.get("metadata", {}), "total_size": total}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = "float32"
    (dst / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for p in src.iterdir():
        if p.is_file() and not p.name.startswith("model") and p.name != "config.json":
            shutil.copy2(p, dst / p.name)
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(dst.iterdir()) if p.is_file()}


def check(lock, cases_n, reference):
    """Parity: float32 copy vs recorded 4B dev+cal final-layer letter logits (a structural-readout
    predictions file written by scripts/structural_readout.py; per-item files are not shipped)."""
    import psutil
    from scripts.structural_readout import layer_forward, read_jsonl, CORPUS
    from typed_decisions.qwen import QwenDecisionModel
    model = QwenDecisionModel(lock_file=lock)
    peak = psutil.Process().memory_info().peak_wset / 2**30
    recorded = {(r["id"], r["format"]): r for r in read_jsonl(reference)}
    cases = [c for c in read_jsonl(CORPUS) if c["split"] == "development"][:cases_n]
    worst = 0.
    for case in cases:
        for fmt in ("letter", "json"):
            mine = layer_forward(model, case, fmt)["layer_logits"][-1]
            ref = recorded[(case["id"], fmt)]["layer_logits"][-1]
            worst = max(worst, max(abs(a - b) for x, y in zip(mine, ref) for a, b in zip(x, y)))
    return {"cases": cases_n, "max_logit_error_vs_recorded": worst, "load_peak_working_set_gib": peak}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["convert", "check"])
    parser.add_argument("--source-lock", type=Path, default=ROOT / "qwen-4b.lock.json")
    parser.add_argument("--lock", type=Path, default=ROOT / "qwen-4b-fp32.lock.json")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--reference", type=Path, help="check: recorded predictions-devcal.jsonl to compare against")
    args = parser.parse_args()
    if args.phase == "convert":
        source = json.loads(args.source_lock.read_text(encoding="utf-8"))
        dst_rel = source["path"] + "-fp32"
        manifest = convert(ROOT / source["path"], ROOT / dst_rel)
        lock = {**source, "path": dst_rel,
                "derived": f"float32 copy of {source['path']} (exact bf16->fp32 cast), scripts/convert_fp32.py"}
        args.lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        (ROOT / f"reports/manifests/model-manifest-{Path(dst_rel).name}.json").write_text(
            json.dumps({"model": lock, "files": manifest}, indent=2), encoding="utf-8")
        print(json.dumps({"converted": dst_rel, "files": len(manifest)}), flush=True)
    else:
        if not args.reference:
            parser.error("check requires --reference")
        print(json.dumps(check(args.lock, args.cases, args.reference)), flush=True)
