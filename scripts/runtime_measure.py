"""Real measurement of the llama-server runtime on test-2 (pre-registered; see docs/RESULTS.md, "Runtime measurement").

Does not change runtime code. Uses the same path as the CLI (`load_config` -> `create_backend` -> `predict` ->
`close`), in-process, so that the llama-server processes are always shut down by the runtime itself. Samples VRAM once
per second through WMI (Windows only). Writes to reports/runtime-measurement/ (summary.json plus per-mode JSONL, which
is not committed).

Argmax agreement with the original test-2 predictions needs the per-item fast-path predictions (not shipped; pass
--test2 with a folder that has fastpath-4b-q8_0/ and fastpath-8b-q8_0/). Without them that comparison is skipped.

    python scripts/runtime_measure.py [--modes ultra-fast fast medium slow routed] [--server-exe tools/llama-b11205-vulkan/llama-server.exe]

The published summary also measured the former alias of `medium` (3 cases, identical implementation and timings
within noise); that alias no longer exists and is not part of the plan.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typed_decisions.llama_server import create_backend, load_config  # noqa: E402

OUT = ROOT / "reports/runtime-measurement"
CFG = OUT / "configs"
EXE = ROOT / "tools/llama-b11205-vulkan/llama-server.exe"
CASES = ROOT / "examples/test-2-scenarios.jsonl"
PLAN = (("ultra-fast", 168), ("medium", 3), ("fast", 168), ("slow", 3), ("routed", 168))


def configs(exe):
    CFG.mkdir(parents=True, exist_ok=True)
    done = {}
    for mode, _ in PLAN:
        c = json.loads((ROOT / f"examples/llama-{mode}.json").read_text(encoding="utf-8"))
        for s in c["servers"].values():
            s["tokenizer_path"] = str((ROOT / "examples" / s["tokenizer_path"]).resolve())
            s["launch"]["executable"] = str(Path(exe).resolve())
            s["launch"]["model_path"] = str((ROOT / "examples" / s["launch"]["model_path"]).resolve())
        if mode == "routed":
            c["routes"].setdefault("random", "fast")  # pre-registered: without it the 12 random cases return 422
        p = CFG / f"llama-{mode}.json"
        p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
        done[mode] = p
    return done


def answer_argmax(q, a):
    if q["type"] == "noul":
        return "true" if a["noul"] > 0.5 else "false"
    pr = a["probabilities"]
    return max(pr, key=pr.get)


def test2_argmax(row):
    p = row["probabilities"]
    return row["keys"][max(range(len(p)), key=p.__getitem__)]


def gold(case):
    t = case["target"]
    return max(t, key=t.get)


class VRAM(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        cmd = ("while($true){ [int]((Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory "
               "| Measure-Object DedicatedUsage -Sum).Sum/1MB); Start-Sleep 1 }")
        self.samples = []
        try:
            self.p = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", cmd], stdout=subprocess.PIPE, text=True)
        except OSError:
            self.p = None  # not on Windows: no VRAM sampling

    def run(self):
        if self.p is None:
            return
        for line in self.p.stdout:
            try:
                self.samples.append((time.time(), int(line.strip())))
            except ValueError:
                pass

    def stop(self):
        if self.p is not None:
            self.p.kill()


def run_mode(mode, cfg, cases, with_domain):
    model = create_backend(**load_config(cfg))
    lines, t0 = [], time.perf_counter()
    try:
        for c in cases:
            q = {"q": c["question"]}
            opts = {"domains": {"q": c["domain"]}} if with_domain else {}
            t = time.perf_counter()
            try:
                r = model.predict(c["state"], q, **opts)
                error = None
            except Exception as exc:  # noqa: BLE001 - a runtime error is recorded, it does not stop the measurement
                r, error = None, f"{type(exc).__name__}: {exc}"
            s = time.perf_counter() - t
            line = {"id": c["id"], "domain": c["domain"], "s": s, "error": error}
            if r:
                if "q" not in r.get("answers", {}):
                    line["error"] = json.dumps(r.get("errors", {}).get("q"))
                else:
                    line["argmax"] = answer_argmax(c["question"], r["answers"]["q"])
                    line["answer"] = r["answers"]["q"]
                line["usage"] = r["usage"]
                rt = r.get("metadata", {}).get("routing", {}).get("q")
                if rt:
                    line["routing"] = rt
            lines.append(line)
            print(json.dumps({"mode": mode, "id": c["id"], "s": round(s, 2), "error": line["error"]}), flush=True)
    finally:
        model.close()
    return lines, time.perf_counter() - t0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=[m for m, _ in PLAN], choices=[m for m, _ in PLAN])
    ap.add_argument("--server-exe", type=Path, default=EXE, help="llama-server executable (llama.cpp b11205 Vulkan)")
    ap.add_argument("--test2", type=Path, default=ROOT / "reports/test-2", help="folder with per-item test-2 fast-path predictions (optional)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = configs(args.server_exe)
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(cases) == 168
    ref = {}
    for mode, folder in (("ultra-fast", "fastpath-4b-q8_0"), ("fast", "fastpath-8b-q8_0")):
        p = args.test2 / folder / "predictions.jsonl"
        if p.exists():
            ref[mode] = {r["id"]: test2_argmax(r) for r in map(json.loads, p.open(encoding="utf-8"))}
    vram = VRAM()
    vram.start()
    summary_path = OUT / "summary.json"  # separate runs per model accumulate in the same summary
    res = (json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists()
           else {"preregistration": "docs/RESULTS.md (runtime measurement)", "steps": {}})
    try:
        for mode, n in [(m, n) for m, n in PLAN if m in args.modes]:
            start = time.time()
            lines, total = run_mode(mode, cfg[mode], cases[:n], mode == "routed")
            end = time.time()
            (OUT / f"{mode}.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
            ok = [l for l in lines if not l["error"]]
            step = {"cases": n, "errors": n - len(ok), "total_s": round(total, 1),
                    "mean_s": round(sum(l["s"] for l in lines) / n, 3),
                    "median_s": round(sorted(l["s"] for l in lines)[n // 2], 3),
                    "vram_peak_mb": max((v for t, v in vram.samples if start <= t <= end), default=None)}
            if mode in ref:
                same = sum(1 for l in ok if l["argmax"] == ref[mode][l["id"]])
                step["argmax_same_as_test2"] = f"{same}/{n}"
                step["criterion_99"] = same >= 0.99 * n
            if mode == "routed":
                labeled = [l for l in ok if l["domain"] != "random"]
                golds = {c["id"]: gold(c) for c in cases}
                step["correct_156"] = sum(1 for l in labeled if l["argmax"] == golds[l["id"]])
                step["criterion_144_plus_minus_2"] = 142 <= step["correct_156"] <= 146
                by_dom = {}
                for l in labeled:
                    d = by_dom.setdefault(l["domain"], [0, 0])
                    d[0] += l["argmax"] == golds[l["id"]]
                    d[1] += 1
                step["by_domain"] = {k: f"{a}/{b}" for k, (a, b) in sorted(by_dom.items())}
                models = [l["routing"]["model_key"] for l in ok if "routing" in l]
                switches = [i for i in range(1, len(models)) if models[i] != models[i - 1]]
                step["model_loads"] = len(switches) + 1  # the first load counts too
                step["s_on_requests_with_load"] = [round(ok[0]["s"], 1)] + [round(ok[i]["s"], 1) for i in switches]
                by_mode = {}
                for l in ok:
                    by_mode.setdefault(l["routing"]["mode"], []).append(l["s"])
                step["median_s_by_mode"] = {m: round(sorted(v)[len(v) // 2], 2) for m, v in by_mode.items()}
            res["steps"][mode] = step
            summary_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        vram.stop()
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
