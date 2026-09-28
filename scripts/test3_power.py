"""test-3 phase 0: power analysis from existing test-2 data (no new runs).

1. Paired Jev x local-model disagreement on test-2 (156 labeled): the rate d = (b + c) / n used by McNemar.
2. Inter-labeler agreement on test-2: label A (corpus author, the `target`) x blind label B
   (examples/test-2-labels-b.jsonl).
3. Minimum detectable difference (MDE) for two-sided McNemar, alpha 0.05, power 0.80, for several N and d;
   and the half-width of the 95 % CI of the paired difference (basis of the non-inferiority criterion).

Item 1 needs per-item test-2 predictions, which are not shipped (local ones can be regenerated with the test-2
runners; Jev baseline outputs are withheld because they come from a commercial API). Without them only items 2-3 run.
The committed output is reports/test-3/power.json.

    python scripts/test3_power.py                                  # print the tables
    python scripts/test3_power.py --jev <jev predictions.jsonl> --model "4B Q8 fast=<predictions.jsonl>" --json F
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Z_A, Z_B = 1.959964, 0.841621  # alpha 0.05 two-sided; power 0.80


def correct_by_id(path: Path) -> dict[str, bool]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("target_kind") == "known_distribution":
            continue
        if "correct" in r:
            out[r["id"]] = bool(r["correct"])
            continue
        p = r["probabilities"]
        if p and isinstance(p[0], list):  # Jev: two letter permutations -> mean
            p = [sum(v[i] for v in p) / len(p) for i in range(len(p[0]))]
        t = r["target"]
        if isinstance(t, dict):
            t = [t[k] for k in r["keys"]]
        out[r["id"]] = max(range(len(p)), key=p.__getitem__) == max(range(len(t)), key=t.__getitem__)
    return out


def disagreement(ref: dict, mod: dict) -> dict:
    ids = sorted(set(ref) & set(mod))
    b = sum(ref[i] and not mod[i] for i in ids)
    c = sum(mod[i] and not ref[i] for i in ids)
    return {"n": len(ids), "ref_correct": sum(ref[i] for i in ids), "model_correct": sum(mod[i] for i in ids),
            "b_only_ref": b, "c_only_model": c, "d": round((b + c) / len(ids), 4)}


def mde(n: int, d: float) -> float:
    """Smallest detectable delta = p10 - p01: n = (za*sqrt(d) + zb*sqrt(d - delta^2))^2 / delta^2 (Connor 1987), solved for delta."""
    lo, hi = 1e-6, d
    for _ in range(100):
        m = (lo + hi) / 2
        req = (Z_A * math.sqrt(d) + Z_B * math.sqrt(max(d - m * m, 0))) ** 2 / (m * m)
        lo, hi = (lo, m) if req <= n else (m, hi)
    return hi


def half_width(n: int, d: float) -> float:
    return Z_A * math.sqrt(d) / math.sqrt(n)


def labeler_agreement() -> dict:
    cases = {json.loads(l)["id"]: json.loads(l) for l in
             (ROOT / "examples/test-2-scenarios.jsonl").read_text(encoding="utf-8").splitlines()}
    by_dom: dict[str, list[int]] = {}
    for line in (ROOT / "examples/test-2-labels-b.jsonl").read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        c = cases[a["id"]]
        t = c["target"]
        gold = max(t, key=t.get)
        if c["question"]["type"] == "score":  # label by position
            gold = str(list(t).index(gold)) if gold not in map(str, range(10)) else gold
        by_dom.setdefault(a["domain"], [0, 0])
        by_dom[a["domain"]][0] += str(a["label"]) == str(gold)
        by_dom[a["domain"]][1] += 1
    tot = [sum(v[0] for v in by_dom.values()), sum(v[1] for v in by_dom.values())]
    return {"by_domain": {k: f"{v[0]}/{v[1]}" for k, v in by_dom.items()},
            "total": f"{tot[0]}/{tot[1]}", "disagreement": round(1 - tot[0] / tot[1], 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--jev", type=Path, help="per-item Jev test-2 predictions (not shipped)")
    ap.add_argument("--model", action="append", default=[], help="NAME=path to per-item test-2 predictions (repeatable)")
    ap.add_argument("--json", type=Path, help="write the tables to this file")
    args = ap.parse_args()
    res = {"disagreement_vs_jev": {}}
    if args.jev:
        jev = correct_by_id(args.jev)
        for spec in args.model:
            name, path = spec.split("=", 1)
            res["disagreement_vs_jev"][name] = disagreement(jev, correct_by_id(Path(path)))
    res["labelers_test2"] = labeler_agreement()
    ds = [0.06, 0.08, 0.10, 0.12]
    ns = [100, 300, 600, 800, 900, 1000, 1200]
    res["mde_points"] = {str(n): {str(d): round(100 * mde(n, d), 1) for d in ds} for n in ns}
    res["ci95_half_width_points"] = {str(n): {str(d): round(100 * half_width(n, d), 1) for d in ds} for n in ns}

    print("paired disagreement on test-2 (156 labeled) against the official Jev run:")
    for k, v in res["disagreement_vs_jev"].items():
        print(f"  {k:<26} Jev {v['ref_correct']}/{v['n']}  model {v['model_correct']}/{v['n']}  "
              f"b={v['b_only_ref']} c={v['c_only_model']}  d={v['d']:.3f}")
    r = res["labelers_test2"]
    print(f"labelers on test-2 (label A x blind label B): {r['total']} agree "
          f"(disagreement {100 * r['disagreement']:.1f} %) - {r['by_domain']}")
    print("\nMDE (percentage points; McNemar, alpha 0.05 two-sided, power 0.80)")
    print("     N  " + "".join(f"  d={d:<5}" for d in ds))
    for n in ns:
        print(f"  {n:>5} " + "".join(f"  {res['mde_points'][str(n)][str(d)]:>6.1f} " for d in ds))
    print("\nhalf-width of the 95 % CI of the paired difference (points)")
    for n in ns:
        print(f"  {n:>5} " + "".join(f"  {res['ci95_half_width_points'][str(n)][str(d)]:>6.1f} " for d in ds))
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
