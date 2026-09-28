"""Per-domain router (chosen post hoc on test-3), validated on test-2 with existing per-item predictions.

Pre-registered before this script was written (see docs/METHODOLOGY.md, "Router validation"). No model runs.
Writes reports/test-3/router-test2.json and .md.

Configurations:
  E (router)        numeric, sentence -> 4b-q8-think; factual, deterministic, sentiment -> 8b-q8-short;
                    subjective_tone, robotic_style, noul_refund, score_urgency -> 8b-q8-fast
  V (cheap variant) 8b-q8-fast everywhere except numeric -> 4b-q8-think

Inputs are per-item predictions, which are NOT shipped: local test-2 predictions can be regenerated with the test-2
runners, test-3 timings with scripts/test3_arm.py, and the Jev baseline per-item outputs are withheld because they
come from a commercial API (regenerate them with your own access).

    python scripts/test3_router.py --test2 <dir with per-arm test-2 outputs> --test3 <dir with test-3 arm outputs> --jev <jev test-2 predictions.jsonl>
    python scripts/test3_router.py --render     # rebuild the .md from the committed .json
"""
import argparse
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/test-3"
SEED, N_BOOT = 20260928, 10_000

E = {"numeric": "4b-q8-think", "sentence": "4b-q8-think",
     "factual": "8b-q8-short", "deterministic": "8b-q8-short", "sentiment": "8b-q8-short",
     "subjective_tone": "8b-q8-fast", "robotic_style": "8b-q8-fast", "noul_refund": "8b-q8-fast",
     "score_urgency": "8b-q8-fast"}
V = {d: ("4b-q8-think" if d == "numeric" else "8b-q8-fast") for d in E}
ARMS = ("8b-q8-fast", "4b-q8-fast", "8b-q8-short", "4b-q8-think")


def read(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"{p}: per-item predictions not found (not shipped; see the module docstring)")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def correct_from_probs(rows):
    return {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in rows
            if r["target_kind"] != "known_distribution"}


def load(t2, jev_path):
    ac = {"8b-q8-fast": correct_from_probs(read(t2 / "fastpath-8b-q8_0/predictions.jsonl")),
          "4b-q8-fast": correct_from_probs(read(t2 / "fastpath-4b-q8_0/predictions.jsonl"))}
    ac["4b-q8-think"] = {r["id"]: bool(r["own_correct"]) for r in read(t2 / "thinking-4b-q8/thoughts.jsonl")
                         if r["target_kind"] != "known_distribution"}
    ac["8b-q8-short"] = {r["id"]: bool(r["reads"]["128"]["correct"]) for r in
                         read(t2 / "thinking-8b-q8-short/thoughts-k0.jsonl") if r["target_kind"] != "known_distribution"}
    jev = {}
    for r in read(jev_path):
        if r.get("target_kind") == "known_distribution":
            continue
        p = r["probabilities"]
        mean = [sum(v[i] for v in p) / len(p) for i in range(len(p[0]))] if isinstance(p[0], list) else p
        t = r["target"] if isinstance(r["target"], list) else [r["target"][k] for k in r["keys"]]
        jev[r["id"]] = argmax(mean) == argmax(t)
    ac["jev"] = jev
    domain = {r["id"]: r["domain"] for r in read(t2 / "fastpath-8b-q8_0/predictions.jsonl")}
    ids = sorted(ac["jev"])
    for k, v in ac.items():
        assert set(v) == set(ids), f"{k}: cases differ from Jev's ({len(v)} vs {len(ids)})"
    return ac, domain, ids


def route(rule, ac, domain, ids):
    return {i: ac[rule[domain[i]]][i] for i in ids}


def mcnemar(a, b, ids):
    b10 = sum(a[i] and not b[i] for i in ids)
    b01 = sum(b[i] and not a[i] for i in ids)
    n = b10 + b01
    if n == 0:
        return b10, b01, 1.0
    k = min(b10, b01)
    return b10, b01, min(1.0, 2 * sum(comb(n, j) for j in range(k + 1)) / 2 ** n)


def boot_ci(a, b, ids, rng):
    d = [int(a[i]) - int(b[i]) for i in ids]
    n = len(d)
    samples = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return 100 * samples[int(0.025 * N_BOOT)], 100 * samples[int(0.975 * N_BOOT) - 1]


def compare(name_a, a, name_b, b, ids, rng):
    only_a, only_b, p = mcnemar(a, b, ids)
    lo, hi = boot_ci(a, b, ids, rng)
    return {"a": name_a, "b": name_b, "n": len(ids), "correct_a": sum(a[i] for i in ids), "correct_b": sum(b[i] for i in ids),
            "delta_points": round(100 * (sum(a[i] for i in ids) - sum(b[i] for i in ids)) / len(ids), 1),
            "only_a": only_a, "only_b": only_b, "mcnemar_p": round(p, 4), "ci95_bootstrap": [round(lo, 1), round(hi, 1)]}


def gpu_times(t3):
    """Mean elapsed_ms per arm and domain, measured on the RX 7600 in test-3 (test split)."""
    t = {}
    for b in ARMS:
        for r in read(t3 / b / "predictions.jsonl"):
            t.setdefault(b, {}).setdefault(r["domain"], []).append(r["elapsed_ms"])
    return {b: {d: sum(v) / len(v) / 1000 for d, v in ds.items()} for b, ds in t.items()}


def mean_time(rule, times, mix):
    tot = sum(mix.values())
    return sum(times[rule[d]][d] * n for d, n in mix.items()) / tot


def render_md(res):
    L = ["# Per-domain router: validation on test-2", "",
         f"Pre-registration: {res['preregistration']}. {res['n']} labeled test-2 cases. Times: RX 7600, measured on test-3.", "",
         "| configuration | correct /156 | /138 without numeric | s/question (test-2 mix) | s/question (test-3 mix) |",
         "|---|---:|---:|---:|---:|"]
    for k in ("E", "V") + ARMS + ("jev",):
        t = res["seconds_per_question"].get(k, {})
        L.append(f"| {k} | {res['correct'][k]['156']} | {res['correct'][k]['138_without_numeric']} | "
                 f"{t.get('test2_mix', '-')} | {t.get('test3_mix', '-')} |")
    L += ["", "| comparison | delta (points) | 95 % bootstrap CI | only A / only B | McNemar p |", "|---|---:|---|---|---:|"]
    for c in res["comparisons"]:
        L.append(f"| {c['a']} - {c['b']} | {c['delta_points']:+} | [{c['ci95_bootstrap'][0]}; {c['ci95_bootstrap'][1]}] | "
                 f"{c['only_a']} / {c['only_b']} | {c['mcnemar_p']} |")
    L += ["", "| domain | E | V | Jev | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * (3 + len(ARMS))]
    for d, v in res["by_domain"].items():
        L.append(f"| {d} | {v['E']} | {v['V']} | {v['jev']} | " + " | ".join(v[b] for b in ARMS) + " |")
    L += ["", "E = router, V = cheap variant (see scripts/test3_router.py)."]
    return "\n".join(L) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--test2", type=Path, default=ROOT / "reports/test-2", help="folder with per-arm test-2 per-item outputs")
    ap.add_argument("--test3", type=Path, default=REPORTS, help="folder with test-3 per-arm predictions (timings)")
    ap.add_argument("--jev", type=Path, default=ROOT / "reports/test-2/jev-official/predictions.jsonl",
                    help="Jev test-2 per-item predictions (withheld: commercial API)")
    ap.add_argument("--render", action="store_true", help="only rebuild router-test2.md from router-test2.json")
    args = ap.parse_args()
    if args.render:
        res = json.loads((REPORTS / "router-test2.json").read_text(encoding="utf-8"))
        (REPORTS / "router-test2.md").write_text(render_md(res), encoding="utf-8")
        return
    rng = random.Random(SEED)
    ac, domain, ids = load(args.test2, args.jev)
    ac["E"] = route(E, ac, domain, ids)
    ac["V"] = route(V, ac, domain, ids)
    no_num = [i for i in ids if domain[i] != "numeric"]
    res = {"preregistration": "docs/METHODOLOGY.md (router validation)", "n": len(ids),
           "correct": {k: {"156": sum(v[i] for i in ids), "138_without_numeric": sum(v[i] for i in no_num)} for k, v in ac.items()},
           "comparisons": [], "by_domain": {}, "seconds_per_question": {}}
    pairs = [("E", "jev"), ("V", "jev")] + [("E", b) for b in ARMS] + [("V", "8b-q8-fast")]
    for a, b in pairs:
        res["comparisons"].append(compare(a, ac[a], b, ac[b], ids, rng))
    doms = sorted({domain[i] for i in ids})
    for d in doms:
        di = [i for i in ids if domain[i] == d]
        res["by_domain"][d] = {k: f"{sum(ac[k][i] for i in di)}/{len(di)}" for k in ("E", "V", "jev") + ARMS}
    times = gpu_times(args.test3)
    mix_t2 = {d: sum(1 for i in ids if domain[i] == d) for d in doms}
    q3 = [json.loads(l) for l in (ROOT / "examples/test-3/questions.jsonl").read_text(encoding="utf-8").splitlines()]
    mix_t3 = {d: sum(1 for q in q3 if q["split"] == "test" and q["domain"] == d) for d in doms}
    for name, rule in [("E", E), ("V", V)] + [(b, {d: b for d in E}) for b in ARMS]:
        res["seconds_per_question"][name] = {"test2_mix": round(mean_time(rule, times, mix_t2), 2),
                                             "test3_mix": round(mean_time(rule, times, mix_t3), 2)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "router-test2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_md(res)
    (REPORTS / "router-test2.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
