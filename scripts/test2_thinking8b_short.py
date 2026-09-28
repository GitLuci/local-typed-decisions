"""Qwen3-8B Q8_0 thinks over the frozen test-2 corpus with a short budget (cap 128), read by itself at 64 (primary) and 128.

Protocol: docs/METHODOLOGY.md (test-2). Same machinery as scripts/test2_thinking4b_q4.py; the thinker is
devcal_thinkers's `qwen3-8b-q8_0` (sealed hash, n_ctx 2048, no mmap). Seed 20260925 + case index, sampling of
the thinking study, one thought per case generated with max 128 tokens; the same thought is read truncated at 64 tokens
(= a 64-token budget run, given the fixed seed) and at 128 (natural close or forced). Nothing trained.

Phases: think [--fragment k --fragments n] | merge | summary
"""
import argparse
import json
from pathlib import Path
try:  # Unix only; on Windows the peak RSS is recorded as null
    import resource
except ImportError:
    resource = None
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.devcal_thinkers as cd
import scripts.test2_thinking4b_q4 as t4
from scripts.devcal_thinkers import GGUFThinker, LETTERS, THINKERS, letter_ids, base_provenance, truncated_body
from scripts.structural_readout import argmax, read_jsonl, sha256

CAP, PRIMARY, SECONDARY = 128, 64, 128
SPEC = THINKERS["qwen3-8b-q8_0"]
LOCK = ROOT / SPEC["lock"]
OUT = ROOT / "reports/test-2/thinking-8b-q8-short"
OFFICIAL = ROOT / "reports/test-2/thinking-4b-q8/thoughts.jsonl"  # 4B Q8 thinking, the official model (own reading)
FASTPATH_8B_TOTALS = {"156": 132, "138_no_numeric": 126}  # reports/test-2/fastpath-8b-q8_0/summary.json (no per-case file)

t4.OUT, t4.FASTPATH_Q8, t4.FASTPATH_NAME = OUT, OFFICIAL, "4B Q8 thinking (official model)"
cd.CAP = CAP


def phase_think(args):
    assert 0 <= args.fragment < args.fragments
    t4.FRAGMENTS = args.fragments
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / f"thoughts-k{args.fragment}.jsonl"
    cases = [c for c in t4.cases_test2() if c["index"] % args.fragments == args.fragment]
    done = {r["id"] for r in read_jsonl(raw)} if raw.exists() else set()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cd.THREADS = args.threads
    thinker = GGUFThinker(str(ROOT / lock["path"]), ROOT / SPEC["gguf"], n_ctx=SPEC["n_ctx"], expected_sha256=SPEC["gguf_sha256"],
                          use_mmap=SPEC["use_mmap"])
    thinker.letters = letter_ids(thinker.tok, len(LETTERS))
    prov = base_provenance("qwen3-8b-q8_0-short", {"backend": "gguf"}, lock)
    prov.update({"protocol": "docs/METHODOLOGY.md (test-2, 8B Q8_0 short thinking)", "corpus": t4.CORPUS.name, "corpus_sha256": t4.CORPUS_SHA256,
                 "cap": CAP, "primary_budget": PRIMARY, "secondary_budget": SECONDARY, "fragment": args.fragment, "fragments": args.fragments,
                 "cases": [c["id"] for c in cases], "thinker_info": thinker.info, "resumed_after_cases": len(done), "reader": "own",
                 "n_ctx": SPEC["n_ctx"], "use_mmap": SPEC["use_mmap"]})
    (OUT / f"provenance-k{args.fragment}.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    with raw.open("a", encoding="utf-8") as log:
        for case in cases:
            if case["id"] in done:
                continue
            row = t4.row_base(case)
            row.update(thinker.think(case, case["index"]))
            row["reads"] = {}
            for budget in (PRIMARY, SECONDARY):
                body, used = truncated_body(thinker.tok, row, budget if budget < CAP else None)
                r = thinker.read_own(case, body)
                r["thinker_tokens_used"] = used
                r["probabilities"] = t4.softmax(r["own_logits"])
                r["correct"] = argmax(r["own_logits"]) == argmax(row["target"])
                row["reads"][str(budget)] = r
            primary = row["reads"][str(PRIMARY)]
            row["probabilities"], row["own_logits"], row["own_correct"] = primary["probabilities"], primary["own_logits"], primary["correct"]
            row["own_letter_mass"], row["own_read_s"] = primary["own_letter_mass"], sum(r["own_read_s"] for r in row["reads"].values())
            row["peak_rss_gib"] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20 if resource else None)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"case": case["id"], "index": case["index"], "tokens": row["thinking_tokens"], "closed": row["closed"],
                              "s": round(row["generation_s"], 1), "tok_per_s": round(row["generated_tokens"] / row["generation_s"], 2),
                              "correct_64": row["reads"][str(PRIMARY)]["correct"], "correct_128": row["reads"][str(SECONDARY)]["correct"]}), flush=True)


def phase_summary(args):
    cases = t4.cases_test2()
    rows = read_jsonl(OUT / "thoughts.jsonl")
    assert len(rows) == 168
    jev = read_jsonl(t4.JEV)
    labels_b = {r["id"]: r["label"] for r in read_jsonl(t4.LABEL_B)}
    jev_mean = {j["id"]: [sum(v[i] for v in j["probabilities"]) / len(j["probabilities"]) for i in range(len(j["keys"]))] for j in jev}
    jev_correct = {j["id"]: argmax(jev_mean[j["id"]]) == argmax(j["target"]) for j in jev}
    official_correct = {r["id"]: argmax(r["probabilities"]) == argmax(r["target"]) for r in read_jsonl(OFFICIAL)}
    subsets = {"156_all": lambda r: True, "138_no_numeric": lambda r: r["domain"] != "numeric",
               "120_no_numeric_no_tone": lambda r: r["domain"] not in ("numeric", "subjective_tone")}

    def report(budget):
        pred = [{**{k: r[k] for k in ("id", "domain", "question_type", "target_kind", "keys", "target")},
                 "probabilities": r["reads"][str(budget)]["probabilities"]} for r in rows]
        out = {"budget": budget,
               "all": t4.evaluate(pred, cases, jev, labels_b),
               "domains": {d: t4.evaluate([r for r in pred if r["domain"] == d], cases, jev, labels_b) for d in sorted({r["domain"] for r in pred})},
               "subsets": {name: t4.subset_report(pred, cases, jev, labels_b, official_correct, jev_correct, name, keep) for name, keep in subsets.items()}}
        s138 = out["subsets"]["138_no_numeric"]
        thinking = s138["correct_A"]
        ci = s138["vs_jev"]["paired_bootstrap_95ci"]
        out["criteria"] = {"primary_pays": thinking >= FASTPATH_8B_TOTALS["138_no_numeric"] + 7,
                           "primary_rule": "138 without numeric >= 8B fast path (126) + 7 = 133",
                           "jev_level": (ci[0] <= 0 <= ci[1]) and (s138["vs_jev"]["reference_correct"] - thinking <= 5),
                           "jev_rule": "95% CI of the paired difference contains 0 and Jev - thinking <= 5 on the 138",
                           "beats_official_by_4": thinking >= s138["vs_fastpath_q8"]["reference_correct"] + 4,
                           "official_rule": "138 >= 4B Q8 thinking (129) + 4 = 133; cost per case must also be <= the official's (~30 s on this VM)"}
        for name in out["subsets"]:
            out["subsets"][name]["vs_official_4b_q8_thinking"] = out["subsets"][name].pop("vs_fastpath_q8")
        return out

    summary = {"protocol": "docs/METHODOLOGY.md (test-2, 8B Q8_0 short thinking)", "n": 168, "cap": CAP, "primary_budget": PRIMARY,
               "primary_read_64": report(PRIMARY), "secondary_read_128": report(SECONDARY),
               "fastpath_8b_totals": FASTPATH_8B_TOTALS,
               "closed_by_128": sum(r["closed"] for r in rows), "median_thinking_tokens": sorted(r["thinking_tokens"] for r in rows)[84],
               "median_generation_s": sorted(r["generation_s"] for r in rows)[84], "mean_generation_s": sum(r["generation_s"] for r in rows) / 168,
               "median_reads_s": sorted(r["own_read_s"] for r in rows)[84],
               "median_case_s_64_budget_estimate": sorted(r["generation_s"] * min(1.0, PRIMARY / max(r["generated_tokens"], 1)) + r["reads"][str(PRIMARY)]["own_read_s"] for r in rows)[84],
               "median_tok_per_s": sorted(r["generated_tokens"] / r["generation_s"] for r in rows)[84],
               "peak_rss_gib_think": max(r["peak_rss_gib"] for r in rows), "thoughts_sha256": sha256(OUT / "thoughts.jsonl"),
               "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    brief = {k: summary[k] for k in ("closed_by_128", "median_thinking_tokens", "median_generation_s", "median_case_s_64_budget_estimate")}
    for who in ("primary_read_64", "secondary_read_128"):
        r = summary[who]
        brief[who] = {name: (s["correct_A"], s["n"], "jev", s["jev_correct_A"], "official", s["vs_official_4b_q8_thinking"]["reference_correct"])
                      for name, s in r["subsets"].items()} | {"criteria": {k: v for k, v in r["criteria"].items() if not k.endswith("rule")},
                      "domains": {d: (v.get("correct_A"), v["n_labelled"]) for d, v in r["domains"].items()}}
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["think", "merge", "summary"])
    p.add_argument("--fragment", type=int, default=0)
    p.add_argument("--fragments", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    t4.FRAGMENTS = a.fragments
    {"think": phase_think, "merge": t4.phase_merge, "summary": phase_summary}[a.phase](a)
