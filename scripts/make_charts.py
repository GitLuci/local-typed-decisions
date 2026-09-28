"""Benchmark charts (PNG) from the committed aggregate reports; re-run after any report changes.

Inputs: reports/test-3/results.json (arms, per-domain, cost), reports/test-3/router-test2.json (test-2 per arm and
Jev), reports/runtime-measurement/summary.json (routed, measured end to end), examples/llama-routed.json (routing map)
and reports/comparison.json (other systems; rows with null values are skipped until their numbers are filled in).

    python scripts/make_charts.py            # writes docs/img/*.png

Requires matplotlib (not needed by the runtime).
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/img"
ARMS = {"4b-q8-fast": "ultra-fast", "8b-q8-fast": "fast", "4b-q8-think": "medium", "8b-q8-short": "slow"}
MODE_ARM = {v: k for k, v in ARMS.items()}
COLORS = {"jev": "#d0822a", "ours": "#2f6db5", "routed": "#6fa3dc", "other": "#8a8a8a"}


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def data():
    res, router = load("reports/test-3/results.json"), load("reports/test-3/router-test2.json")
    runtime = load("reports/runtime-measurement/summary.json")["steps"]["routed"]
    routes = load("examples/llama-routed.json")["routes"]
    arms = res["arms"]
    anyarm = next(iter(arms.values()))
    rows3 = [{"name": "Jev 1.13 (API)", "k": anyarm["global"]["correct_jev"], "kind": "jev", "s": None}]
    for arm, mode in ARMS.items():
        a = arms[arm]
        rows3.append({"name": mode, "k": a["global"]["correct_model"], "kind": "ours", "s": a["cost"]["median_ms"] / 1000})
    routed = sum(arms[MODE_ARM[routes[d]]]["by_domain"][d]["correct_model"] for d in anyarm["by_domain"])
    # routed: mean (not median) time estimated for the test-3 domain mix; most questions are fast, a few think
    rows3.append({"name": "routed (post hoc; mean time)", "k": routed, "kind": "routed",
                  "s": router["seconds_per_question"]["E"]["test3_mix"]})
    rows2 = [{"name": "Jev 1.13 (API)", "k": router["correct"]["jev"]["156"], "kind": "jev"}]
    rows2 += [{"name": m, "k": router["correct"][a]["156"], "kind": "ours"} for a, m in ARMS.items()]
    rows2.append({"name": "routed (measured)", "k": runtime["correct_156"], "kind": "routed"})
    for s in load("reports/comparison.json")["systems"]:
        if s.get("test3_900") is not None:
            rows3.append({"name": s["name"], "k": s["test3_900"], "kind": "other", "s": s.get("median_s")})
        if s.get("test2_156") is not None:
            rows2.append({"name": s["name"], "k": s["test2_156"], "kind": "other"})
    domains = {}
    for d, v in anyarm["by_domain"].items():
        domains[d] = {"Jev": v["correct_jev"],
                      "medium": arms[MODE_ARM["medium"]]["by_domain"][d]["correct_model"],
                      "fast": arms[MODE_ARM["fast"]]["by_domain"][d]["correct_model"],
                      "routed": arms[MODE_ARM[routes[d]]]["by_domain"][d]["correct_model"]}
    return rows3, rows2, domains


def bars(rows, n, title, path):
    rows = sorted(rows, key=lambda r: -r["k"])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4))
    y = list(range(len(rows)))[::-1]
    acc = [100 * r["k"] / n for r in rows]
    ci = [wilson(r["k"], n) for r in rows]
    err = [[a - lo for a, (lo, _) in zip(acc, ci)], [hi - a for a, (_, hi) in zip(acc, ci)]]
    ax.barh(y, acc, xerr=err, color=[COLORS[r["kind"]] for r in rows], capsize=3)
    for yi, r, (_, hi) in zip(y, rows, ci):
        ax.text(hi + 0.3, yi, f"{r['k']}/{n}", va="center", fontsize=9)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlim(max(0, min(acc) - 8), 100)
    ax.set_xlabel("accuracy (%) — error bars: Wilson 95 % CI")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    jev = next(r for r in rows if r["kind"] == "jev")
    ax.axhline(100 * jev["k"] / 900, color=COLORS["jev"], ls="--", lw=1.2,
               label=f"Jev 1.13 (API): {jev['k']}/900 (latency: network, not comparable)")
    for r in rows:
        if r["kind"] == "jev" or r.get("s") is None:
            continue
        a = 100 * r["k"] / 900
        ax.scatter(r["s"], a, s=60, color=COLORS[r["kind"]], zorder=3)
        ax.annotate(r["name"], (r["s"], a), textcoords="offset points", xytext=(6, -4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median seconds per question (log scale; local modes on RX 7600 8 GB)")
    ax.set_ylabel("test-3 accuracy (%)")
    ax.set_title("Accuracy vs latency (test-3, 900 sealed questions)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_domain(domains, path):
    systems = ["Jev", "routed", "medium", "fast"]
    colors = {"Jev": COLORS["jev"], "routed": COLORS["routed"], "medium": COLORS["ours"], "fast": "#9bbf5a"}
    names = sorted(domains, key=lambda d: -domains[d]["Jev"])
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    for i, s in enumerate(systems):
        ax.bar([x + (i - 1.5) * w for x in range(len(names))], [domains[d][s] for d in names], w, label=s,
               color=colors[s])
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_ylim(40, 101)
    ax.set_ylabel("correct / 100")
    ax.set_title("test-3 per domain (routed = per-domain best mode, chosen post hoc)")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows3, rows2, domains = data()
    bars(rows3, 900, "test-3: 900 sealed questions, independent gold", OUT / "test3_accuracy.png")
    scatter(rows3, OUT / "accuracy_vs_latency.png")
    per_domain(domains, OUT / "test3_per_domain.png")
    bars(rows2, 156, "test-2: 156 labeled cases", OUT / "test2_accuracy.png")
    print(json.dumps({"test3": {r["name"]: r["k"] for r in rows3}, "test2": {r["name"]: r["k"] for r in rows2}}))


if __name__ == "__main__":
    main()
