"""Benchmark charts (PNG) from the committed aggregate reports; re-run after any report changes.

Inputs: reports/test-3/results.json (arms, per-domain, cost), reports/test-3/router-test2.json (test-2 per arm and
Jev), reports/runtime-measurement/summary.json (routed, measured end to end), examples/llama-routed.json (routing map)
and reports/comparison.json (other systems; rows with null values are skipped until their numbers are filled in);
reports/test-4/results.json (test-4 aggregates, written by scripts/export_test4.py).

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
    rows3.append({"name": "routed (post hoc)", "k": routed, "kind": "routed",
                  "s": router["seconds_per_question"]["E"]["test3_mix"]})
    rows2 = [{"name": "Jev 1.13 (API)", "k": router["correct"]["jev"]["156"], "kind": "jev"}]
    rows2 += [{"name": m, "k": router["correct"][a]["156"], "kind": "ours"} for a, m in ARMS.items()]
    rows2.append({"name": "routed (measured)", "k": runtime["correct_156"], "kind": "routed"})
    for s in load("reports/comparison.json")["systems"]:
        if s.get("test3_900") is not None:
            rows3.append({"name": s["name"], "k": s["test3_900"], "kind": "other", "s": s.get("median_s"),
                          "hw": "CPU" if "CPU" in (s.get("hardware") or "") else s.get("hardware")})
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
    rows = sorted(rows, key=lambda r: -r["k"] / r.get("n", n))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4))
    y = list(range(len(rows)))[::-1]
    acc = [100 * r["k"] / r.get("n", n) for r in rows]
    ci = [wilson(r["k"], r.get("n", n)) for r in rows]
    err = [[a - lo for a, (lo, _) in zip(acc, ci)], [hi - a for a, (_, hi) in zip(acc, ci)]]
    ax.barh(y, acc, xerr=err, color=[COLORS[r["kind"]] for r in rows], capsize=3)
    for yi, r, (_, hi) in zip(y, rows, ci):
        ax.text(hi + 0.3, yi, f"{r['k']}/{r.get('n', n)}", va="center", fontsize=9)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlim(max(0, min(acc) - 8), 100)
    ax.set_xlabel("accuracy (%) — error bars: Wilson 95 % CI")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter(rows, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    jev = next(r for r in rows if r["kind"] == "jev")
    ax.axhline(100 * jev["k"] / 900, color=COLORS["jev"], ls="--", lw=1.2,
               label=f"Jev 1.13 (API): {jev['k']}/900 (latency: network, not comparable)")
    for r in rows:
        if r["kind"] == "jev" or r.get("s") is None:
            continue
        a = 100 * r["k"] / 900
        ax.scatter(r["s"], a, s=60, color=COLORS[r["kind"]], zorder=3)
        left = r["kind"] == "routed" or (r["kind"] == "other" and r["s"] > 8)  # keep labels inside the plot
        label = r["name"] + ("; mean time" if r["kind"] == "routed" else "") + (f" ({r['hw']})" if r.get("hw") else "")
        ax.annotate(label, (r["s"], a), textcoords="offset points",
                    xytext=((-8, 6) if r["kind"] == "other" else (-8, -4)) if left else (6, -4), ha="right" if left else "left", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median seconds per question (log scale)\n"
                  "our modes: RX 7600 8 GB GPU; grey: other systems on CPU (Ryzen 5 5600X)")
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


# ------------------------------------------------------------------------------------------------- test-4
T4_ARMS = ["Jev 1.13 (API)", "ultra-fast", "fast", "medium", "slow"]
T4_COLORS = {"Jev 1.13 (API)": COLORS["jev"], "ultra-fast": "#9bbf5a", "fast": "#4f9a3a", "medium": COLORS["ours"],
             "slow": "#1d3f73", "routed (test-3 routes)": COLORS["routed"]}


def t4_label(name, arm):
    return name + (" (2k subsample)" if arm["set"] != "all 10 000" else "")


def test4_accuracy(t4, path):
    rows = []
    for name, a in t4["arms"].items():
        tot = a["accuracy"]["total"]
        rows.append({"name": t4_label(name, a), "k": tot["correct"], "n": tot["n"],
                     "kind": "jev" if name.startswith("Jev") else ("routed" if name.startswith("routed") else "ours")})
    bars(rows, 10000, "test-4: 40 public datasets (EN + PT), gold from the datasets", path)


def test4_per_group(t4, path):
    arms = t4["arms"]
    groups = sorted(t4["counts"]["by_group"], key=lambda g: -arms[T4_ARMS[0]]["accuracy"][f"group:{g}"]["accuracy"])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.16
    for i, name in enumerate(T4_ARMS):
        a = arms[name]
        ax.bar([x + (i - 2) * w for x in range(len(groups))], [100 * a["accuracy"][f"group:{g}"]["accuracy"] for g in groups], w,
               label=t4_label(name, a), color=T4_COLORS[name], hatch="//" if a["set"] != "all 10 000" else None,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(groups)), groups, rotation=25, ha="right")
    ax.set_ylim(30, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("test-4 per task group (sorted by Jev); hatched = 2 000-item subsample")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test4_scatter(t4, path):
    """Same items for every arm: accuracy on the 2 000-item subsample vs median seconds per item."""
    arms = t4["arms"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    jev = arms[T4_ARMS[0]]["accuracy_on_subsample"]
    ax.axhline(100 * jev["accuracy"], color=COLORS["jev"], ls="--", lw=1.2,
               label=f"Jev 1.13 (API): {jev['correct']}/{jev['n']} (latency: network, not comparable)")
    for name, a in arms.items():
        if name.startswith("Jev"):
            continue
        s, acc_ = a["median_s_per_item"], 100 * a["accuracy_on_subsample"]["accuracy"]
        ax.scatter(s, acc_, s=60, color=T4_COLORS[name], zorder=3)
        routed = name.startswith("routed")
        ax.annotate(name + ("; does not transfer" if routed else ""), (s, acc_), textcoords="offset points",
                    xytext=(-8, -12) if routed else (6, -4), ha="right" if routed else "left", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median seconds per item (log scale), RX 7600 8 GB GPU")
    ax.set_ylabel("test-4 accuracy on the 2 000-item subsample (%)")
    ax.set_title("Accuracy vs latency (test-4, same 2 000 items for every arm)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test4_languages(t4, path):
    arms = t4["arms"]
    cuts = [("language:en", "EN"), ("pt:native", "PT native"), ("pt:machine-translated", "PT machine-translated")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    w = 0.16
    for i, name in enumerate(T4_ARMS):
        a = arms[name]
        ax.bar([x + (i - 2) * w for x in range(len(cuts))], [100 * a["accuracy"][c]["accuracy"] for c, _ in cuts], w,
               label=t4_label(name, a), color=T4_COLORS[name], hatch="//" if a["set"] != "all 10 000" else None,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(cuts)), [lab + chr(10) + f"(n = {arms[T4_ARMS[0]]['accuracy'][c]['n']})" for c, lab in cuts])
    ax.set_ylim(50, 90)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("test-4 by language (n for the 10 000-item arms; hatched = 2 000-item subsample)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
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
    t4 = load("reports/test-4/results.json")
    test4_accuracy(t4, OUT / "test4_accuracy.png")
    test4_per_group(t4, OUT / "test4_per_group.png")
    test4_scatter(t4, OUT / "test4_accuracy_vs_latency.png")
    test4_languages(t4, OUT / "test4_languages.png")
    print(json.dumps({"test3": {r["name"]: r["k"] for r in rows3}, "test2": {r["name"]: r["k"] for r in rows2}}))


if __name__ == "__main__":
    main()
