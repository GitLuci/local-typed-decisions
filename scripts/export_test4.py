"""Export the test-4 aggregates (no items, no per-item predictions, no reasoning traces) into reports/test-4/results.json.

test-4 was built, sealed and run in the private development repository; several of its 40 public sources do not allow
redistribution of their texts, so only aggregate numbers are published here. This script reads that repository's
`reports/test-4/analise.json` (aggregates only) and `examples/test-4/selo.json` (seal: hashes, counts, source list)
and writes the public, English-labelled summary used by `scripts/make_charts.py`.

    python scripts/export_test4.py --analysis PATH/analise.json --seal PATH/selo.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = {"jev": "Jev 1.13 (API)", "ultra": "ultra-fast", "rapido": "fast", "medio": "medium", "demorado": "slow",
       "encaminhado": "routed (test-3 routes)"}
GROUP = {"intencao": "intent", "sentimento": "sentiment", "emocao": "emotion", "toxicidade": "toxicity", "nli": "NLI",
         "simnao": "yes/no QA", "escolha": "multiple choice", "topico": "topic", "similaridade": "similarity", "spam": "spam"}


LICENSE_EN = {"nenhuma": "none stated", "académico": "academic/research use only", "não comercial": "non-commercial",
              "CC BY-NC-SA 4.0 (original)": "CC BY-NC-SA 4.0 (original release)", "CC-BY-SA-4.0 (tweets)": "CC BY-SA 4.0 (tweets)",
              "MIT (origem)": "MIT (original release)", "CC-BY-4.0 (UCI)": "CC BY 4.0 (UCI repository)", "unknown": "unknown"}
# This repository's stricter reading (DATA_LICENSES.md §3): GoEmotions texts are not redistributed; the PT translation derives from them.
NOT_REDISTRIBUTED_HERE = {"goemotions", "goemotions-pt"}


def cut(k):
    kind, _, v = k.partition(":")
    if kind == "total":
        return "total"
    if kind == "grupo":
        return f"group:{GROUP[v]}"
    if kind == "lingua":
        return f"language:{v}"
    if kind == "tipo":
        return f"type:{v}"
    if kind == "pt":
        return {"ta": "pt:machine-translated", "nativo": "pt:native"}[v]
    raise ValueError(k)


def acc(a):
    return {"correct": a["certos"], "n": a["n"], "accuracy": a["taxa"], "ci95_wilson": a["ic95"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--seal", required=True)
    a = ap.parse_args()
    res = json.loads(Path(a.analysis).read_text(encoding="utf-8"))
    seal = json.loads(Path(a.seal).read_text(encoding="utf-8"))
    c = seal["contagens"]
    out = {
        "test": "test-4",
        "description": "Broad stress test: 10 000 items from 40 public, already-labelled datasets (EN and PT), gold from the datasets. "
                       "Jev, ultra-fast and fast run on all 10 000; medium, slow and routed on a stratified 2 000-item subsample "
                       "(20 % of every source, nested in the 10 000).",
        "hardware": "local modes: AMD Radeon RX 7600 (8 GB), llama.cpp b11205 Vulkan, batch 1, cold prefill; Jev: public API",
        "seal": {"items_sha256": seal["sha256"]["itens.jsonl"], "manifest_sha256": seal["sha256"]["manifesto.jsonl"],
                 "seed": seal["semente"]},
        "counts": {"total": c["total"], "subsample": c["subamostra"],
                   "by_group": {GROUP[k]: v for k, v in c["por_grupo"].items()},
                   "by_language": c["por_lingua"], "by_type": c["por_tipo"], "pt_machine_translated": c["traducao_automatica"]},
        "arms": {},
        "paired_vs_jev": {},
        "notes": ["routed (test-3 routes) applies the routing map chosen on test-3 (per test-3 domain) to the closest test-4 "
                  "group; it does not transfer and scores below medium.",
                  "Jev choice questions are asked in two option orders; the prediction is the argmax of the mean.",
                  "Options per question are capped at 26 (letter readout): intent sets show 10 candidates (gold + 9 seeded), "
                  "GoEmotions uses the Ekman grouping (7).",
                  "No local arm reaches Jev level on test-4."],
    }
    for k, b in res["bracos"].items():
        out["arms"][ARM[k]] = {"set": "all 10 000" if b["conjunto"] == "todos" else "2 000 subsample", "items": b["itens"],
                               "accuracy": {cut(x): acc(v) for x, v in b["acerto"].items()},
                               "accuracy_on_subsample": acc(b["acerto_na_subamostra"]),
                               "median_s_per_item": b["mediana_s"], "score_mean_level_error": b["score_erro_medio_niveis"]}
    for name, comp in res["comparacoes"].items():
        _, _, rest = name.partition("_vs_")
        arm, _, s = rest.rpartition("_")
        key = f"Jev − {ARM[arm]} ({'all 10 000' if s == 'todos' else '2 000 subsample'})"
        out["paired_vs_jev"][key] = {cut(x): {"n": v["n"], "only_jev_correct": v["so_a"], "only_arm_correct": v["so_b"],
                                              "difference": v["diferenca_a_menos_b"], "ci95_newcombe": v["ic95"],
                                              "p_mcnemar_exact": v["p_mcnemar"]} for x, v in comp.items()}
    out["sources"] = [{"name": f["nome"], "group": GROUP[f["grupo"]], "language": f["lingua"], "items": f["n"],
                       "dataset": f["obter"][1] if isinstance(f["obter"][0], str) else f["obter"][0][1],
                       "license": LICENSE_EN.get(f["licenca"], f["licenca"]),
                       "texts_redistributable": f["redistribui"] and f["nome"] not in NOT_REDISTRIBUTED_HERE,
                       "machine_translated": f["traducao_automatica"]} for f in seal["fontes"]]
    dst = ROOT / "reports/test-4/results.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(dst)


if __name__ == "__main__":
    main()
