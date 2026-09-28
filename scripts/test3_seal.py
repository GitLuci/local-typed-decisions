"""test-3: verify the sealed set shipped in examples/test-3/ (states, questions, gold, levels + manifest.json).

The sealed files were built once, before any model saw them, from sources recorded in `manifest.json` (`sources`:
sha256 of each generator output, authored file, verifier/labeler/adjudicator file and exclusion list). Those source
files and the development history that held them are NOT part of this repository, so the build step cannot be re-run
here. What can be checked on the shipped files is checked:

- sha256 of the four sealed files against the manifest. `states.jsonl` is shipped with the 44 GoEmotions texts set to
  null (no explicit data licence, see DATA_LICENSES.md); that redacted file must match
  `manifest["shipped_redacted"]["sha256"]`, and after `python scripts/fetch_goemotions.py` it must match the sealed
  hash. Both states pass this check; model runs require the restored (sealed) file;
- byte-level determinism: every file is LF-only and equal to its canonical serialization (rows sorted by id, one
  compact JSON object per line, UTF-8);
- referential integrity: unique ids, every question has a state, gold and level row, every state has a question;
- gold rule on the shipped fields: constructed items take the generator gold (stratum "a"); dataset (stratum "b") and
  authored (stratum "c") items take R1 when R1 = R2, otherwise the adjudicator J; `ambiguo_final` items have
  `target: null`, come from J and are outside the primary analysis; `primario` = labeled domain and not ambiguous;
  targets use exactly the question's option keys, and labeled targets are one-hot;
- excluded items (quota and privacy) are absent from the set;
- counts and totals in the manifest match the files, and every labeled domain's test split is within the level quota
  (N1 20 %, N2 40 %, N3 40 %, +-10 points).

Field names inside the sealed files (`nivel`, `estrato`, `primario`, `ambiguo_final`, `n_frases`, ...) are part of the
sealed bytes and are kept as-is: nivel = difficulty level, estrato = stratum, primario = counts in the primary
analysis, ambiguo_final = still ambiguous after adjudication.

    python scripts/test3_seal.py --check      # exit 1 on any failure
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "examples/test-3"
FILES = ("states.jsonl", "questions.jsonl", "gold.jsonl", "levels.jsonl")
QUOTA = {1: 20, 2: 40, 3: 40}
TOLERANCE = 10
STRATUM = {"constructed": "a", "dataset": "b", "judged": "c"}


def jsonl(b: bytes):
    return [json.loads(l) for l in b.decode("utf-8").splitlines() if l.strip()]


def canonical(rows, key):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r[key])).encode("utf-8")


def keys_for(question):
    t = question["type"]
    if t == "choice":
        return list(question["criteria"])
    if t == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(question["criteria"]))]


def load(folder=SEALED):
    folder = Path(folder)
    data = {n: (folder / n).read_bytes() for n in FILES}
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return data, manifest


def check_bytes(data, manifest) -> list[str]:
    fails = []
    redacted = manifest.get("shipped_redacted", {}).get("sha256", {})
    for name, b in data.items():
        digest = hashlib.sha256(b).hexdigest()
        if digest != manifest["sha256"].get(name) and digest != redacted.get(name):
            fails.append(f"{name}: sha256 differs from the manifest")
        if b"\r" in b:
            fails.append(f"{name}: contains CR (must be LF-only)")
        key = "state_id" if name == "states.jsonl" else "id"
        if canonical(jsonl(b), key) != b:
            fails.append(f"{name}: not in canonical serialization")
    return fails


def check_gold_rule(data, manifest) -> list[str]:
    fails = []
    states = {s["state_id"]: s for s in jsonl(data["states.jsonl"])}
    questions = jsonl(data["questions.jsonl"])
    qs = {q["id"]: q for q in questions}
    gold = {g["id"]: g for g in jsonl(data["gold.jsonl"])}
    levels = {l["id"]: l for l in jsonl(data["levels.jsonl"])}
    if len(qs) != len(questions):
        fails.append("repeated question id")
    if set(gold) != set(qs) or set(levels) != set(qs):
        fails.append("gold/levels ids differ from question ids")
    if {q["state_id"] for q in questions} != set(states):
        fails.append("states without questions or questions without a state")
    for qid, q in qs.items():
        g, st = gold.get(qid), states.get(q["state_id"])
        if g is None or st is None:
            continue
        kind = st["source"]["kind"]
        labeled = q["domain"] != "random"
        if g["estrato"] != STRATUM.get(kind):
            fails.append(f"{qid}: stratum {g['estrato']} does not match source kind {kind}")
        if kind == "constructed" and g["gold_from"] != "generator":
            fails.append(f"{qid}: constructed item without generator gold")
        if kind != "constructed" and g["gold_from"] not in ("R1=R2", "J"):
            fails.append(f"{qid}: labeled item gold must come from R1=R2 or J")
        if g["ambiguo_final"]:
            if g["target"] is not None or g["primario"] or g["gold_from"] != "J":
                fails.append(f"{qid}: ambiguo_final must have null target, J gold and be outside the primary analysis")
            continue
        if g["primario"] != labeled:
            fails.append(f"{qid}: primario must equal (labeled domain and not ambiguous)")
        if set(g["target"]) != set(keys_for(q["question"])):
            fails.append(f"{qid}: target keys differ from the question options")
        elif labeled and sorted(g["target"].values()) != [0.0] * (len(g["target"]) - 1) + [1.0]:
            fails.append(f"{qid}: labeled target is not one-hot")
        if labeled and levels[qid]["nivel"] not in (1, 2, 3):
            fails.append(f"{qid}: labeled item without level 1-3")
    excluded = set(manifest["excluded"]["quota"]) | set(manifest["excluded"]["privacy"])
    if excluded & set(qs):
        fails.append("excluded items are present in the sealed set")
    return fails


def counts(data):
    questions = jsonl(data["questions.jsonl"])
    dom = {q["id"]: q["domain"] for q in questions}
    spl = {q["id"]: q["split"] for q in questions}
    out = {}
    for x in jsonl(data["levels.jsonl"]):
        out.setdefault(dom[x["id"]], {}).setdefault(spl[x["id"]], Counter())[str(x["nivel"])] += 1
    return {d: {s: dict(sorted(c.items())) for s, c in sorted(v.items())} for d, v in sorted(out.items())}


def check_counts(data, manifest) -> list[str]:
    fails = []
    if counts(data) != manifest["counts"]:
        fails.append("manifest counts differ from the files")
    questions = jsonl(data["questions.jsonl"])
    gold = jsonl(data["gold.jsonl"])
    totals = {"states": len(jsonl(data["states.jsonl"])), "questions": len(questions),
              "test": sum(1 for q in questions if q["split"] == "test"),
              "ambiguo_final": sum(g["ambiguo_final"] for g in gold)}
    if totals != manifest["totals"]:
        fails.append(f"manifest totals differ from the files: {totals}")
    return fails


def check_quota(manifest) -> list[str]:
    fails = []
    for d, by_split in manifest["counts"].items():
        if d == "random":
            continue
        c = by_split.get("test", {})
        n = sum(c.get(str(k), 0) for k in QUOTA)
        for k, target_pct in QUOTA.items():
            pct = 100 * c.get(str(k), 0) / n
            if abs(pct - target_pct) > TOLERANCE:
                fails.append(f"{d}: N{k} = {pct:.0f} % (target {target_pct} +- {TOLERANCE})")
    return fails


def check(folder=SEALED) -> list[str]:
    data, manifest = load(folder)
    return check_bytes(data, manifest) + check_gold_rule(data, manifest) + check_counts(data, manifest) + check_quota(manifest)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the sealed files (the default action)")
    ap.add_argument("--folder", default=str(SEALED))
    a = ap.parse_args(argv)
    data, manifest = load(a.folder)
    print(json.dumps({"totals": manifest["totals"], "sha256": manifest["sha256"]}, ensure_ascii=False, indent=1))
    fails = check(a.folder)
    if hashlib.sha256((Path(a.folder) / "states.jsonl").read_bytes()).hexdigest() != manifest["sha256"]["states.jsonl"]:
        print("note: states.jsonl is the shipped redacted file (GoEmotions texts null); "
              "run scripts/fetch_goemotions.py before running models")
    for f in fails:
        print("FAIL:", f)
    print("seal check: " + ("FAILED" if fails else "ok (hashes, canonical LF bytes, gold rule, counts, quota within +-10)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
