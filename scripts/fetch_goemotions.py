"""Restore the GoEmotions texts that are not shipped with this repository.

The 44 GoEmotions-derived states of test-3 (and 2 items of the human-audit sample) are shipped with `text: null`
because the GoEmotions data carries no explicit data licence (see DATA_LICENSES.md). This script downloads the
original `test.tsv` from the upstream repository at a pinned commit, checks its SHA-256, fills each text by its
original comment id (`source.orig_id`; surrounding whitespace stripped, as when the set was built) and then verifies
that the restored `examples/test-3/states.jsonl` is byte-identical to the sealed file (manifest hash).

    python scripts/fetch_goemotions.py            # download, restore, verify
    python scripts/fetch_goemotions.py --tsv FILE # use an already downloaded test.tsv
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "2adf640a14f11025ae5a9d0ec493b78530d276d3"
URL = f"https://raw.githubusercontent.com/google-research/google-research/{COMMIT}/goemotions/data/test.tsv"
TSV_SHA256 = "0587b2dd8b27b97352adbfc3fb083d46005c8946657fdc2b1ca8b1cc7f1f8be4"
DATASET = "google-research-datasets/go_emotions (simplified)"
STATES = ROOT / "examples/test-3/states.jsonl"
AUDIT = ROOT / "labels/test-3/audit-sample.jsonl"


def texts_by_id(raw: bytes):
    if hashlib.sha256(raw).hexdigest() != TSV_SHA256:
        raise SystemExit("test.tsv: SHA-256 differs from the pinned upstream file; refusing to restore")
    out = {}
    for line in raw.decode("utf-8").splitlines():
        text, _labels, cid = line.split("\t")
        out[cid] = text.strip()
    return out


def restore_rows(rows, texts, text_key, ref):
    n = 0
    for r in rows:
        src = ref(r)
        if src and src.get("dataset") == DATASET and r.get(text_key) is None:
            r[text_key] = texts[src["orig_id"]]
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tsv", type=Path, help="local copy of goemotions/data/test.tsv")
    a = ap.parse_args(argv)
    raw = a.tsv.read_bytes() if a.tsv else urllib.request.urlopen(URL, timeout=120).read()
    texts = texts_by_id(raw)

    rows = [json.loads(l) for l in STATES.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = restore_rows(rows, texts, "text", lambda r: r.get("source"))
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rows, key=lambda r: r["state_id"])).encode("utf-8")
    sealed = json.loads((STATES.parent / "manifest.json").read_text(encoding="utf-8"))["sha256"]["states.jsonl"]
    if hashlib.sha256(data).hexdigest() != sealed:
        raise SystemExit("restored states.jsonl does not match the sealed hash; nothing written")
    STATES.write_bytes(data)

    audit = [json.loads(l) for l in AUDIT.read_text(encoding="utf-8").splitlines() if l.strip()]
    m = restore_rows(audit, texts, "state_text", lambda r: r.get("state_source"))
    AUDIT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in audit), encoding="utf-8", newline="\n")
    print(json.dumps({"states_restored": n, "audit_items_restored": m, "states_sha256": sealed, "sealed_match": True}))


if __name__ == "__main__":
    sys.exit(main())
