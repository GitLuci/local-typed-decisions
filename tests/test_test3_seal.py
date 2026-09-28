"""test3_seal: the shipped sealed set passes the check, and each check fails when its property is broken."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test3_seal as S  # noqa: E402


@pytest.fixture
def copy(tmp_path):
    folder = tmp_path / "sealed"
    shutil.copytree(S.SEALED, folder)
    return folder


def _rows(folder, name):
    return [json.loads(l) for l in (folder / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write(folder, name, rows, key="id", reseal=True):
    b = S.canonical(rows, key)
    (folder / name).write_bytes(b)
    if reseal:  # keep the manifest hash consistent, so only the property under test can fail
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        man["sha256"][name] = hashlib.sha256(b).hexdigest()
        (folder / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


def test_shipped_set_passes():
    assert S.check() == []
    assert S.main(["--check"]) == 0


def test_hash_change_fails(copy):
    p = copy / "gold.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    assert any("sha256" in f for f in S.check(copy))


def test_crlf_fails(copy):
    p = copy / "levels.jsonl"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    fails = S.check(copy)
    assert any("CR" in f for f in fails)


def test_gold_rule_violations_fail(copy):
    gold = _rows(copy, "gold.jsonl")
    amb = next(g for g in gold if g["ambiguo_final"])
    amb["primario"] = True
    gen = next(g for g in gold if g["gold_from"] == "generator" and g["primario"])
    gen["gold_from"] = "R1=R2"
    _write(copy, "gold.jsonl", gold)
    fails = S.check(copy)
    assert any("ambiguo_final" in f for f in fails) and any("generator gold" in f for f in fails)


def test_non_one_hot_labeled_target_fails(copy):
    gold = _rows(copy, "gold.jsonl")
    g = next(g for g in gold if g["primario"])
    k = list(g["target"])
    g["target"] = {x: 1.0 / len(k) for x in k}
    _write(copy, "gold.jsonl", gold)
    assert any("one-hot" in f for f in S.check(copy))


def test_excluded_item_present_fails(copy):
    man = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
    man["excluded"]["privacy"].append(_rows(copy, "questions.jsonl")[0]["id"])
    (copy / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    assert any("excluded" in f for f in S.check(copy))


def test_counts_mismatch_fails(copy):
    levels = _rows(copy, "levels.jsonl")
    x = next(l for l in levels if l["nivel"] == 1)
    x["nivel"] = 3
    _write(copy, "levels.jsonl", levels)
    assert any("counts" in f for f in S.check(copy))


def test_quota_check_fails_outside_and_passes_at_limit():
    fake = {"counts": {"x": {"test": {"1": 40, "2": 30, "3": 30}}, "random": {"test": {"None": 50}}}}
    assert S.check_quota(fake) and "x: N1" in S.check_quota(fake)[0]
    at_limit = {"counts": {"x": {"test": {"1": 30, "2": 30, "3": 40}}}}
    assert S.check_quota(at_limit) == []  # +-10 inclusive
