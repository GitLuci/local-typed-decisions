"""test3_harness: simulated backend, synthetic sealed set, and the pre-registered analysis on known numbers. No models."""
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test3_harness as M  # noqa: E402

Q_CHOICE = {"type": "choice", "instructions": "Is it on Tuesday?", "criteria": {"yes": "Yes", "no": "No", "unknown": "?"}}
Q_NOUL = {"type": "noul", "instructions": "Refund requested?", "criteria": {"true": "Yes", "false": "No"}}
Q_SCORE = {"type": "score", "instructions": "Urgency?", "criteria": ["0", "1", "2", "3"]}
Q_RANDOM = {"type": "choice", "instructions": "Group?", "criteria": {"a": "A", "b": "B", "c": "C"}}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def synthetic_set(folder, n_states=24, seed=7):
    """States with 1-2 questions, 3 labeled domains + random, levels, one ambiguo_final; manifest with hashes."""
    rng = random.Random(seed)
    states, questions, gold, levels = [], [], [], []
    k = 0
    for s in range(n_states):
        sid = f"e{s:03d}"
        split = "dev" if s < 3 else "test"
        kind = ["constructed", "dataset", "judged"][s % 3]
        states.append({"state_id": sid, "text": f"state {sid} tuesday", "lang": "en", "source": {"kind": kind}, "n_frases": 1, "split": split})
        n_q = 2 if s % 4 == 0 else 1
        for j in range(n_q):
            dom, q = [("factual", Q_CHOICE), ("noul_refund", Q_NOUL), ("score_urgency", Q_SCORE)][(s + j) % 3]
            qid = f"q{k:03d}"
            keys = M.keys_for(q)
            target = rng.randrange(len(keys))
            questions.append({"id": qid, "state_id": sid, "domain": dom, "question": q, "target_kind": "factual", "split": split})
            gold.append({"id": qid, "target": {kk: float(i == target) for i, kk in enumerate(keys)}, "gold_from": "R1=R2" if s % 2 else "generator", "ambiguo_final": qid == "q005"})
            levels.append({"id": qid, "nivel": 1 + (k % 3), "nivel_estrutural": 1 + (k % 3), "nivel_rotulagem": None})
            k += 1
    # random: distribution only
    for r in range(3):
        sid = f"r{r}"
        states.append({"state_id": sid, "text": "random", "lang": "en", "source": {"kind": "constructed"}, "n_frases": 1, "split": "test"})
        qid = f"q{k:03d}"
        questions.append({"id": qid, "state_id": sid, "domain": "random", "question": Q_RANDOM, "target_kind": "known_distribution", "split": "test"})
        gold.append({"id": qid, "target": {"a": 0.5, "b": 0.3, "c": 0.2}, "gold_from": "generator", "ambiguo_final": False})
        k += 1
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (("states.jsonl", states), ("questions.jsonl", questions), ("gold.jsonl", gold), ("levels.jsonl", levels)):
        _jsonl(folder / name, rows)
    man = {"sha256": {name: _sha(folder / name) for name in ("states.jsonl", "questions.jsonl", "gold.jsonl", "levels.jsonl")}, "seal_commit": "synthetic"}
    (folder / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return states, questions, gold, levels


def rule_from_gold(gold, questions, correct):
    """Simulated backend: correct(qid) says whether it answers the gold or the next option. Reads the question from the prompt."""
    target = {g["id"]: M.argmax([g["target"][k] for k in M.keys_for(questions[g["id"]]["question"])]) for g in gold}

    def rule(prompt_text):
        parts = prompt_text.split("|")
        text, rest = parts[1], parts[2]  # rest = instructions (+ thought and close, in thinking mode)
        qid = next(q for q in questions if questions[q]["_state_text"] == text and rest.startswith(questions[q]["question"]["instructions"]))
        n = len(M.keys_for(questions[qid]["question"]))
        pick = target[qid] if correct(qid) else (target[qid] + 1) % n
        return [math.log(0.9) if i == pick else math.log(0.1 / (n - 1)) for i in range(n)]
    return rule


def jev_predictions(out_dir, questions, gold, jev_correct):
    """File in the scripts/baseline_test3.py format: one line per state, variants per question."""
    by_state = {}
    for q in questions.values():
        if q["split"] == "test":
            by_state.setdefault(q["state_id"], []).append(q)
    lines = []
    for sid, qs in sorted(by_state.items()):
        answers = {}
        for q in qs:
            keys = M.keys_for(q["question"])
            g = next(x for x in gold if x["id"] == q["id"])
            target = M.argmax([g["target"][k] for k in keys])
            pick = target if jev_correct(q["id"]) else (target + 1) % len(keys)
            dist = {k: (0.8 if i == pick else 0.2 / (len(keys) - 1)) for i, k in enumerate(keys)}
            answers[q["id"]] = {"variants": [dist, dist] if q["question"]["type"] == "choice" else [dist]}
        lines.append({"state_id": sid, "method": "jev_api", "questions": answers})
    (out_dir / "jev").mkdir(parents=True, exist_ok=True)
    _jsonl(out_dir / "jev" / "predictions.jsonl", lines)


@pytest.fixture
def world(tmp_path):
    folder = tmp_path / "sealed"
    states, questions, gold, levels = synthetic_set(folder)
    texts = {st["state_id"]: st["text"] for st in states}
    qmap = {q["id"]: {**q, "_state_text": texts[q["state_id"]]} for q in questions}
    return SimpleNamespace(folder=folder, out=tmp_path / "reports", states=states, questions=qmap, gold=gold, levels=levels)


def _args(world, arm, split="test"):
    return SimpleNamespace(arm=arm, sealed=str(world.folder), split=split, server=None, gguf=None, backend="simulated", out=str(world.out))


def test_manifest_refuses_different_hash(world):
    (world.folder / "questions.jsonl").write_text((world.folder / "questions.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="sha256 differs"):
        M.read_sealed(world.folder)


def test_refuses_without_manifest(tmp_path):
    with pytest.raises(SystemExit, match="no sealed manifest"):
        M.read_sealed(tmp_path)


def test_run_fast_and_think_with_simulated_backend(world):
    wrong = next(q for q, x in world.questions.items() if x["split"] == "test")
    rule = rule_from_gold(world.gold, world.questions, lambda q: q != wrong)
    for arm in ("8b-q8-fast", "8b-q8-short"):
        backend = M.SimulatedBackend(rule, thinking_tokens=5)
        summary = M.run(_args(world, arm), backend=backend, builder=M.SimulatedBuilder())
        rows = M.read_jsonl(world.out / arm / "predictions.jsonl")
        test_ids = [q for q, x in world.questions.items() if x["split"] == "test"]
        assert summary["questions"] == len(test_ids) and {r["id"] for r in rows} == set(test_ids)
        ok = M.model_correct(rows, {g["id"]: g for g in world.gold})
        assert not ok[wrong] and all(ok[q] for q in ok if q != wrong and world.questions[q]["domain"] != "random")
        assert all(abs(sum(r["probabilities"]) - 1) < 1e-9 for r in rows)
        assert all(r["seed"] == M.SEED_BASE + r["index"] for r in rows)
        if arm == "8b-q8-short":
            assert all(r["cap"] == 128 and r["thinking_tokens"] == 5 and r["closed"] for r in rows)
            assert all(c["stop"] == ["</think>"] for c in backend.calls if c["n_predict"] > 1)
        # no dev question was touched
        assert not any(world.questions[r["id"]]["split"] == "dev" for r in rows)
    prov = json.loads((world.out / "8b-q8-short" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["cap"] == 128 and prov["sampling"] == M.SAMPLING and prov["gguf_sha256"] == M.ARMS["8b-q8-short"]["sha256"]


def test_resume_skips_done_and_dev_smoke_is_separate(world):
    rule = rule_from_gold(world.gold, world.questions, lambda q: True)
    b1 = M.SimulatedBackend(rule)
    M.run(_args(world, "4b-q8-fast"), backend=b1, builder=M.SimulatedBuilder())
    n1 = len(b1.calls)
    b2 = M.SimulatedBackend(rule)
    M.run(_args(world, "4b-q8-fast"), backend=b2, builder=M.SimulatedBuilder())
    assert n1 > 0 and len(b2.calls) == 0  # everything already recorded: no new calls
    M.run(_args(world, "4b-q8-fast", split="dev"), backend=M.SimulatedBackend(rule), builder=M.SimulatedBuilder())
    assert (world.out / "4b-q8-fast-dev" / "predictions.jsonl").exists()
    rows = M.read_jsonl(world.out / "4b-q8-fast-dev" / "predictions.jsonl")
    assert rows and all(world.questions[r["id"]]["split"] == "dev" for r in rows)


def test_different_provenance_does_not_mix(world):
    rule = rule_from_gold(world.gold, world.questions, lambda q: True)
    M.run(_args(world, "8b-q8-fast"), backend=M.SimulatedBackend(rule), builder=M.SimulatedBuilder())
    meta = world.out / "8b-q8-fast" / "provenance.json"
    p = json.loads(meta.read_text(encoding="utf-8"))
    p["gguf_sha256"] = "other"
    meta.write_text(json.dumps(p), encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to mix"):
        M.run(_args(world, "8b-q8-fast"), backend=M.SimulatedBackend(rule), builder=M.SimulatedBuilder())


def test_missing_letters_get_floor_and_are_flagged():
    backend = M.SimulatedBackend(lambda text: [0.0, -1.0])  # only 2 of 3 letters
    r = M.read_letters(backend, M.SimulatedBuilder(), "FAST|x|y", 3, seed=1)
    assert r["letters_missing"] == [2] and M.argmax(r["probabilities"]) == 0 and r["probabilities"][2] < r["probabilities"][1]


def test_first_position_probs_formats():
    j1 = {"completion_probabilities": [{"id": 5, "token": "A", "logprob": -0.1, "top_logprobs": [{"id": 5, "logprob": -0.1}, {"id": 6, "logprob": -2.5}]}]}
    j2 = {"completion_probabilities": [{"id": 5, "token": "A", "prob": 0.9, "probs": [{"id": 5, "prob": 0.9}, {"id": 6, "prob": 0.1}]}]}
    assert M._first_position_probs(j1) == {5: -0.1, 6: -2.5}
    p2 = M._first_position_probs(j2)
    assert set(p2) == {5, 6} and abs(p2[5] - math.log(0.9)) < 1e-12
    assert M._first_position_probs({}) == {}


def test_newcombe_and_mcnemar_known_values():
    # no discordant pairs: delta = 0, symmetric CI containing 0
    d, lo, hi = M.newcombe_paired(80, 0, 0, 20)
    assert d == 0 and lo < 0 < hi
    # the model gets 10 right that Jev misses, Jev 0 that the model misses
    d, lo, hi = M.newcombe_paired(80, 10, 0, 10)
    assert abs(d - 0.10) < 1e-12 and 0 < lo < d < hi
    assert M.mcnemar_exact(10, 0) == pytest.approx(2 / 2 ** 10) and M.mcnemar_exact(0, 0) == 1.0
    assert M.mcnemar_one_sided_better(0, 10) == pytest.approx(1 / 2 ** 10) and M.mcnemar_one_sided_better(5, 5) > 0.5
    # Wilson: 0/10
    lo, hi = M.wilson(0, 10)
    assert lo == 0.0 and 0.2 < hi < 0.35
    assert M.holm([0.01, 0.04, 0.03]) == [pytest.approx(0.03), pytest.approx(0.06), pytest.approx(0.06)]
    assert M.holm([0.2]) == [0.2] and M.holm([]) == []


def test_state_bootstrap_is_deterministic_and_covers_delta():
    pairs = [(f"s{i // 2}", i % 3 != 0, i % 4 != 0) for i in range(120)]
    a = M.bootstrap_by_state(pairs, n_boot=500)
    b = M.bootstrap_by_state(pairs, n_boot=500)
    delta = (sum(m for _, m, _ in pairs) - sum(j for _, _, j in pairs)) / len(pairs)
    assert a == b and a[0] <= delta <= a[1]


def test_full_analysis_verdicts_and_ceiling(world):
    gold = {g["id"]: g for g in world.gold}
    # primary model: all correct; Jev misses 10 -> the model is better; 4B misses many -> not at Jev level
    primary = [q for q, x in world.questions.items() if x["split"] == "test" and x["domain"] != "random" and not gold[q]["ambiguo_final"]]
    jev_misses = set(primary[::2][:10])
    jev_predictions(world.out, world.questions, world.gold, lambda q: q not in jev_misses)
    M.run(_args(world, "8b-q8-fast"), backend=M.SimulatedBackend(rule_from_gold(world.gold, world.questions, lambda q: True)), builder=M.SimulatedBuilder())
    M.run(_args(world, "4b-q8-fast"), backend=M.SimulatedBackend(rule_from_gold(world.gold, world.questions, lambda q: int(q[1:]) % 2 == 0)), builder=M.SimulatedBuilder())
    M.run(_args(world, "8b-q8-short"), backend=M.SimulatedBackend(rule_from_gold(world.gold, world.questions, lambda q: q != primary[0])), builder=M.SimulatedBuilder())
    res = M.analyze(SimpleNamespace(sealed=str(world.folder), arms=list(M.ARMS), jev=str(world.out / "jev"), out=str(world.out)))
    assert res["holm_family_arms"] == ["8b-q8-fast", "4b-q8-fast", "8b-q8-short"]  # the 4B thinking arm did not run: excluded
    prim = res["arms"]["8b-q8-fast"]["global"]
    assert prim["primary"] and prim["correct_model"] == prim["n"] and prim["only_jev"] == 0 and prim["only_model"] == len(jev_misses)
    assert prim["jev_level"] and prim["better"] is True and prim["ci95_conservative"][0] > 0
    q4 = res["arms"]["4b-q8-fast"]["global"]
    assert not q4["jev_level"] and q4["better"] is False
    # the ambiguo_final (q005) and random stay outside the primary analysis
    assert res["n_primary"] == sum(1 for q, x in world.questions.items() if x["split"] == "test" and x["domain"] != "random" and not gold[q]["ambiguo_final"])
    # levels and domains with Holm; secondary metrics present; addendum-4 descriptive
    assert set(res["arms"]["8b-q8-fast"]["by_level"]) == {"1", "2", "3"}
    assert all("p_holm" in g for g in res["arms"]["8b-q8-fast"]["by_domain"].values())
    sec = res["arms"]["8b-q8-fast"]["secondary"]
    assert sec["score_mean_level_error"] == 0 and sec["noul_recall"] == {"false": 1.0, "true": 1.0} and not sec["noul_collapse"] and "random_mean_tv" in sec
    assert "descriptive_8b_short_vs_8b_fast" in res and res["descriptive_8b_short_vs_8b_fast"]["only_jev"] == 1
    # ceiling: where Jev is right on every item of a domain, "better" is "not measurable (ceiling)"
    ceilings = [g for g in res["arms"]["8b-q8-fast"]["by_domain"].values() if g["acc_jev"] >= M.CEILING]
    assert all(g["better"] == M.NOT_MEASURABLE for g in ceilings)
    assert (world.out / "results.md").exists() and "8b-q8-fast (primary)" in (world.out / "results.md").read_text(encoding="utf-8")


def test_analyze_fails_clearly_without_baseline(world):
    with pytest.raises(SystemExit, match="baseline predictions not found"):
        M.analyze(SimpleNamespace(sealed=str(world.folder), arms=list(M.ARMS), jev=str(world.out / "absent"), out=str(world.out)))


def test_noul_collapse_is_flagged(world):
    gold = {g["id"]: g for g in world.gold}
    nouls = [q for q, x in world.questions.items() if x["domain"] == "noul_refund" and x["split"] == "test"]
    # always answers "false": recall of "true" = 0 -> collapse
    rows = [{"id": q, "question_type": "noul", "keys": ["false", "true"], "probabilities": [0.9, 0.1]} for q in nouls]
    sec = M.secondary(rows, gold, world.questions)
    assert sec["noul_recall"]["true"] == 0.0 and sec["noul_collapse"]


def test_cli_requires_gguf_with_llama_server(world):
    with pytest.raises(SystemExit):
        M.main(["run", "--arm", "8b-q8-fast", "--sealed", str(world.folder)])
