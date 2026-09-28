"""test-3: harness for the local arms (llama-server, Vulkan) over the SEALED set, plus the pre-registered analysis.

Protocol: docs/METHODOLOGY.md (test-3: arms, criteria, sealed files, run order; addendum 4 adds the "8B short 128" arm).
The harness refuses any file whose sha256 does not match examples/test-3/manifest.json. Nothing is trained; prompts
and readouts are those of the final recorded configurations.

Arms, all with cold prefill (`cache_prompt: false`), batch 1, one pass per question:
  8b-q8-fast    Qwen3-8B Q8_0 `bc7efafe...`, fast path (letter readout, thinking off)      - PRIMARY
  4b-q8-fast    Qwen3-4B Q8_0 `78bea492...`, fast path
  4b-q8-think   Qwen3-4B Q8_0, thinks up to 1024 tokens, reads its own answer (natural or forced close)
  8b-q8-short   Qwen3-8B Q8_0, thinks up to 128 tokens, reads its own answer at 128 (addendum 4)
Thinking sampling: T 0.6, top_p 0.95, top_k 20, min_p 0; seed 20260927 + index of the question in the sealed file
(also applied to the 4B thinking arm, recorded in the provenance).

Backend: an already running `llama-server` (`--server http://127.0.0.1:8080`) with the arm's GGUF; the harness checks
the sha256 of the GGUF file (`--gguf`) and records `/props`. Letters are read with `n_predict: 1` and `n_probs`
(full-distribution logprobs, `post_sampling_probs: false`); a thought is a `/completion` with `stop: ["</think>"]`.
`--backend simulated` is a fake backend used only by the tests (never for results).

  python scripts/test3_harness.py run     --arm 8b-q8-fast --server http://127.0.0.1:8080 --gguf models/qwen3-8b-gguf/qwen3-8b-Q8_0.gguf [--split dev]
  python scripts/test3_harness.py analyze [--arms 8b-q8-fast 4b-q8-fast 4b-q8-think 8b-q8-short] --jev <dir with baseline predictions.jsonl>

Output: reports/test-3/<arm>/{predictions.jsonl, provenance.json, summary.json}; reports/test-3/results.{json,md}.
Per-item predictions are not shipped in the repository. The Jev baseline per-item outputs are withheld because they
come from a commercial API; `analyze` needs them (produce them with scripts/baseline_test3.py).
"""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEALED = ROOT / "examples/test-3"
REPORTS = ROOT / "reports/test-3"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
SEED_BASE = 20260927
CLOSE = "\n</think>\n\n"
N_PROBS = 20
ARMS = {
    "8b-q8-fast": {"gguf": "models/qwen3-8b-gguf/qwen3-8b-Q8_0.gguf", "sha256": "bc7efafeb86690a3efe8c0f9babc48a4e747cd4d9cec490c4bec1e298f446506",
                   "tokenizer": "models/qwen3-8b", "mode": "fast", "primary": True},
    "4b-q8-fast": {"gguf": "models/qwen3-4b-gguf/qwen3-4b-Q8_0.gguf", "sha256": "78bea492bbdc8667db4bc7f3affcbc619d3a28b88daf212cd1334441ab6e5a70",
                   "tokenizer": "models/qwen3-4b", "mode": "fast"},
    "4b-q8-think": {"gguf": "models/qwen3-4b-gguf/qwen3-4b-Q8_0.gguf", "sha256": "78bea492bbdc8667db4bc7f3affcbc619d3a28b88daf212cd1334441ab6e5a70",
                    "tokenizer": "models/qwen3-4b", "mode": "think", "cap": 1024},
    "8b-q8-short": {"gguf": "models/qwen3-8b-gguf/qwen3-8b-Q8_0.gguf", "sha256": "bc7efafeb86690a3efe8c0f9babc48a4e747cd4d9cec490c4bec1e298f446506",
                    "tokenizer": "models/qwen3-8b", "mode": "think", "cap": 128},
}
PRIMARY_ARM = "8b-q8-fast"
MARGINS = {"global": 3.0, "level": 5.0, "domain": 8.0}  # percentage points (pre-registered)
CEILING = 0.97
NOT_MEASURABLE = "not measurable (ceiling)"
BOOTSTRAP_N, BOOTSTRAP_SEED = 10_000, 20260927
LABELED_DOMAINS = ("factual", "numeric", "deterministic", "sentence", "sentiment", "subjective_tone", "robotic_style", "noul_refund", "score_urgency")


# ------------------------------------------------------------------------------------------------- utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def keys_for(question):
    t = question["type"]
    if t == "choice":
        return list(question["criteria"])
    if t == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def softmax_from_logprobs(lp):
    m = max(lp)
    p = [math.exp(x - m) for x in lp]
    s = sum(p)
    return [x / s for x in p]


# ------------------------------------------------------------------------------------------------- sealed set
def read_sealed(folder, needed=("states.jsonl", "questions.jsonl"), gold=False, levels=False):
    """Read the sealed set, checking every file against the manifest. Refuse without a manifest or on hash mismatch."""
    folder = Path(folder)
    man_path = folder / "manifest.json"
    if not man_path.exists():
        raise SystemExit(f"{man_path}: no sealed manifest; refusing to run")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    hashes = man.get("sha256", man)
    names = list(needed) + (["gold.jsonl"] if gold else []) + (["levels.jsonl"] if levels else [])
    out = {}
    for name in names:
        p = folder / name
        if name not in hashes:
            raise SystemExit(f"{name}: not in the sealed manifest; refusing to run")
        if not p.exists():
            raise SystemExit(f"{name}: file missing")
        if sha256_file(p) != hashes[name]:
            if sha256_file(p) == man.get("shipped_redacted", {}).get("sha256", {}).get(name):
                raise SystemExit(f"{name}: GoEmotions texts are not restored; run scripts/fetch_goemotions.py first")
            raise SystemExit(f"{name}: sha256 differs from the sealed manifest; refusing to run")
        out[name] = read_jsonl(p)
    out["manifest"] = man
    return out


def questions_of_split(sealed, split):
    """Questions in sealed-file order (the index gives the seed), with their state attached."""
    states = {s["state_id"]: s for s in sealed["states.jsonl"]}
    out = []
    for index, q in enumerate(sealed["questions.jsonl"]):
        if q.get("split", "test") != split:
            continue
        if q["state_id"] not in states:
            raise SystemExit(f"{q['id']}: state {q['state_id']} does not exist")
        out.append({"index": index, "question_row": q, "state": states[q["state_id"]]})
    return out


# ------------------------------------------------------------------------------------------------- prompts
class Builder:
    """Prompts of the final configurations: fast path (production) and thinking (thinking_readout.prompt_text)."""

    def __init__(self, tokenizer_dir):
        from scripts.test2_q8_fastpath import PromptOnly
        self.base = PromptOnly(str(tokenizer_dir))
        self.tok = self.base.tok
        self.letters = self.base.alias_token_ids
        self.close_ids = self.tok.encode(CLOSE, add_special_tokens=False)

    def fast_ids(self, state_text, question):
        seqs, sizes, _ = self.base._sequences(state_text, {"q": question})
        return seqs[0], sizes[0]

    def think_ids(self, state_text, question):
        from scripts.thinking_readout import prompt_text
        text = prompt_text(self.base, {"state": state_text, "question": question})
        return self.tok.encode(text, add_special_tokens=False), len(question["criteria"]) if question["type"] != "noul" else 2

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)


# ------------------------------------------------------------------------------------------------- backends
class LlamaServer:
    """An already running `llama-server`. `/completion` with the prompt as ids; letters via n_probs (full distribution)."""

    def __init__(self, url, timeout=3600.0):
        import httpx
        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def props(self):
        r = self.client.get(f"{self.url}/props")
        r.raise_for_status()
        return r.json()

    def complete(self, ids, n_predict, seed=0, stop=None, sampling=None, n_probs=0):
        body = {"prompt": ids, "n_predict": n_predict, "seed": seed, "cache_prompt": False, "n_probs": n_probs,
                "post_sampling_probs": False, "stop": stop or []}
        body.update(sampling or {"temperature": 0.0})
        started = time.perf_counter()
        r = self.client.post(f"{self.url}/completion", json=body)
        r.raise_for_status()
        j = r.json()
        return {"content": j.get("content", ""), "tokens_predicted": j.get("tokens_predicted", 0),
                "stopped_word": j.get("stop_type") == "word" or bool(j.get("stopped_word", False)),
                "first_probs": _first_position_probs(j), "elapsed_ms": (time.perf_counter() - started) * 1000,
                "timings": j.get("timings", {})}


def _first_position_probs(j):
    """{token_id: logprob} of the first generated position, in the formats llama-server has used."""
    cp = j.get("completion_probabilities") or []
    if not cp:
        return {}
    first = cp[0]
    cands = first.get("top_logprobs") or first.get("top_probs") or first.get("probs") or []
    out = {}
    for c in cands:
        tid = c.get("id")
        if tid is None:
            continue
        if "logprob" in c:
            out[int(tid)] = float(c["logprob"])
        elif "prob" in c:
            out[int(tid)] = math.log(max(float(c["prob"]), 1e-300))
    if "id" in first and first["id"] not in out and "logprob" in first:
        out[int(first["id"])] = float(first["logprob"])
    return out


class SimulatedBackend:
    """Fake backend for tests: answers by a rule over the prompt text. Never used for results."""

    def __init__(self, rule, thinking_tokens=5):
        self.rule = rule            # rule(prompt_text) -> list of logprobs per letter (A, B, C, ...)
        self.thinking_tokens = thinking_tokens
        self.calls = []

    def props(self):
        return {"model_path": "simulated", "simulated": True}

    def complete(self, ids, n_predict, seed=0, stop=None, sampling=None, n_probs=0):
        text = ids if isinstance(ids, str) else " ".join(map(str, ids))
        self.calls.append({"n_predict": n_predict, "seed": seed, "stop": stop})
        if n_predict > 1:  # thought
            return {"content": " thought.", "tokens_predicted": min(self.thinking_tokens, n_predict), "stopped_word": self.thinking_tokens < n_predict,
                    "first_probs": {}, "elapsed_ms": 1.0, "timings": {}}
        lp = self.rule(text)
        return {"content": "", "tokens_predicted": 1, "stopped_word": False,
                "first_probs": {i: v for i, v in enumerate(lp)}, "elapsed_ms": 1.0, "timings": {}}


class SimulatedBuilder:
    """Toy prompts: the "prompt" is the text plus the question instructions, and the letters are ids 0..9."""
    letters = list(range(10))
    close_ids = [99]

    def __init__(self, ids=None):
        pass

    def fast_ids(self, state_text, question):
        return f"FAST|{state_text}|{question['instructions']}", len(keys_for(question))

    def think_ids(self, state_text, question):
        return f"THINK|{state_text}|{question['instructions']}", len(keys_for(question))

    def encode(self, text):
        return [len(text)]


# ------------------------------------------------------------------------------------------------- run one arm
def read_letters(backend, builder, ids, size, seed):
    res = backend.complete(ids, n_predict=1, seed=seed, n_probs=N_PROBS)
    probs = res["first_probs"]
    letter_ids = builder.letters[:size]
    lp = [probs.get(t, None) for t in letter_ids]
    missing = [i for i, v in enumerate(lp) if v is None]
    floor = (min(probs.values()) - 5.0) if probs else -50.0
    lp = [floor if v is None else v for v in lp]
    return {"logprobs": lp, "probabilities": softmax_from_logprobs(lp), "letters_missing": missing, "elapsed_ms": res["elapsed_ms"], "timings": res["timings"]}


def run_question(arm, backend, builder, item):
    q, state = item["question_row"]["question"], item["state"]
    keys = keys_for(q)
    seed = SEED_BASE + item["index"]
    row = {"id": item["question_row"]["id"], "state_id": state["state_id"], "index": item["index"], "domain": item["question_row"].get("domain"),
           "question_type": q["type"], "keys": keys, "seed": seed}
    if arm["mode"] == "fast":
        ids, size = builder.fast_ids(state["text"], q)
        r = read_letters(backend, builder, ids, size, seed)
        row.update({"probabilities": r["probabilities"], "logprobs": r["logprobs"], "letters_missing": r["letters_missing"],
                    "prompt_tokens": len(ids) if not isinstance(ids, str) else None, "elapsed_ms": r["elapsed_ms"], "timings": r["timings"]})
        return row
    ids, size = builder.think_ids(state["text"], q)
    t = backend.complete(ids, n_predict=arm["cap"], seed=seed, stop=["</think>"], sampling=SAMPLING)
    thought = t["content"]
    read_ids = (ids + builder.encode(thought.rstrip()) + builder.close_ids) if not isinstance(ids, str) else ids + thought + CLOSE
    r = read_letters(backend, builder, read_ids, size, seed)
    row.update({"probabilities": r["probabilities"], "logprobs": r["logprobs"], "letters_missing": r["letters_missing"],
                "thought": thought, "thinking_tokens": t["tokens_predicted"], "closed": t["stopped_word"], "cap": arm["cap"],
                "prompt_tokens": len(ids) if not isinstance(ids, str) else None,
                "generation_ms": t["elapsed_ms"], "read_ms": r["elapsed_ms"], "elapsed_ms": t["elapsed_ms"] + r["elapsed_ms"], "timings": t["timings"]})
    return row


def run(args, backend=None, builder=None):
    arm_name = args.arm
    arm = ARMS[arm_name]
    sealed = read_sealed(args.sealed)
    items = questions_of_split(sealed, args.split)
    if not items:
        raise SystemExit(f"no questions in split {args.split}")
    out = Path(args.out) / (arm_name if args.split == "test" else f"{arm_name}-{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    if backend is None:
        if args.backend == "simulated":
            raise SystemExit("the simulated backend exists only for tests; pass it in code")
        gguf = Path(args.gguf)
        got = sha256_file(gguf)
        if got != arm["sha256"]:
            raise SystemExit(f"GGUF {gguf.name}: sha256 {got[:12]}... != {arm['sha256'][:12]}... for arm {arm_name}; refusing to run")
        backend = LlamaServer(args.server)
        builder = Builder(ROOT / arm["tokenizer"])
    props = backend.props()
    manifest = sealed["manifest"]
    provenance = {"protocol": "docs/METHODOLOGY.md, test-3 arms and criteria + addendum 4", "arm": arm_name, "mode": arm["mode"], "cap": arm.get("cap"),
                  "gguf": arm["gguf"], "gguf_sha256": arm["sha256"], "split": args.split, "server": getattr(args, "server", None), "server_props": props,
                  "seal": manifest.get("seal_commit", manifest.get("commit")),
                  "sha256": {k: manifest.get("sha256", manifest).get(k) for k in ("states.jsonl", "questions.jsonl")},
                  "sampling": SAMPLING if arm["mode"] == "think" else None, "seed": f"{SEED_BASE} + index of the question in the sealed file",
                  "readout": "letters via n_probs of the full distribution, n_predict 1, cache_prompt false, batch 1",
                  "head_commit": _head(), "script_sha256": sha256_file(Path(__file__)), "weights_trained": False, "command": " ".join(sys.argv)}
    meta = out / "provenance.json"
    if meta.exists():
        old = json.loads(meta.read_text(encoding="utf-8"))
        for k in ("gguf_sha256", "sha256", "split", "mode", "cap"):
            if old.get(k) != provenance.get(k):
                raise SystemExit(f"provenance differs from the run already started ({k}); refusing to mix runs")
    meta.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    raw = out / "predictions.jsonl"
    done = read_jsonl(raw) if raw.exists() else []
    seen = {r["id"] for r in done}
    with raw.open("a", encoding="utf-8") as log:
        for item in items:
            if item["question_row"]["id"] in seen:
                continue
            row = run_question(arm, backend, builder, item)
            done.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"id": row["id"], "done": len(done), "ms": round(row["elapsed_ms"])}), flush=True)
    ms = sorted(r["elapsed_ms"] for r in done)
    summary = {"arm": arm_name, "split": args.split, "questions": len(done), "median_ms": ms[len(ms) // 2], "mean_ms": sum(ms) / len(ms),
               "letters_missing": sum(1 for r in done if r["letters_missing"]),
               **({"median_thinking_tokens": sorted(r["thinking_tokens"] for r in done)[len(done) // 2],
                   "closed": sum(r["closed"] for r in done)} if arm["mode"] == "think" else {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def _head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_paired(a, b, c, d):
    """Newcombe (1998, method 10, hybrid score) 95 % CI for p1 - p2 on paired data.
    a: both correct; b: only the first correct; c: only the second correct; d: both wrong. p1 = (a+b)/n, p2 = (a+c)/n."""
    n = a + b + c + d
    if n == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    m1, m2, n1, n2 = a + b, c + d, a + c, b + d
    if min(m1, m2, n1, n2) == 0:
        phi = 0.0
    else:
        phi = (a * d - b * c) / math.sqrt(m1 * m2 * n1 * n2)
        phi = max(phi, 0.0)  # Newcombe: negative correlations are treated as 0
    delta = p1 - p2
    lower = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2)))
    upper = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 + (p2 - l2) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2)))
    return delta, lower, upper


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mcnemar_one_sided_better(b, c):
    """p of "the model is right more often" (c only-model > b only-Jev), exact binomial."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(c, n + 1)) / 2 ** n


def bootstrap_by_state(pairs, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """pairs: [(state_id, model_ok, jev_ok)]; resample states with replacement; percentile CI of the accuracy difference."""
    by_state = {}
    for sid, m, j in pairs:
        by_state.setdefault(sid, []).append((m, j))
    states = list(by_state.values())
    rng = random.Random(seed)
    diffs = []
    k = len(states)
    for _ in range(n_boot):
        tm = tj = tn = 0
        for _ in range(k):
            for m, j in states[rng.randrange(k)]:
                tm += m
                tj += j
                tn += 1
        diffs.append((tm - tj) / tn if tn else 0.0)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]


def holm(pvals):
    """Holm correction: returns adjusted p-values in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def compare(pairs, margin):
    """pairs: [(state_id, model_ok, jev_ok)]. Delta in points, Newcombe and state-bootstrap CIs, McNemar, verdicts."""
    a = sum(1 for _, m, j in pairs if m and j)
    b = sum(1 for _, m, j in pairs if not m and j)   # only Jev
    c = sum(1 for _, m, j in pairs if m and not j)   # only the model
    d = sum(1 for _, m, j in pairs if not m and not j)
    n = len(pairs)
    delta, lo, hi = newcombe_paired(a, c, b, d)       # p1 = model (a+c), p2 = Jev (a+b)
    blo, bhi = bootstrap_by_state(pairs) if n else (0.0, 0.0)
    lo_cons, hi_cons = min(lo, blo), max(hi, bhi)     # if they disagree on the decision, the more conservative wins
    acc_m, acc_j = (a + c) / n if n else 0.0, (a + b) / n if n else 0.0
    return {"n": n, "correct_model": a + c, "correct_jev": a + b, "acc_model": acc_m, "acc_jev": acc_j,
            "delta_points": 100 * delta, "ci95_newcombe": [100 * lo, 100 * hi], "ci95_state_bootstrap": [100 * blo, 100 * bhi],
            "ci95_conservative": [100 * lo_cons, 100 * hi_cons], "only_jev": b, "only_model": c,
            "mcnemar_exact_p": mcnemar_exact(b, c), "p_one_sided_better": mcnemar_one_sided_better(b, c),
            "ceiling": acc_j >= CEILING, "jev_level": (100 * lo_cons) > -margin, "better_raw": (100 * lo_cons) > 0}


def jev_correct(jev_dir, gold, questions):
    """Per-question Jev correctness from <jev_dir>/predictions.jsonl (mean over letter-order variants)."""
    path = Path(jev_dir) / "predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"{path}: baseline predictions not found. Per-item Jev outputs are not shipped (commercial API); "
                         "produce them with scripts/baseline_test3.py and pass --jev")
    ok = {}
    for line in read_jsonl(path):
        for qid, info in line["questions"].items():
            if gold.get(qid, {}).get("target") is None:  # ambiguo_final: no gold, outside the primary analysis
                continue
            keys = keys_for(questions[qid]["question"])
            vs = info["variants"]
            mean = [sum(v[k] for v in vs) / len(vs) for k in keys]
            ok[qid] = argmax(mean) == argmax([gold[qid]["target"][k] for k in keys])
    return ok


def model_correct(pred_rows, gold):
    return {r["id"]: argmax(r["probabilities"]) == argmax([gold[r["id"]]["target"][k] for k in r["keys"]]) for r in pred_rows
            if r["id"] in gold and gold[r["id"]].get("target") is not None}


def secondary(pred_rows, gold, questions):
    out = {}
    scores = [r for r in pred_rows if r["question_type"] == "score" and r["id"] in gold]
    if scores:
        out["score_mean_level_error"] = sum(abs(argmax(r["probabilities"]) - argmax([gold[r["id"]]["target"][k] for k in r["keys"]])) for r in scores) / len(scores)
    nouls = [r for r in pred_rows if r["question_type"] == "noul" and r["id"] in gold]
    if nouls:
        rec = {}
        for cls, idx in (("false", 0), ("true", 1)):
            pos = [r for r in nouls if argmax([gold[r["id"]]["target"][k] for k in r["keys"]]) == idx]
            rec[cls] = (sum(argmax(r["probabilities"]) == idx for r in pos) / len(pos)) if pos else None
        out["noul_recall"] = rec
        out["noul_collapse"] = any(v is not None and v < 0.80 for v in rec.values())
    rnd = [r for r in pred_rows if questions[r["id"]].get("domain") == "random" and r["id"] in gold]
    if rnd:
        out["random_mean_tv"] = sum(0.5 * sum(abs(p - gold[r["id"]]["target"][k]) for p, k in zip(r["probabilities"], r["keys"])) for r in rnd) / len(rnd)
    return out


def analyze(args):
    sealed = read_sealed(args.sealed, gold=True, levels=True)
    questions = {q["id"]: q for q in sealed["questions.jsonl"]}
    states = {s["state_id"]: s for s in sealed["states.jsonl"]}
    gold = {g["id"]: g for g in sealed["gold.jsonl"]}
    levels = {l["id"]: l for l in sealed["levels.jsonl"]}
    primary_ids = [qid for qid, q in questions.items() if q.get("split", "test") == "test" and q.get("domain") != "random"
                   and qid in gold and not gold[qid].get("ambiguo_final", False)]
    jev_ok = jev_correct(args.jev, gold, questions)
    reports = Path(args.out)
    result = {"protocol": "docs/METHODOLOGY.md, test-3 criteria + addendum 4", "n_primary": len(primary_ids), "arms": {}, "holm_family_arms": []}
    arms = [a for a in args.arms if (reports / a / "predictions.jsonl").exists()]
    if not arms:
        raise SystemExit("no arm with predictions.jsonl")
    result["holm_family_arms"] = arms  # fixed before reading any result: only the arms that ran
    ok_by_arm = {}
    for arm in arms:
        rows = read_jsonl(reports / arm / "predictions.jsonl")
        ok = model_correct(rows, gold)
        ok_by_arm[arm] = ok
        missing = [q for q in primary_ids if q not in ok or q not in jev_ok]
        ids = [q for q in primary_ids if q in ok and q in jev_ok]
        pairs = [(questions[q]["state_id"], ok[q], jev_ok[q]) for q in ids]
        r = {"missing_questions": len(missing), "global": compare(pairs, MARGINS["global"]), "by_level": {}, "by_domain": {}, "strata": {}}
        for level in (1, 2, 3):
            sub = [(questions[q]["state_id"], ok[q], jev_ok[q]) for q in ids if levels.get(q, {}).get("nivel") == level]
            if sub:
                r["by_level"][str(level)] = compare(sub, MARGINS["level"])
        for dom in LABELED_DOMAINS:
            sub = [(questions[q]["state_id"], ok[q], jev_ok[q]) for q in ids if questions[q].get("domain") == dom]
            if sub:
                r["by_domain"][dom] = compare(sub, MARGINS["domain"])
        for kind in ("constructed", "dataset", "judged"):
            sub = [(questions[q]["state_id"], ok[q], jev_ok[q]) for q in ids if states[questions[q]["state_id"]].get("source", {}).get("kind") == kind]
            if sub:
                r["strata"][kind] = compare(sub, MARGINS["domain"])
        unan = [(questions[q]["state_id"], ok[q], jev_ok[q]) for q in ids if gold[q].get("gold_from") == "R1=R2"]
        if unan:
            r["strata"]["unanimous_R1=R2"] = compare(unan, MARGINS["domain"])
        # Holm within the arm: levels (3) and domains (9), over the one-sided "better" p-value
        for group in (r["by_level"], r["by_domain"]):
            names = list(group)
            adj = holm([group[k]["p_one_sided_better"] for k in names]) if names else []
            for k, p in zip(names, adj):
                group[k]["p_holm"] = p
                group[k]["better"] = NOT_MEASURABLE if group[k]["ceiling"] else bool(group[k]["better_raw"] and p < 0.05)
                group[k]["jev_level"] = bool(group[k]["jev_level"])
        r["secondary"] = secondary(rows, gold, questions)
        times = sorted(x["elapsed_ms"] for x in rows)
        r["cost"] = {"median_ms": times[len(times) // 2], "questions": len(rows)}
        if rows and "thinking_tokens" in rows[0]:
            r["cost"]["median_thinking_tokens"] = sorted(x["thinking_tokens"] for x in rows)[len(rows) // 2]
            r["cost"]["closed"] = sum(x["closed"] for x in rows)
        result["arms"][arm] = r
    # Holm across arms (global "better" claims)
    adj = holm([result["arms"][a]["global"]["p_one_sided_better"] for a in arms])
    for a, p in zip(arms, adj):
        g = result["arms"][a]["global"]
        g["p_holm_arms"] = p
        g["better"] = NOT_MEASURABLE if g["ceiling"] else bool(g["better_raw"] and p < 0.05)
        g["jev_level"] = bool(g["jev_level"])
        g["primary"] = a == PRIMARY_ARM
    if "8b-q8-short" in ok_by_arm and "8b-q8-fast" in ok_by_arm:  # addendum 4: descriptive only, no verdict
        ids = [q for q in primary_ids if q in ok_by_arm["8b-q8-short"] and q in ok_by_arm["8b-q8-fast"]]
        pairs = [(questions[q]["state_id"], ok_by_arm["8b-q8-short"][q], ok_by_arm["8b-q8-fast"][q]) for q in ids]
        result["descriptive_8b_short_vs_8b_fast"] = {k: v for k, v in compare(pairs, MARGINS["global"]).items() if k not in ("jev_level", "better_raw", "ceiling")}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports / "results.md").write_text(report_md(result), encoding="utf-8")
    print(json.dumps({a: {"acc": r["global"]["correct_model"], "jev": r["global"]["correct_jev"], "n": r["global"]["n"], "delta": round(r["global"]["delta_points"], 2),
                          "ci": [round(x, 2) for x in r["global"]["ci95_conservative"]], "jev_level": r["global"]["jev_level"], "better": r["global"]["better"]}
                      for a, r in result["arms"].items()}, ensure_ascii=False, indent=1))
    return result


def _f(x, nd=1):
    return f"{x:.{nd}f}"


def report_md(res):
    lines = ["# test-3 results, generated by scripts/test3_harness.py analyze", "",
             f"Primary: {res['n_primary']} sealed labeled questions, excluding `ambiguo_final`. Holm family of arms: {', '.join(res['holm_family_arms'])}.", "",
             "| arm | correct | Jev | delta (points) | conservative 95 % CI | only Jev / only model | McNemar p | Holm p | Jev level | better |", "|---|---:|---:|---:|---|---|---:|---:|---|---|"]
    for a, r in res["arms"].items():
        g = r["global"]
        lines.append(f"| {a}{' (primary)' if g.get('primary') else ''} | {g['correct_model']}/{g['n']} | {g['correct_jev']} | {_f(g['delta_points'])} | [{_f(g['ci95_conservative'][0])}; {_f(g['ci95_conservative'][1])}] | {g['only_jev']} / {g['only_model']} | {_f(g['mcnemar_exact_p'], 3)} | {_f(g['p_holm_arms'], 3)} | {'yes' if g['jev_level'] else 'no'} | {g['better']} |")
    for a, r in res["arms"].items():
        lines += ["", f"## {a}", "", "| stratum | n | correct | Jev | delta | 95 % CI | Jev level | better (Holm) |", "|---|---:|---:|---:|---:|---|---|---|"]
        for group, label in (("by_level", "level"), ("by_domain", "domain"), ("strata", "stratum")):
            for k, g in r[group].items():
                lines.append(f"| {label} {k} | {g['n']} | {g['correct_model']} | {g['correct_jev']} | {_f(g['delta_points'])} | [{_f(g['ci95_conservative'][0])}; {_f(g['ci95_conservative'][1])}] | {'yes' if g['jev_level'] else 'no'} | {g.get('better', '-')} |")
        lines += ["", f"Secondary: {json.dumps(r['secondary'], ensure_ascii=False)}", f"Cost: {json.dumps(r['cost'], ensure_ascii=False)}"]
    if "descriptive_8b_short_vs_8b_fast" in res:
        d = res["descriptive_8b_short_vs_8b_fast"]
        lines += ["", f"Descriptive (addendum 4): 8B short - 8B fast = {_f(d['delta_points'])} points, CI [{_f(d['ci95_conservative'][0])}; {_f(d['ci95_conservative'][1])}], no verdict."]
    lines += ["", "Strata (level/domain/source) are descriptive; the yes/no marks there apply the global margin rule and carry no verdict."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)
    c = sub.add_parser("run", help="one arm over the sealed set (or over dev, as a format smoke test)")
    c.add_argument("--arm", choices=sorted(ARMS), required=True)
    c.add_argument("--sealed", default=str(SEALED))
    c.add_argument("--split", choices=["test", "dev"], default="test")
    c.add_argument("--server", default="http://127.0.0.1:8080")
    c.add_argument("--gguf", help="path of the GGUF loaded in the server (sha256 checked against the arm)")
    c.add_argument("--backend", choices=["llama-server", "simulated"], default="llama-server")
    c.add_argument("--out", default=str(REPORTS))
    a = sub.add_parser("analyze", help="delta vs Jev with Newcombe + state-bootstrap CI, McNemar, Holm, ceiling")
    a.add_argument("--sealed", default=str(SEALED))
    a.add_argument("--arms", nargs="+", default=list(ARMS))
    a.add_argument("--jev", default=str(REPORTS / "jev"), help="folder with the baseline predictions.jsonl (not shipped)")
    a.add_argument("--out", default=str(REPORTS))
    args = p.parse_args(argv)
    if args.phase == "run":
        if args.backend == "llama-server" and not args.gguf:
            p.error("--gguf is required with llama-server (sha256 check)")
        return run(args)
    return analyze(args)


if __name__ == "__main__":
    main()
