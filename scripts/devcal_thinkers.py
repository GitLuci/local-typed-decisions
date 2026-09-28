"""Candidate thinkers on dev+cal: each thinks once per case; Qwen3-0.6B reads the letters (primary), the thinker too.

Protocol: docs/HISTORY.md (candidate thinkers). The thinker uses its own chat template (same system prompt and user content
as production, lettered options), thinking on, Qwen's thinking-mode sampling, seed 20260925 + case index, up to
1024 tokens or its own </think>. Phases (separate invocations):

  think --thinker NAME     generate one thought per labelled dev+cal case (resumable), own reading right after
  read  --thinker NAME     Qwen3-0.6B FP32 reads the thought (own_think placement) at budgets 64/128/256/512/full
  compare                  criterion for every thinker with a reading, speed ratio against the reference thinker

No training. Development/calibration only; the test split is never loaded here.
"""
import argparse
import gc
import json
import os
from pathlib import Path
try:  # Unix only; on Windows the peak RSS is recorded as null
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typed_decisions.qwen import SYSTEM, option_descriptions, render
from scripts.structural_readout import CORPUS, argmax, mcnemar_exact, paired_bootstrap, read_jsonl, sha256
from scripts.thinking_readout import CLOSE, SAMPLING, SEED, prompt_text

LETTERS = "ABCDEFGHIJ"
CAP = 1024
BUDGETS = (64, 128, 192, 256, 512, None)  # None = the full natural thought (or the forced close at CAP)
SELECTION_BUDGETS = (64, 128, 192, 256, None)  # pre-registered list for the 1.7B budget choice
SELECTION_MAX_LOSS = 1
THINKERS = {
    "lfm2.5-1.2b-thinking": {"backend": "hf", "lock": "lfm2.5-1.2b-thinking.lock.json"},
    "qwen3.5-2b": {"backend": "hf", "lock": "qwen3.5-2b.lock.json"},
    "qwen3-4b-thinking-2507-q4_k_m": {"backend": "gguf", "lock": "qwen3-4b-thinking-2507.lock.json",
                                      "gguf": "models/gguf/qwen3-4b-thinking-2507-Q4_K_M.gguf",
                                      "manifest": "reports/manifests/gguf-manifest-qwen3-4b-thinking-2507.json"},
    "qwen3-1.7b-forced128": {"backend": "hf", "lock": "qwen-1.7b.lock.json", "min_think": 128},
    "qwen3-8b-q8_0": {"backend": "gguf", "lock": "qwen-8b.lock.json", "gguf": "models/gguf/qwen3-8b-Q8_0.gguf",
                      "manifest": "reports/manifests/gguf-manifest-qwen3-8b.json", "n_ctx": 2048, "use_mmap": False,
                      "gguf_sha256": "bc7efafeb86690a3efe8c0f9babc48a4e747cd4d9cec490c4bec1e298f446506",
                      "fastpath_parity": "reports/quant-8b/q8_0/parity.jsonl",
                      "protocol": "docs/HISTORY.md (8B Q8_0 thinker on dev+cal)"},
}
REFERENCE = "qwen3-1.7b-forced128"
SELECTION_THINKERS = (REFERENCE, "qwen3-8b-q8_0")  # thinkers whose budget is chosen by the sealed rule
OUT = ROOT / "reports/devcal-thinkers"
NO_THINK_4B = ROOT / "reports/structural-readout-4b/predictions-devcal.jsonl"  # from structural_readout.py; not distributed
THINK_06B = ROOT / "reports/thinking-06b-devcal/predictions.jsonl"  # from thinking_readout.py; not distributed
THREADS = 4


def sha256_file(path):
    """Streaming SHA-256 (structural_readout.sha256 reads the whole file into memory: 8.7 GB for the 8B GGUF)."""
    import hashlib
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cases_devcal():
    rows = [c for c in read_jsonl(CORPUS) if c["split"] in ("development", "calibration")
            and c["target_kind"] != "known_distribution"]
    assert len(rows) == 45
    return rows


def user_content(case):
    q = case["question"]
    options = "\n".join(f"{LETTERS[i]}: {s}" for i, s in enumerate(option_descriptions(q)))
    return (f"STATE:\n{render(case['state'])}\n\nQUESTION:\n" + render(q["instructions"]) +
            "\n\nOPTIONS:\n" + options + "\n\nAnswer with the letter code only.")


def thinker_prompt(tok, case):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_content(case)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return text, text.rstrip().endswith("<think>")


def letter_ids(tok, n=3):
    ids = []
    for letter in LETTERS[:n]:
        enc = tok.encode(letter, add_special_tokens=False)
        assert len(enc) == 1, (letter, enc)
        ids.append(enc[0])
    return ids


def split_thought(text):
    """Body of the thought and whether the thinker closed it itself."""
    closed = "</think>" in text
    body = text.split("</think>")[0] if closed else text
    if body.lstrip().startswith("<think>"):
        body = body.lstrip()[len("<think>"):]
    return body.strip("\n"), closed


def peak_rss_gib():
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2 if resource else None)


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()


def base_provenance(name, spec, lock):
    prov = {"protocol": "docs/HISTORY.md (candidate thinkers)", "thinker": name, "backend": spec["backend"], "lock": lock,
            "cap": CAP, "sampling": SAMPLING, "seed_base": SEED, "min_think": spec.get("min_think", 0),
            "forced_close": CLOSE, "threads": THREADS, "corpus_sha256": sha256(CORPUS), "head_commit": git_head(),
            "script_sha256": sha256(Path(__file__)), "weights_trained": False, "imatrix": False,
            "typed_decisions_sha256": {p.name: sha256(p) for p in sorted((ROOT / "typed_decisions").glob("*.py"))},
            "python": sys.version.split()[0]}
    try:
        import torch, transformers, numpy
        prov["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                            "numpy": numpy.__version__}
    except ImportError:
        pass
    return prov


# ----------------------------------------------------------------------------------------------- think (HF)
class HFThinker:
    def __init__(self, path, min_think):
        import torch, transformers
        torch.set_num_threads(THREADS)
        self.torch = torch
        self.tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
        try:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32)
        except (ValueError, KeyError):
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(path, local_files_only=True,
                                                                                  dtype=torch.float32)
        self.model.eval()
        self.min_think = min_think
        self.letters = letter_ids(self.tok)
        closer = self.tok.encode("</think>", add_special_tokens=False)
        self.closer = closer[0] if len(closer) == 1 else None
        self.info = {"model_class": type(self.model).__name__, "closer_single_token": self.closer is not None,
                     "letter_ids": self.letters}

    def think(self, case, index):
        t, tok = self.torch, self.tok
        text, pre_opened = thinker_prompt(tok, case)
        ids = tok.encode(text, add_special_tokens=False)
        t.manual_seed(SEED + index)
        stop = {"eos_token_id": self.closer} if self.closer is not None else {"stop_strings": ["</think>"], "tokenizer": tok}
        started = time.perf_counter()
        with t.inference_mode():
            gen = self.model.generate(t.tensor([ids]), attention_mask=t.ones(1, len(ids), dtype=t.long),
                                      max_new_tokens=CAP, min_new_tokens=self.min_think, **stop, **SAMPLING)[0, len(ids):].tolist()
        seconds = time.perf_counter() - started
        raw = tok.decode(gen, skip_special_tokens=False)
        body, closed = split_thought(raw)
        n_body = len(tok.encode(body, add_special_tokens=False))
        return {"prompt_tokens": len(ids), "pre_opened": pre_opened, "generated_ids": gen, "generated_tokens": len(gen),
                "thinking_tokens": n_body, "closed": closed, "generation_s": seconds, "thought": body}

    def read_own(self, case, body):
        t, tok = self.torch, self.tok
        text, pre_opened = thinker_prompt(tok, case)
        prefix = text + ("" if pre_opened else "<think>\n") + body.rstrip() + CLOSE
        ids = tok.encode(prefix, add_special_tokens=False)
        started = time.perf_counter()
        with t.inference_mode():
            try:
                logits = self.model(input_ids=t.tensor([ids]), use_cache=False, logits_to_keep=1).logits[0, -1]
            except TypeError:
                logits = self.model(input_ids=t.tensor([ids]), use_cache=False).logits[0, -1]
            size = len(case["question"]["criteria"])
            letters = logits[self.letters[:size]].float().tolist()
            mass = logits.float().log_softmax(-1)[self.letters[:size]].exp().sum().item()
        return {"own_logits": letters, "own_letter_mass": mass, "own_read_s": time.perf_counter() - started,
                "own_read_tokens": len(ids)}


# --------------------------------------------------------------------------------------------- think (GGUF)
class GGUFThinker:
    def __init__(self, tokenizer_dir, gguf, n_ctx=4096, expected_sha256=None, use_mmap=True):
        import numpy as np
        from llama_cpp import Llama
        import transformers
        self.np = np
        digest = sha256_file(gguf)  # one streaming pass (8.7 GB for the 8B; the disk of this VM can be slow)
        if expected_sha256 is not None:
            assert digest == expected_sha256, "GGUF differs from the sealed hash"
        self.tok = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        # use_mmap=False for the 8B: with mmap, llama.cpp repacks Q8_0 weights for AVX-512 into anonymous memory and the
        # file pages stay resident too (13.3 GiB, OOM on this 15 GB VM); without mmap it is 9.4 GiB.
        self.llm = Llama(model_path=str(gguf), n_ctx=n_ctx, n_threads=THREADS, n_batch=512, logits_all=False,
                         verbose=False, seed=0, use_mmap=use_mmap)
        self.letters = letter_ids(self.tok)
        self.min_think = 0
        import llama_cpp
        self.info = {"llama_cpp_python": llama_cpp.__version__, "gguf": str(gguf), "gguf_sha256": digest,
                     "letter_ids": self.letters, "n_ctx": n_ctx, "use_mmap": use_mmap}

    def think(self, case, index):
        text, pre_opened = thinker_prompt(self.tok, case)
        ids = self.tok.encode(text, add_special_tokens=False)
        assert self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True) == ids, "llama.cpp tokens differ from HF"
        started = time.perf_counter()
        res = self.llm.create_completion(ids, max_tokens=CAP, stop=["</think>"], seed=SEED + index,
                                         temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"],
                                         top_k=SAMPLING["top_k"], min_p=0.0, repeat_penalty=1.0)
        seconds = time.perf_counter() - started
        raw = res["choices"][0]["text"]
        closed = res["choices"][0]["finish_reason"] == "stop"
        body, _ = split_thought(raw + ("</think>" if closed else ""))
        n = res["usage"]["completion_tokens"]
        return {"prompt_tokens": len(ids), "pre_opened": pre_opened, "generated_tokens": n,
                "thinking_tokens": len(self.tok.encode(body, add_special_tokens=False)), "closed": closed,
                "generation_s": seconds, "thought": body, "generated_ids": None}

    def read_own(self, case, body):
        text, pre_opened = thinker_prompt(self.tok, case)
        prefix = text + ("" if pre_opened else "<think>\n") + body.rstrip() + CLOSE
        ids = self.tok.encode(prefix, add_special_tokens=False)
        started = time.perf_counter()
        self.llm.reset()
        self.llm.eval(ids)
        logits = self.np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.llm._n_vocab,)).astype("float64")
        size = len(case["question"]["criteria"])
        letters = [float(logits[i]) for i in self.letters[:size]]
        z = logits - logits.max()
        mass = float(self.np.exp(z[self.letters[:size]]).sum() / self.np.exp(z).sum())
        return {"own_logits": letters, "own_letter_mass": mass, "own_read_s": time.perf_counter() - started,
                "own_read_tokens": len(ids)}


def phase_think(args):
    name = args.thinker
    spec = THINKERS[name]
    lock = json.loads((ROOT / spec["lock"]).read_text(encoding="utf-8"))
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "thoughts.jsonl"
    cases = cases_devcal()
    if args.smoke:
        cases = [c for c in cases if c["split"] == "development"][:args.smoke]
        raw = out / "smoke-thoughts.jsonl"
        if raw.exists():
            raw.unlink()
    done = {r["id"] for r in read_jsonl(raw)} if raw.exists() else set()
    if done and not args.resume:
        raise FileExistsError(raw)
    prov = base_provenance(name, spec, lock)
    prov["smoke"] = bool(args.smoke)
    if spec["backend"] == "hf":
        thinker = HFThinker(str(ROOT / lock["path"]), spec.get("min_think", 0))
    else:
        thinker = GGUFThinker(str(ROOT / lock["path"]), ROOT / spec["gguf"], n_ctx=spec.get("n_ctx", 4096),
                              expected_sha256=spec.get("gguf_sha256"), use_mmap=spec.get("use_mmap", True))
        prov["gguf_manifest"] = json.loads((ROOT / spec["manifest"]).read_text(encoding="utf-8"))
        prov["n_ctx"] = spec.get("n_ctx", 4096)
        if "protocol" in spec:
            prov["protocol"] = spec["protocol"]
    prov["thinker_info"] = thinker.info
    prov["resumed_after_cases"] = len(done)
    (out / ("smoke-provenance-think.json" if args.smoke else "provenance-think.json")).write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    with raw.open("a", encoding="utf-8") as log:
        for index, case in enumerate(cases):
            if case["id"] in done:
                continue
            row = {"id": case["id"], "domain": case["domain"], "split": case["split"], "index": index,
                   "keys": list(case["question"]["criteria"]),
                   "target": [case["target"][k] for k in case["question"]["criteria"]]}
            row.update(thinker.think(case, index))
            row["own"] = {}
            for budget in BUDGETS:
                body, used = truncated_body(thinker.tok, row, budget)
                if budget is not None and used == row["thinking_tokens"] and "None" in row["own"]:
                    row["own"][str(budget)] = dict(row["own"]["None"])
                    continue
                if budget is not None and used == row["thinking_tokens"]:
                    reading = thinker.read_own(case, row["thought"])
                else:
                    reading = thinker.read_own(case, body)
                reading["thinker_tokens_used"] = used
                reading["correct"] = argmax(reading["own_logits"]) == argmax(row["target"])
                row["own"][str(budget)] = reading
                if budget is not None and used == row["thinking_tokens"]:
                    row["own"]["None"] = dict(reading)
            row.update({k: v for k, v in row["own"]["None"].items() if k.startswith("own_")})
            row["own_correct"] = row["own"]["None"]["correct"]
            row["peak_rss_gib"] = peak_rss_gib()
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps({"case": case["id"], "tokens": row["thinking_tokens"], "closed": row["closed"],
                              "s": round(row["generation_s"], 1),
                              "tok_per_s": round(row["generated_tokens"] / row["generation_s"], 2),
                              "own_correct": row["own_correct"], "own_mass": round(row["own_letter_mass"], 3),
                              "own_curve": [row["own"][str(b)]["correct"] for b in BUDGETS]}), flush=True)
            if args.smoke:
                print(row["thought"][:1200], flush=True)


# ----------------------------------------------------------------------------------------------- read (0.6B)
def truncated_body(tok, row, budget):
    if budget is None or row["thinking_tokens"] <= budget:
        return row["thought"], row["thinking_tokens"]
    ids = tok.encode(row["thought"], add_special_tokens=False)[:budget]
    return tok.decode(ids), budget


def phase_read(args):
    name = args.thinker
    spec = THINKERS[name]
    lock = json.loads((ROOT / spec["lock"]).read_text(encoding="utf-8"))
    out = OUT / name
    raw = out / ("smoke-thoughts.jsonl" if args.smoke else "thoughts.jsonl")
    rows = read_jsonl(raw)
    if not args.smoke:
        assert len(rows) == 45, len(rows)
    target = out / ("smoke-readings.jsonl" if args.smoke else "readings.jsonl")
    if target.exists() and not args.smoke:
        raise FileExistsError(target)
    import transformers
    thinker_tok = transformers.AutoTokenizer.from_pretrained(str(ROOT / lock["path"]), local_files_only=True)
    from typed_decisions.qwen import QwenDecisionModel
    reader = QwenDecisionModel(lock_file=ROOT / "qwen.lock.json", tree_kernel="dense")
    from scripts.crossmodel_thinking import read_letters
    by_id = {c["id"]: c for c in cases_devcal()}
    prov = base_provenance(name, spec, lock)
    prov.update({"phase": "read", "reader": reader.lock_data, "placement": "own_think", "budgets": BUDGETS,
                 "thoughts_sha256": sha256(raw), "smoke": bool(args.smoke)})
    (out / ("smoke-provenance-read.json" if args.smoke else "provenance-read.json")).write_text(
        json.dumps(prov, indent=2), encoding="utf-8")
    readings = []
    with target.open("w", encoding="utf-8") as log:
        for row in rows:
            case = by_id[row["id"]]
            size = len(case["question"]["criteria"])
            rec = {"id": row["id"], "domain": row["domain"], "target": row["target"], "budgets": {}}
            for budget in BUDGETS:
                body, used = truncated_body(thinker_tok, row, budget)
                text = prompt_text(reader, case) + "<think>\n" + body.rstrip() + CLOSE
                logits, seconds, tokens = read_letters(reader, text, size)
                rec["budgets"][str(budget)] = {"logits": logits, "correct": argmax(logits) == argmax(row["target"]),
                                               "thinker_tokens_used": used, "read_s": seconds, "reader_tokens": tokens}
            rec["peak_rss_gib"] = peak_rss_gib()
            readings.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps({"case": row["id"], **{b: rec["budgets"][b]["correct"] for b in rec["budgets"]}}), flush=True)
    if not args.smoke:
        write_summary(name, rows, readings, prov)


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else None


def per_domain(flags):
    out = {}
    for domain, ok in flags:
        out[domain] = out.get(domain, 0) + int(ok)
    return dict(sorted(out.items()))


def write_summary(name, rows, readings, prov):
    ids = [r["id"] for r in rows]
    by_read = {r["id"]: r for r in readings}
    reader = {}
    for budget in BUDGETS:
        flags = [(r["domain"], by_read[r["id"]]["budgets"][str(budget)]["correct"]) for r in rows]
        reader[str(budget)] = {"correct": sum(ok for _, ok in flags), "domains": per_domain(flags),
                               "median_thinker_tokens_used": median(by_read[i]["budgets"][str(budget)]["thinker_tokens_used"] for i in ids)}
    own = [(r["domain"], r["own_correct"]) for r in rows]
    own_curve = {str(b): {"correct": sum(r["own"][str(b)]["correct"] for r in rows),
                          "domains": per_domain((r["domain"], r["own"][str(b)]["correct"]) for r in rows),
                          "median_thinker_tokens_used": median(r["own"][str(b)]["thinker_tokens_used"] for r in rows)}
                 for b in BUDGETS}
    mine = {r["id"]: by_read[r["id"]]["budgets"]["None"]["correct"] for r in rows}

    def paired(ref):
        assert set(ref) == set(mine)
        b = sum(ref[i] and not mine[i] for i in ids)
        c = sum(mine[i] and not ref[i] for i in ids)
        return {"reference_correct": sum(ref.values()), "candidate_correct": sum(mine.values()), "reference_only": b,
                "candidate_only": c, "mcnemar_exact_p": mcnemar_exact(b, c),
                "paired_bootstrap_95ci": paired_bootstrap([ref[i] for i in ids], [mine[i] for i in ids])}
    ref_4b = {r["id"]: argmax(r["layer_logits"][-1][0]) == argmax(r["target"]) for r in read_jsonl(NO_THINK_4B)
              if r["format"] == "letter" and r["id"] in mine}
    ref_06b = {r["id"]: r["budgets"]["full"]["correct"] for r in read_jsonl(THINK_06B) if r["id"] in mine}
    summary = {
        "thinker": name, "n": len(rows), "cap": CAP,
        "reader_06b": reader, "primary_correct_06b_full": reader["None"]["correct"],
        "own_reading": {"correct": sum(ok for _, ok in own), "domains": per_domain(own), "curve": own_curve,
                        "min_letter_mass": min(r["own_letter_mass"] for r in rows),
                        "median_letter_mass": median(r["own_letter_mass"] for r in rows),
                        "median_own_read_s": median(r["own_read_s"] for r in rows)},
        "agreement_06b_vs_own_full": sum(argmax(by_read[r["id"]]["budgets"]["None"]["logits"]) == argmax(r["own_logits"]) for r in rows),
        "closed_naturally": sum(r["closed"] for r in rows), "hit_cap": sum(not r["closed"] for r in rows),
        "median_thinking_tokens": median(r["thinking_tokens"] for r in rows),
        "median_generation_s": median(r["generation_s"] for r in rows),
        "mean_generation_s": sum(r["generation_s"] for r in rows) / len(rows),
        "median_tok_per_s": median(r["generated_tokens"] / r["generation_s"] for r in rows),
        "median_reader_06b_read_s": median(by_read[i]["budgets"]["None"]["read_s"] for i in ids),
        "peak_rss_gib_think": max(r["peak_rss_gib"] for r in rows),
        "peak_rss_gib_read": max(r["peak_rss_gib"] for r in readings),
        "vs_4b_no_thinking_devcal": paired(ref_4b), "vs_06b_thinking_devcal": paired(ref_06b),
        "criterion_38": reader["None"]["correct"] >= 38,
        "criterion_36_needs_speed": reader["None"]["correct"] >= 36,
        "provenance": prov}
    if spec_fastpath := THINKERS[name].get("fastpath_parity"):
        cases_by_id = {c["id"]: c for c in cases_devcal()}
        fast = {}
        for r in read_jsonl(ROOT / spec_fastpath):
            if r["id"] in mine:
                c = cases_by_id[r["id"]]
                fast[r["id"]] = r["argmax"] == max(range(len(r["keys"])), key=lambda i: c["target"][r["keys"][i]])
        own_full = {r["id"]: r["own_correct"] for r in rows}
        summary["vs_fastpath_devcal"] = {"source": spec_fastpath, "reader_06b": paired(fast),
                                         "own": {"reference_correct": sum(fast.values()), "candidate_correct": sum(own_full.values()),
                                                 "reference_only": sum(fast[i] and not own_full[i] for i in ids),
                                                 "candidate_only": sum(own_full[i] and not fast[i] for i in ids),
                                                 "mcnemar_exact_p": mcnemar_exact(sum(fast[i] and not own_full[i] for i in ids), sum(own_full[i] and not fast[i] for i in ids)),
                                                 "fastpath_wrong": sorted(i for i in ids if not fast[i]),
                                                 "own_wrong": sorted(i for i in ids if not own_full[i])}}
    if name in SELECTION_THINKERS:
        full = own_curve["None"]["correct"]
        chosen = next((b for b in SELECTION_BUDGETS if own_curve[str(b)]["correct"] >= full - SELECTION_MAX_LOSS), None)
        selection = {"rule": "shortest budget in 64/128/192/256/full whose own-read (1.7B) dev+cal correct count is "
                             ">= full - 1; none -> full (pre-registered before running)",
                     "reader": f"the thinker itself ({name})", "budgets_considered": SELECTION_BUDGETS,
                     "own_curve": {str(b): own_curve[str(b)]["correct"] for b in SELECTION_BUDGETS},
                     "reader_06b_curve_aside": {str(b): reader[str(b)]["correct"] for b in SELECTION_BUDGETS},
                     "full_correct": full, "chosen_budget": chosen if chosen is not None else "full",
                     "thoughts_sha256": prov["thoughts_sha256"], "head_commit": prov["head_commit"]}
        summary["budget_selection"] = selection
        (OUT / name / "budget-selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    (OUT / name / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "provenance"}, indent=2), flush=True)


def phase_summarize(args):
    """Recompute summary.json from the recorded thoughts and readings (no model is loaded)."""
    out = OUT / args.thinker
    rows, readings = read_jsonl(out / "thoughts.jsonl"), read_jsonl(out / "readings.jsonl")
    assert len(rows) == len(readings) == 45
    prov = json.loads((out / "provenance-read.json").read_text(encoding="utf-8"))
    write_summary(args.thinker, rows, readings, prov)


def phase_compare(args):
    summaries = {}
    for name in THINKERS:
        path = OUT / name / "summary.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    ref = summaries.get(REFERENCE)
    table = {}
    for name, s in summaries.items():
        acc = s["primary_correct_06b_full"]
        ratio = (ref["median_generation_s"] / s["median_generation_s"]) if ref else None
        table[name] = {"correct_06b": acc, "own_correct": s["own_reading"]["correct"], "closed": s["closed_naturally"],
                       "median_thinking_tokens": s["median_thinking_tokens"], "median_generation_s": s["median_generation_s"],
                       "median_tok_per_s": s["median_tok_per_s"], "speed_ratio_vs_reference": ratio,
                       "passes_38": acc >= 38, "passes_36_and_2x": (acc >= 36 and ratio is not None and ratio >= 2.0),
                       "verdict": "worth_it" if (acc >= 38 or (acc >= 36 and ratio is not None and ratio >= 2.0)) else
                                  ("pending: no reference speed" if (acc >= 36 and ratio is None) else "not_worth_it"),
                       "domains_06b": s["reader_06b"]["None"]["domains"],
                       "budget_curve_06b": {b: s["reader_06b"][b]["correct"] for b in s["reader_06b"]}}
    result = {"protocol": "docs/HISTORY.md (candidate thinkers)", "reference": REFERENCE, "reference_present": ref is not None,
              "criterion": ">= 38/45 (0.6B reader) or >= 36/45 with median generation seconds <= half of the reference",
              "thinkers": table, "head_commit": git_head()}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["think", "read", "summarize", "compare"])
    parser.add_argument("--thinker", choices=sorted(THINKERS))
    parser.add_argument("--smoke", type=int, default=0, help="technical check on the first N development cases only")
    parser.add_argument("--resume", action="store_true")
    a = parser.parse_args()
    if a.phase != "compare" and not a.thinker:
        parser.error("--thinker is required")
    {"think": phase_think, "read": phase_read, "summarize": phase_summarize, "compare": phase_compare}[a.phase](a)
