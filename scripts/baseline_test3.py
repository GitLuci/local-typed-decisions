"""test-3: the Jev baseline run (Jev 1.13 by TypeSafe, via its commercial API), once, over the SEALED test cases.

- **One call per state**, with all of its questions. A `choice` question is sent in two letter orders (as in
  test-2, so a letter preference cannot pass as content); `noul` and `score` are sent as written.
- **Resumable:** states already recorded in `predictions.jsonl` are skipped.
- Raw responses are recorded locally (`response`) next to per-key probabilities. They come from a commercial API and
  are not redistributed with this repository; only aggregate numbers are published.
- **Credentials** come only from the `TYPESAFE_TOKEN` environment variable and are never printed or written. An HTTP
  error stops the run without recording the response body.
- **Cost before running:** `--estimate` reads no credentials and uses no network. It estimates input tokens from the
  tokens/character ratio measured on test-2 (80,902 tokens over 168 requests) and applies `--price` (USD per million
  input tokens; the 0.042 used for test-2 is an assumption to confirm with the vendor before running).
- **Cap:** stops if real tokens exceed 1.5x the estimate.
- **Without `--confirm` no calls are made.**

    python scripts/baseline_test3.py --estimate --price 0.042
    python scripts/baseline_test3.py --confirm --price 0.042        # only after the cases are sealed
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
CASES = ROOT / "examples/test-3"
OUT = ROOT / "reports/test-3/jev"
TEST2 = ROOT / "examples/test-2-scenarios.jsonl"
TEST2_TOKENS, TEST2_REQUESTS = 80_902, 168  # measured in the test-2 baseline run (aggregate)
CREDENTIAL_ENV = "TYPESAFE_TOKEN"


def probabilities(answer, question):
    """Validate one answer and return probabilities in request-key order."""
    if answer["type"] != question["type"]:
        raise ValueError("unexpected answer type")
    if question["type"] == "noul":
        p = [1 - float(answer["noul"]), float(answer["noul"])]
    else:
        keys = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(question["criteria"]))]
        if set(answer["probabilities"]) != set(keys):
            raise ValueError("returned option keys differ from request")
        p = [float(answer["probabilities"][k]) for k in keys]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > .001:
        raise ValueError("invalid response probability distribution")
    return [v / sum(p) for v in p]


def variants(pid: str, q: dict):
    """{sent key: (sent question, map sent option -> corpus option)}."""
    if q["type"] == "choice":
        items = list(q["criteria"].items())
        out = {}
        for i in (0, 1):
            order = items[i:] + items[:i]
            out[f"{pid}_v{i}"] = ({**q, "criteria": {chr(65 + j): f"{k}: {v}" for j, (k, v) in enumerate(order)}},
                                  {chr(65 + j): k for j, (k, _) in enumerate(order)})
        return out
    if q["type"] == "noul":
        return {f"{pid}_v0": (q, {"false": "false", "true": "true"})}
    return {f"{pid}_v0": (q, {str(i): str(i) for i in range(len(q["criteria"]))})}


def requests_for(states, questions):
    """One request per test state: {state_id, payload, map}. Stable order by state_id."""
    by_state = {}
    for q in questions:
        if q["split"] == "test":
            by_state.setdefault(q["state_id"], []).append(q)
    st = {s["state_id"]: s for s in states}
    out = []
    for sid in sorted(by_state):
        sent, mapping = {}, {}
        for n, q in enumerate(by_state[sid]):
            for key, (question, m) in variants(f"q{n}", q["question"]).items():
                sent[key] = question
                mapping[key] = {"id": q["id"], "map": m, "type": q["question"]["type"]}
        out.append({"state_id": sid, "payload": {"model": MODEL, "state": st[sid]["text"], "questions": sent},
                    "map": mapping})
    return out


def test2_ratio() -> float:
    """Input tokens per payload character, measured on test-2 (same request shape)."""
    chars = 0
    for line in TEST2.read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        sent = {k: v for k, (v, _) in variants("v", c["question"]).items()}
        chars += len(json.dumps({"state": c["state"], "questions": sent}, ensure_ascii=False))
    return TEST2_TOKENS / chars


def estimate(reqs, price: float) -> dict:
    r = test2_ratio()
    chars = sum(len(json.dumps({k: p["payload"][k] for k in ("state", "questions")}, ensure_ascii=False)) for p in reqs)
    tokens = int(chars * r)
    return {"requests": len(reqs), "questions": sum(len({m["id"] for m in p["map"].values()}) for p in reqs),
            "estimated_input_tokens": tokens, "tokens_per_char_test2": round(r, 4),
            "price_usd_per_million": price, "estimated_cost_usd": round(tokens * price / 1e6, 4),
            "token_cap": int(tokens * 1.5)}


def read_sealed(folder: Path):
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    files = {}
    for name in ("states.jsonl", "questions.jsonl"):
        data = (folder / name).read_bytes()
        if hashlib.sha256(data).hexdigest() == man.get("shipped_redacted", {}).get("sha256", {}).get(name):
            raise SystemExit(f"{name}: GoEmotions texts are not restored; run scripts/fetch_goemotions.py first")
        if hashlib.sha256(data).hexdigest() != man["sha256"][name]:
            raise SystemExit(f"{name}: sha256 differs from the sealed manifest; refusing to run")
        files[name] = [json.loads(l) for l in data.decode("utf-8").splitlines() if l.strip()]
    return files["states.jsonl"], files["questions.jsonl"], man


def run(reqs, est, client, out: Path, provenance: dict, pause=0.0):
    out.mkdir(parents=True, exist_ok=True)
    meta, path = out / "provenance.json", out / "predictions.jsonl"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) != provenance:
        raise SystemExit("provenance differs from the run already started; refusing to mix runs")
    meta.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    done = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    seen = {r["state_id"] for r in done}
    tokens = sum(r["usage"].get("input_tokens", 0) for r in done)
    with path.open("a", encoding="utf-8") as log:
        for p in reqs:
            if p["state_id"] in seen:
                continue
            if tokens > est["token_cap"]:
                raise SystemExit(f"token cap reached ({tokens} > {est['token_cap']}); stopped, resumable")
            t0 = time.perf_counter()
            resp = client.post(ENDPOINT, json=p["payload"])
            ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                raise SystemExit(f"HTTP {resp.status_code}; stopped without recording the body; resumable")
            body = resp.json()
            if body.get("model") != MODEL or set(body["answers"]) != set(p["payload"]["questions"]):
                raise SystemExit("unexpected model or answer ids; stopped")
            per_question = {}
            for key, info in p["map"].items():
                sent = p["payload"]["questions"][key]
                prob = probabilities(body["answers"][key], sent)
                sent_keys = list(sent["criteria"]) if sent["type"] == "choice" else list(info["map"])
                per_key = {info["map"][k]: v for k, v in zip(sent_keys, prob)}
                per_question.setdefault(info["id"], []).append(per_key)
            line = {"state_id": p["state_id"], "method": "jev_api", "elapsed_ms": ms, "usage": body.get("usage", {}),
                    "questions": {qid: {"variants": v} for qid, v in per_question.items()},
                    "response": {k: body[k] for k in ("model", "answers", "usage") if k in body}}
            log.write(json.dumps(line, ensure_ascii=False) + "\n")
            log.flush()
            tokens += line["usage"].get("input_tokens", 0)
            print(json.dumps({"state": p["state_id"], "done": len(seen) + 1, "tokens": tokens}), flush=True)
            seen.add(p["state_id"])
            if pause:
                time.sleep(pause)
    return {"requests": len(seen), "input_tokens": tokens}


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default=str(CASES), help="folder with the sealed states.jsonl, questions.jsonl and manifest.json")
    ap.add_argument("--price", type=float, required=True, help="USD per million input tokens (confirm first)")
    ap.add_argument("--estimate", action="store_true", help="estimate only: no credentials, no network")
    ap.add_argument("--confirm", action="store_true", help="without this no calls are made")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    states, questions, man = read_sealed(Path(a.cases))
    reqs = requests_for(states, questions)
    est = estimate(reqs, a.price)
    print(json.dumps({"estimate": est}, ensure_ascii=False, indent=1))
    if a.estimate or not a.confirm:
        if not a.estimate:
            print("no --confirm: no calls made")
        return 0
    import httpx
    credential = os.environ.get(CREDENTIAL_ENV)
    if not credential:
        raise SystemExit(f"set {CREDENTIAL_ENV} in the environment to call the baseline API")
    provenance = {"model": MODEL, "endpoint": ENDPOINT, "seal": man.get("seal_commit"),
                  "states_sha256": man["sha256"]["states.jsonl"], "questions_sha256": man["sha256"]["questions.jsonl"],
                  "protocol": "docs/METHODOLOGY.md (test-3)", "split": "test", "estimate": est,
                  "choice": "two letter orders per question; noul/score as written"}
    try:
        with httpx.Client(timeout=60, follow_redirects=False, headers={"Authorization": f"Bearer {credential}"}) as client:
            res = run(reqs, est, client, Path(a.out), provenance)
    except httpx.HTTPError:
        raise SystemExit("transport error; what was recorded stays; no credentials in the log") from None
    res["cost_usd_at_given_price"] = round(res["input_tokens"] * a.price / 1e6, 4)
    (Path(a.out) / "summary.json").write_text(json.dumps({"run": res, "estimate": est}, indent=2), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
