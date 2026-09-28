"""Calibration uses probability mass, never the entropy concentration field."""
import math
import random


def summarize(rows, bins=10):
    if not rows:
        raise ValueError("no observations")
    n = len(rows)
    correct, brier, nll, confidence = [], [], [], []
    buckets = [[] for _ in range(bins)]
    for r in rows:
        p, target = r["p"], r["target"]
        if len(p) != len(target) or abs(sum(p) - 1) > 1e-6 or abs(sum(target) - 1) > 1e-6:
            raise ValueError("invalid probability/target distribution")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [*p, *target]):
            raise ValueError("invalid probability value")
        pred = max(range(len(p)), key=p.__getitem__)
        correct.append(target[pred])
        confidence.append(p[pred])
        brier.append(sum((a-b)**2 for a,b in zip(p,target)))
        nll.append(-sum(b*math.log(max(a, 1e-15)) for a,b in zip(p,target)))
        buckets[min(bins-1, int(p[pred]*bins))].append((p[pred], target[pred]))
    reliability = []
    ece = 0.
    for i, bucket in enumerate(buckets):
        c = sum(v[0] for v in bucket) / len(bucket) if bucket else None
        a = sum(v[1] for v in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket)/n * abs(c-a)
        reliability.append({"lower": i/bins, "upper": (i+1)/bins, "n": len(bucket), "p_max": c, "accuracy": a})
    return {"n": n, "accuracy": sum(correct)/n, "mean_p_max": sum(confidence)/n,
            "brier_sum": sum(brier)/n, "nll": sum(nll)/n, "ece_10": ece, "reliability": reliability}


def cluster_bootstrap(rows, seed=42, repeats=500):
    groups = {}
    for row in rows:
        groups.setdefault(row["case_id"], []).append(row)
    rng = random.Random(seed)
    keys = list(groups)
    samples = {k: [] for k in ("accuracy", "brier_sum", "ece_10")}
    for _ in range(repeats):
        batch = [r for key in rng.choices(keys, k=len(keys)) for r in groups[key]]
        summary = summarize(batch)
        for k in samples:
            samples[k].append(summary[k])
    return {k: [sorted(v)[int(.025*repeats)], sorted(v)[min(repeats-1, int(.975*repeats))]] for k,v in samples.items()}
