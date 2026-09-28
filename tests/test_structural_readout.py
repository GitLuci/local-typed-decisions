import math

from scripts.structural_readout import candidates, log_softmax, mcnemar_exact, paired_bootstrap, pooled


def record(fmt, layer_logits):
    return {"id": "x", "domain": "d", "split": "development", "target_kind": "factual", "keys": ["a", "b"],
            "target": [1., 0.], "format": fmt, "layer_logits": layer_logits}


def test_candidate_list_is_closed_and_uses_no_fitted_weights():
    specs = candidates(28)
    assert set(specs) == {"letter", "json", "letter_orders", "letter_json", "letter_json_orders",
                          "letter_late_layers", "all_late_layers", "letter_dola"}
    assert specs["letter"] == [(1., "letter", 0, 28)]
    assert [v[3] for v in specs["letter_late_layers"]] == list(range(22, 29))
    assert specs["letter_dola"] == [(1., "letter", 0, 28), (-1., "letter", 0, 14)]
    for name, spec in specs.items():
        if name != "letter_dola":
            assert math.isclose(sum(w for w, *_ in spec), 1.)


def test_pooling_averages_log_probabilities_and_baseline_is_identity():
    layers = [[[0., 0.], [0., 0.]], [[2., 0.], [0., 3.]]]
    by_format = {"letter": record("letter", layers), "json": record("json", layers)}
    base = pooled(by_format, [(1., "letter", 0, 2)])["logits"][0]
    assert base == log_softmax([2., 0.])
    orders = pooled(by_format, [(.5, "letter", 0, 2), (.5, "letter", 1, 2)])["logits"][0]
    expected = [(x + y) / 2 for x, y in zip(log_softmax([2., 0.]), log_softmax([0., 3.]))]
    assert orders == expected


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact(0, 0) == 1.
    assert math.isclose(mcnemar_exact(0, 5), 2 / 32)
    assert mcnemar_exact(3, 3) == 1.
    lo, hi = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0])
    assert lo == hi == 0.
