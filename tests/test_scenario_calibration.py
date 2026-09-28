import math
import pytest

from scripts.audit_scenarios import cross_entropy, fit_temperature, metrics, questions_for, softmax
from scripts.build_scenario_cases import build


def test_temperature_fit_uses_only_calibration_and_recovers_known_distribution():
    # softmax([log 4, 0]/T) is [2/3,1/3] exactly when T=2.
    row={'id':'analytic','split':'calibration','logits':[[math.log(4),0]],'target':[2/3,1/3]}
    fitted=fit_temperature([row])
    assert fitted['temperature']==pytest.approx(2.,rel=1e-5)
    assert fitted['nll_after']<fitted['nll_before']
    with pytest.raises(ValueError):fit_temperature([{**row,'split':'test'}])
    with pytest.raises(ValueError):fit_temperature([])
    assert cross_entropy([1000,0],[0,1],1)==pytest.approx(1000)


def test_corpus_splits_are_disjoint_and_random_targets_are_not_accuracy():
    cases=build()
    assert len(cases)==108
    assert len({c['state'] for c in cases})==108
    assert [sum(c['split']==s for c in cases) for s in ['development','calibration','test']]==[18,36,54]
    for c in cases:
        assert set(c['target'])==set(c['question']['criteria'])
        assert sum(c['target'].values())==pytest.approx(1)
    r={'logits':[[0,0,0]],'target':[1/3]*3,'target_kind':'known_distribution'}
    m=metrics([r])
    assert m['accuracy'] is None
    assert m['expected_success_of_argmax']==pytest.approx(1/3)
    assert m['mean_total_variation']==pytest.approx(0)


def test_candidate_questions_have_no_answer_dependencies_and_cover_all_options():
    case=build()[0]
    qs,variants=questions_for(case,'candidate')
    assert len(qs)==6
    for i,options in enumerate(variants):
        for key,description in options:
            q=qs[f'v{i}_{key}']
            assert q['instructions']['all_options']==dict(options)
            assert q['instructions']['candidate']=={'key':key,'description':description}
            assert q['type']=='noul'
    assert max(range(3),key=softmax([3,1,-2],10).__getitem__)==0
    for value in [0,-1,float('nan')]:
        with pytest.raises(ValueError):softmax([1,2],value)
