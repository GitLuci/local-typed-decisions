from scripts.audit_quality import metrics


def test_zero_agreement_reports_zero_coverage_and_undefined_selective_accuracy():
    case={'questions':{'decision':{'criteria':{'a':'A','b':'B'}}},'targets':{'decision':'a'}}
    answer={'status':'order_sensitive','variants':[
        {'probabilities':{'a':.9,'b':.1}}, {'probabilities':{'a':.1,'b':.9}}],
        'mean_probabilities':{'a':.5,'b':.5},'total_variation_max':.8}
    r=metrics([{'case':case,'result':{'answers':{'decision':answer},'metadata':{'elapsed_ms':1}}}])
    assert r['agreement_count']==0 and r['agreement_coverage']==0
    assert r['agreed']['accuracy'] is None
    assert r['confidence_filter_same_coverage']['accuracy'] is None
