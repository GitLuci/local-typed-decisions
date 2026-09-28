import pytest
from typed_decisions.consistency import predict_consistent


class Fake:
    choice_order='input'
    def __init__(self, positional=False):
        self.positional=positional
        self.calls=0
    def predict(self,state,qs):
        self.calls+=1
        self.received=qs
        answers={}; logits={}
        for k,q in qs.items():
            keys=list(q['criteria'])
            win=keys[0] if self.positional else 'good'
            answers[k]={'type':'choice','choice':win,'probabilities':{x:float(x==win) for x in keys}}
            logits[k]=[float(x==win) for x in keys]
        return {'model':'stub','answers':answers,'logits':logits,'usage':{},'metadata':{'backbone_forward_passes':1}}


def questions():
    return {'id':{'type':'choice','instructions':'Pick the matching option.',
                  'criteria':{'bad':'Incorrect','good':'Correct','other':'Other'}}}


def test_semantic_consistency_survives_changed_codes_in_one_call():
    m=Fake()
    r=predict_consistent(m,'state',questions())
    a=r['answers']['id']
    assert m.calls==1
    assert a['choice']=='good' and a['status']=='agreement'
    assert a['mean_probabilities']['good']==1 and a['total_variation_max']==0
    original=list(m.received['q0_r0']['criteria']); shifted=list(m.received['q0_r1']['criteria'])
    assert all(a!=b for a,b in zip(original,shifted))


def test_same_letter_bias_causes_abstention_not_false_agreement():
    r=predict_consistent(Fake(positional=True),'state',questions())
    assert r['answers']['id']['choice'] is None
    assert r['answers']['id']['status']=='order_sensitive'
    assert r['answers']['id']['total_variation_max']==1


def test_canonical_order_cannot_silently_nullify_the_experiment():
    m=Fake(); m.choice_order='canonical'
    with pytest.raises(ValueError,match='canonical'):
        predict_consistent(m,'state',questions())


def test_score_and_repeated_permutations_rejected():
    with pytest.raises(ValueError,match='Choice only'):
        predict_consistent(Fake(),'state',{'q':{'type':'score','instructions':'Level?','criteria':['low','high']}})
    with pytest.raises(ValueError,match='distinct'):
        predict_consistent(Fake(),'state',questions(),rounds=4)


def test_changed_state_cannot_silently_use_an_old_handle():
    class Handle:
        _state_text='original'
    with pytest.raises(ValueError,match='differs'):
        predict_consistent(Fake(),'changed',questions(),prepared=Handle())


def test_request_cannot_exceed_parallel_branch_budget():
    qs={f'q{i}':questions()['id'] for i in range(17)}
    m=Fake()
    with pytest.raises(ValueError,match='32-branch'):
        predict_consistent(m,'state',qs)
    assert m.calls==0
