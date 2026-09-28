import copy
import ipaddress
import json
from pathlib import Path
import socket

import pytest
from fastapi.testclient import TestClient

from conformance.contract import ContractError, Profile, assert_request, assert_response, exercise_backend, assert_rejected
from typed_decisions.runtime import answer_from_probabilities, validate
from typed_decisions.server import create_app

FIXTURES = json.loads((Path(__file__).parent/'fixtures/contract/cases.json').read_text(encoding='utf-8'))['cases']


@pytest.fixture(autouse=True)
def prohibit_external_network(monkeypatch):
    for method in ('connect', 'connect_ex'):
        original = getattr(socket.socket, method)
        def guarded(sock, address, _original=original):
            if sock.family in (socket.AF_INET, socket.AF_INET6):
                host = address[0]
                try:
                    local = ipaddress.ip_address(host).is_loopback
                except ValueError:
                    local = host == 'localhost'
                if not local:
                    raise AssertionError('Conformance suite prohibits external networking')
            return _original(sock, address)
        monkeypatch.setattr(socket.socket, method, guarded)


@pytest.mark.parametrize('case', FIXTURES, ids=lambda x:x['id'])
def test_frozen_request_response_pairs(case):
    original = copy.deepcopy(case)
    assert_request(case['request'])
    assert_response(case['request'], case['response'], Profile(fast_path=True))
    assert case == original


def set_path(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


BAD_RESPONSE = [
    (('answers',), {}),
    (('answers','friday?','type'), 'choice'),
    (('answers','friday?','noul'), True),
    (('answers','friday?','noul'), '0.75'),
    (('answers','friday?','noul'), -0.01),
    (('answers','friday?','noul'), float('nan')),
    (('answers','friday?','noul'), float('inf')),
    (('answers','friday?','noul'), 10**400),
    (('answers','naïve/中文','choice'), 'nonexistent'),
    (('answers','naïve/中文','choice'), 'Lisbon'),
    (('answers','naïve/中文','choice'), None),
    (('answers','naïve/中文','confidence'), 1.01),
    (('answers','naïve/中文','probabilities'), {'Évora':1}),
    (('answers','naïve/中文','probabilities'), {'Évora':0.9,'Lisbon':0.2}),
    (('answers','naïve/中文','probabilities'), {'Évora':1.1,'Lisbon':-0.1}),
    (('answers','intensity','score'), 2),
    (('answers','intensity','score'), 0.75),
    (('answers','intensity','legend'), {'1':'Low','2':'Medium','3':'High'}),
    (('answers','intensity','probabilities'), {'1':0.1,'2':0.3,'3':0.6}),
    (('usage','input_tokens'), True),
    (('usage','input_tokens'), -1),
    (('usage','output_tokens'), 1.5),
    (('model',), ''),
]


@pytest.mark.parametrize('path,value', BAD_RESPONSE)
def test_detects_corrupt_backend_response(path, value):
    case = copy.deepcopy(FIXTURES[0])
    set_path(case['response'], path, value)
    with pytest.raises(ContractError):
        assert_response(case['request'], case['response'])


BAD_REQUEST = [
    (('state',), None), (('state',), True), (('state',), 42),
    (('questions',), {}), (('questions',), []),
    (('questions','friday?','type'), 'Noul'),
    (('questions','friday?','type'), ['noul']),
    (('questions','friday?','instructions'), ''),
    (('questions','friday?','instructions'), 123),
    (('questions','naïve/中文','criteria'), {'only':None}),
    (('questions','naïve/中文','criteria'), {'a':42,'b':None}),
    (('questions','intensity','criteria'), ['a',None]),
    (('questions','intensity','criteria'), ['a']*11),
    (('questions','friday?','criteria'), False),
    (('model',), []),
]


@pytest.mark.parametrize('path,value', BAD_REQUEST)
def test_detects_invalid_request(path, value):
    request = copy.deepcopy(FIXTURES[0]['request'])
    set_path(request, path, value)
    with pytest.raises(ContractError):
        assert_request(request)


def test_nonfinite_nested_json_unknown_fields_and_duplicate_routing():
    for field, value in [('extra', 1), ('state', {'x':float('nan')}), ('questions', {'':{'type':'noul','instructions':'Q'}})]:
        request = {**copy.deepcopy(FIXTURES[0]['request']), field:value}
        with pytest.raises(ContractError):
            assert_request(request)
    response = copy.deepcopy(FIXTURES[0]['response'])
    response['answers']['friday?']['confidence'] = 0.5
    with pytest.raises(ContractError):
        assert_response(FIXTURES[0]['request'], response)


def test_limits_at_boundary_without_model_tokens():
    question = {'type':'choice','instructions':'Choose','criteria':{str(i):None for i in range(255)}}
    request = {'state':[], 'questions':{f'q{i}':copy.deepcopy(question) for i in range(32)}}
    assert_request(request)
    request['questions']['q32'] = question
    with pytest.raises(ContractError):
        assert_request(request)
    request['questions'] = {'q':question}
    question['criteria']['255'] = None
    with pytest.raises(ContractError):
        assert_request(request)
    question['criteria'] = {str(i):None for i in range(21)}
    with pytest.raises(ContractError):
        assert_request(request, Profile(max_choice_options=20))


def test_official_noul_shape_and_local_extensions_are_explicit():
    case = FIXTURES[3]
    with pytest.raises(ContractError):
        assert_request(case['request'], Profile(noul_clarifications=False))
    with pytest.raises(ContractError):
        assert_request(case['request'], Profile(require_model=True))


def test_wire_usage_is_not_assumed_to_be_zero_and_logits_are_private():
    case = copy.deepcopy(FIXTURES[0])
    case['response']['usage']['output_tokens'] = 69
    assert_response(case['request'],case['response'])
    with pytest.raises(ContractError):
        assert_response(case['request'],case['response'],Profile(fast_path=True))
    case['response']['logits'] = {'private':[1,2]}
    assert_response(case['request'],case['response'],public=False)
    with pytest.raises(ContractError):
        assert_response(case['request'],case['response'])


class FixtureBackend:
    """In-memory contract double; makes no claim about semantic inference."""
    model_id = 'contract-fixture'
    def predict(self, state, questions):
        validate(state, questions, max_options=255)
        answers = {}
        for qid,q in questions.items():
            p = [0.25,0.75] if q['type']=='noul' else list(range(1,len(q['criteria'])+1))
            answers[qid] = answer_from_probabilities(q,p)
        return {'model':self.model_id,'answers':answers,'usage':{'input_tokens':1,'output_tokens':0},'logits':{}}


def test_existing_readout_and_http_transport_obey_response_contract():
    backend = FixtureBackend()
    with TestClient(create_app(model=backend)) as client:
        for case in FIXTURES:
            result = client.post('/v1/systemone',json=case['request'])
            assert result.status_code == 200
            assert_response(case['request'],result.json(),Profile(fast_path=True),expected_model=backend.model_id)
            assert 'logits' not in result.json()


@pytest.mark.parametrize('body', [None, [], {}, {'state':'x'}, {'state':'x','questions':{}},
                                  {'state':'x','questions':{},'unexpected':1}])
def test_http_error_profile_invalid_envelopes(body):
    with TestClient(create_app(model=FixtureBackend())) as client:
        result = client.post('/v1/systemone',content=json.dumps(body),headers={'content-type':'application/json'})
        assert result.status_code == 422
        assert isinstance(result.json()['detail'],(str,list,dict))
        assert result.json()['detail']


def test_http_errors_and_utf8_byte_limit():
    with TestClient(create_app(model=FixtureBackend())) as client:
        for raw in ('{', b'\xff'):
            assert client.post('/v1/systemone',content=raw).status_code == 422
        assert client.post('/v1/systemone',content=('é'*128001).encode()).status_code == 413
        body = copy.deepcopy(FIXTURES[0]['request'])
        body['model'] = 'not-installed'
        assert client.post('/v1/systemone',json=body).status_code == 422
        body['model'] = 'local'
        assert client.post('/v1/systemone',json=body).status_code == 200
        body = {'state':'','questions':{'q':{'type':'noul','instructions':'Q?'}}}
        encode = lambda b:json.dumps(b,separators=(',',':')).encode()
        body['state'] = 'x'*(256000-len(encode(body)))
        assert len(encode(body)) == 256000
        assert client.post('/v1/systemone',content=encode(body)).status_code == 200
        body['state'] += 'x'
        assert client.post('/v1/systemone',content=encode(body)).status_code == 413


def test_independence_checks_detect_id_and_batch_leaks_and_mutation():
    request = FIXTURES[0]['request']
    backend = FixtureBackend()
    assert exercise_backend(backend.predict,request)['comparisons'] == 16
    def id_leak(state, questions):
        r = backend.predict(state, questions)
        for qid,a in r['answers'].items():
            if a['type']=='noul':
                a['noul'] = 0.9 if qid.startswith('renamed') else 0.1
        return r
    def batch_leak(state, questions):
        r = backend.predict(state,questions)
        for a in r['answers'].values():
            if a['type']=='noul': a['noul'] = 0.9 if len(questions)>1 else 0.1
        return r
    def mutate(state, questions):
        r = backend.predict(state,questions)
        questions.clear()
        return r
    for bad in (id_leak,batch_leak,mutate):
        with pytest.raises(ContractError):
            exercise_backend(bad,request)


def test_external_network_is_blocked():
    with socket.socket() as sock, pytest.raises(AssertionError, match='external networking'):
        sock.connect(('192.0.2.1',443))


def test_selected_backend_contract(selected_backend, pytestconfig):
    profile = Profile(max_choice_options=pytestconfig.getoption('--td-max-options'),
                      fast_path=not pytestconfig.getoption('--td-generates-tokens'))
    for case in FIXTURES:
        exercise_backend(selected_backend.predict,case['request'],profile,
                         independence_atol=pytestconfig.getoption('--td-independence-atol'),
                         expected_model=getattr(selected_backend,'model_id',None))
    for path,value in BAD_REQUEST:
        if path[0] == 'model':
            continue  # model selection belongs to the transport adapter
        body = copy.deepcopy(FIXTURES[0]['request'])
        set_path(body,path,value)
        assert_rejected(selected_backend.predict,body,profile)


def test_nested_legend_and_input_types_are_not_silently_coerced():
    case = copy.deepcopy(FIXTURES[1])
    case['request']['questions']['grades_ñ']['criteria'][0] = {'value':True}
    case['response']['answers']['grades_ñ']['legend']['0'] = {'value':1}
    with pytest.raises(ContractError):
        assert_response(case['request'],case['response'])
    def mutate(state, questions):
        result = FixtureBackend().predict(state,questions)
        state['fragile'] = 1  # Python equality alone would miss this mutation.
        return result
    with pytest.raises(ContractError,match='mutated'):
        exercise_backend(mutate,FIXTURES[0]['request'])


def test_invalid_native_requests_are_not_accepted_as_success():
    bad = {'state':None,'questions':{}}
    with pytest.raises(ContractError,match='accepted'):
        assert_rejected(lambda state,qs:{},bad)
    assert_rejected(FixtureBackend().predict,bad)


def make_strict_fixture_backend():
    """Used only to exercise opt-in test plumbing without a real model."""
    class Strict(FixtureBackend):
        def predict(self,state,questions):
            try:
                assert_request({'state':state,'questions':questions})
            except ContractError as error:
                raise ValueError(str(error)) from error
            return super().predict(state,questions)
    return Strict()
