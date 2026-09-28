"""Backend-independent assertions for the documented Jev/local-v1 contract.

Standard library only. No inference, networking, normalization, or repairs.
An assertion identifies the failing JSON path; inputs are never mutated.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math


class ContractError(AssertionError):
    pass


@dataclass(frozen=True)
class Profile:
    max_questions: int = 32  # local resource policy, not a documented Jev ceiling
    max_choice_options: int = 255
    max_score_levels: int = 10
    require_model: bool = False  # local endpoint permits omission
    noul_clarifications: bool = True  # local legacy extension
    probability_atol: float = 1e-6
    score_atol: float = 1e-6
    fast_path: bool = False
    question_errors: bool = False  # explicit local extension; base Jev shape stays strict


def need(condition, path, message):
    if not condition:
        raise ContractError(f'{path}: {message}')


def same_json(a, b):
    # Unlike Python equality, this does not equate true with 1 in nested data.
    return json.dumps(a, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False, allow_nan=False)


def json_value(value, path='$'):
    """Reject Python-only values, non-string keys and non-finite JSON numbers."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        need(math.isfinite(value), path, 'number must be finite')
    elif type(value) is list:
        for i, item in enumerate(value):
            json_value(item, f'{path}[{i}]')
    elif type(value) is dict:
        for key, item in value.items():
            need(type(key) is str, path, 'object keys must be strings')
            json_value(item, f'{path}.{key}')
    else:
        raise ContractError(f'{path}: not a JSON value')


def content(value, path, *, nullable=False, nonempty=False):
    need((nullable and value is None) or type(value) in (str, dict, list), path,
         'expected text, object or array'+(' or null' if nullable else ''))
    if nonempty:
        need(bool(value), path, 'must not be empty')


def fields(value, required, optional, path):
    need(type(value) is dict, path, 'expected object')
    need(set(required) <= value.keys(), path, 'missing fields: '+str(sorted(set(required)-value.keys())))
    need(value.keys() <= set(required)|set(optional), path,
         'unknown fields: '+str(sorted(value.keys()-set(required)-set(optional))))


def assert_request(request, profile=Profile()):
    json_value(request)
    required = {'state', 'questions'} | ({'model'} if profile.require_model else set())
    fields(request, required, {'model'}, '$')
    content(request['state'], '$.state')
    if 'model' in request:
        need(type(request['model']) is str and bool(request['model']), '$.model', 'expected nonempty string')
    questions = request['questions']
    need(type(questions) is dict, '$.questions', 'expected object')
    need(1 <= len(questions) <= profile.max_questions, '$.questions', 'question count outside profile')
    for qid, q in questions.items():
        path = '$.questions.'+qid
        need(bool(qid), path, 'ID must not be empty')
        fields(q, {'type', 'instructions'}, {'criteria'}, path)
        t = q['type']
        need(type(t) is str and t in ('choice', 'score', 'noul'), path+'.type', 'unknown question type')
        content(q['instructions'], path+'.instructions', nonempty=True)
        criteria = q.get('criteria')
        if t == 'choice':
            need(type(criteria) is dict, path+'.criteria', 'expected option map')
            need(2 <= len(criteria) <= profile.max_choice_options, path+'.criteria', 'option count outside profile')
            for key, value in criteria.items():
                need(bool(key), path+'.criteria', 'option IDs must not be empty')
                content(value, path+'.criteria.'+key, nullable=True)
        elif t == 'score':
            need(type(criteria) is list, path+'.criteria', 'expected ordered levels')
            need(2 <= len(criteria) <= profile.max_score_levels, path+'.criteria', 'level count outside profile')
            for i, value in enumerate(criteria):
                content(value, path+f'.criteria[{i}]')
        elif criteria is not None:
            if type(criteria) is dict and criteria.keys() <= {'true', 'false'}:
                for key, value in criteria.items():
                    content(value, path+'.criteria.'+key)
            else:
                need(profile.noul_clarifications, path+'.criteria', 'expected true/false map')
                content(criteria, path+'.criteria')


def number(value, path, low=0, high=1):
    need(type(value) in (int, float) and (type(value) is int or math.isfinite(value)), path, 'expected finite number, not bool/string')
    need(low <= value <= high, path, f'expected value in [{low}, {high}]')


def assert_response(request, response, profile=Profile(), *, public=True, expected_model=None):
    assert_request(request, profile)
    json_value(response)
    fields(response, {'model', 'answers', 'usage'}, {'metadata'} | (set() if public else {'logits'}) |
           ({'errors'} if profile.question_errors else set()), '$')
    need(type(response['model']) is str and bool(response['model']), '$.model', 'expected nonempty model ID')
    if expected_model is not None:
        need(response['model'] == expected_model, '$.model', 'unexpected model identity')
    if 'metadata' in response:
        need(type(response['metadata']) is dict, '$.metadata', 'expected object')
    fields(response['usage'], {'input_tokens', 'output_tokens'}, set(), '$.usage')
    for key, value in response['usage'].items():
        need(type(value) is int and value >= 0, '$.usage.'+key, 'expected nonnegative integer')
    if profile.fast_path:
        need(response['usage']['output_tokens'] == 0, '$.usage.output_tokens', 'local fast path must not generate tokens')
    answers = response['answers']
    errors = response.get('errors', {})
    need(type(answers) is dict and type(errors) is dict, '$', 'answers/errors must be objects')
    need(not (answers.keys() & errors.keys()), '$', 'a question cannot have both an answer and an error')
    need((answers.keys() | errors.keys()) == request['questions'].keys(), '$.answers', 'answer/error IDs must cover every question exactly')
    for qid, q in request['questions'].items():
        if qid in errors:
            path, error = '$.errors.'+qid, errors[qid]
            fields(error, {'code', 'message', 'missing_options', 'n_probs'}, set(), path)
            need(error['code'] == 'missing_option_probabilities', path+'.code', 'unknown question error')
            need(type(error['message']) is str and bool(error['message']), path+'.message', 'expected nonempty text')
            need(type(error['n_probs']) is int and 1 <= error['n_probs'] <= 200000, path+'.n_probs', 'invalid probability count')
            keys = (list(q['criteria']) if q['type'] == 'choice' else
                    [str(i) for i in range(len(q['criteria']))] if q['type'] == 'score' else ['false', 'true'])
            missing = error['missing_options']
            need(type(missing) is list and bool(missing) and all(type(k) is str and k in keys for k in missing),
                 path+'.missing_options', 'expected supplied option IDs')
            need(len(set(missing)) == len(missing), path+'.missing_options', 'duplicate option ID')
            continue
        path = '$.answers.'+qid
        a, t = answers[qid], q['type']
        required = {'type', 'noul'} if t == 'noul' else {'type', 'confidence', 'probabilities'} | (
            {'choice'} if t == 'choice' else {'score', 'legend'})
        fields(a, required, set(), path)
        need(a['type'] == t, path+'.type', 'must match question type')
        if t == 'noul':
            number(a['noul'], path+'.noul')
            continue
        keys = list(q['criteria']) if t == 'choice' else [str(i) for i in range(len(q['criteria']))]
        probs = a['probabilities']
        need(type(probs) is dict and set(probs) == set(keys), path+'.probabilities', 'must contain every option/level and no others')
        for key, value in probs.items():
            number(value, path+'.probabilities.'+key)
        need(abs(math.fsum(probs.values())-1) <= profile.probability_atol, path+'.probabilities', 'probabilities must sum to one')
        number(a['confidence'], path+'.confidence')
        if t == 'choice':
            need(type(a['choice']) is str and a['choice'] in probs, path+'.choice', 'unknown option')
            need(probs[a['choice']] >= max(probs.values())-profile.probability_atol,
                 path+'.choice', 'choice must maximize probability; any exact tie winner is allowed')
        else:
            number(a['score'], path+'.score', 0, len(keys)-1)
            expected = math.fsum(i*probs[str(i)] for i in range(len(keys)))
            need(abs(a['score']-expected) <= profile.score_atol, path+'.score', 'must equal zero-based expectation')
            need(type(a['legend']) is dict and same_json(a['legend'], dict(zip(keys, q['criteria']))),
                 path+'.legend', 'must preserve ordered criterion descriptions (local structured legend profile)')


def exercise_backend(predict, request, profile=Profile(), *, independence_atol=1e-5, expected_model=None):
    """Check shape, no mutation, repeat/ID/order/batch isolation for one request.

    Does not assert semantic truth or option-permutation invariance. Caller
    owns backend lifetime and must explicitly opt into model-backed execution.
    """
    number(independence_atol, '$.independence_atol')
    original = deepcopy(request)
    def call(body):
        body = deepcopy(body)
        snapshot = deepcopy(body)
        result = predict(body['state'], body['questions'])
        need(same_json(body, snapshot), '$', 'backend mutated request')
        assert_response(body, result, profile, public=False, expected_model=expected_model)
        return result
    baseline = call(deepcopy(request))
    comparisons = 0
    def outcome(result, qid):
        return result['answers'].get(qid, result.get('errors', {}).get(qid))

    def same(first, second, path):
        nonlocal comparisons
        if 'code' in first or 'code' in second:
            need(same_json(first, second), path, 'question error depends on ID/order/sibling/history')
            comparisons += 1
            return
        need(first['type'] == second['type'], path, 'type changed')
        keys = ['noul'] if first['type'] == 'noul' else ['confidence']
        for key in keys:
            need(abs(first[key]-second[key]) <= independence_atol, path+'.'+key, 'depends on ID/order/sibling/history')
        if first['type'] != 'noul':
            need(first['probabilities'].keys() == second['probabilities'].keys(), path, 'option keys changed')
            for key, value in first['probabilities'].items():
                need(abs(value-second['probabilities'][key]) <= independence_atol, path+'.probabilities.'+key, 'depends on ID/order/sibling/history')
        comparisons += 1
    qs = request['questions']
    for tag, transformed in [('repeat', deepcopy(qs)), ('reverse', dict(reversed(list(qs.items()))))]:
        r = call({**request, 'questions':transformed})
        for qid in qs:
            same(outcome(baseline, qid), outcome(r, qid), tag+'.'+qid)
    names = {qid: f'renamed-{i}' for i, qid in enumerate(qs)}
    r = call({**request, 'questions':{names[k]:deepcopy(v) for k,v in qs.items()}})
    for qid, q in qs.items():
        same(outcome(baseline, qid), outcome(r, names[qid]), 'rename.'+qid)
        isolated = call({**request, 'questions':{qid:deepcopy(q)}})
        same(outcome(baseline, qid), outcome(isolated, qid), 'isolate.'+qid)
    target = next(iter(qs))
    siblings = deepcopy(qs)
    other = next((qid for qid in qs if qid != target), None)
    if other is None and len(qs) < profile.max_questions:
        other = 'contract-extra'
        while other in qs:
            other += '-x'
    if other is not None:
        siblings[other] = {'type':'noul','instructions':'Does the supplied state mention a purple satellite?'}
        r = call({**request, 'questions':siblings})
        same(outcome(baseline, target), outcome(r, target), 'changed-sibling.'+target)
    # Interleave a different state before repeating the original: catches stale caches.
    call({**request, 'state':{'contract_probe':'unrelated state; no answer is prescribed'}})
    r = call(deepcopy(request))
    for qid in qs:
        same(outcome(baseline, qid), outcome(r, qid), 'after-other-state.'+qid)
    need(same_json(request, original), '$', 'conformance probe mutated request')
    return {'questions':len(qs), 'comparisons':comparisons, 'status':'passed'}


def assert_rejected(predict, request, profile=Profile()):
    """Invalid native requests must fail explicitly, without mutating the input."""
    try:
        assert_request(request, profile)
    except ContractError:
        pass
    else:
        raise ContractError('Rejection fixture is a valid request')
    request = deepcopy(request)
    snapshot = deepcopy(request)
    try:
        predict(request['state'], request['questions'])
    except (ValueError, TypeError):
        need(same_json(request, snapshot), '$', 'backend mutated invalid request')
    else:
        raise ContractError('$: backend accepted an invalid request')
