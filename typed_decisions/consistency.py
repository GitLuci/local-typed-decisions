"""Optional option-order diagnostic: independent variants in one shared forward.

Agreement is stability under the tested permutations, not proof of correctness or
a calibrated estimate of uncertainty. No answer is fed into another branch.
"""
from .runtime import validate
from .qwen import render


def predict_consistent(model, state, questions, *, rounds=2, prepared=None):
    if model.choice_order != "input":
        raise ValueError("permutation diagnostics require choice_order='input'; canonical ordering hides the intervention")
    qs = validate(state, questions, max_options=255)
    if prepared is not None and render(state) != prepared._state_text:
        raise ValueError("supplied state differs from the prepared state snapshot")
    if not isinstance(rounds,int) or isinstance(rounds,bool) or rounds<2:
        raise ValueError("rounds must be an integer >= 2")
    if len(qs)*rounds>32:
        raise ValueError("all variants must fit the 32-branch limit; use fewer questions or rounds")
    expanded={}
    for i,(qid,q) in enumerate(qs.items()):
        if q['type']!='choice':
            raise ValueError("option permutation supports Choice only; Score levels have semantic order")
        options=list(q['criteria'].items())
        if rounds>len(options):
            raise ValueError("rounds must not exceed the number of distinct cyclic option orders")
        for r in range(rounds):
            # A one-place cyclic shift changes the code of EVERY option.
            shifted=options[r:]+options[:r]
            expanded[f'q{i}_r{r}']={**q,'criteria':dict(shifted)}
    result=model.predict_prepared(prepared,expanded) if prepared is not None else model.predict(state,expanded)
    answers={}
    for i,(qid,q) in enumerate(qs.items()):
        variants=[result['answers'][f'q{i}_r{r}'] for r in range(rounds)]
        logits=[result['logits'][f'q{i}_r{r}'] for r in range(rounds)]
        selected=[v['choice'] for v in variants]
        stable=len(set(selected))==1
        probabilities={k:sum(v['probabilities'][k] for v in variants)/rounds for k in q['criteria']}
        variation=max(.5*sum(abs(a['probabilities'][k]-b['probabilities'][k]) for k in q['criteria'])
                      for a in variants for b in variants)
        answers[qid]={'status':'agreement' if stable else 'order_sensitive',
                      'choice':selected[0] if stable else None,
                      'mean_probabilities':probabilities,'total_variation_max':variation,
                      'variants':variants,
                      'variant_logits':[{k:z for k,z in zip(expanded[f'q{i}_r{r}']['criteria'],row)}
                                        for r,row in enumerate(logits)]}
    return {'model':result['model'],'answers':answers,'usage':result['usage'],
            'metadata':{**result['metadata'],'diagnostic':'cyclic-option-permutations',
                        'rounds':rounds,'original_questions':len(qs),'inference_branches':len(expanded),
                        'agreement_is_correctness':False,'thinking':False}}
