"""Measure option-order agreement, selective accuracy and readout ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from typed_decisions.qwen import QwenDecisionModel
from typed_decisions.consistency import predict_consistent
from typed_decisions.metrics import summarize
from scripts.evaluate import resource_record


def metrics(records):
    rows=[]; rotated=[]; averaged=[]; agreed=[]
    for r in records:
        case,a=r['case'],r['result']['answers']['decision']
        keys=list(case['questions']['decision']['criteria']); target=case['targets']['decision']
        y=[float(k==target) for k in keys]
        def row(p): return {'p':[p[k] for k in keys],'target':y}
        rows.append(row(a['variants'][0]['probabilities']))
        rotated.append(row(a['variants'][1]['probabilities']))
        averaged.append(row(a['mean_probabilities']))
        if a['status']=='agreement': agreed.append(rows[-1])
    # Same-coverage confidence filter: useful control for apparent selective gains.
    by_confidence=sorted(rows,key=lambda r:max(r['p']),reverse=True)[:len(agreed)]
    return {'n':len(rows),'base':summarize(rows),'rotated':summarize(rotated),
            'averaged':summarize(averaged),'agreement_count':len(agreed),
            'agreement_coverage':len(agreed)/len(rows),
            'agreed':summarize(agreed) if agreed else {'n':0,'accuracy':None},
            'confidence_filter_same_coverage':summarize(by_confidence) if by_confidence else {'n':0,'accuracy':None},
            'mean_total_variation':statistics.mean(r['result']['answers']['decision']['total_variation_max'] for r in records),
            'median_ms_two_variants':statistics.median(r['result']['metadata']['elapsed_ms'] for r in records)}


def run(args):
    corpus_path=ROOT/'examples/audit-002.jsonl'
    cases=[json.loads(s) for s in corpus_path.read_text(encoding='utf-8').splitlines()]
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; summaries={}
    with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as log, resource_record(args.memory_gib,'Qwen order-sensitivity and readout audit'):
        model=QwenDecisionModel(lock_file=args.lock)
        with patch.object(model.model,'generate',side_effect=AssertionError('thinking/generation forbidden')):
            for split in ('development','test'):
                summaries[split]={}
                for mode in ('letter','json'):
                    model.readout=mode
                    current=[]
                    for case in [c for c in cases if c['split']==split]:
                        result=predict_consistent(model,case['state'],case['questions'])
                        record={'readout':mode,'case':case,'result':result}
                        rows.append(record); current.append(record)
                        log.write(json.dumps(record,ensure_ascii=False)+'\n'); log.flush()
                        a=result['answers']['decision']
                        print(json.dumps({'split':split,'mode':mode,'id':case['id'],'choices':[v['choice'] for v in a['variants']],
                                          'target':case['targets']['decision'],'status':a['status']}),flush=True)
                    summaries[split][mode]=metrics(current)
                if split=='development':
                    # Readout selection is frozen BEFORE any test inference.
                    selected=min(summaries[split],key=lambda m:(-summaries[split][m]['base']['accuracy'],summaries[split][m]['base']['nll']))
                    source='this model development families'
                    if args.selected_from:
                        selected=json.loads(args.selected_from.read_text())['selected_readout']
                        source=str(args.selected_from)
                    selection={'selected_readout':selected,'source':source,
                               'rule':'development base accuracy, then NLL; no test tuning',
                               'development':summaries[split]}
                    (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
            model.readout=selected
            model.choice_order='canonical'
            canonical=[]
            max_order_delta=0.
            for case in [c for c in cases if c['split']=='test']:
                qs=case['questions']; q=qs['decision']
                swapped={'decision':{**q,'criteria':dict(reversed(list(q['criteria'].items())))}}
                a=model.predict(case['state'],qs); b=model.predict(case['state'],swapped)
                pa=a['answers']['decision']['probabilities']; pb=b['answers']['decision']['probabilities']
                order_delta=max(abs(pa[k]-pb[k]) for k in pa)
                assert order_delta<1e-7
                max_order_delta=max(max_order_delta,order_delta)
                assert a['answers']['decision']['choice']==b['answers']['decision']['choice']
                canonical.append({'p':[pa[k] for k in q['criteria']],
                                  'target':[float(k==case['targets']['decision']) for k in q['criteria']]})
            summaries['test']['canonical_selected_readout']={'metrics':summarize(canonical),'max_order_delta':max_order_delta,
                                                             'scope':'invariance by deterministic ordering; not a bias diagnostic'}
        summary={'model':model.lock_data,'thinking':False,'weights_trained':False,'selected_readout':selected,
                 'corpus_sha256':hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                 'scope':'48 synthetic bilingual Choice cases; task families split; not broad calibration',
                 'results':summaries}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps({'selected':selected,'test':{m:{'base_acc':v['base']['accuracy'],'agreement_coverage':v['agreement_coverage'],
                          'agreement_acc':v['agreed']['accuracy']} for m,v in summaries['test'].items() if 'base' in v}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path)
    p.add_argument('--selected-from',type=Path);p.add_argument('--memory-gib',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
