"""Single entry point; all new outputs stay inside this project."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ['UACR_ROOT'] = str(ROOT)
os.environ['UACR_LEGACY_ROOT'] = str(ROOT/'legacy')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
for path in ['script','eval','route_compare_test/scripts']:
    sys.path.insert(0,str(ROOT/path))
MODELS={'qwen35_9b':'Qwen3.5-9B','qwen25vl_7b':'Qwen2.5-VL-7B-Instruct'}
DATASETS=['goodsad_80p','mvtec_ad_80p','mvtec_loco_80p','visa_80p','ksdd2_mvtlike']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['check','replay','train','infer','routes','summary'])
p.add_argument('--model',choices=list(MODELS),default='qwen25vl_7b')
p.add_argument('--dataset',choices=DATASETS,default='mvtec_ad_80p')
p.add_argument('--split',choices=['val','test'],default='test')
p.add_argument('--pixels',type=int,choices=[65536,147456],default=147456)
p.add_argument('--adapter-root',type=Path,help='Explicit adapter root; inference default is archived aligned adapters')
p.add_argument('--base',action='store_true',help='Unadapted base with the SAME row prompt and pixel budget')
p.add_argument('--limit',type=int,default=0,help='Smoke only; must not be mixed with full predictions')
p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
p.add_argument('--cache-root',type=Path)
p.add_argument('--eff-scores',type=Path,help='Regenerated local score table; default is archived scores')
p.add_argument('--saec-run',type=Path,help='Run directory containing regenerated route_compare_test/routes_cached/saec_mod')
p.add_argument('--run',default='reproduction')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if not a.run or Path(a.run).name!=a.run or a.run in ['.','..']:
    p.error('--run must be a single directory name')
out=ROOT/'runs'/a.run

def command(args):
    import shlex
    print(shlex.join(map(str,args)),flush=True)
    if not a.dry_run:
        subprocess.run(list(map(str,args)),check=True,cwd=ROOT)

def dump_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def routes(cache):
    module=importlib.import_module('evaluate_pixel_variant_routes')
    module.DATASETS=a.datasets
    if a.eff_scores:
        module.route.EFF_SCORES=a.eff_scores
    sys.argv=['routes','--cache-root',str(cache),'--result-root',str(out/'uacr'),'--variant-name',f'pixels_{a.pixels}']
    module.main()
    saec=importlib.import_module('evaluate_saec_same_new_adapter')
    saec.DATASETS=a.datasets
    if a.saec_run:
        saec.ROOT=a.saec_run
    os.environ['UACR_PREDICTIONS']=str(cache)
    os.environ['UACR_SAEC_OUTPUT']=str(out/'saec')
    saec.main()

def summary():
    rows=[]
    metadata=json.loads((out/'inputs.json').read_text()) if (out/'inputs.json').exists() else {}
    datasets=metadata.get('datasets',a.datasets)
    for method in ['uacr','saec']:
        with (out/method/'test_summary.csv').open() as f:
            for row in csv.DictReader(f):
                row['method']=method
                av=row.get('defect_type_exact_abnormal'); pv=row.get('defect_type_hit_pred_abnormal')
                if av and pv:
                    av,pv=float(av),float(pv)
                    row['type_f1']=2*av*pv/(av+pv) if av+pv else 0
                else:
                    row['type_f1']=''
                rows.append(row)
    from eval_three_field_qwen import aggregate, row_metrics
    from evaluate_pixel_variant_routes import add_type_p
    import tune_per_dataset_q_20260630 as route
    if a.eff_scores or metadata.get('eff_scores'):
        route.EFF_SCORES=Path(a.eff_scores or metadata['eff_scores'])
    eff=route.load_eff_scores()
    for dataset in datasets:
        manifest=route.read_jsonl(ROOT/'data/three_field_qwen'/dataset/'test.jsonl')
        correct=sum(int(eff[(dataset,'test',r['id'])]['eff_pred'])==int(r['target']['label']=='abnormal') for r in manifest)
        rows.append({'method':'efficientad','model_key':'local','dataset':dataset,'n':len(manifest),'label_accuracy':correct/len(manifest),'qwen_rate':0})
        for model in MODELS:
            base=out/'base_predictions'/model/dataset/'test/predictions.csv'
            if metadata.get('archived_base',a.stage=='replay' and a.pixels==147456):
                archives=json.loads((ROOT/'provenance/archives.json').read_text())
                base=Path(archives['anomaly_detection2'])/'route_compare_test/base_qwen_three_field_current_prompt'/model/dataset/'predictions.csv'
            if not base.exists():
                continue
            with base.open() as f:
                predicted=list(csv.DictReader(f))
            if len(predicted)!=len(manifest) or {r['id'] for r in predicted}!={r['id'] for r in manifest}:
                raise RuntimeError(f'Incomplete base predictions: {base}')
            item=add_type_p(aggregate(predicted),predicted)
            item.update(method='qwen_base',model_key=model,dataset=dataset,qwen_rate=1)
            av=item.get('defect_type_exact_abnormal');pv=item.get('defect_type_hit_pred_abnormal')
            item['type_f1']=(2*av*pv/(av+pv) if av+pv else 0) if av is not None and pv is not None else ''
            rows.append(item)
    for row in rows:
        row['large_model_invocation_rate']=row.get('qwen_rate','')
        if row.get('denom_type_abnormal') in (0,'0'):
            row['defect_type_exact_abnormal']=''
            row['defect_type_hit_pred_abnormal']=''
            row['type_f1']=''
        if row.get('denom_explanation_abnormal') in (0,'0'):
            row['explanation_keyword_hit_abnormal']=''
    dump_csv(out/'metrics.csv',rows)
    print('Metrics:',out/'metrics.csv')

if a.stage=='check':
    command([sys.executable,ROOT/'checks.py'])
elif a.stage=='train':
    if a.dataset=='ksdd2_mvtlike':
        p.error('KSDD2 has no dedicated three-field training manifest; use a frozen transfer adapter.')
    if a.pixels!=147456:
        p.error('Canonical new training uses 147456; 65536 is the archived evaluation ablation.')
    target=out/('smoke_adapters' if a.limit else 'adapters')/a.model/a.dataset
    if (target/'TRAINING_DONE').exists():
        raise SystemExit('Already complete; use a new --run for a separate experiment.')
    args=[sys.executable,ROOT/'script/train_three_field_qwen_lora.py','--train-manifest',ROOT/'data/three_field_qwen'/a.dataset/'train.jsonl','--model',ROOT/'resources/model_qwen_base'/MODELS[a.model],'--output-dir',target,'--epochs',1,'--batch-size',2,'--grad-accum',4,'--learning-rate','2e-4','--max-pixels',147456,'--max-length',4096,'--max-samples',a.limit,'--normal-abnormal-ratio',3 if a.dataset=='visa_80p' else 2,'--label-loss-weight',10,'--defect-loss-weight',3,'--explanation-loss-weight',1,'--lora-r',8,'--lora-alpha',16,'--lora-dropout',0.05,'--seed',42,'--save-strategy','steps','--save-steps',100,'--save-total-limit',2,'--logging-steps',10]
    checkpoints=sorted(target.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
    if checkpoints:
        args+=['--resume-from-checkpoint',checkpoints[-1]]
    elif target.exists() and any(target.iterdir()):
        raise SystemExit('Nonempty incomplete output without checkpoint; choose a new --run.')
    command(args)
elif a.stage=='infer':
    family='base_predictions' if a.base else 'predictions'
    if a.limit:
        family='smoke_'+family
    target=out/family/a.model/a.dataset/a.split
    args=[sys.executable,ROOT/'eval/eval_three_field_qwen.py','--data-root',ROOT/'data/three_field_qwen','--dataset',a.dataset,'--split',a.split,'--model',ROOT/'resources/model_qwen_base'/MODELS[a.model],'--out-dir',target,'--batch-size',4,'--max-new-tokens',180,'--max-pixels',a.pixels,'--prompt-source','row','--limit',a.limit]
    adapter=None
    if not a.base:
        adapter=(a.adapter_root or ROOT/f'resources/adapters_{a.pixels}')/a.model/('mvtec_ad_80p' if a.dataset=='ksdd2_mvtlike' else a.dataset)
        args+=['--adapter',adapter]
    signature={'model':a.model,'dataset':a.dataset,'split':a.split,'pixels':a.pixels,'adapter':str(adapter),'limit':a.limit}
    config=target/'invocation.json'
    if config.exists() and json.loads(config.read_text())!=signature:
        raise SystemExit('Output configuration differs; choose a new --run.')
    if not a.dry_run:
        target.mkdir(parents=True,exist_ok=True)
        config.write_text(json.dumps(signature,indent=2))
    command(args)
elif a.stage in ['replay','routes']:
    cache=a.cache_root or (ROOT/f'resources/predictions_{a.pixels}' if a.stage=='replay' else out/'predictions')
    if a.dry_run:
        print('Cache:',cache,'Output:',out)
    else:
        # Exact ID coverage is checked before searching q; partial smoke caches cannot pass.
        for model in MODELS:
            for dataset in a.datasets:
                for split in ['val','test']:
                    expected={json.loads(x)['id'] for x in (ROOT/'data/three_field_qwen'/dataset/f'{split}.jsonl').read_text().splitlines() if x.strip()}
                    with (cache/model/dataset/split/'predictions.csv').open() as f:
                        actual=[r['id'] for r in csv.DictReader(f)]
                    if len(actual)!=len(set(actual)) or set(actual)!=expected:
                        raise RuntimeError(f'Incomplete or duplicate cache: {model}/{dataset}/{split}')
        metadata={'cache_root':str(cache.resolve()),'eff_scores':str((a.eff_scores or ROOT/'data/splits/hybrid_unified_five/merged_val_test_scores.csv').resolve()),'saec_run':str(a.saec_run or ROOT),'datasets':a.datasets,'pixels':a.pixels,'archived_base':a.stage=='replay' and a.pixels==147456}
        out.mkdir(parents=True,exist_ok=True)
        record=out/'inputs.json'
        if record.exists() and json.loads(record.read_text())!=metadata:
            raise RuntimeError('Run input configuration changed; use a new --run.')
        record.write_text(json.dumps(metadata,indent=2))
        routes(cache)
        summary()
elif a.stage=='summary':
    summary()
