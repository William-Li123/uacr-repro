"""Regenerate local detector weights/scores; never overwrite archived scores."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
os.environ['UACR_LEGACY_ROOT']=str(ROOT/'legacy')
sys.path.insert(0,str(ROOT/'legacy/scripts'))
ARCHIVES=json.loads((ROOT/'provenance/archives.json').read_text())
A=Path(ARCHIVES['anomaly_detection'])
B=Path(ARCHIVES['anomaly_detection2'])
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['train','score','saec-routes'])
p.add_argument('--dataset',required=True,choices=['goodsad_80p','mvtec_ad_80p','mvtec_loco_80p','visa_80p','ksdd2_mvtlike'])
p.add_argument('--category',help='Train one category; omitted trains all categories')
p.add_argument('--run',default='local_retrain')
p.add_argument('--weights-root',type=Path,help='Default: archived EfficientAD weights')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if Path(a.run).name!=a.run or a.run in ['.','..']:
    p.error('Invalid run name')
out=ROOT/'runs'/a.run
def rows(split):
    return [json.loads(x) for x in (ROOT/'data/three_field_qwen'/a.dataset/f'{split}.jsonl').read_text().splitlines() if x.strip()]
def write(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)));w.writeheader();w.writerows(items)

if a.stage=='train':
    import yaml
    data=B/'data/data_adapters'/a.dataset
    if not data.exists():
        raise FileNotFoundError(data)
    categories=sorted({r['category'] for r in rows('val')})
    if a.category:
        if a.category not in categories:
            p.error('Unknown category')
        categories=[a.category]
    for cat in categories:
        target=out/'efficientad'/a.dataset/cat
        config=yaml.safe_load((ROOT/'legacy/configs/efficientad_mvtlike_5001.yaml').read_text())
        config['Datasets']['train']['root']=str(data)
        config['Datasets']['eval']['root']=str(data)
        config['category']=cat
        config['ckpt_dir']=str(target)
        cfg=out/'configs'/f'{a.dataset}_{cat}.yaml'
        args=[sys.executable,str(ROOT/'legacy/tools/EfficientAD/train_reduced_student.py'),'-c',str(cfg)]
        print(args,flush=True)
        if a.dry_run:
            continue
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f'Output exists: {target}; use another --run')
        target.mkdir(parents=True)
        cfg.parent.mkdir(parents=True,exist_ok=True)
        cfg.write_text(yaml.safe_dump(config))
        shutil.copy2(config['Model']['checkpoints'],target/'best_teacher.pth')
        subprocess.run(args,check=True)
        # Use the last checkpoint, not whichever checkpoint maximized evaluation AUROC.
        for suffix in ['student.pth','autoencoder.pth','quantiles.npy']:
            stem,ext=suffix.rsplit('.',1)
            last=target/f'{cat}_{stem}_last.{ext}'
            if not last.exists():
                raise FileNotFoundError(last)
            shutil.copy2(last,target/f'{cat}_{suffix}')
elif a.stage=='score':
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from analyze_efficientad_mvtlike import EfficientADInference
    from run_unified_hybrid_five import best_threshold
    key='ksdd2' if a.dataset=='ksdd2_mvtlike' else a.dataset
    weights=(a.weights_root/a.dataset if a.weights_root else ROOT/'resources/model_efficient_AD'/key)
    transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    items=[]
    manifests={split:rows(split) for split in ['val','test']}
    for cat in sorted({r['category'] for r in manifests['val']}):
        infer=EfficientADInference(weights/cat,cat)
        for split,manifest in manifests.items():
            selected=[r for r in manifest if r['category']==cat]
            for start in range(0,len(selected),16):
                batch=selected[start:start+16]
                tensors=[]
                for row in batch:
                    with Image.open(row['image']) as im:
                        tensors.append(transform(im.convert('RGB')))
                values=infer.score_batch(torch.stack(tensors)).tolist()
                for row,value in zip(batch,values):
                    items.append({'id':row['id'],'dataset':key,'category':cat,'split2':split,'score':value,'label_int':int(row['target']['label']=='abnormal')})
        val=[r for r in items if r['category']==cat and r['split2']=='val']
        thr=best_threshold(np.array([r['label_int'] for r in val]),np.array([r['score'] for r in val]))
        for row in items:
            if row['category']==cat:
                row['thr']=thr;row['eff_pred']=int(row['score']>=thr)
        del infer
        torch.cuda.empty_cache()
    write(out/'local_scores'/f'{a.dataset}.csv',items)
    combined=[]
    for path in sorted((out/'local_scores').glob('*.csv')):
        with path.open() as f:
            combined.extend(csv.DictReader(f))
    write(out/'merged_val_test_scores.csv',combined)
    print(out/'merged_val_test_scores.csv')
elif a.stage=='saec-routes':
    import numpy as np
    sys.path.insert(0,str(ROOT/'route_compare_test/scripts'))
    os.environ['UACR_ROOT']=str(ROOT)
    import prepare_cached_baseline_routes as s
    s.OUT_ROOT=out/'route_compare_test'
    s.YOLO_WEIGHTS=B/'route_compare_test/yolo_weights/yolo11s-cls.pt'
    test=rows('test')
    values=s.complexity_scores(a.dataset,'test',test)
    threshold=float(np.quantile(list(values.values()),.7))
    low=[r for r in test if values[r['id']]<threshold]
    yolo=s.yolo_confidence_rows(low,a.dataset)
    result=[]
    for row in test:
        high=values[row['id']]>=threshold
        good=bool(int(float(yolo.get(row['id'],{}).get('yolo_confident_good',0))))
        item=s.route_row_base(row,a.dataset,'saec_mod')
        item.update(qwen_called=int(high or not good),route_source='complexity_qwen' if high else ('yolo_confident_good' if good else 'yolo_defect_or_uncertain'))
        result.append(item)
    write(out/'route_compare_test/routes_cached/saec_mod'/f'{a.dataset}.csv',result)
