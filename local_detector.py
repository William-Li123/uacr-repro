"""Train the local detector, calibrate on validation, and rebuild SAEC routes."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from project import ROOT, WORK, DATA, DATASETS, configure, read_rows, run_dir

configure()
sys.path.insert(0,str(ROOT/'legacy/scripts'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['train','score','saec-routes'])
p.add_argument('--dataset',required=True,choices=['goodsad_80p','mvtec_ad_80p','mvtec_loco_80p','visa_80p','ksdd2_mvtlike'])
p.add_argument('--category',help='Train one category; omitted trains all categories')
p.add_argument('--run',default='local_retrain')
p.add_argument('--weights-root',type=Path,help='Default: this run\'s trained EfficientAD weights')
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
if Path(a.run).name!=a.run or a.run in ['.','..']:
    p.error('Invalid run name')
out=run_dir(a.run)
def rows(split):
    return read_rows(DATA/a.dataset/f'{split}.jsonl')
def write(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in items for k in r)));w.writeheader();w.writerows(items)

if a.stage=='train':
    import yaml
    data=WORK/'data/data_adapters'/a.dataset
    if not data.exists():
        raise FileNotFoundError(data)
    categories=sorted({r['category'] for r in rows('val')})
    if a.category:
        if a.category not in categories:
            p.error('Unknown category')
        categories=[a.category]
    for cat in categories:
        target=out/'efficientad'/a.dataset/cat
        config=yaml.safe_load((ROOT/'configs/efficientad.yaml').read_text())
        config['Datasets']['train']['root']=str(data)
        config['Datasets']['eval']['root']=str(data)
        config['Datasets']['imagenet']['root']=str(WORK/'raw/imagenette2-320/train')
        config['Model']['checkpoints']=str(WORK/'models/teacher/best_teacher.pth')
        config['category']=cat
        config['ckpt_dir']=str(target)
        cfg=out/'configs'/f'{a.dataset}_{cat}.yaml'
        args=[sys.executable,str(ROOT/'legacy/tools/EfficientAD/train_reduced_student.py'),'-c',str(cfg)]
        print(args,flush=True)
        if a.dry_run:
            continue
        if (target/'TRAINING_DONE').is_file():
            print(f'Already complete: {target}')
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
        (target/'TRAINING_DONE').write_text('done\n')
elif a.stage=='score':
    if a.dry_run:
        print(f'Score {a.dataset} using {a.weights_root or out/"efficientad"}')
        raise SystemExit(0)
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from analyze_efficientad_mvtlike import EfficientADInference
    from sklearn.metrics import balanced_accuracy_score
    def best_threshold(labels, scores):
        best=None
        for threshold in np.unique(scores):
            pred=scores>=threshold
            acc=float((pred==labels).mean())
            bal=float(balanced_accuracy_score(labels,pred)) if len(np.unique(labels))>1 else acc
            rank=(bal,acc,bal)
            if best is None or rank>best[0]:
                best=(rank,float(threshold))
        if best is None:
            raise ValueError('Cannot calibrate an empty category')
        return best[1]
    key='ksdd2' if a.dataset=='ksdd2_mvtlike' else a.dataset
    weights=(a.weights_root or out/'efficientad')/a.dataset
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
    if a.dry_run:
        print(f'Rebuild SAEC routes for {a.dataset}')
        raise SystemExit(0)
    import numpy as np
    sys.path.insert(0,str(ROOT/'route_compare_test/scripts'))
    import prepare_cached_baseline_routes as s
    s.OUT_ROOT=out/'route_compare_test'
    s.YOLO_WEIGHTS=WORK/'models/saec/yolo11s-cls.pt'
    if not s.YOLO_WEIGHTS.is_file():
        raise FileNotFoundError('Run python download_resources.py --resource yolo first')
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
