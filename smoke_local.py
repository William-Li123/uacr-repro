import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
os.environ['UACR_LEGACY_ROOT']=str(ROOT/'legacy')
sys.path.insert(0,str(ROOT/'legacy/scripts'))
import torch
from PIL import Image
from torchvision import transforms
from analyze_efficientad_mvtlike import EfficientADInference

row=json.loads((ROOT/'data/three_field_qwen/mvtec_ad_80p/val.jsonl').read_text().splitlines()[0])
model=EfficientADInference(ROOT/'resources/model_efficient_AD/mvtec_ad_80p'/row['category'],row['category'])
with Image.open(row['image']) as image:
    tensor=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])(image.convert('RGB'))
score=float(model.score_batch(tensor.unsqueeze(0))[0])
with (ROOT/'data/splits/hybrid_unified_five/merged_val_test_scores.csv').open() as f:
    original=next(x for x in csv.DictReader(f) if x['id']==row['id'] and x['dataset']=='mvtec_ad_80p')
reference=float(original['score'])
threshold=float(original['thr'])
result={'id':row['id'],'score':score,'archived_score':reference,'abs_difference':abs(score-reference),'device':str(model.device),'finite_forward':bool(torch.isfinite(torch.tensor(score))),'score_matches_at_1e_4':abs(score-reference)<1e-4,'decision_matches':(score>=threshold)==(reference>=threshold),'threshold':threshold}
(ROOT/'provenance/local_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
sys.exit(0 if result['finite_forward'] and result['decision_matches'] else 1)
