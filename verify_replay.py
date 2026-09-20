import csv
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parent
checks=[]
for method,reference in [('uacr','reference_routes_147456'),('saec','reference_saec')]:
    def read(path):
        with path.open() as f:
            return {(r['model_key'],r['dataset']):r for r in csv.DictReader(f)}
    actual=read(root/'runs/archived147'/method/'test_summary.csv')
    expected=read(root/'resources'/reference/'test_summary.csv')
    assert actual.keys()==expected.keys(),(method,'row keys differ')
    fields=['n','label_accuracy','defect_type_exact_abnormal','defect_type_hit_pred_abnormal','explanation_keyword_hit_abnormal','qwen_rate','qwen_called']
    for key,old in expected.items():
        for field in fields:
            x,y=actual[key].get(field,''),old.get(field,'')
            assert (x==y) or (x and y and math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-12)),(method,key,field,x,y)
    checks.append({'method':method,'rows':len(actual),'fields':fields,'status':'MATCH','tolerance':1e-12})
(root/'provenance/replay_verification.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
