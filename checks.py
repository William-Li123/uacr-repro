import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
errors=[]
counts={}
for folder in sorted((ROOT/'data/three_field_qwen').iterdir()):
    split_images={}
    for path in sorted(folder.glob('*.jsonl')):
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        counts[f'{folder.name}/{path.stem}']=len(rows)
        ids=[r['id'] for r in rows]
        if len(ids)!=len(set(ids)):
            errors.append(f'Duplicate IDs: {path}')
        split_images[path.stem]={r['image'] for r in rows}
        for r in rows:
            for image in [r['image']]+r.get('ref_images',[]):
                if not Path(image).is_file():
                    errors.append(f'Missing image: {image}')
    for left,right in [('train','val'),('train','test'),('val','test')]:
        overlap=split_images.get(left,set()) & split_images.get(right,set())
        if overlap:
            errors.append(f'Image overlap {folder.name} {left}/{right}: {len(overlap)}')
report={'manifest_counts':counts,'errors':list(dict.fromkeys(errors))}
(ROOT/'provenance/data_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'counts':counts,'error_count':len(report['errors']),'first_errors':report['errors'][:10]},indent=2))
sys.exit(bool(errors))
