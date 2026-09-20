"""Check prepared query/reference paths and split separation."""
import json
from pathlib import Path
from project import WORK, DATA, read_rows

def main():
    if not (WORK/'prepared.json').is_file():
        raise SystemExit('Run python bootstrap.py --raw-root YOUR_DATA_DIRECTORY first.')
    config=json.loads((WORK/'prepared.json').read_text())
    counts={}
    checked=set()
    for dataset in config['datasets']:
        images={}
        for split in ['train','val','test']:
            rows=read_rows(DATA/dataset/f'{split}.jsonl')
            ids=[r['id'] for r in rows]
            assert len(ids)==len(set(ids)), (dataset, split, 'duplicate IDs')
            images[split]={r['image'] for r in rows}
            for row in rows:
                for image in [row['image']]+row.get('ref_images',[]):
                    if image in checked:
                        continue
                    if not Path(image).is_file():
                        raise FileNotFoundError(image)
                    checked.add(image)
            counts[f'{dataset}/{split}']=len(rows)
        for a,b in [('train','val'),('train','test'),('val','test')]:
            assert not images[a]&images[b], (dataset,a,b,'overlap')
    print(json.dumps({'status':'PASS','counts':counts},indent=2))

if __name__=='__main__':
    main()
