"""Prepare the fixed experiment splits from publicly downloaded datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from project import ROOT, WORK, DATA, DATASETS, read_rows

def resolve_image(raw, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or not part.parts:
        raise ValueError(f'Unsafe dataset path: {relative}')
    path = raw.joinpath(*part.parts)
    if not path.is_file():
        raise FileNotFoundError(f'Missing image: {path}. See docs/DATASETS.md for extraction layout.')
    return path.resolve()

def put_link(source, target, copy=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source:
            return
        if target.is_file() and os.path.samefile(source, target):
            return
        if target.is_file() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest():
            return
        raise FileExistsError(f'Refusing to overwrite: {target}')
    if copy:
        shutil.copy2(source, target)
    else:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', type=Path, default=WORK/'raw')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--copy', action='store_true', help='Copy detector images instead of linking')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    raw = args.raw_root.expanduser().resolve()
    counts, prepared, checksums = {}, {}, {}
    resolved = {}
    def image_path(relative):
        if relative not in resolved:
            resolved[relative] = str(resolve_image(raw, relative))
        return resolved[relative]
    for dataset in args.datasets:
        images = {}
        for split in ['train', 'val', 'test']:
            template = ROOT/'data/three_field_qwen'/dataset/f'{split}.jsonl'
            rows = read_rows(template)
            ids = [row['id'] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate IDs: {dataset}/{split}')
            for row in rows:
                row['image'] = image_path(row['image'])
                row['ref_images'] = [image_path(x) for x in row.get('ref_images', [])]
            images[split] = {r['image'] for r in rows}
            prepared[(dataset, split)] = rows
            counts[f'{dataset}/{split}'] = len(rows)
            checksums[f'{dataset}/{split}'] = hashlib.sha256(template.read_bytes()).hexdigest()
            print(f'Checked {dataset}/{split}: {len(rows)} samples', flush=True)
        for left, right in [('train','val'), ('train','test'), ('val','test')]:
            if images[left] & images[right]:
                raise ValueError(f'Query overlap: {dataset} {left}/{right}')
    if args.check_only:
        print(json.dumps({'counts':counts, 'status':'all paths and splits checked'}, indent=2))
        return
    record = WORK/'prepared.json'
    signature = {'raw_root':str(raw), 'datasets':args.datasets, 'manifest_sha256':checksums}
    if record.exists() and json.loads(record.read_text()) != signature:
        raise RuntimeError('Prepared configuration changed. Use a new UACR_WORKDIR.')
    for (dataset, split), rows in prepared.items():
        output = DATA/dataset/f'{split}.jsonl'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        # The local trainer sees normal training queries and validation queries only.
        if split in ['train', 'val']:
            for row in rows:
                normal = row['target']['label'] == 'normal'
                if split == 'train' and not normal:
                    continue
                source = Path(row['image'])
                if Path(row['id']).name != row['id'] or Path(row['category']).name != row['category']:
                    raise ValueError('Unsafe sample ID or category')
                branch = 'train' if split == 'train' else 'test'
                target = WORK/'data/data_adapters'/dataset/row['category']/branch/('good' if normal else 'defect')/(row['id']+source.suffix)
                put_link(source, target, args.copy)
    score = WORK/'data/splits/hybrid_unified_five/merged_val_test_scores.csv'
    score.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/'data/splits/hybrid_unified_five/merged_val_test_scores.csv', score)
    masks = WORK/'route_compare_test/routes_cached/saec_mod'
    masks.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        shutil.copy2(ROOT/'route_compare_test/routes_cached/saec_mod'/f'{dataset}.csv', masks/f'{dataset}.csv')
    record.write_text(json.dumps(signature, indent=2))
    print(json.dumps({'work_dir':str(WORK), 'counts':counts}, indent=2))

if __name__ == '__main__':
    main()
