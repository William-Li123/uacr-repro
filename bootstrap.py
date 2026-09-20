"""Extract the reproducible core without modifying either source archive."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--archive-parent', default='/mnt/aoss-250010161/cloudplatform/models')
p.add_argument('--dest', default='/data/yuzheng/uacr_repro')
a = p.parse_args()
dest = Path(a.dest)
if dest.exists():
    raise SystemExit(f'Refusing to overwrite existing project: {dest}')
dest.mkdir(parents=True)
roots = {n: Path(a.archive_parent)/n/'archive_20260903_full' for n in ['anomaly_detection', 'anomaly_detection2']}
links = {}
for name, root in roots.items():
    for line in (root.parent/'archive_20260903_full.symlinks.tsv').read_text().splitlines():
        rel, target = line.split('\t', 1)
        links[f'/data/yuzheng/{name}/{rel}'] = target

def resolve(value, seen=None):
    seen = set() if seen is None else seen
    if value in seen:
        raise ValueError(f'Circular archived link: {value}')
    seen.add(value)
    if value in links:
        target = links[value]
        if not target.startswith('/'):
            target = str(Path(value).parent/target)
        return resolve(target, seen)
    for name in ['anomaly_detection2', 'anomaly_detection']:
        prefix = f'/data/yuzheng/{name}/'
        if value.startswith(prefix):
            return str(roots[name]/value[len(prefix):])
    return value

provenance = []
def copy_code(name, rel, out=None):
    src = roots[name]/rel
    target = dest/(out or rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_bytes()
    text = raw.decode('utf-8-sig')
    text = text.replace('Path("/data/yuzheng/anomaly_detection2")', 'Path(__import__("os").environ["UACR_ROOT"])')
    text = text.replace('Path("/data/yuzheng/anomaly_detection")', 'Path(__import__("os").environ["UACR_LEGACY_ROOT"])')
    if rel == 'script/evaluate_saec_same_new_adapter.py':
        text = text.replace('ROOT / "pixel_alignment_acp" / "predictions_147456"', 'Path(__import__("os").environ["UACR_PREDICTIONS"])')
        text = text.replace('ROOT / "pixel_alignment_acp" / "saec_same_new_adapter"', 'Path(__import__("os").environ["UACR_SAEC_OUTPUT"])')
    target.write_text(text, encoding='utf-8')
    provenance.append({'source':str(src),'output':str(target.relative_to(dest)), 'source_sha256':hashlib.sha256(raw).hexdigest(), 'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})

for rel in ['script/train_three_field_qwen_lora.py', 'script/build_three_field_qwen_data.py', 'script/evaluate_pixel_variant_routes.py', 'script/evaluate_saec_same_new_adapter.py', 'eval/eval_three_field_qwen.py', 'eval/EVALUATION_RULES.md']:
    copy_code('anomaly_detection2', rel)
for name in ['tune_per_dataset_q_20260630.py','compare_random50_saec_ours_20260630.py','prepare_cached_baseline_routes.py']:
    copy_code('anomaly_detection2', 'route_compare_test/scripts/'+name)
for src in (roots['anomaly_detection']/'tools/EfficientAD').glob('*.py'):
    copy_code('anomaly_detection', str(src.relative_to(roots['anomaly_detection'])), 'legacy/tools/EfficientAD/'+src.name)
copy_code('anomaly_detection','tools/EfficientAD/requirements.txt','legacy/tools/EfficientAD/requirements.txt')
copy_code('anomaly_detection','scripts/analyze_efficientad_mvtlike.py','legacy/scripts/analyze_efficientad_mvtlike.py')
copy_code('anomaly_detection','scripts/run_unified_hybrid_five.py','legacy/scripts/run_unified_hybrid_five.py')
for src in (roots['anomaly_detection']/'configs').glob('efficientad*.yaml'):
    target = dest/'legacy/configs'/src.name
    target.parent.mkdir(parents=True,exist_ok=True)
    text = src.read_text().replace('/data/yuzheng/anomaly_detection',str(roots['anomaly_detection']))
    target.write_text(text)

def link(src, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(src, target_is_directory=Path(src).is_dir())

for rel in ['model_qwen_base','model_efficient_AD']:
    link(roots['anomaly_detection2']/rel,dest/'resources'/rel)
link(roots['anomaly_detection2']/'model_qwen_adapter/sft_lora_px147456', dest/'resources/adapters_147456')
link(roots['anomaly_detection2']/'model_qwen_adapter/sft_lora', dest/'resources/adapters_65536')
link(roots['anomaly_detection2']/'pixel_alignment_acp/predictions_147456', dest/'resources/predictions_147456')
link(roots['anomaly_detection2']/'pixel_alignment_acp/predictions_65536', dest/'resources/predictions_65536')
link(roots['anomaly_detection2']/'pixel_alignment_acp/routes_147456', dest/'resources/reference_routes_147456')
link(roots['anomaly_detection2']/'pixel_alignment_acp/saec_same_new_adapter',dest/'resources/reference_saec')

counts = {}
for src in (roots['anomaly_detection2']/'data/three_field_qwen').glob('*/*.jsonl'):
    target = dest/'data/three_field_qwen'/src.parent.name/src.name
    target.parent.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    for row in rows:
        row['image'] = resolve(row['image'])
        row['ref_images'] = [resolve(x) for x in row.get('ref_images',[])]
    target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf-8')
    counts[str(target.relative_to(dest))] = len(rows)
for rel in ['data/splits/hybrid_unified_five/merged_val_test_scores.csv']:
    target = dest/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(roots['anomaly_detection2']/rel,target)
shutil.copytree(roots['anomaly_detection2']/'route_compare_test/routes_cached/saec_mod',dest/'route_compare_test/routes_cached/saec_mod')

# Materialize only image links for the preserved EfficientAD training layout.
prefix='/data/yuzheng/anomaly_detection2/data/data_adapters/'
for old in links:
    if old.startswith(prefix):
        link(resolve(old),dest/'data/data_adapters'/old[len(prefix):])
(dest/'provenance').mkdir()
(dest/'provenance/source_files.json').write_text(json.dumps(provenance,indent=2))
(dest/'provenance/manifest_counts.json').write_text(json.dumps(counts,indent=2))
(dest/'provenance/archives.json').write_text(json.dumps({k:str(v) for k,v in roots.items()},indent=2))
for name in ['reproduce.py','README.md','checks.py','local_detector.py','environment.py','verify_replay.py','smoke_local.py','run_all.sh','开始复现.md']:
    shutil.copy2(Path(__file__).with_name(name),dest/name)
shutil.copy2(__file__,dest/'bootstrap.py')
print(json.dumps({'project':str(dest),'source_files':len(provenance),'manifests':counts},indent=2))
