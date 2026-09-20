"""Portable project paths and shared experiment identifiers."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get('UACR_WORKDIR', str(ROOT / 'work'))).expanduser().resolve()
DATA = WORK / 'data/three_field_qwen'
DATASETS = ['goodsad_80p', 'mvtec_ad_80p', 'mvtec_loco_80p', 'visa_80p', 'ksdd2_mvtlike']
MODELS = {'qwen35_9b': 'Qwen3.5-9B', 'qwen25vl_7b': 'Qwen2.5-VL-7B-Instruct'}

def configure():
    os.environ['UACR_ROOT'] = str(WORK)
    os.environ['UACR_LEGACY_ROOT'] = str(ROOT / 'legacy')
    os.environ['UACR_WORKDIR'] = str(WORK)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def run_dir(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be one directory name')
    return WORK / 'runs' / name
