"""CPU-only manifest and pipeline wiring checks; no training quality claims."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import project


class ManifestTests(unittest.TestCase):
    def test_templates_are_portable_and_disjoint(self):
        total = 0
        for dataset in project.DATASETS:
            sets = {}
            for split in ('train', 'val', 'test'):
                rows = project.read_rows(ROOT/'data/three_field_qwen'/dataset/f'{split}.jsonl')
                self.assertTrue(rows)
                self.assertEqual(len(rows), len({r['id'] for r in rows}))
                sets[split] = {r['image'] for r in rows}
                total += len(rows)
                for row in rows:
                    self.assertIn(row['target']['label'], ('normal', 'abnormal'))
                    for value in [row['image'], *row['ref_images']]:
                        self.assertFalse(PurePosixPath(value).is_absolute(), value)
                        self.assertFalse(PureWindowsPath(value).drive, value)
                        self.assertNotIn('..', PurePosixPath(value).parts)
            self.assertFalse(sets['train'] & sets['val'])
            self.assertFalse(sets['train'] & sets['test'])
            self.assertFalse(sets['val'] & sets['test'])
        self.assertEqual(total, 29041)

    def test_fixed_tables_cover_evaluation_ids(self):
        with (ROOT/'data/splits/hybrid_unified_five/merged_val_test_scores.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        keys = [(r['dataset'], r['split2'], r['id']) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        keys = set(keys)
        for dataset in project.DATASETS:
            score_key = 'ksdd2' if dataset == 'ksdd2_mvtlike' else dataset
            for split in ('val', 'test'):
                manifest = project.read_rows(ROOT/'data/three_field_qwen'/dataset/f'{split}.jsonl')
                for row in manifest:
                    self.assertIn((score_key, split, row['id']), keys)
            with (ROOT/'route_compare_test/routes_cached/saec_mod'/f'{dataset}.csv').open(encoding='utf-8') as f:
                masks = list(csv.DictReader(f))
            self.assertEqual({r['id'] for r in masks}, {r['id'] for r in manifest})
            self.assertEqual(len(masks), len(manifest))


class PreparationTests(unittest.TestCase):
    def test_prepare_rerun_and_no_test_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'repo'
            work = Path(temporary)/'work'
            raw = Path(temporary)/'raw'
            ds = 'mvtec_ad_80p'
            directory = root/'data/three_field_qwen'/ds
            directory.mkdir(parents=True)
            raw.mkdir()
            for split in ('train', 'val', 'test'):
                (raw/f'{split}.png').write_bytes(b'synthetic fixture, not a decoded image')
                row = {'id':split, 'category':'object', 'image':f'{split}.png',
                       'ref_images':['train.png'],
                       'target':{'label':'normal' if split=='train' else 'abnormal'}}
                (directory/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
            for name in ['data/splits/hybrid_unified_five/merged_val_test_scores.csv',
                         f'route_compare_test/routes_cached/saec_mod/{ds}.csv']:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('id\n')
            args = ['bootstrap.py','--raw-root',str(raw),'--datasets',ds,'--copy']
            with patch.multiple(bootstrap, ROOT=root, WORK=work, DATA=work/'data/three_field_qwen'):
                with patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()):
                    bootstrap.main()
                    bootstrap.main()
                self.assertTrue((work/'prepared.json').is_file())
                detector = work/'data/data_adapters'/ds/'object'
                self.assertTrue((detector/'train/good/train.png').is_file())
                self.assertTrue((detector/'test/defect/val.png').is_file())
                self.assertFalse(list(detector.rglob('test.png')))
                # Re-run after automatic copy fallback is also idempotent.
                bootstrap.put_link((raw/'train.png').resolve(), detector/'train/good/train.png')
            with self.assertRaises(ValueError):
                bootstrap.resolve_image(raw, '../outside.png')
            with self.assertRaises(FileNotFoundError):
                bootstrap.resolve_image(raw, 'missing.png')

    def test_pipeline_plans_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, UACR_WORKDIR=temporary)
            for script in ['bootstrap.py','download_resources.py','pipeline.py','reproduce.py','local_detector.py']:
                result = subprocess.run([sys.executable,str(ROOT/script),'--help'],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            command = [sys.executable,str(ROOT/'pipeline.py'),'--dry-run','--datasets','mvtec_ad_80p']
            result = subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('local_detector.py',result.stdout)
            self.assertIn('--eff-scores',result.stdout)
            self.assertIn('--base',result.stdout)
            result = subprocess.run(command+['--protocol','fixed-local'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('local_detector.py',result.stdout)


if __name__ == '__main__':
    unittest.main()
