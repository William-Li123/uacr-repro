"""Sequential, resumable reproduction on one selected CUDA device."""
import argparse
import os
import shlex
import subprocess
import sys
from project import ROOT, WORK, DATASETS, MODELS, run_dir

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='full_reproduction')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--protocol',choices=['retrain-local','fixed-local'],default='retrain-local')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    out=run_dir(a.run)
    if 'ksdd2_mvtlike' in a.datasets and 'mvtec_ad_80p' not in a.datasets:
        p.error('KSDD2 uses the MVTec AD adapter; include mvtec_ad_80p.')
    env=dict(os.environ,UACR_WORKDIR=str(WORK))
    def run(script,*arguments):
        command=[sys.executable,str(ROOT/script),*map(str,arguments)]
        print(shlex.join(command),flush=True)
        if not a.dry_run:
            subprocess.run(command,cwd=ROOT,env=env,check=True)
    run('checks.py')
    if a.protocol=='retrain-local':
        for dataset in a.datasets:
            run('local_detector.py','train','--dataset',dataset,'--run',a.run)
            run('local_detector.py','score','--dataset',dataset,'--run',a.run)
            run('local_detector.py','saec-routes','--dataset',dataset,'--run',a.run)
    for model in MODELS:
        for dataset in a.datasets:
            if dataset=='ksdd2_mvtlike':
                continue
            if not (out/'adapters'/model/dataset/'TRAINING_DONE').is_file():
                run('reproduce.py','train','--model',model,'--dataset',dataset,'--run',a.run)
        for dataset in a.datasets:
            for split in ['val','test']:
                arguments=['--model',model,'--dataset',dataset,'--split',split,'--run',a.run]
                run('reproduce.py','infer',*arguments)
                run('reproduce.py','infer',*arguments,'--base')
    arguments=['routes','--run',a.run,'--datasets',*a.datasets]
    if a.protocol=='retrain-local':
        arguments+=['--eff-scores',str(out/'merged_val_test_scores.csv'),'--saec-run',str(out)]
    run('reproduce.py',*arguments)

if __name__=='__main__':
    main()
