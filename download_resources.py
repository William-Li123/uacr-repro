"""Download public model/auxiliary resources; datasets requiring forms stay manual."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
from project import ROOT, WORK

TEACHER_SHA='6e5e1220841d1604ee2830d55fb792a0363f955b8dff710691ca3f99aa884b5d'
TEACHER_URL='https://media.githubusercontent.com/media/rximg/EfficientAD/61d06b83667d28e306c88a592869f9f7b3e1b4f0/ckptSmall/best_teacher.pth'
RESOURCES={
    'teacher':(TEACHER_URL,'models/teacher/best_teacher.pth',TEACHER_SHA),
    'yolo':('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-cls.pt','models/saec/yolo11s-cls.pt','e2b605d1c8c212b434a75a32759a6f7adf1d2b29c35f76bdccd4c794cb653cf2'),
    'imagenette':('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz','downloads/imagenette2-320.tgz',None),
    'visa':('https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar','downloads/VisA_20220922.tar',None),
}

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def fetch(url,path,expected=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.part')
        request=urllib.request.Request(url,headers={'User-Agent':'uacr-repro-resource-downloader'})
        print(f'Downloading {url}',flush=True)
        with urllib.request.urlopen(request,timeout=120) as src, partial.open('wb') as dst:
            while True:
                block=src.read(8*1024*1024)
                if not block:
                    break
                dst.write(block)
        actual=sha256(partial)
        if expected and actual!=expected:
            raise RuntimeError(f'Checksum mismatch: {path}')
        partial.replace(path)
    actual=sha256(path)
    if expected and actual!=expected:
        raise RuntimeError(f'Checksum mismatch: {path}')
    return actual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource',required=True,choices=['qwen25vl_7b','qwen35_9b',*RESOURCES])
    p.add_argument('--extract',action='store_true',help='Extract VisA/Imagenette with safe tar filtering')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.extract and args.resource not in ['visa','imagenette']:
        p.error('--extract is only valid for visa and imagenette')
    models=json.loads((ROOT/'configs/models.json').read_text())
    if args.resource in models:
        spec=models[args.resource]
        output=WORK/'models/base'/spec['directory']
        if args.dry_run:
            print(json.dumps({**spec,'output':str(output)},indent=2));return
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec['repo_id'],revision=spec['revision'],local_dir=str(output))
        record={**spec,'output':str(output)}
    else:
        url,rel,expected=RESOURCES[args.resource]
        output=WORK/rel
        if args.dry_run:
            print(json.dumps({'url':url,'output':str(output),'expected_sha256':expected},indent=2));return
        digest=fetch(url,output,expected)
        record={'url':url,'output':str(output),'sha256':digest}
        if args.extract:
            if args.resource not in ['visa','imagenette']:
                p.error('--extract is only valid for visa and imagenette')
            dest=WORK/'raw/visa' if args.resource=='visa' else WORK/'raw'
            marker=WORK/'downloads'/f'{args.resource}.extracted.sha256'
            if not marker.exists() or marker.read_text()!=digest:
                dest.mkdir(parents=True,exist_ok=True)
                with tarfile.open(output) as archive:
                    archive.extractall(dest,filter='data')
                marker.write_text(digest)
    records=WORK/'downloads'
    records.mkdir(parents=True,exist_ok=True)
    (records/f'{args.resource}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    main()
