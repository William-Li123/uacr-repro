import importlib.metadata as m
import json
import platform
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
names=['torch','torchvision','transformers','peft','accelerate','qwen-vl-utils','numpy','pandas','Pillow','scikit-learn','opencv-python','PyYAML','tqdm','safetensors','huggingface-hub','tokenizers','ultralytics','bitsandbytes']
versions={}
for name in names:
    try:
        versions[name]=m.version(name)
    except m.PackageNotFoundError:
        versions[name]=None
result={'python':sys.version,'platform':platform.platform(),'packages':versions}
(root/'provenance/environment.json').write_text(json.dumps(result,indent=2))
(root/'requirements.lock.txt').write_text(''.join(f'{name}=={version}\n' for name,version in versions.items() if version))
print(json.dumps(result,indent=2))
