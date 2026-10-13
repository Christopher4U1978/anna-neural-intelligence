import hashlib, json, os, platform, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import sklearn,matplotlib,scipy

def seed_all(seed, threads=2):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def environment():
    def read(path):
        try: return Path(path).read_text().strip()
        except OSError: return None
    return {"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__,
        "platform":platform.platform(),"cpu":next((x.split(":",1)[1].strip() for x in (read('/proc/cpuinfo') or '').splitlines() if x.startswith('model name')),platform.processor()),
        "logical_cpus":os.cpu_count(),"cpu_quota":read('/sys/fs/cgroup/cpu.max'),
        "memory_limit_bytes":read('/sys/fs/cgroup/memory.max'),
        "cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,
        "quantized_engines":torch.backends.quantized.supported_engines,
        "torch_threads":torch.get_num_threads(),"torch_build":torch.__config__.show(),
        "utc_time":__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}

def source_manifest(root):
    root=Path(root)
    paths=sorted([*root.glob('src/**/*.py'),*root.glob('tests/**/*.py'),*root.glob('configs/*.json'),root/'pyproject.toml'])
    return {str(x.relative_to(root)):sha256(x) for x in paths}
