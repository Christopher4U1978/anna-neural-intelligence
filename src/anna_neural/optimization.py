import copy
from pathlib import Path
import torch
from torch.ao.quantization import fuse_modules,prepare,convert,get_default_qconfig
from .data import loader

def prepared_model(model,engine='x86'):
    if engine not in torch.backends.quantized.supported_engines:raise ValueError('unsupported INT8 backend '+engine)
    torch.backends.quantized.engine=engine
    m=copy.deepcopy(model).cpu().eval()
    fuse_modules(m,[['conv1','relu1'],['conv2','relu2'],['fc1','relu3']],inplace=True)
    m.qconfig=get_default_qconfig(engine);return prepare(m,inplace=False)

@torch.inference_mode()
def quantize(model,training_dataset,calibration_samples=512,engine='x86'):
    if calibration_samples<1:raise ValueError('positive calibration sample count required')
    m=prepared_model(model,engine)
    ids=list(range(min(calibration_samples,len(training_dataset))))
    for x,_ in loader(torch.utils.data.Subset(training_dataset,ids),64):m(x)
    return convert(m,inplace=False).eval()

def save_quantized(path,model,engine='x86'):
    torch.save({'format_version':1,'engine':engine,'state':model.state_dict()},path)

def load_quantized(path,fp32):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format_version']!=1:raise ValueError('unsupported quantized checkpoint')
    m=prepared_model(fp32,payload['engine'])
    # A dummy call initializes observers; checkpoint overwrites quantization parameters.
    with torch.inference_mode():m(torch.zeros(1,1,8,8))
    m=convert(m,inplace=False);m.load_state_dict(payload['state']);return m.eval()

# PyTorch 2.5.1 quantized_resize_cpu_ is not registered as deterministic.
# Disable strict enforcement only inside an explicitly scoped inference call.
from contextlib import contextmanager
@contextmanager
def inference_context(model):
    is_int8=any('quantized' in type(m).__module__ for m in model.modules())
    strict=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    if is_int8:torch.use_deterministic_algorithms(False)
    try:
        with torch.inference_mode():yield
    finally:
        torch.use_deterministic_algorithms(strict,warn_only=warn)
