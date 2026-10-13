"""Preallocated CPU forward-only wall-clock timing; no hidden data loading."""
import copy,json,resource,time,statistics
from pathlib import Path
import numpy as np
import torch
from .common import environment,write_json,sha256
from .training import load_checkpoint
from .optimization import load_quantized

def summarize(times_ms,batch):
    if batch<1 or not times_ms or min(times_ms)<=0:raise ValueError('positive timings and batch required')
    a=np.asarray(times_ms,dtype=float)
    if not np.isfinite(a).all():raise ValueError('finite timings required')
    return {'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'std_ms':float(a.std(ddof=1)) if len(a)>1 else 0.,'min_ms':float(a.min()),'max_ms':float(a.max()),'throughput_images_s':float(batch*1000/a.mean())}

@torch.inference_mode()
def measure(model,x,warmup,iterations,repeats):
    if warmup<0 or iterations<1 or repeats<2:raise ValueError('invalid measurement protocol')
    t=time.perf_counter_ns();model(x);cold=(time.perf_counter_ns()-t)/1e6
    for _ in range(warmup):model(x)
    runs=[]
    for _ in range(repeats):
        times=[]
        for _ in range(iterations):
            t=time.perf_counter_ns();model(x);times.append((time.perf_counter_ns()-t)/1e6)
        runs.append({'summary':summarize(times,len(x)),'raw_ms':times})
    medians=[r['summary']['median_ms'] for r in runs]
    return {'first_forward_ms':cold,'first_forward_scope':'first call of this model/input pair in an already imported worker process; not boot or import cold start','warmup':warmup,'iterations_per_repeat':iterations,'repeats':repeats,'aggregate':summarize([t for r in runs for t in r['raw_ms']],len(x)),'repeat_median_min_ms':min(medians),'repeat_median_max_ms':max(medians),'repeat_median_cv':statistics.stdev(medians)/statistics.mean(medians),'runs':runs}

def benchmark_worker(checkpoint,quantized_path,variant,batch,threads,out,warmup=30,iterations=100,repeats=5):
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.manual_seed(369);start=time.perf_counter();model,_=load_checkpoint(checkpoint)
    if variant=='int8':model=load_quantized(quantized_path,model)
    elif variant=='fp32_channels_last':model=model.to(memory_format=torch.channels_last)
    elif variant!='fp32':raise ValueError('unknown benchmark variant')
    load_seconds=time.perf_counter()-start
    # Identical seeded values for each variant. Normalized synthetic input is for timing only.
    x=torch.randn(batch,1,8,8)
    if variant=='fp32_channels_last':x=x.contiguous(memory_format=torch.channels_last)
    m=measure(model,x,warmup,iterations,repeats)
    # Linux ru_maxrss is KB; includes interpreter, library, model and previous allocations.
    m.update({'variant':variant,'batch_size':batch,'threads':threads,'model_load_seconds':load_seconds,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'memory_scope':'fresh process peak RSS including Python and PyTorch; not isolated activation memory','input_bytes':x.numel()*x.element_size(),'input_shape':list(x.shape),'input_kind':'seeded synthetic values, same across variants; accuracy measured separately on real held-out data','environment':environment(),'checkpoint_sha256':sha256(checkpoint),'quantized_sha256':sha256(quantized_path) if variant=='int8' else None})
    write_json(out,m);return m

def profile_model(model,x,out):
    model.eval()
    with torch.inference_mode():
        for _ in range(10):model(x)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as prof:
            for _ in range(20):model(x)
    Path(out).write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=30))
    prof.export_chrome_trace(str(Path(out).with_suffix('.trace.json')))
