"""Sequential fresh workers avoid concurrent training/timing interference."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--int8',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
if out.exists() and any(out.iterdir()):raise SystemExit('Use a new output directory to preserve raw measurements')
out.mkdir(parents=True,exist_ok=True)
rows=[];start=time.perf_counter()
# Interleave variants at every condition rather than run all baseline then all INT8.
for threads in [1,2,4]:
 for batch in [1,8,32,128]:
  for variant in ['fp32','int8','fp32_channels_last']:
   path=out/f'{variant}-b{batch}-t{threads}.json'
   subprocess.run([sys.executable,'-m','anna_neural.cli','benchmark-worker','--checkpoint',a.checkpoint,'--int8',a.int8,'--variant',variant,'--batch',str(batch),'--threads',str(threads),'--out',str(path)],check=True)
   r=json.loads(path.read_text());rows.append({k:r[k] for k in ['variant','batch_size','threads','aggregate','repeat_median_min_ms','repeat_median_max_ms','repeat_median_cv','first_forward_ms','model_load_seconds','peak_process_rss_bytes']});print(variant,batch,threads,r['aggregate']['median_ms'],flush=True)
(out/'summary.json').write_text(json.dumps({'status':'locally_measured_CPU','matrix_seconds':time.perf_counter()-start,'rows':rows},indent=2)+'\n')
