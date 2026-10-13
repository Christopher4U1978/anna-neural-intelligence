"""Fresh process import + model load + one inference; distinct from warm timing."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--int8',required=True);p.add_argument('--out',required=True);a=p.parse_args()
worker='''import time,json,sys
start=time.perf_counter()
import torch
from anna_neural.training import load_checkpoint
from anna_neural.optimization import load_quantized
after_import=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1)
m,_=load_checkpoint(sys.argv[1])
if sys.argv[3]=='int8':m=load_quantized(sys.argv[2],m)
after_load=time.perf_counter()
with torch.inference_mode():y=m(torch.zeros(1,1,8,8))
end=time.perf_counter()
print(json.dumps({'import_ms':(after_import-start)*1000,'load_and_configure_ms':(after_load-after_import)*1000,'first_forward_ms':(end-after_load)*1000,'worker_total_ms':(end-start)*1000,'output_shape':list(y.shape)}))
'''
rows=[]
for repeat in range(3):
 for variant in ['fp32','int8']:
  t=time.perf_counter();r=subprocess.run([sys.executable,'-c',worker,a.checkpoint,a.int8,variant],capture_output=True,text=True,check=True)
  v=json.loads(r.stdout);v.update({'variant':variant,'repeat':repeat,'parent_launch_to_exit_ms':(time.perf_counter()-t)*1000});rows.append(v)
Path(a.out).write_text(json.dumps({'conditions':'one thread, batch 1, zero input, separate fresh processes, warm OS page cache possible','rows':rows},indent=2)+'\n')
