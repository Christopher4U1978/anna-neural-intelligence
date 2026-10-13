import argparse,json
from pathlib import Path
from .common import write_json,seed_all
from .training import train,load_checkpoint
from .data import prepare_data
from .evaluation import evaluate
from .optimization import quantize,save_quantized,load_quantized,inference_context
from .benchmarking import benchmark_worker,profile_model

def main():
    p=argparse.ArgumentParser(description='Standalone CPU CNN engineering demonstration')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--config',default='configs/cpu.json');t.add_argument('--data',default='data');t.add_argument('--out',required=True);t.add_argument('--seed',type=int)
    for name in ['evaluate','quantize','profile','predict']:
        q=sub.add_parser(name);q.add_argument('--checkpoint',required=True);q.add_argument('--data',default='data');q.add_argument('--out',required=True)
        if name=='evaluate':q.add_argument('--int8')
        if name=='predict':q.add_argument('--index',type=int,default=0)
    b=sub.add_parser('benchmark-worker');b.add_argument('--checkpoint',required=True);b.add_argument('--int8',required=True);b.add_argument('--variant',choices=['fp32','fp32_channels_last','int8'],required=True);b.add_argument('--batch',type=int,required=True);b.add_argument('--threads',type=int,required=True);b.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='train':
        c=json.loads(Path(a.config).read_text())
        if a.seed is not None:c['seed']=a.seed
        result=train(c,a.data,a.out);print(json.dumps({k:result[k] for k in ['selected_epoch','parameter_count','training_seconds']}));return
    if a.command=='benchmark-worker':benchmark_worker(a.checkpoint,a.int8,a.variant,a.batch,a.threads,a.out);return
    model,checkpoint=load_checkpoint(a.checkpoint);c=checkpoint['config'];seed_all(c['seed'],c['threads']);datasets,metadata=prepare_data(a.data,c)
    if metadata!=checkpoint['data']:raise ValueError('data provenance differs from training checkpoint')
    if a.command=='evaluate':
        if a.int8:model=load_quantized(a.int8,model)
        result={'validation':evaluate(model,datasets['val']),'test':evaluate(model,datasets['test'])};write_json(a.out,result);print(json.dumps(result['test']));return
    if a.command=='quantize':
        model=quantize(model,datasets['train'],c['calibration_samples']);Path(a.out).parent.mkdir(parents=True,exist_ok=True);save_quantized(a.out,model)
        restored=load_quantized(a.out,load_checkpoint(a.checkpoint)[0])
        import torch
        x=datasets['test'].tensors[0][:8]
        with inference_context(model):torch.testing.assert_close(model(x),restored(x),rtol=0,atol=0)
        write_json(str(a.out)+'.metrics.json',{'status':'locally_verified_INT8','calibration_source':'training only','calibration_samples':min(c['calibration_samples'],len(datasets['train'])),'validation':evaluate(restored,datasets['val']),'test':evaluate(restored,datasets['test']),'serialized_bytes':Path(a.out).stat().st_size,'reload_identical_logits':True});return
    if a.command=='profile':profile_model(model,datasets['test'].tensors[0][:32],a.out);return
    if a.command=='predict':
        import torch
        x,y=datasets['test'][a.index]
        with torch.inference_mode():prob=torch.softmax(model(x.unsqueeze(0)),dim=1)[0]
        result={'status':'local_inference','dataset':'UCI optical digits official held-out test','index':a.index,'truth':int(y),'prediction':int(prob.argmax()),'probabilities':prob.tolist(),'note':'softmax score is not calibrated certainty'};write_json(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
