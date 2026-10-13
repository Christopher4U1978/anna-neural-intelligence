import json,time
from pathlib import Path
import torch
from .common import seed_all,write_json,environment,sha256,source_manifest
from .data import prepare_data,loader
from .model import ForgeCNN,parameter_count
from .evaluation import evaluate

def save_checkpoint(path,model,optimizer,config,metadata,epoch):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'format_version':1,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'config':config,'data':metadata,'epoch':epoch},path)

def load_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['format_version']!=1: raise ValueError('unsupported checkpoint')
    model=ForgeCNN();model.load_state_dict(checkpoint['model_state']);model.eval()
    return model,checkpoint

def train(config,data_root,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('output directory must be new or empty; preserve earlier runs')
    out.mkdir(parents=True,exist_ok=True)
    if config['epochs']<1:raise ValueError('epochs must be positive')
    seed_all(config['seed'],config['threads']);datasets,metadata=prepare_data(data_root,config)
    model=ForgeCNN();optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    train_loader=loader(datasets['train'],config['batch_size'],True,config['seed'])
    curves=[];best=float('inf');start=time.perf_counter()
    write_json(out/'config.json',config);write_json(out/'data_manifest.json',metadata)
    env=environment();write_json(out/'environment.json',env)
    manifest=source_manifest(Path(__file__).resolve().parents[2]);write_json(out/'source_manifest.json',manifest)
    initial=evaluate(model,datasets['val'])
    for epoch in range(1,config['epochs']+1):
        model.train();loss_sum=0.;correct=0;n=0;epoch_start=time.perf_counter()
        for x,y in train_loader:
            optimizer.zero_grad(set_to_none=True);logits=model(x)
            loss=torch.nn.functional.cross_entropy(logits,y);loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();n+=len(y)
        val=evaluate(model,datasets['val']);row={'epoch':epoch,'train_loss':loss_sum/n,'train_accuracy':correct/n,'val_loss':val['loss'],'val_accuracy':val['accuracy'],'seconds':time.perf_counter()-epoch_start};curves.append(row)
        write_json(out/'learning_curves.json',curves)
        if val['loss']<best:
            best=val['loss'];save_checkpoint(out/'best.pt',model,optimizer,config,metadata,epoch)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val_acc={val['accuracy']:.4f}",flush=True)
    training_wall_seconds=time.perf_counter()-start
    if source_manifest(Path(__file__).resolve().parents[2])!=manifest:raise RuntimeError('source changed during training; retain run as unverified')
    model,checkpoint=load_checkpoint(out/'best.pt')
    # Test is evaluated once, after training and validation-only checkpoint selection.
    results={'status':'locally_verified_training','config':config,'environment':env,'initial_validation':initial,'validation':evaluate(model,datasets['val']),'test':evaluate(model,datasets['test']),'selected_epoch':checkpoint['epoch'],'training_seconds':training_wall_seconds,'parameter_count':parameter_count(model),'checkpoint_bytes':(out/'best.pt').stat().st_size,'checkpoint_sha256':sha256(out/'best.pt'),'source_manifest':manifest,'data':metadata}
    weights=out/'fp32-weights.pt';torch.save(model.state_dict(),weights);results['weights_bytes']=weights.stat().st_size;results['weights_sha256']=sha256(weights)
    write_json(out/'training.json',results);return results
