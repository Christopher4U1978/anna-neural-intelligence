"""Official UCI train/test files; validation comes only from official training."""
import hashlib, io, urllib.request, zipfile
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

URL='https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip'
ARCHIVE_SHA='0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0'
HASHES={'optdigits.tra':'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd','optdigits.tes':'6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8'}

def ensure_data(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not all((root/name).exists() for name in HASHES):
        with urllib.request.urlopen(URL, timeout=60) as response: raw=response.read()
        if hashlib.sha256(raw).hexdigest()!=ARCHIVE_SHA: raise ValueError('Dataset archive hash mismatch; review provenance before updating.')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in HASHES: (root/name).write_bytes(z.read(name))
    for name,expected in HASHES.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected: raise ValueError('Dataset file hash mismatch: '+name)
    return root

def split_training(labels, seed, val_fraction):
    if not 0<val_fraction<1: raise ValueError('validation fraction must be between zero and one')
    return train_test_split(np.arange(len(labels)),test_size=val_fraction,random_state=seed,stratify=labels)

def training_stats(x):
    mean=float(x.mean()); std=float(x.std())
    if std<=0: raise ValueError('constant training data')
    return mean,std

def prepare_data(root, config):
    root=ensure_data(root)
    tr=np.loadtxt(root/'optdigits.tra',delimiter=',',dtype=np.int64)
    te=np.loadtxt(root/'optdigits.tes',delimiter=',',dtype=np.int64)
    if tr.shape!=(3823,65) or te.shape!=(1797,65): raise ValueError('unexpected dataset schema')
    for a in (tr,te):
        if not ((a[:,:64]>=0).all() and (a[:,:64]<=16).all() and (a[:,64]>=0).all() and (a[:,64]<10).all()): raise ValueError('invalid pixel or label')
    train_ids,val_ids=split_training(tr[:,64],config['split_seed'],config['val_fraction'])
    x=tr[:,:64].astype('float32').reshape(-1,1,8,8)/16
    mean,std=training_stats(x[train_ids])
    def dataset(images,labels):return TensorDataset(torch.from_numpy((images-mean)/std),torch.tensor(labels,dtype=torch.long))
    datasets={'train':dataset(x[train_ids],tr[train_ids,64]),'val':dataset(x[val_ids],tr[val_ids,64]),'test':dataset(te[:,:64].astype('float32').reshape(-1,1,8,8)/16,te[:,64])}
    # Detect exact image overlap. No test labels or statistics are used by training.
    train_keys={a.tobytes() for a in tr[:,:64]}
    overlap=sum(a.tobytes() in train_keys for a in te[:,:64])
    metadata={'source_url':URL,'archive_sha256':ARCHIVE_SHA,'file_sha256':HASHES,'license':'CC BY 4.0','train_indices':train_ids.tolist(),'validation_indices':val_ids.tolist(),'counts':{k:len(v) for k,v in datasets.items()},'normalization':{'pixel_divisor':16,'mean':mean,'std':std,'fit_on':'training subset only'},'exact_image_overlap_official_train_test':overlap,'split_seed':config['split_seed']}
    return datasets,metadata

def loader(dataset,batch_size,shuffle=False,seed=369):
    if batch_size<=0:raise ValueError('batch size must be positive')
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(seed))
