import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from .data import loader
from .optimization import inference_context

@torch.inference_mode()
def evaluate(model,dataset,batch_size=128):
    model.eval();total_loss=0.;true=[];pred=[]
    for x,y in loader(dataset,batch_size):
        with inference_context(model):logits=model(x)
        total_loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
        true.extend(y.tolist());pred.extend(logits.argmax(1).tolist())
    if not true:raise ValueError('empty evaluation dataset')
    precision,recall,f1,support=precision_recall_fscore_support(true,pred,labels=list(range(10)),zero_division=0)
    correct=int((np.asarray(true)==pred).sum());n=len(true)
    # Wilson binomial interval, not a generalization or writer-independence guarantee.
    z=1.96;p=correct/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/den
    return {'samples':n,'correct':correct,'accuracy':p,'accuracy_wilson_95':[centre-half,centre+half],'loss':total_loss/n,'confusion_matrix':confusion_matrix(true,pred,labels=list(range(10))).tolist(),'per_class':[{'class':i,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(10)]}
