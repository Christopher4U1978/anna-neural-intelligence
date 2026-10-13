import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub

class ForgeCNN(nn.Module):
    """1x8x8 ->16x8x8 ->32x4x4 ->512 ->64 ->10 logits."""
    def __init__(self):
        super().__init__()
        self.quant=QuantStub();self.dequant=DeQuantStub()
        self.conv1=nn.Conv2d(1,16,3,padding=1);self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(16,32,3,padding=1);self.relu2=nn.ReLU()
        self.fc1=nn.Linear(32*4*4,64);self.relu3=nn.ReLU()
        self.fc2=nn.Linear(64,10)
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu');nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.quant(x);x=self.pool(self.relu1(self.conv1(x)))
        x=self.relu2(self.conv2(x));x=torch.flatten(x,1)
        x=self.relu3(self.fc1(x));return self.dequant(self.fc2(x))

def parameter_count(model): return sum(p.numel() for p in model.parameters())
