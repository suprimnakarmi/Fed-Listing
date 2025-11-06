import torch
from torch import nn

class Dense(nn.Module):
    def __init__(self,features_in,features_out,params):
        super().__init__()
        self.fc = nn.Linear(features_in,features_out)
        self.fc.weight = params[0]
        self.fc.bias = params[1]
    
    def forward(self,x):
        return self.fc(x)