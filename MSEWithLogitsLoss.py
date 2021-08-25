import torch
from torch import nn as nn

class MSEWithLogitsLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    

    def forward(self,output, target):
        output = nn.functional.sigmoid(output)
        loss = nn.functional.mse_loss(output,target)
        return loss