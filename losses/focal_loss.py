import torch.nn as nn 
import torch.nn.functional as F 
import torch 

#PyTorch. Adapted from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss


ALPHA = 0.99
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        # BCE_EXP = torch.exp(-BCE)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_t * (1-p_t)**gamma * BCE
                       
        # this averages only over the positive examples               
        return focal_loss.mean()