import torch.nn as nn
import torch

class NegativeSamplingLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, positive_score, negative_scores):
        
        positive_loss= -torch.log(torch.sigmoid(positive_score)+1e-9)
        
        negative_loss= -torch.sum(torch.log(torch.sigmoid(-negative_scores)+1e-9), dim=1)
        
        return (positive_loss + negative_loss).mean()