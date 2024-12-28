import torch
from torch.utils.data import Dataset

class SkipgramDataset(Dataset):
    def __init__(self, pairs):
        self.pairs= pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context= self.pairs[idx]
        
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)