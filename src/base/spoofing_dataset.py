import torch
import numpy as np
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

class MySpoofing(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.classes = [0,1]
        # self.dataset_name = dataset_name
        
        # self.flags = flags
        # self.signals = signals

        
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index
    
    def __len__(self):
        return len(self.targets)

