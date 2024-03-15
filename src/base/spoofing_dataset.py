import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

class MySpoofing(Dataset):
    def __init__(self, root, dataset_name, train, random_state) -> None:
        super().__init__()
        self.classes = [0,1]
        self.train = train
        self.dataset_name = dataset_name
        test_ratio = 0.2

        path = os.path.join(root,'data/balanced_attack')
        signals = np.load(os.path.join(path,'innovations_batched.npy'))
        flags = np.load(os.path.join(path,'flags_batched.npy'))
        self.flags = flags
        self.signals = signals

        idx_norm = flags==0
        idx_out = flags==1

        X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(signals[idx_norm], flags[idx_norm],
                                                                        test_size=test_ratio, random_state=random_state)
                                                                                
        X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(signals[idx_out], flags[idx_out],
                                                                            test_size=0.4, random_state=random_state)

        X_train = np.concatenate([X_train_norm, X_train_out])
        X_test = np.concatenate([X_test_norm, X_test_out])
        y_train = np.concatenate([y_train_norm, y_train_out])
        y_test = np.concatenate((y_test_norm, y_test_out))

        # # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        # scaler = StandardScaler().fit(X_train)
        # X_train_stand = scaler.transform(X_train)
        # X_test_stand = scaler.transform(X_test)

        # # Scale to range [0,1]
        # minmax_scaler = MinMaxScaler().fit(X_train_stand)
        # X_train_scaled = minmax_scaler.transform(X_train_stand)
        # X_test_scaled = minmax_scaler.transform(X_test_stand)
        
        if self.train:
            self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

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

