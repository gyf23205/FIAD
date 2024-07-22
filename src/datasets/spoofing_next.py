from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.spoofing_dataset_next import MySpoofingNext
from base.spoofing_dataset import MySpoofing
from .preprocessing import create_semisupervised_setting
import torch
import os
import numpy as np
from sklearn.model_selection import train_test_split

from pathlib import Path



class SpoofingDatasetFlat(BaseADDataset):
    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)

        # Define normal and outlier classes

        self.n_classes = 2 # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)
        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)

        # Load data
        test_ratio = 0.2
        path = os.path.join(root,'spoofing')
        signals = np.load(os.path.join(path,'data_batched.npy'))
        signals_next = np.load(os.path.joint(path, 'next_batched.npy'))
        flags = np.load(os.path.join(path,'labels_batched.npy'))
        idx_norm = flags==0
        idx_out = flags==1

         # The model only learn the dynamics of the system during training
        X_train_norm, X_test_norm, y_train_norm, y_test_norm, next_train_norm, _ = train_test_split(signals[idx_norm], flags[idx_norm], signals_next[idx_norm],
                                                                        test_size=test_ratio, random_state=random_state)
                                                                                
        X_train_out, X_test_out, y_train_out, y_test_out, next_train_out, _ = train_test_split(signals[idx_out], flags[idx_out],
                                                                            test_size=0.4, random_state=random_state)

        X_train = np.concatenate([X_train_norm, X_train_out])
        X_test = np.concatenate([X_test_norm, X_test_out])
        y_train = np.concatenate([y_train_norm, y_train_out])
        y_test = np.concatenate((y_test_norm, y_test_out))
        next_train = np.concatenate([next_train_norm, next_train_out])
        # Construct validation set
        val_ratio = 0.5
        idx_val = np.random.choice(len(y_test), size=int(val_ratio*len(y_test)), replace=False)
        mask = np.ones(len(y_test),dtype=bool)
        mask[[idx_val]] = False
        X_val = X_test[~mask]
        y_val = y_test[~mask]
        

        X_test = X_test[mask]
        y_test = y_test[mask]
        
        # Get training set
        train_set = MySpoofingNext(X_train, y_train, next_train)

        # Creat semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                        self.outlier_classes, self.known_outlier_classes,
                                                        ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)

        # Subset train_+set to semi_supervised setup
        self.train_set = Subset(train_set, idx)
        self.val_set = MySpoofing(X_val, y_val)
        
        #Get test set
        self.test_set = MySpoofing(X_test, y_test)
        


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size,shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, val_loader, test_loader

