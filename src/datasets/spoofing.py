from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.spoofing_dataset import MySpoofing
from .preprocessing import create_semisupervised_setting
import torch

from pathlib import Path



class SpoofingDataset(BaseADDataset):
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
        
        # Get training set
        train_set = MySpoofing(root, dataset_name, True, random_state)

        # Creat semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                        self.outlier_classes, self.known_outlier_classes,
                                                        ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)

        # Subset train_+set to semi_supervised setup
        self.train_set = Subset(train_set, idx)
        
        #Get test set
        self.test_set = MySpoofing(root, dataset_name, False, random_state)
        


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader

