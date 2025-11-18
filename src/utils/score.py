import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

def compute_weighted_score(weights, dataset, model, device):
    '''
    Compute the weighted score considering 
    '''
