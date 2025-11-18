import numpy as np
import torch
import torch.nn as nn
from networks.mlp import MLP_Physical, MLP

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

class DummyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DummyMLP, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = nn.ReLU()(z)
        return z, self.predictor(z)


def update_center(data, net, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = torch.zeros(n_outlier_classes+1)
    net.eval()
    centroids = {'c_normal': torch.zeros((64,)),
                 'c_outlier_1': torch.zeros((64,)),
                 'c_outlier_2': torch.zeros((64,))}
    with torch.no_grad():
        # get the inputs of the batch
        inputs, semi_target, data_next = data
        for i in range(n_outlier_classes + 1):
            inputs_temp = inputs[semi_target == (i)]
            outputs_temp = net(inputs_temp)
            n_samples[i] += outputs_temp.shape[0]
            if i == 0:
                centroids['c_normal'] += torch.sum(outputs_temp, dim=0)
            else:
                centroids['c_outlier_{}'.format(i)] += torch.sum(outputs_temp, dim=0)

    for i in range(n_outlier_classes + 1):
        if i == 0:
            centroids['c_normal'] /= n_samples[i]
            centroids['c_normal'][(abs(centroids['c_normal']) < eps) & (centroids['c_normal'] < 0)] = -eps
            centroids['c_normal'][(abs(centroids['c_normal']) < eps) & (centroids['c_normal'] > 0)] = eps
        else:
            centroids['c_outlier_{}'.format(i)] /= n_samples[i]
            centroids['c_outlier_{}'.format(i)][(abs(centroids['c_outlier_{}'.format(i)]) < eps) & (centroids['c_outlier_{}'.format(i)] < 0)] = -eps
            centroids['c_outlier_{}'.format(i)][(abs(centroids['c_outlier_{}'.format(i)]) < eps) & (centroids['c_outlier_{}'.format(i)] > 0)] = eps

    return centroids

if __name__ == "__main__":
    model = MLP(x_dim=1200, h_dims=[128, 64], rep_dim=64, bias=False)
    data_all = np.load('data/spoofing/data_multi_sin_batched.npy')
    labels_all = np.load('data/spoofing/labels_multi_sin_batched.npy')
    next_all = np.load('data/spoofing/next_multi_sin_batched.npy')

    idx_norm = labels_all==0
    idx_out1 = labels_all==1
    idx_out2 = labels_all==2

    data_normal = torch.tensor(data_all[idx_norm][:100], dtype=torch.float32)
    data_out1 = torch.tensor(data_all[idx_out1][:100], dtype=torch.float32)
    data_out2 = torch.tensor(data_all[idx_out2][:100], dtype=torch.float32)
    data = torch.concat([data_normal, data_out1, data_out2], dim=0)
    labels_normal = torch.tensor(labels_all[idx_norm][:100], dtype=torch.int64)
    labels_out1 = torch.tensor(labels_all[idx_out1][:100], dtype=torch.int64)
    labels_out2 = torch.tensor(labels_all[idx_out2][:100], dtype=torch.int64)
    labels = torch.concat([labels_normal, labels_out1, labels_out2], dim=0)
    next_normal = torch.tensor(next_all[idx_norm][:100], dtype=torch.float32)
    next_out1 = torch.tensor(next_all[idx_out1][:100], dtype=torch.float32)
    next_out2 = torch.tensor(next_all[idx_out2][:100], dtype=torch.float32)
    next_data = torch.concat([next_normal, next_out1, next_out2], dim=0)

    labels[:20] = -1
    labels[100:120] = -1
    labels[220:240] = -1

    # print(labels)

    coeff = {
        'normal': 1.0,
        'outlier_1': 1.0,
        'outlier_2': 1.0,
        'outlier': 1.0,
        'unlabeled': 1.0,
        'pred': 1.0
    }

    n_outlier_classes = 2
    eps = 0.1
    MSE_loss = nn.MSELoss()
    centroids = update_center((data, labels, next_data), model, eps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = 0.0
    # Labeled normals loss
    outputs = model(data)
    dist_normal = torch.sum((outputs[labels == 0] - centroids['c_normal']) ** 2, dim=1)
    loss_normal = coeff['normal'] * (dist_normal + eps) if len(dist_normal)>0 else 0
    # Labeled outliers loss
    loss_outlier = 0
    for i in range(1, n_outlier_classes+1):
        dist_outlier = torch.sum((outputs[labels == i] - centroids['c_outlier_{}'.format(i)]) ** 2, dim=1)
        loss_outlier += coeff['outlier_{}'.format(i)] * (dist_outlier + eps) if len(dist_outlier)>0 else 0
    loss_outlier = coeff['outlier'] * loss_outlier
    # Compute the distance between all the unlabeled samples and the closest centroid
    dist_unlabeled = torch.zeros((len(outputs[labels == -1]), n_outlier_classes+1))
    for i in range(n_outlier_classes+1):
        if i == 0:
            dist_unlabeled[:, i] = torch.sum((outputs[labels == -1] - centroids['c_normal']) ** 2, dim=1)
        else:
            dist_unlabeled[:, i] = torch.sum((outputs[labels == -1] - centroids['c_outlier_{}'.format(i)]) ** 2, dim=1)
    min_dist_unlabeled, _ = torch.min(dist_unlabeled, dim=1)
    loss_unlabeled = coeff['unlabeled'] * (min_dist_unlabeled + eps) if len(min_dist_unlabeled)>0 else 0
    # # Physics-informed loss
    # loss_pred = MSE_loss(signal_pred, next_data)

    loss += torch.mean(loss_unlabeled) # torch.mean(loss_normal) + torch.mean(loss_outlier) + torch.mean(loss_unlabeled) + loss_pred
    # ############################### End compute losses ################################
    # optimizer.zero_grad()
    loss.backward()
    print(compute_grad_norm(model))