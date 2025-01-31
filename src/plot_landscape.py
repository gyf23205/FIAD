import os
import copy
import torch 
import setting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FuncFormatter

from DeepSAD import DeepSAD
from torch.nn import MSELoss
from pyhessian import hessian
from datasets.main import load_dataset
from networks.main import build_network_physical, build_network

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def loss_physical(semi_targets,  outputs, signal_next, signal_pred):
    dist = torch.sum((outputs - c) ** 2, dim=1)
    losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps) ** semi_targets.float()))
    loss = torch.mean(losses)
    MSE_loss = MSELoss()
    loss_pred = MSE_loss(signal_pred, signal_next)
    loss += weight_pred * loss_pred
    return loss

def loss_vanilla(outputs, semi_targets):
    dist = torch.sum((outputs - c) ** 2, dim=1)
    losses = torch.where(semi_targets == 0, dist, eta * ((dist + eps) ** semi_targets.float()))
    loss = torch.mean(losses)
    return loss


if __name__=='__main__':

    # Set constants
    physical = True
    seed = 4
    setting.init([512, 512, 1024])
    dataset_name = 'spoofing_physical'
    net_name = 'spoof_mlp'
    data_path = './data'
    ratio_known_outlier = 0.005
    ratio_pollution = 0.2
    normal_class = 0
    known_outlier_class = 1
    n_known_outlier_classes = 1
    ratio_known_normal=0.0
    eta = 6.9264986318494515
    # eta=5
    weight_pred = 9.111514123138956
    # weight_pred = 1
    eps = 1e-6
    if physical:
        model_path = '/home/yifan/git/FIAD/model/physical/model_with_pred_0005_02/model_physical.tar'
    else:
        model_path = '/home/yifan/git/FIAD/model/vanilla/model_0005_02/model.tar'

    # Load model
    deepSAD = DeepSAD(eta)
    if physical:
        deepSAD.net = build_network_physical(net_name)
    else:
        deepSAD.net = build_network(net_name)
    deepSAD.load_model(model_path=model_path, load_ae=False, map_location='cuda')
    model = deepSAD.net
    model.eval()
    c = torch.tensor(deepSAD.c, dtype=float).cuda()

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))
    
    loader, _, _ = dataset.loaders(batch_size=256, num_workers=0)
    for data in loader:
        inputs, _, semi_targets, _, signal_next = data
        inputs, semi_targets, signal_next = inputs.cuda(), semi_targets.cuda(), signal_next.cuda()
        break

    # Set Hessian info
    if physical:
        hessian_comp = hessian(model, loss_physical, data=(inputs, semi_targets, signal_next), cuda=True)
    else:
        hessian_comp = hessian(model, loss_vanilla, data=(inputs, semi_targets), cuda=True)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    print(top_eigenvalues)

    # # Perturb the model
    lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    n_perb = len(lams)
    land = np.empty((n_perb, n_perb))
    model_perb = copy.deepcopy(model)

    for i in range(n_perb):
         for j in range(n_perb):
            model_perb = get_params(model, model_perb, top_eigenvector[0], lams[i])
            model_perb = get_params(model_perb, model_perb, top_eigenvector[1], lams[j])
            if physical:
                outputs, signal_pred = model_perb(inputs)
                land[i, j] = loss_physical(semi_targets, outputs, signal_next, signal_pred).item()
            else:
                outputs = model_perb(inputs)
                land[i, j] = loss_vanilla(outputs, semi_targets).item()

    if physical:
        np.save('land_physical.npy', land)
    else:
        np.save('land_vanilla.npy', land)


# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
# X = Y = lams
# X, Y = np.meshgrid(X, Y)
# # Z = np.load('land_vanilla.npy')
# Z = np.load('land_physical.npy')
# Z = Z/1e5
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# # Customize the z axis.
# # ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')
# ax.set_xlabel('v2', labelpad=13, size=20)
# ax.set_ylabel('v1', labelpad=13, size=20)
# ax.set_zlabel('Loss value ($\\times 10^5$)', labelpad=13, size=20)

# ax.tick_params(axis='both', which='major', labelsize=18)
# ax.tick_params(axis='both', which='minor', labelsize=18)

# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()

# loss_list = []

# # create a copy of the model

# for lam in lams:
#     model_perb = get_params(model, model_perb, top_eigenvector[0], lam)
#     loss_list.append(criterion(model_perb(inputs), targets).item())

# np.save('lams_temp.npy', np.array(lams))
# np.save('loss_list_temp.npy', np.array(loss_list))
