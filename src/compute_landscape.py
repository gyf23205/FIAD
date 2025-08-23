import torch 
import setting
import numpy as np

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
    method = 2
    seed = 4

    data_path = './data'
    ratio_known_outlier = 0.003
    ratio_pollution = 0.12
    normal_class = 0
    known_outlier_class = 1
    n_known_outlier_classes = 1
    ratio_known_normal=0.0
    if method == 0:
        eta=5
    elif method == 1:
        eta = 6.9264986318494515
        weight_pred = 9.111514123138956
    elif method == 2:
        eta = 9.6450178070626
        weight_pred = 7.343276666841756

    eps = 1e-6
    deepSAD = DeepSAD(eta)

    if method == 0:
        setting.init([512, 512, 1024])
        model_path = '/home/yifan/git/FIAD/model/vanilla/model_0003_012/model.tar'
        net_name = 'spoof_mlp'
        dataset_name = 'spoofing_physical'
        deepSAD.net = build_network(net_name)
    elif method == 1:
        setting.init([512, 512, 1024])
        model_path = '/home/yifan/git/FIAD/model/physical/model_0003_012/model_physical_with_pred.tar'
        net_name = 'spoof_mlp'
        dataset_name = 'spoofing_physical'
        deepSAD.net = build_network_physical(net_name)
    elif method==2:
        setting.init([256, 256, 512])
        model_path = '/home/yifan/git/FIAD/model/physical_res/model_0003_012/model_physical_res_with_pred.tar'
        net_name = 'spoof_mlp_res'
        dataset_name = 'spoofing_state_only'
        deepSAD.net = build_network_physical(net_name)
    else:
        pass

        
    deepSAD.load_model(model_path=model_path, load_ae=False, map_location='cuda')
    model = deepSAD.net
    model.eval()
    c = torch.tensor(deepSAD.c, dtype=float).cuda()

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))
    
    loader, _, _ = dataset.loaders(batch_size=1024, num_workers=0)
    avg = []
    n_iter = 3
    for i in range(n_iter):
        data = next(iter(loader))
        inputs, _, semi_targets, _, signal_next = data
        inputs, semi_targets, signal_next = inputs.cuda(), semi_targets.cuda(), signal_next.cuda()

        # Set Hessian info
        if method != 0:
            hessian_comp = hessian(model, loss_physical, data=(inputs, semi_targets, signal_next), cuda=True)
        else:
            hessian_comp = hessian(model, loss_vanilla, data=(inputs, semi_targets), cuda=True)
        
        
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=100)
        avg.append(top_eigenvalues[0]/top_eigenvalues[-1])

    print('Average ratio of top eigenvalue to bottom eigenvalue:', np.mean(avg))

    # # # Perturb the model
    # lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    # n_perb = len(lams)
    # land = np.empty((n_perb, n_perb))
    # model_perb = copy.deepcopy(model)

    # for i in range(n_perb):
    #      for j in range(n_perb):
    #         model_perb = get_params(model, model_perb, top_eigenvector[0], lams[i])
    #         model_perb = get_params(model_perb, model_perb, top_eigenvector[1], lams[j])
    #         if method != 0:
    #             outputs, signal_pred = model_perb(inputs)
    #             land[i, j] = loss_physical(semi_targets, outputs, signal_next, signal_pred).item()
    #         else:
    #             outputs = model_perb(inputs)
    #             land[i, j] = loss_vanilla(outputs, semi_targets).item()

        
    # if method == 0:
    #     np.save('land_vanilla.npy', land)
    # elif method == 1:
    #     np.save('land_physical.npy', land)
    # elif method==2:
    #     np.save('land_physical_res.npy', land)
    # else:
    #     pass

