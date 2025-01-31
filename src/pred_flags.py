import torch
import numpy as np
import os
import setting
from DeepSAD import DeepSAD
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# import matplotlib.pyplot as plt

def choose_best_r(fpr, tpr, thresholds):
    ratio = []
    eps = 1e-7  # In case divide by 0
    for i in range(len(tpr)-1):
        ratio.append((tpr[i+1]-tpr[i])/(fpr[i+1]-fpr[i]+eps))
    
    # return thresholds[np.argmax(ratio)]
    return thresholds[1]

if __name__=='__main__':
    method = 1
    data_path = './data/spoofing'
    if method == 0:
        model_path = '/home/yifan/git/FIAD/model/vanilla/model_0005_02/model.tar'
        setting.init([512, 512, 1024])
    elif method == 1:
        model_path = '/home/yifan/git/FIAD/model/physical/model_0005_01/model_physical_normalized.tar'
        setting.init([512, 512, 1024])
    elif method == 2:
        model_path = '/home/yifan/git/FIAD/model/physical_res/model_0005_02/model_physical_res.tar'
        setting.init([256, 256, 512])
    elif method == 3:
        model_path = '/home/yifan/git/FIAD/model/physical/model_0005_02/model_physical_state_only.tar'
        setting.init([512, 512, 1024])
    else:
        model_path = '/home/yifan/git/FIAD/model/physical_init/model_0005_02/model_physics_init.tar'
        setting.init([512, 512, 1024])
    net_name = 'spoof_mlp'

    # Load data
    attack_type = 2
    if attack_type == 0:
        signals = np.load(os.path.join(data_path, 'data_unscaled_multi_noise.npy'))
        labels = np.load(os.path.join(data_path, 'labels_unscaled_multi_noise.npy'))
    elif attack_type == 1:
        signals = np.load(os.path.join(data_path, 'data_unscaled_sin_attack.npy'))
        labels = np.load(os.path.join(data_path, 'labels_unscaled_sin_attack.npy'))
    elif attack_type == 2:
        signals = np.load(os.path.join(data_path, 'data_unscaled_log_attack.npy'))
        labels = np.load(os.path.join(data_path, 'labels_unscaled_log_attack.npy'))


    scaler = StandardScaler().fit(signals)
    signals_standard = scaler.transform(signals)

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals = minmax_scaler.transform(signals_standard)
    
    seq_len = 100
    n_dim = 12
    signals_batched = np.zeros((len(signals)-seq_len+1, seq_len*n_dim))
    labels_batched = np.zeros(len(signals)-seq_len+1)
    for i in range(len(labels_batched)):
        signals_batched[i,:] = signals[i:i+seq_len,:].reshape((seq_len*n_dim,))
        labels_batched[i] = 1 if np.sum(labels[i:i+seq_len])>0 else 0

    signals_batched = torch.tensor(signals_batched, dtype=torch.float).to('cuda')
    # Load model
    deepSAD = DeepSAD(eta=1.0)
    deepSAD.set_network(net_name)
    deepSAD.load_model(model_path=model_path, load_ae=False, map_location='cuda')
    net = deepSAD.net.to('cuda')
    c = torch.tensor(deepSAD.c, dtype=float)
    fpr, tpr, thresholds = deepSAD.roc_curve

    r = choose_best_r(fpr, tpr, thresholds)

    net.eval()
    with torch.no_grad():
        outputs = net(signals_batched).cpu()
        dist = torch.sum((outputs-c)**2, dim=1)
        y_pred = np.array(dist>r, dtype=float)
    print('AUC-ROC: ', 100. * roc_auc_score(labels_batched, dist))
    assert len(y_pred) == len(labels_batched)
    if method==0:
        np.save('y_pred.npy', y_pred)
        np.save('labels.npy', labels_batched)
    elif method==1:
        np.save('y_pred_physical.npy', y_pred)
        np.save('labels_physical.npy', labels_batched)
    elif method == 2:
        np.save('y_pred_physical_res.npy', y_pred)
        np.save('labels_physical_res.npy', labels_batched)
    elif method == 3:
        np.save('y_pred_physical_state_only.npy', y_pred)
        np.save('labels_physical_state_only.npy', labels_batched)
    else:
        np.save('y_pred_physical_init.npy', y_pred)
        np.save('labels_physical_init.npy', labels_batched)
    # plt.plot(y_pred)
    # plt.plot(labels_batched)
    # plt.show()
    
