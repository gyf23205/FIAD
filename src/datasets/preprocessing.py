import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import pickle

def create_semisupervised_setting(labels, normal_classes, unknown_outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param unknown_outlier_classes: tuple with unknown anomaly class labels
    :param known_outlier_classes: tuple with known anomaly class labels
    :param ratio_known_normal: the desired ratio of labeled normal samples
    :param ratio_known_outlier: the desired ratio of labeled anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    (-1: unlabeled, 0: known normal, >1: known anomaly)
    """
    logger = logging.getLogger()

    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_unknown_outlier = np.argwhere(np.isin(labels, unknown_outlier_classes)).flatten()
    idx_known_outlier = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()
    idx_all_outlier = np.concatenate((idx_unknown_outlier, idx_known_outlier), axis=0)

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)
    

    # Get number of samples
    n_labeled_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_labeled_outlier = int(x[3])
    logger.info(f'In semi setting: n_labeled_normal: {n_labeled_normal}, n_unlabeled_normal: {n_unlabeled_normal}, n_unlabeled_outlier: {n_unlabeled_outlier}, n_labeled_outlier: {n_labeled_outlier}')

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_labeled_outlier = np.random.permutation(len(idx_known_outlier))
    # perm_unknown_outlier = np.random.permutation(len(idx_unknown_outlier))
    # perm_known_outlier = np.random.permutation(len(idx_known_outlier))

    idx_labeled_normal = idx_normal[perm_normal[:n_labeled_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_labeled_normal:n_labeled_normal+n_unlabeled_normal]].tolist()
    idx_labeled_outlier = idx_known_outlier[perm_labeled_outlier[:n_labeled_outlier]].tolist()
    # Exclude labeled outliers from all outliers
    idx_unlabeled_outlier = np.setdiff1d(idx_all_outlier, idx_labeled_outlier)
    perm_unlabeled_outlier = np.random.permutation(len(idx_unlabeled_outlier))
    idx_unlabeled_outlier = idx_unlabeled_outlier[perm_unlabeled_outlier[:n_unlabeled_outlier]].tolist()


    # Get original class labels
    labels_labeled_normal = labels[idx_labeled_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_labeled_outlier = labels[idx_labeled_outlier].tolist()
    # Check that all known outlier classes appear at least once in labeled outliers
    known_outlier_classes_in_labeled = np.unique(labels_labeled_outlier)
    missing_classes = np.setdiff1d(known_outlier_classes, known_outlier_classes_in_labeled)
    for mc in missing_classes:
        # insert one sample of the missing class into the labeled set
        idx_sample_mc = np.argwhere(labels==mc).flatten()[0]
        labels_labeled_outlier.append(mc)
        idx_labeled_outlier.append(idx_sample_mc)

    # Get semi-supervised setting labels
    semi_labels_labeled_normal = np.zeros(n_labeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = (-np.ones(n_unlabeled_normal)).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = (-np.ones(n_unlabeled_outlier)).astype(np.int32).tolist()
    semi_labels_labeled_outlier = labels_labeled_outlier


    # Create final lists
    list_idx = idx_labeled_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_labeled_outlier
    list_labels = labels_labeled_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_labeled_outlier
    list_semi_labels = (semi_labels_labeled_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_labeled_outlier)

    return list_idx, list_labels, list_semi_labels

def normalization(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    eps = 1e-7
    data_std = (data - mu) / (std + eps)
    data_scaled = (data_std - np.min(data_std, axis=0))/(np.max(data_std, axis=0) - np.min(data_std, axis=0) + eps)
    params = {'mu':mu, 'std':std, 'min':np.min(data_std, axis=0), 'max':np.max(data_std, axis=0)}
    with open('/home/yifan/git/qualisys_drone_sdk/examples/params.pkl', 'wb') as f:
        pickle.dump(params, f)
    return data_scaled

def batch_sequential():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    print(signals.shape)
    n_channels = signals.shape[-1]

    signals_scaled = normalization(signals)
    signals_batched = np.zeros((len(signals_scaled)-seq_len+1, seq_len, n_channels))
    flags_batched = np.zeros((len(signals_scaled)-seq_len+1, seq_len))

    for i in range(len(signals_scaled)-seq_len+1):
        signals_batched[i,:,:] = signals_scaled[i:i+seq_len,:]
        flags_batched[i, :] = flags[i:i+seq_len]
    np.save(os.path.join(root, 'data_wind_seq.npy'), signals_batched)
    np.save(os.path.join(root, 'labels_wind_seq.npy'),flags_batched)

def build_next(root):
    signal_path = os.path.join(root,'data_unscaled_multi_noise.npy')
    flags_path = os.path.join(root,'labels_unscaled_multi_noise.npy')
    signals = np.squeeze(np.load(signal_path))
    flags = np.load(flags_path)
    seq_len = 100
    n_samples = signals.shape[0] - 1
    n_channels = 12
    next_real = np.empty((n_samples, n_channels))
    pad = np.zeros((seq_len - 1,))
    flags = np.concatenate([pad, flags], axis=0)
    labels = np.empty((n_samples,))
    for i in range(n_samples):
        next_real[i, :] = signals[i+1, -12:]
        labels[i] = 1 if np.sum(flags[i:i+seq_len]) > 0 else 0
    np.save(os.path.join(root, 'data_real.npy'), signals[:-1, :])
    np.save(os.path.join(root, 'next_real.npy'), next_real)
    np.save(os.path.join(root, 'labels_real.npy'),labels)


def batch_sequential_flat():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)

    signals_scaled = normalization(signals)
    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_channels))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        signals_next_batched[i, :] = signals_scaled[i+seq_len,:]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(target, 'data_wind.npy'), signals_batched)
    np.save(os.path.join(target, 'next_wind.npy'), signals_next_batched)

def batch_sequential_flat_state_only():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)
   
    signals_scaled = normalization(signals)
    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_state))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        signals_next_batched[i, :] = signals_scaled[i+seq_len, 3:6]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(target, 'data_wind_state_only.npy'), signals_batched)
    np.save(os.path.join(target, 'next_wind_state_only.npy'), signals_next_batched)
    np.save(os.path.join(target, 'labels_wind_state_only.npy'),labels_batched)


if __name__=='__main__':
    source = '' # 
    target = '/home/yifan/git/FIAD/data/spoofing'
    root = '/home/yifan/git/FIAD/data/spoofing'

    batch_sequential_flat_state_only()
    batch_sequential_flat()