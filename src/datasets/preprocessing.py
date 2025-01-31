import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    logger = logging.getLogger()

    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)
    

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])
    logger.info(f'In semi setting: n_known_normal: {n_known_normal}, n_unlabled_normal: {n_unlabeled_normal}, n_unlabeled_outlier: {n_unlabeled_outlier}, n_known_outlier: {n_known_outlier}')

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist() # A minor bug here when outlier_class and known_outlier_class overlap

    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)

    return list_idx, list_labels, list_semi_labels


def batch_sequential(root):
    seq_len = 100
    signal_path = os.path.join(root,'innovations.npy')
    flags_path = os.path.join(root,'flags.npy')
    signals = np.load(signal_path)
    flags = np.load(flags_path)
    print(signals.shape)

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    scaler = StandardScaler().fit(signals)
    signals_standard = scaler.transform(signals)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals_scaled = minmax_scaler.transform(signals_standard)

    signals_batched = np.zeros((len(signals_scaled)-seq_len+1, seq_len, 8))
    flags_batched = np.zeros(len(signals_scaled)-seq_len+1)
    for i in range(len(signals_scaled)-seq_len+1):
        signals_batched[i,:,:] = signals_scaled[i:i+seq_len,:]
        flags_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(root, 'innovations_batched.npy'), signals_batched)
    np.save(os.path.join(root, 'flags_batched.npy'),flags_batched)

def batch_sequential_flat(root):
    seq_len = 100
    # signal_path = os.path.join(root,'data.npy')
    signal_path = os.path.join(root,'data_unscaled_multi_noise.npy')
    flags_path = os.path.join(root,'labels_unscaled_multi_noise.npy')
    signals = np.load(signal_path)
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    scaler = StandardScaler().fit(signals)
    signals_standard = scaler.transform(signals)

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals_scaled = minmax_scaler.transform(signals_standard)

    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_channels))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        # for j in range(num_channels):
        #     signals_batched[i,j*seq_len:(j+1)*seq_len] = signals[i:i+seq_len,j]
        signals_next_batched[i, :] = signals_scaled[i+seq_len,:]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(root, 'data_multi_noise_batched.npy'), signals_batched)
    np.save(os.path.join(root, 'next_multi_noise_batched.npy'), signals_next_batched)
    np.save(os.path.join(root, 'labels_multi_noise_batched.npy'),labels_batched)
    # np.save(os.path.join(root, 'data_unscaled_multi_noise_batched.npy'), signals_batched)
    # np.save(os.path.join(root, 'next_unscaled_multi_noise_batched.npy'), signals_next_batched)
    # np.save(os.path.join(root, 'labels_unscaled_multi_noise_batched.npy'),labels_batched)

def batch_sequential_flat_state_only(root):
    seq_len = 100
    # signal_path = os.path.join(root,'data.npy')
    signal_path = os.path.join(root,'data_unscaled_multi_noise.npy')
    flags_path = os.path.join(root,'labels_unscaled_multi_noise.npy')
    signals = np.load(signal_path)
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    scaler = StandardScaler().fit(signals)
    signals_standard = scaler.transform(signals)

    # # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(signals_standard)
    signals_scaled = minmax_scaler.transform(signals_standard)

    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_state))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        # for j in range(num_channels):
        #     signals_batched[i,j*seq_len:(j+1)*seq_len] = signals[i:i+seq_len,j]
        signals_next_batched[i, :] = signals_scaled[i+seq_len, 6:9]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(root, 'data_multi_noise_state_only_batched.npy'), signals_batched)
    np.save(os.path.join(root, 'next_multi_noise_state_only_batched.npy'), signals_next_batched)
    np.save(os.path.join(root, 'labels_multi_noise_state_only_batched.npy'),labels_batched)

    # np.save(os.path.join(root, 'data_state_only.npy'), signals_batched)
    # np.save(os.path.join(root, 'next_state_only.npy'), signals_next_batched)
    # np.save(os.path.join(root, 'labels_state_only.npy'),labels_batched)

def batch_sequential_flat_normalize_per_sample(root):
    seq_len = 100
    # signal_path = os.path.join(root,'data.npy')
    signal_path = os.path.join(root,'data_unscaled_multi_noise.npy')
    flags_path = os.path.join(root,'labels_unscaled_multi_noise.npy')
    signals = np.load(signal_path)
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)

    # Each sample move forward one time step
    num_samples = len(signals)-seq_len
    # num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_channels))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        temp = signals[i:i+seq_len+1,:]
        scaler = StandardScaler().fit(temp)
        temp_standard = scaler.transform(temp)
        minmax_scaler = MinMaxScaler().fit(temp_standard)
        temp_scaled = minmax_scaler.transform(temp_standard)
        signals_batched[i,:] = temp_scaled[:seq_len,:].reshape((seq_len*num_channels,))
        # for j in range(num_channels):
        #     signals_batched[i,j*seq_len:(j+1)*seq_len] = signals[i:i+seq_len,j]
        signals_next_batched[i, :] = temp_scaled[seq_len, :]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(root, 'data_multi_noise_batched_normalize_per_sample.npy'), signals_batched)
    np.save(os.path.join(root, 'next_multi_noise_batched_normalize_per_sample.npy'), signals_next_batched)
    np.save(os.path.join(root, 'labels_multi_noise_batched_normalize_per_sample.npy'),labels_batched)
    # np.save(os.path.join(root, 'data_unscaled_multi_noise_batched.npy'), signals_batched)
    # np.save(os.path.join(root, 'next_unscaled_multi_noise_batched.npy'), signals_next_batched)
    # np.save(os.path.join(root, 'labels_unscaled_multi_noise_batched.npy'),labels_batched)


if __name__=='__main__':
    root = '/home/yifan/git/FIAD/data/spoofing'
    batch_sequential_flat_normalize_per_sample(root)
    # batch_sequential_flat_state_only(root)
    # batch_sequential_flat(root)
    # x = np.load(os.path.join(root, 'innovations_batched_flat.npy'))
    # print(x.shape)