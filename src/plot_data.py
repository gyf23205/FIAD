import torch
import logging
from datetime import datetime
import numpy as np
import os
import setting

from sklearn.metrics import roc_curve
from DeepSAD import DeepSAD
from datasets.main import load_dataset
from matplotlib import pyplot as plt
from base.spoofing_dataset import MySpoofing

if __name__=='__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_name = 'spoofing_physical'
    net_name = 'spoof_mlp'
    xp_path = './log/DeepSAD/spoofing_physical/test'
    model_path = './model/physical/model_physical.tar'
    data_path = './data/spoofing'
    setting.init([])

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    log_file = xp_path + '/log-'+ time + '.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Load model
    deepSAD = DeepSAD(eta=1.0)
    deepSAD.set_network(net_name)
    deepSAD.load_model(model_path=model_path, load_ae=False, map_location=device)
    net = deepSAD.net
    c = deepSAD.c

    # Load data
    signals = np.load(os.path.join(data_path, 'data_unscaled_multi_noise.npy'))
    labels = np.load(os.path.join(data_path, 'labels_unscaled_multi_noise.npy'))

    dataset = MySpoofing(signals, labels)
    scores = []
    # Predict
    for signal, _, _,_ in dataset:
        output = net(signal)
        dist = (output - c) ** 2
        scores.append(dist.numpy())
    
    scores = np.array(scores)
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)

    print(fpr)
    print(tpr)
    print(threshold)