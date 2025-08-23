import os
import numpy as np
import pandas as pd
import torch
import glob

import time
import baselines.utils_RoSAS
from baselines.RoSAS import RoSAS
# from datasets.spoofing_physical import SpoofingDatasetPhysical
from datasets.main import load_dataset
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
params = {
            'nbatch_per_epoch': 16,
            'epochs': 200,
            'batch_size': 32,
            'lr': 0.005,
            'n_emb': 128,
            'alpha': 0.5,
            'margin': 1,
            'beta': 1,
            'score_loss': 'smooth'
}
root_path = './'


def run_model(df, dataset_name, runs):
    model_name = args['algo']

    print("------------------------------------ Dataset: [%s] ------------------------------------" % dataset_name)
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df.fillna(method='ffill', inplace=True)
    # x = np.load('./data/spoofing/data_multi_noise_batched.npy')
    # y = np.load('./data/spoofing/labels_multi_noise_batched.npy')

    # x_train, y_train, x_test, y_test, x_val, y_val = baselines.utils_RoSAS.split_train_test_val(x, y,
    #                                                                             test_ratio=0.2,
    #                                                                             val_ratio=0.2,
    #                                                                             random_state=2021,
    #                                                                             del_features=True)
    # args['n_known'] = int(args['ratio_known_outlier'] * sum (y_train))
    # semi_y = baselines.utils_RoSAS.semi_setting(y_train, n_known_outliers=args['n_known'])

    # # # this is to control contamination rate and estimate the robustness
    # if args['contamination'] is not None:
        # x_train, y_train, semi_y = baselines.utils_RoSAS.adjust_contamination(x_train, y_train, semi_y,
    #                                                           adjust_cont_r=args['contamination'],
    #                                                           random_state=2021)

    # Load dataset
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           args['ratio_known_normal'], args['ratio_known_outlier'], args['contamination'],
                           random_state=np.random.RandomState(seed))
    x_train, y_train, semi_y,x_test, y_test, x_val, y_val = dataset.data_direct()
    # Align the labels
    semi_y[semi_y == 1] = 0
    semi_y[semi_y == -1] = 1  # 1 for labeled outliers

    rauc, raucpr, rtime = np.zeros(runs), np.zeros(runs), np.zeros(runs)
    for i in range(runs):
        st = time.time()

        params['use_es'] = args['use_es']
        params['seed'] = 42 + i

        model = RoSAS(**params)
        model.fit(x_train, semi_y, val_x=x_val, val_y=y_val)
        score = model.predict(x_test)

        auroc, aupr, roc_curve = baselines.utils_RoSAS.evaluate(y_test, score)
        rtime[i] = time.time() - st
        rauc[i] = auroc
        raucpr[i] = aupr

        txt = f'{dataset_name}, AUC-ROC: {auroc:.4f}, AUC-PR: {aupr:.4f}, ' \
              f'time: {rtime[i]:.1f}, runs: [{i+1}/{runs}]'

        print(txt)
        doc1 = open(args['res_path'] + f'@raw_{model_name}{args["flag"]}.csv', 'a')
        print(txt, file=doc1)
        doc1.close()

    print_text = f"{dataset_name}, AUC-ROC, {np.average(rauc):.4f}, {np.std(rauc):.4f}," \
                 f" AUC-PR, {np.average(raucpr):.4f}, {np.std(raucpr):.4f}, {np.average(rtime):.1f}," \
                 f" {runs}runs, {args['ratio_known_outlier']*100}percent known, {args['contamination']:.2f}cont."
    wandb.log({'auc_roc': np.average(rauc)})
    print(print_text, end='\n\n\n')

    if not args['debug']:
        doc1 = open(args['res_path'] + f'{model_name}{args["flag"]}.csv', 'a')
        print(print_text, file=doc1)
        doc1.close()
    net_dict = model.basenet.state_dict()
    torch.save({'net_dict': net_dict,
                'roc': roc_curve}, 'saved_model/RoSAS/model.pth')
    return np.average(rauc), np.average(raucpr)


if __name__ == '__main__':
    args = {'path': 'data',
            'datasets': 'spoofing_physical',
            'algo': 'rosas',
            'flag': '',
            'ratio_known_outlier': 0.003,
            'ratio_known_normal': 0.001,  # Ratio of labeled normal train samples
            'contamination': 0.05,
            'runs': 1,
            'log_avg': False,
            'use_es': True,
            'debug': False,
            'res_path': 'log/'}
    os.makedirs('log/', exist_ok=True)
    os.makedirs(args['res_path'], exist_ok=True)

    wandb.init(
        project='PIAD',
        name='Physical_sweep_RoSAS',
    )
    # args['ratio_known_outlier'] = wandb.config.ratio_known_outlier
    # args['contamination'] = wandb.config.ratio_pollution
    dataset_name = 'spoofing_physical' 
    data_path = './data'
    normal_class = 0
    seed = 4
    known_outlier_class = 1
    n_known_outlier_classes = 1  # Number of known outlier classes. If 0, no anomalies are known.
    path = os.path.join(root_path, args['path'])
    t1 = time.time()
    datasets_auc = []
    datasets_aupr = []


    df = None
    auroc, aupr = run_model(df, dataset_name=args['datasets'], runs=args['runs'])
    datasets_auc.append(auroc)
    datasets_aupr.append(aupr)


    avg1 = np.average(datasets_auc)
    avg2 = np.average(datasets_aupr)
    avg = f"avg, AUC-ROC, {avg1:.3f}, AUC-PR, {avg2:.3f}, {time.time()-t1:.1f}s"
    print(avg)

    if args['log_avg'] and not args['debug']:
        doc = open(args['res_path'] + f'{args["algo"]}{args["flag"]}.csv', 'a')
        print("", file=doc)
        print(avg, file=doc)
        doc.close()
