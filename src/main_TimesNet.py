# from data_provider.data_factory import data_provider
from base.exp_basic import Exp_Basic
from baselines.util_TimesNet import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from datasets.main import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import wandb

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        # arg should be passed as a dict
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.args = args
        self.device = torch.device(
            'cuda:{}'.format(self.args['gpu']) if self.args['use_gpu'] else 'cpu')
        self.model = self._build_model().to(self.device)
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()


    def _build_model(self):
        model = self.model_dict['TimesNet'].Model(self.args).float()

        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
        return model

    def get_data(self, dataset_name):
        # Load my own data loader here.
        # dataset_name = 'spoofing_physical'
        datasets = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           self.args['ratio_known_normal'], self.args['ratio_known_outlier'], self.args['ratio_pollution'],
                           random_state=np.random.RandomState(seed))
        return datasets.loaders(batch_size=self.args['batch_size'])

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=30, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args['train_epochs'] - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = model_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test=0):
        # test_data, test_loader = self._get_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pth')))

        attens_energy = []
        
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1) # suspecious
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y, _) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args['ratio_pollution']*100)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        # pred = pred.reshape(-1, 100)
        # pred = (np.sum(pred, axis=1) > 0).astype(int)
        # gt = gt.reshape(-1, 100)
        # gt = (np.sum(gt, axis=1) > 0).astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        roc_auc = roc_auc_score(gt, test_energy - threshold)
        wandb.log({
            'precision': precision,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'f_score': f_score,
            'recall': recall
        })
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, ROC AUC : {:0.4f}".format(
            accuracy, precision,
            recall, f_score, roc_auc))

        f = open("result_anomaly_detection.txt", 'a')
        # f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, ROC AUC : {:0.4f}".format(
            accuracy, precision,
            recall, f_score, roc_auc))
        f.write('\n')
        f.write('\n')
        f.close()
        return threshold
    
if __name__=='__main__':
    seed = 4
    dataset_name = 'spoofing_unsupervised'  # Change to 'spoofing_physical' or 'spoofing' as needed
    data_path = './data'
    # ratio_known_outlier = 0.005 
    # ratio_known_normal = 0
    # ratio_pollution = 0.2 # Replace with anomaly ratio in args
    # batch_size = 128
    normal_class = 0
    known_outlier_class = 1
    n_known_outlier_classes = 1
    model_path = './saved_model/TimesNet'
    log_path = './log/TimesNet'

    args = {
        # entries from args
        'use_gpu': True,
        'gpu_type': 'cuda',
        'gpu': 0,
        'use_multi_gpu': False,
        # 'devices': [0],
        'train_epochs': 20,
        'learning_rate': 1e-4,
        'lradj': 'type2',  # 'type1', 'type2', 'type3', 'cosine'
        'features': 'M',
        'ratio_pollution': 0.05, # Change later to the desired anomaly ratio
        'ratio_known_normal': 0,
        'ratio_known_outlier': 0,  # Known outlier ratio
        # entries from configs
        'seq_len': 100,
        'pred_len': 0,
        'd_model': 128,
        'd_ff': 128,
        'top_k': 3,
        'num_kernels': 3,
        'e_layers': 3,
        'enc_in': 12,
        'embed': 'fixed',
        'freq': 'h',
        'dropout': 0.0,
        'c_out': 12,
        'batch_size': 128
    }
    wandb.init(
        project='PIAD',
        name='Physical_sweep_TimesNet',
    )
    # args['ratio_pollution'] = wandb.config.ratio_pollution
    exp = Exp_Anomaly_Detection(args)
    train_loader, vali_loader, test_loader = exp.get_data(dataset_name)
    exp.train()
    threshold = exp.test(test=1)
    net_dict = exp.model.state_dict()
    torch.save({'net_dict': net_dict,
                'threshold': threshold}, model_path + '/model.pth')

