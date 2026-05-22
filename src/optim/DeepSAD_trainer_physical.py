from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from torch.nn import MSELoss
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import setting
import wandb
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import copy


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def pairwise_apply(A: torch.Tensor, f):
    """
    Returns F with shape (M, M) where F[i, j] = f(A[i], A[j]).
    """
    # vmap over the first argument (rows i) and second argument (rows j)
    f_i = torch.vmap(f, in_dims=(0, None))         # map over i, keep v fixed
    F   = torch.vmap(f_i, in_dims=(None, 0))(A, A) # then map over j
    return F  # (M, M)

def cos_sim(u, v):
    # return torch.log(torch.nn.functional.sigmoid(torch.dot(u, v)/setting.rep))
    den = (torch.norm(u) * torch.norm(v)).clamp_min(1e-8)
    return torch.dot(u, v) / den

# def pair_values_by_label(A: torch.Tensor, y: torch.Tensor, f, exclude_self=True):
#     """
#     Compute f(u, v) for all pairs, split into same-label and different-label.
#     Returns:
#       same_vals: 1-D tensor of values for pairs with same labels
#       diff_vals: 1-D tensor of values for pairs with different labels
#     """
#     F = pairwise_apply(A, f)           # (M, M)
#     M = A.shape[0]

#     same_mask = (y.unsqueeze(0) == y.unsqueeze(1))  # (M, M)
#     # print(same_mask)
#     if exclude_self:
#         # keep only i != j (unordered pairs use upper triangle)
#         tri = torch.ones(M, M, dtype=torch.bool, device=A.device).triu(diagonal=1)
#         same_vals = F[same_mask & tri]
#         diff_vals = F[(~same_mask) & tri]
#     else:
#         same_vals = F[same_mask]
#         diff_vals = F[~same_mask]

#     return same_vals, diff_vals

def l_contrastive(A, y):
    """
    A:  (k, k) matrix with A[i, j] = (f_i · f_j / tau)
    y:  (k,) integer labels
    returns: scalar loss
    """
    k = A.size(0)
    device = A.device
    y = y.view(-1)

    # Masks
    same = (y.unsqueeze(0) == y.unsqueeze(1))               # same class (incl. self)
    eye  = torch.eye(k, dtype=torch.bool, device=device)
    pos_mask = same & ~eye                                  # positives P(i)
    neg_mask = ~same                                        # negatives A(i)

    # logsumexp over negatives (masked)
    # neg_logits = A.masked_fill(~neg_mask, float('-inf'))
    logits_not_same = A.masked_fill(eye, float('-inf'))
    lse_not_same = torch.logsumexp(logits_not_same, dim=1)            # (k,)

    # average over positives
    pos_counts = pos_mask.sum(dim=1)
    valid_pos = pos_counts > 0                              # anchors with positives
    sum_pos = (A * pos_mask.float()).sum(dim=1)
    mean_pos = torch.zeros_like(sum_pos)
    mean_pos[valid_pos] = sum_pos[valid_pos] / pos_counts[valid_pos]

    # check which anchors have negatives
    valid_neg = neg_mask.sum(dim=1) > 0                     # anchors with negatives

    # final valid anchors
    valid = valid_pos & valid_neg
    if not valid.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # loss per anchor: L(i) = logsumexp_neg(i) - mean_pos(i)
    loss_i = lse_not_same - mean_pos
    loss = loss_i[valid].mean()
    return loss

# def l_contrastive(logits, y): # This should be equivalent to the above function, only easier to read
#     mask = y.unsqueeze(0) == y.unsqueeze(1)
#     eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
#     logits = logits - torch.max(logits, dim=1, keepdim=True).values  # stability
#     exp_logits = torch.exp(logits) * ~eye
#     denom = exp_logits.sum(dim=1, keepdim=True)  # all except self

#     pos_mask = mask & ~eye
#     pos_count = pos_mask.sum(dim=1)
#     valid = pos_count > 0
#     log_prob = logits - torch.log(denom + 1e-12)
#     loss = -(log_prob * pos_mask).sum(dim=1) / (pos_count + 1e-12)
#     return loss[valid].mean() if valid.any() else logits.new_tensor(0.0)

def get_all_centroids(A: torch.Tensor, y: torch.Tensor):
    """
    Compute centroids for all classes in y.
    Returns:
      centroids: dict mapping label -> centroid tensor
    """
    # Differentiable computation of centroids using scatter_add
    labels = torch.unique(y)
    centroids = {}
    for label in labels:
        mask = (y == label)
        count = mask.sum()
        # Avoid division by zero
        if count > 0:
            centroid = (A * mask.unsqueeze(1)).sum(dim=0) / count
            centroids[label.item()] = centroid
        else:
            centroids[label.item()] = torch.zeros_like(A[0])
    return centroids

class DeepSADTrainerPhysical(BaseTrainer):

    def __init__(self, n_known_outlier_classes: int, known_outlier_classes, coeff: dict, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, tau=0.1):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.n_known_outlier_classes = n_known_outlier_classes
        self.known_outlier_classes = known_outlier_classes
        self.class_ids = [0] + list(self.known_outlier_classes)
        self.centroids = {
            'c_normal': torch.zeros((setting.rep,), device=self.device)
        }
        for i in range(n_known_outlier_classes):
            self.centroids['c_outlier_{}'.format(i+1)] = torch.zeros((setting.rep,), device=self.device)
        self.roc_curve = None
        self.coeff = coeff
        self.n_aug = 2
        self.tau = tau

        # Optimization parameters
        self.eps = 1e-6
        self.MSE_loss = MSELoss()

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.test_f1_macro = None
        self.test_f1_weighted = None
        self.test_acc = None
        self.test_recall = None

    def loss_all(self, outputs, semi_targets, signal_pred, signal_next):
        loss = 0.0
        ########## Original Deep SAD loss ##########
        dist = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
        normal_mask = (semi_targets <= 0)
        
        if normal_mask.any():
            loss_normal = torch.mean(dist[normal_mask])
        else:
            loss_normal = torch.tensor(0.0, device=self.device)

        anomaly_mask = (semi_targets > 0)
        if anomaly_mask.any():
            loss_anomaly = torch.mean(torch.reciprocal(dist[anomaly_mask] + self.eps))
        else:
            loss_anomaly = torch.tensor(0.0, device=self.device)
        loss_sad = loss_normal + loss_anomaly
        ######### Physics-informed loss ##########
        loss_pred = self.MSE_loss(signal_pred, signal_next)

        ########## Directional loss ##############
        idx_labeled = (semi_targets>=0)

        if torch.sum(idx_labeled)<=1:
            loss_dir = torch.tensor(0.0).to(self.device)
        else:
            # A = pairwise_apply(outputs[idx_labeled]-self.centroids['c_normal'], cos_sim)
            A = pairwise_apply(outputs[idx_labeled], cos_sim)
            y_tensor = semi_targets[idx_labeled]
            loss_dir = l_contrastive(A/self.tau, y_tensor)

        ########### Clustering loss #############
        # Labeled normals loss
        dist_normal = torch.sum((outputs[semi_targets == 0] - self.centroids['c_normal']) ** 2, dim=1)
        loss_normal = torch.mean(dist_normal) if len(dist_normal)>0 else torch.tensor(0.0).to(self.device)

        # Labeled outliers loss
        loss_outlier = 0.0
        for i in range(1, self.n_known_outlier_classes+1):
            dist_outlier = torch.sum((outputs[semi_targets == i] - self.centroids['c_outlier_{}'.format(i)]) ** 2, dim=1)
            loss_outlier += torch.mean(dist_outlier) if len(dist_outlier)>0 else torch.tensor(0.0).to(self.device)

        # Compute the distance between all the unlabeled samples and the closest centroid
        dist_unlabeled = torch.zeros((len(outputs[semi_targets == -1]), self.n_known_outlier_classes+1)).to(self.device)
        for i in range(self.n_known_outlier_classes+1):
            if i == 0:
                dist_unlabeled[:, i] = torch.sum((outputs[semi_targets == -1] - self.centroids['c_normal']) ** 2, dim=1)
            else:
                dist_unlabeled[:, i] = torch.sum((outputs[semi_targets == -1] - self.centroids['c_outlier_{}'.format(i)]) ** 2, dim=1)
        min_dist_unlabeled, _ = torch.min(dist_unlabeled, dim=1)
        loss_unlabeled = torch.mean(min_dist_unlabeled) if len(min_dist_unlabeled)>0 else torch.tensor(0.0).to(self.device)
        loss_cluster = loss_normal + loss_outlier + loss_unlabeled

        loss += self.coeff['sad'] * loss_sad \
                + self.coeff['pred'] * loss_pred \
                + self.coeff['dir'] * loss_dir \
                + self.coeff['cluster'] * loss_cluster
        return loss, loss_sad, loss_pred, loss_dir , loss_cluster

    def train(self, dataset: BaseADDataset, net: BaseNet, model_path=None, save=False):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        logger.info(f'Training set size: {len(train_loader)}')
        logger.info(f'Validation set size: {len(val_loader)}')
        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)

        logger.info('Initializing centers...')
        self.update_center(train_loader, net)
        logger.info('Centers initialized.')
        # Training
        logger.info('Starting training physically...')
        start_time = time.time()
        # best_val_loss = np.inf
        best_auc = -np.inf
        net_store = None
        centroids_store = None
        for epoch in range(self.n_epochs):
            net.train()
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            epoch_loss_pred = 0.0
            epoch_loss_sad = 0.0
            epoch_loss_dir = 0.0
            epoch_loss_cluster = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _, signal_next = data
                inputs, semi_targets, signal_next = inputs.to(self.device), semi_targets.to(self.device), signal_next.to(self.device)

                # Online data augmentation
                inputs, semi_targets, signal_next = self.data_augmentation(inputs, semi_targets, signal_next)

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs, signal_pred = net(inputs)
                # outputs_proj = net.project(outputs)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(outputs, semi_targets, signal_pred, signal_next)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                # print(compute_grad_norm(net))
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_pred += loss_pred.item()
                epoch_loss_sad += loss_sad.item()
                epoch_loss_dir += loss_dir.item()
                epoch_loss_cluster += loss_cluster.item()
                n_batches += 1

            # Update centroids each epoch
            with torch.no_grad():
                self.update_center(train_loader, net)

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
                
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            wandb.log({'Train loss': epoch_loss / n_batches, 'Train time': epoch_train_time, 
                       'loss_pred':epoch_loss_pred/n_batches, 'loss_sad':epoch_loss_sad/n_batches,
                       'loss_dir':epoch_loss_dir/n_batches, 'loss_cluster':epoch_loss_cluster/n_batches})

            # Validation
            if epoch%100 == 0 and epoch > 0:
                val_auc, roc_curve = self.val(val_loader, net)
                wandb.log({'Validation AUC': val_auc})
                if save:
                    net_dict = net.state_dict()
                    # net_wo_pred_dict = self.net_wo_pred.state_dict() if hasattr(self, 'net_wo_pred') else None
                    # ae_net_dict = self.ae_net.state_dict() if save_ae else None

                    torch.save({'centroids': self.centroids.copy(),
                                'roc': self.roc_curve,
                                'net_dict': net_dict.copy()}, model_path + f'/model_physical_{epoch}.tar')
                if val_auc>best_auc:
                    logger.info("Find better model.")
                    self.roc_curve = roc_curve
                    best_auc = val_auc
                    net_store = copy.deepcopy(net)
                    centroids_store = copy.deepcopy(self.centroids)
              
        self.centroids = centroids_store
        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net_store, best_auc

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        logger.info(f'Test set size: {len(test_loader)}')
        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx, signal_next = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)
                signal_next = signal_next.to(self.device)

                outputs, signal_pred = net(inputs)
                # outputs_proj = net.project(outputs)
                dist_norm = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(outputs, semi_targets, signal_pred, signal_next)

                scores = dist_norm

                # Save tuples of (idx, label, score, outputs) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            outputs.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores, outputs = zip(*idx_label_score)
        labels = np.array(labels)
        labels_bin = np.zeros_like(labels)
        labels_bin[labels>0] = 1 # Treat all outlier classes as one class, may be changed to compute per-class AUC
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels_bin, scores)

        # Compute F1 score based on the best threshold from ROC curve and dist to each centroid
        fpr, tpr, thresholds = self.roc_curve
        youden_index = tpr - fpr
        best_threshold = thresholds[np.argmax(youden_index)]
        # If score < best_threshold, predict as normal (0); else predict base on closest centroid
        samples_normal = scores <= best_threshold
        samples_anomaly = scores > best_threshold
        y_pred = np.zeros_like(labels)
        y_pred[samples_normal] = 0
        if self.n_known_outlier_classes > 0:
            centroid_stack = torch.stack([
                self.centroids[f'c_outlier_{idx}']
                for idx in range(1, self.n_known_outlier_classes + 1)
            ]).to(self.device)
            outputs_tensor = torch.tensor(outputs, device=self.device)
            dist_anomaly = torch.norm(outputs_tensor.unsqueeze(1) - centroid_stack.unsqueeze(0), dim=2)
            nearest_idx = torch.argmin(dist_anomaly, dim=1).cpu().numpy()
            anomaly_labels = np.array(self.class_ids[1:])
            y_pred[samples_anomaly] = anomaly_labels[nearest_idx[samples_anomaly]]
        else:
            y_pred[samples_anomaly] = 1
        # Compute F1 score
        self.test_f1_macro = f1_score(labels, y_pred, average='macro')
        self.test_f1_weighted = f1_score(labels, y_pred, average='weighted')
        # Compute acc and recall
        self.test_acc = np.mean(labels == y_pred)
        self.test_recall = np.sum((labels>=1) & (y_pred==labels)) / np.sum(labels>=1)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Test F1 score (macro): {:.4f}'.format(self.test_f1_macro))
        logger.info('Test F1 score (weighted): {:.4f}'.format(self.test_f1_weighted))
        logger.info('Test Accuracy: {:.4f}'.format(self.test_acc))
        logger.info('Test Recall: {:.4f}'.format(self.test_recall))
        logger.info('Finished testing.')

    
    # def test_all(self, X_all, y_all, net: BaseNet):
    #     logger = logging.getLogger()
    #     # Set device for network
    #     net = net.to(self.device)

    #     # Testing
    #     logger.info('###################################')
    #     logger.info('Starting testing on all classes...')
    #     start_time = time.time()
    #     idx_label_score = []
    #     net.eval()
    #     with torch.no_grad():
    #         inputs = torch.tensor(X_all, device=self.device)
    #         labels = torch.tensor(y_all, device=self.device)
    #         semi_targets = semi_targets.to(self.device)
    #         idx = idx.to(self.device)
    #         signal_next = signal_next.to(self.device)

    #         outputs, signal_pred = net(inputs)
    #         dist_norm = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
    #         loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(outputs, semi_targets, signal_pred, signal_next)

    #         scores = dist_norm

    #         # Save tuples of (idx, label, score, outputs) in a list
    #         idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
    #                                     labels.cpu().data.numpy().tolist(),
    #                                     scores.cpu().data.numpy().tolist(),
    #                                     outputs.cpu().data.numpy().tolist()))

    #         # epoch_loss += loss.item()
    #         # n_batches += 1

    #     self.test_time = time.time() - start_time
    #     self.test_scores = idx_label_score

    #     # Compute AUC
    #     _, labels, scores, outputs = zip(*idx_label_score)
    #     labels = np.array(labels)
    #     labels[labels>0] = 1 # Treat all outlier classes as one class, may be changed to compute per-class AUC
    #     scores = np.array(scores)
    #     self.test_auc = roc_auc_score(labels, scores)

    #     # Compute F1 score based on the best threshold from ROC curve and dist to each centroid
    #     fpr, tpr, thresholds = self.roc_curve
    #     youden_index = tpr - fpr
    #     best_threshold = thresholds[np.argmax(youden_index)]
    #     # If score < best_threshold, predict as normal (0); else predict base on closest centroid
    #     samples_normal = scores <= best_threshold
    #     samples_anomaly = scores > best_threshold
    #     y_pred = np.zeros_like(labels)
    #     y_pred[samples_normal] = 0
    #     if self.n_known_outlier_classes > 0:
    #         centroid_stack = torch.stack([
    #             self.centroids[f'c_outlier_{idx}']
    #             for idx in range(1, self.n_known_outlier_classes + 1)
    #         ]).to(self.device)
    #         outputs_tensor = torch.tensor(outputs, device=self.device)
    #         dist_anomaly = torch.norm(outputs_tensor.unsqueeze(1) - centroid_stack.unsqueeze(0), dim=2)
    #         nearest_idx = torch.argmin(dist_anomaly, dim=1).cpu().numpy()
    #         anomaly_labels = np.array(self.class_ids[1:])
    #         y_pred[samples_anomaly] = anomaly_labels[nearest_idx[samples_anomaly]]
    #     else:
    #         y_pred[samples_anomaly] = 1
    #     # Compute F1 score
    #     self.test_f1_macro = f1_score(labels, y_pred, average='macro')
    #     self.test_f1_weighted = f1_score(labels, y_pred, average='weighted')
    #     # Compute acc and recall
    #     self.test_acc = np.mean(labels == y_pred)
    #     self.test_recall = np.sum((labels>=1) & (y_pred==labels)) / np.sum(labels>=1)

    #     # Log results
    #     logger.info('Test Loss: {:.6f}'.format(loss.item()))
    #     logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
    #     logger.info('Test Time: {:.3f}s'.format(self.test_time))
    #     logger.info('Test F1 score (macro): {:.4f}'.format(self.test_f1_macro))
    #     logger.info('Test F1 score (weighted): {:.4f}'.format(self.test_f1_weighted))
    #     logger.info('Test Accuracy: {:.4f}'.format(self.test_acc))
    #     logger.info('Test Recall: {:.4f}'.format(self.test_recall))
    #     logger.info('Finished testing.')

    def val(self, val_loader, net: BaseNet):
        logger = logging.getLogger()

        # Validation
        logger.info('Starting validation...')
        epoch_loss = 0.0
        n_batches = 0
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels, semi_targets, idx, signal_next = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)
                signal_next = signal_next.to(self.device)

                outputs, signal_pred = net(inputs)
                # outputs_proj = net.project(outputs)
                dist = torch.sum((outputs - self.centroids['c_normal']) ** 2, dim=1)
                loss, loss_sad, loss_pred, loss_dir, loss_cluster = self.loss_all(outputs, semi_targets, signal_pred, signal_next)

                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        labels[labels>0] = 1 # Treat all outlier classes as one class, may be changed to compute per-class AUC
        scores = np.array(scores)
        val_auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        # r = self.choose_best_r(fpr, tpr, thresholds)

        # Log results
        # logger.info(f'fpr: {fpr[1]}, tpr: {tpr[1]}')
        logger.info('Val Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Val AUC: {:.2f}%'.format(100. * val_auc))
        logger.info('Finished validation.')
        return val_auc, (fpr, tpr, thresholds)

        # return epoch_loss / n_batches

    def update_center(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_classes = len(self.class_ids)
        n_samples = torch.zeros(n_classes, device=self.device)

        # Reset accumulators before computing new means
        self.centroids['c_normal'].zero_()
        for idx in range(1, self.n_known_outlier_classes + 1):
            self.centroids[f'c_outlier_{idx}'].zero_()

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs, target, semi_target, index, data_next = data
                for idx, class_id in enumerate(self.class_ids):
                    class_mask = (semi_target == class_id)
                    if class_mask.any():
                        inputs_temp = inputs[class_mask]
                        outputs_temp, _ = net(inputs_temp.to(self.device))
                        n_samples[idx] += outputs_temp.shape[0]
                        if idx == 0:
                            self.centroids['c_normal'] += torch.sum(outputs_temp, dim=0)
                        else:
                            self.centroids[f'c_outlier_{idx}'] += torch.sum(outputs_temp, dim=0)
            
            if any(n_samples == 0):
                raise ValueError("At least one sample need to be labeled in the training set for each class.")

            # # If there are no samples for a class, set the center to unlabeled samples
            # if any(n_samples == 0):
            #     count = 0
            #     c = torch.zeros((setting.rep,), device=self.device)
            #     for data in train_loader:
            #         inputs, target, semi_target, index, data_next = data
            #         inputs_unlabeled = inputs[semi_target == -1]
            #         if inputs_unlabeled.shape[0] == 0:
            #             continue
            #         outputs_unlabeled, _ = net(inputs_unlabeled.to(self.device))
            #         count += outputs_unlabeled.shape[0]
            #         c += torch.sum(outputs_unlabeled, dim=0)
            #     if count > 0:
            #         c /= count
            #     else:
            #         c = torch.zeros((setting.rep,), device=self.device)

        for idx in range(n_classes):
            # if n_samples[idx] == 0:
            #     if idx == 0:
            #         self.centroids['c_normal'] = c.clone()
            #     else:
            #         self.centroids[f'c_outlier_{idx}'] = c.clone()
            # else:
            if idx == 0:
                self.centroids['c_normal'] /= n_samples[idx]
                self.centroids['c_normal'][(abs(self.centroids['c_normal']) < eps) & (self.centroids['c_normal'] < 0)] = -eps
                self.centroids['c_normal'][(abs(self.centroids['c_normal']) < eps) & (self.centroids['c_normal'] > 0)] = eps
            else:
                self.centroids[f'c_outlier_{idx}'] /= n_samples[idx]
                centroid_ref = self.centroids[f'c_outlier_{idx}']
                centroid_ref[(abs(centroid_ref) < eps) & (centroid_ref < 0)] = -eps
                centroid_ref[(abs(centroid_ref) < eps) & (centroid_ref > 0)] = eps
    
    def data_augmentation(self, inputs, semi_targets, signal_next):
        labeled_idx = semi_targets >= 0
        inputs_aug_list = [inputs]
        semi_targets_aug_list = [semi_targets]
        signal_next_aug_list = [signal_next]
        for i in range(self.n_aug):
            # Add Gaussian noise to labeled samples
            noise = torch.randn_like(inputs[labeled_idx]) * 0.05
            inputs_aug = inputs[labeled_idx].clone()
            inputs_aug += noise
            semi_targets_aug = semi_targets[labeled_idx].clone()
            signal_next_aug = signal_next[labeled_idx].clone() + torch.randn_like(signal_next[labeled_idx]) * 0.05
            inputs_aug_list.append(inputs_aug)
            semi_targets_aug_list.append(semi_targets_aug)
            signal_next_aug_list.append(signal_next_aug)
        
        inputs_aug = torch.cat(inputs_aug_list, dim=0)
        semi_targets_aug = torch.cat(semi_targets_aug_list, dim=0)
        signal_next_aug = torch.cat(signal_next_aug_list, dim=0)
        return inputs_aug, semi_targets_aug, signal_next_aug