import torch
import logging
import random
import numpy as np
import wandb
from datetime import datetime
import setting

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset

def main(dataset_name, net_name, xp_path, data_path, load_config=None, load_model=None, load_path=None, eta=1.0,
         ratio_known_normal=0.0, ratio_known_outlier=0.0, ratio_pollution=0.0, device='cuda', seed=-1,
         optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestone=50, batch_size=128, weight_decay=1e-6,
         pretrain=True, pre_optimizer_name='adam', pre_lr=0.001, pre_n_epochs=100, pre_lr_milestone=[0], pre_batch_size=128, pre_weight_decay=1e-6,
         num_threads=0, n_jobs_dataloader=0, normal_class=0, known_outlier_class=1, n_known_outlier_classes=0):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

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

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    # print(seed)
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(eta)
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_path, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_path)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Initilizing with physics knowledge.')
        logger.info('Pretraining learning rate: %g' % pre_lr)
        logger.info('Pretraining epochs: %d' % pre_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % (pre_lr_milestone,))
        logger.info('Pretraining batch size: %d' % pre_batch_size)
        logger.info('Pretraining weight decay: %g' % pre_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain_physical(dataset,
                    optimizer_name=optimizer_name,
                    lr=pre_lr,
                    n_epochs=pre_n_epochs,
                    lr_milestones=lr_milestone,
                    batch_size=pre_batch_size,
                    weight_decay=weight_decay,
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    weight_pred=weight_pred)

        # # Save pretraining results
        # deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (lr_milestone,))
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)

        # Train model on dataset
    deepSAD.train(dataset,
                optimizer_name=optimizer_name,
                lr=lr,
                n_epochs=n_epochs,
                lr_milestones=lr_milestone,
                batch_size=batch_size,
                weight_decay=weight_decay,
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path + '/results_physics_init.json')
    deepSAD.save_model(export_model=xp_path + '/model_physics_init.tar', save_ae=False)
    cfg.save_config(export_json=xp_path + '/config_physics_init.json')


if __name__ == '__main__':
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')

    dataset_name = 'spoofing_physical'
    net_name = 'spoof_mlp'
    xp_path = './log/DeepSAD/spoofing' # Log path
    data_path = './data'
    ratio_known_outlier = 0.001
    ratio_pollution = 0.1
    lr = 0.0001
    n_epochs = 300
    lr_milestone = [50]
    batch_size = 128
    weight_decay = 0.5e-6
    pretrain = True
    pre_lr = 0.0001
    pre_n_epochs = 150
    pre_batch_size = 128
    pre_weight_decay = 0.5e-6
    normal_class = 0
    known_outlier_class = 1
    weight_pred = 9.111514123138956
    seed = 4
    n_known_outlier_classes = 1 # Number of known outlier classes. If 0, no anomalies are known. 
                                # If 1, outlier class as specified in --known_outlier_class option.
                                # If > 1, the specified number of outlier classes will be sampled at random.
    
    
    wandb.init(
        project='PIAD',
        name='Physical_init_sweep',
        config={
            'dataset':'unscaled',
           'lr': lr,
           'batch size': batch_size,
           'weight decay': weight_decay,
           'physical': True,
           'pretrain': pretrain,
        }
    )

    ratio_known_outlier = wandb.config.ratio_known_outlier
    ratio_pollution = wandb.config.ratio_pollution
    seed = wandb.config.seed
    setting.init([])
    # Make the code deterministic

    main(dataset_name, net_name, xp_path, data_path, ratio_known_outlier=ratio_known_outlier,
          ratio_pollution=ratio_pollution, lr=lr, n_epochs=n_epochs, lr_milestone=lr_milestone,
          weight_decay=weight_decay, pretrain=pretrain, pre_lr=pre_lr, pre_n_epochs=pre_n_epochs,
          batch_size=batch_size, pre_batch_size=pre_batch_size, pre_weight_decay=pre_weight_decay, normal_class=normal_class,
          known_outlier_class=known_outlier_class, n_known_outlier_classes=n_known_outlier_classes,seed=seed
         )
