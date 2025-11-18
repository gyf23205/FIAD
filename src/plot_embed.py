import torch
import setting
import numpy as np
import matplotlib.pyplot as plt

from networks.mlp import MLP_Physical
from datasets.main import load_dataset
from sklearn.decomposition import PCA

def choose_sample(n_samples, X, y):
    normal_indices = np.where(y == 0)[0]
    anomaly_indices_1 = np.where(y == 1)[0]
    anomaly_indices_2 = np.where(y == 2)[0]
    chosen_normal_indices = np.random.choice(normal_indices, n_samples_per_class, replace=False)
    chosen_anomaly_indices_1 = np.random.choice(anomaly_indices_1, n_samples_per_class, replace=False)
    chosen_anomaly_indices_2 = np.random.choice(anomaly_indices_2, n_samples_per_class, replace=False)
    chosen_indices = np.concatenate((chosen_normal_indices, chosen_anomaly_indices_1, chosen_anomaly_indices_2))
    inputs = torch.tensor(X[chosen_indices], dtype=torch.float32)
    labels = torch.tensor(y[chosen_indices], dtype=torch.long)
    return inputs, labels

if __name__ == '__main__':
    # Setup
    dataset_name = 'alfa'
    data_path = './data'
    ratio_known_outlier = 0.3
    ratio_known_normal = 0.2
    ratio_pollution = 0.1
    rko = str(ratio_known_outlier).replace('.','')
    rp = str(ratio_pollution).replace('.','')
    model_path = f'./saved_model/physical/model_{rko}_{rp}'
    setting.init([512, 512, 1024, 2.0])
    normal_class = 0
    known_outlier_class = [1, 2]
    n_known_outlier_classes = len(known_outlier_class)
    seed = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples_per_class = 50

    # Load trained model
    model = MLP_Physical(x_dim=1200, h_dims=[setting.hd1, setting.hd2], rep_dim=setting.rep, bias=False)
    results_dict = torch.load(f'{model_path}/model_physical.tar', map_location=device)
    centroids = results_dict['centroids']
    # Compute the angle between centroids
    v_normal_anom1 = (centroids['c_outlier_1'] - centroids['c_normal']).cpu().numpy()
    v_normal_anom2 = (centroids['c_outlier_2'] - centroids['c_normal']).cpu().numpy()

    angle_anom1_anom2 = np.arccos(np.clip(np.dot(v_normal_anom1, v_normal_anom2) / (np.linalg.norm(v_normal_anom1) * np.linalg.norm(v_normal_anom2)), -1.0, 1.0))
    print(f'Angle between Anomaly Type 1 and Anomaly Type 2: {np.degrees(angle_anom1_anom2):.2f} degrees')
    print(f'Norm of Normal Centroid: {np.linalg.norm(centroids["c_normal"].cpu().numpy()):.4f}')
    print(f'Norm of Anomaly Type 1 Centroid: {np.linalg.norm(centroids["c_outlier_1"].cpu().numpy()):.4f}')
    print(f'Norm of Anomaly Type 2 Centroid: {np.linalg.norm(centroids["c_outlier_2"].cpu().numpy()):.4f}')

    model.load_state_dict(results_dict['net_dict'])
    model = model.to(device)
    model.eval()

    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                            ratio_known_normal, ratio_known_outlier, ratio_pollution,
                            random_state=np.random.RandomState(seed))

    X_train, y_train, semi_y, X_test, y_test, X_val, y_val = dataset.data_direct()
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)

    n_batches = 3

    fig, axs = plt.subplots(1, n_batches, figsize=(5*n_batches, 5))
    centroids_colors = ['g', 'r', 'b', 'c', 'm', 'y']
    for i in range(n_batches):
        if i >= n_batches:
            break

        # Randomly choose sufficient number of samples from normal and all the anomalies
        inputs, labels = choose_sample(n_samples_per_class, X_all, y_all)

        with torch.no_grad():
            inputs = inputs.to(device)
            z, pred = model(inputs)
            z = z.cpu().numpy()
            # Convert z to 2D using PCA or t-SNE if needed
            if z.shape[1] > 2:
                pca = PCA(n_components=2)
                z = pca.fit_transform(z)
            normals = np.where(labels.numpy()==0)[0]
            anomlies_1 = np.where(labels.numpy()==1)[0]
            anomlies_2 = np.where(labels.numpy()==2)[0]
            # print(f'Batch {i+1}: Normals: {len(normals)}, Anomaly Type 1: {len(anomlies_1)}, Anomaly Type 2: {len(anomlies_2)}')
            # Plot embeddings
            axs[i].scatter(z[normals,0], z[normals,1], c='g', label='Normal', alpha=0.5)
            axs[i].scatter(z[anomlies_1,0], z[anomlies_1,1], c='r', label='Anomaly Type 1', alpha=0.5)
            axs[i].scatter(z[anomlies_2,0], z[anomlies_2,1], c='b', label='Anomaly Type 2', alpha=0.5)
            # Plot centroids
            axs[i].legend()
            axs[i].set_title(f'Batch {i+1} Embedding')

    for i in range(n_batches):
        for j, (k, v) in enumerate(centroids.items()):
            v = v.cpu().numpy().reshape(1, -1)
            if v.shape[1] > 2:
                v = pca.transform(v)
            axs[i].scatter(v[0, 0], v[0, 1], c=centroids_colors[j], marker='x', s=100, label=f'Centroid {k}')
    plt.show()
    fig.savefig('imgs/embedding_batches.png')