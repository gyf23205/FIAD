import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

method = 1
data_path = './data/spoofing'
signals = np.load(os.path.join(data_path, 'data_unscaled_log_attack.npy'))
scaler = StandardScaler().fit(signals)
signals_standard = scaler.transform(signals)
# Scale to range [0,1]
minmax_scaler = MinMaxScaler().fit(signals_standard)
signals_scaled = minmax_scaler.transform(signals_standard)
if method==0:
    y_pred = np.load('y_pred.npy')
    labels = np.load('labels.npy')
elif method==1:
    y_pred = np.load('y_pred_physical.npy')
    labels = np.load('labels_physical.npy')
elif method == 2:
    y_pred = np.load('y_pred_physical_res.npy')
    labels = np.load('labels_physical_res.npy')
elif method == 3:
    y_pred = np.load('y_pred_physical_state_only.npy')
    labels = np.load('labels_physical_state_only.npy')
else:
    y_pred = np.load('y_pred_physical_init.npy')
    labels = np.load('labels_physical_init.npy')

tpr = np.sum(np.logical_and(y_pred==1, labels==1))/np.sum(y_pred==1)
fpr = np.sum(np.logical_and(y_pred==1, labels==0))/np.sum(y_pred==1)
print('tpr: ', tpr)
print('fpr: ', fpr)
plt.plot(signals_scaled[:, 3:5])
# print(y_pred.shape)
plt.plot(y_pred)
plt.plot(labels)
plt.legend(['Residual Y', 'Residual X', 'Predicted', 'Ground truth'])
plt.show()