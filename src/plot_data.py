import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib as mpl
# mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True

inputs = np.load('/home/yifan/git/FIAD/data/spoofing/data_unscaled_multi_noise.npy')
labels = np.load('/home/yifan/git/FIAD/data/spoofing/labels_unscaled_multi_noise.npy')
scaler = StandardScaler().fit(inputs)
inputs_standard = scaler.transform(inputs)
minmax_scaler = MinMaxScaler().fit(inputs_standard)
inputs = minmax_scaler.transform(inputs_standard)

legend = ['SP Y', 'SP X', 'SP Z', 'Residual Y', 'Res X', 'Res Z', 'Est Y', 'Est X', 'Est Z', 'GPS latitude', 'GPS Lon', 'GPS Alt']
# fig, axs = plt.subplots(2, 2)
# for i in range(4):
#     temp = inputs[10*i:10*i+100, :]
#     scaler = StandardScaler().fit(temp)
#     temp_standard = scaler.transform(temp)
#     minmax_scaler = MinMaxScaler().fit(temp_standard)
#     temp = minmax_scaler.transform(temp_standard)
#     axs[i//2, int(i%2)].plot(temp[:, 3:6])
labels[labels==0]=-1
labels = -labels
plt.figure(figsize=(12, 5))
plt.plot(inputs[:, 3])
plt.plot(inputs[:, 9])
st = np.where(labels==-1)[0][0]
ed = np.where(labels==-1)[0][-1]
plt.axvspan(st, ed, color='red', alpha=0.3)
# plt.plot(labels)
plt.xticks(fontsize=16)
plt.yticks([-1, 0, 1], fontsize=16)
plt.xlabel('Time steps', size=19)
plt.ylabel('Normalized values', size=19)
plt.legend([legend[3], legend[9], 'Attacked region'], fontsize=17)
# plt.legend(['Attack flags'])
plt.savefig('example_flight.pdf', bbox_inches='tight')
plt.show()