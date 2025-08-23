import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Low pollution
# res = np.array([[0.9971, 0.9981, 0.9990, 0.9993],
#                 [0.9826, 0.9941, 0.9973, 0.9982],
#                 [0.9733, 0.9928, 0.9963, 0.9964],
#                 [0.9541, 0.9877, 0.9934, 0.9953],
#                 [0.9665, 0.9738, 0.9927, 0.9927]])

# base = np.array([[0.9949, 0.9960, 0.9944, 0.9975],
#                 [0.9842, 0.9958, 0.9968, 0.9952],
#                 [0.9645, 0.9851, 0.9952, 0.9964],
#                 [0.9563, 0.9676, 0.9775, 0.9960],
#                 [0.8940, 0.9355, 0.9687, 0.9827]])

# piad_all = np.array([[0.9968, 0.9956, 0.9973, 0.9983],
#                      [0.9924, 0.9877, 0.9918, 0.9959],
#                      [0.9748, 0.9878, 0.9917, 0.9957],
#                      [0.9527, 0.9754, 0.9860, 0.9955],
#                      [0.8820, 0.9713, 0.9788, 0.9907]])
# rkp = np.array([0, 0.001, 0.002, 0.003])
# rp = np.flip(np.array([0,  0.02, 0.04, 0.06, 0.08]))
# title = 'Performance gap under low pollution rates'


# Linear High pollution
# res = np.array([[0.9101, 0.9797, 0.9832, 0.9885, 0.9959, 0.9949],
#                 [0.7751, 0.9453, 0.9369, 0.9682, 0.9808, 0.9811],
#                 [0.5774, 0.8731, 0.8448, 0.9077, 0.9407, 0.9833]])

# base = np.array([[0.9265, 0.9774, 0.9501, 0.9866, 0.9888,0.9965],
#                  [0.6689, 0.8545, 0.8939, 0.9311, 0.9541, 0.9881],
#                  [0.7014, 0.7958, 0.8873, 0.9510, 0.9059, 0.9631]])

# piad_all = np.array([[0.9283, 0.9383, 0.9698, 0.9871, 0.9922, 0.9942],
#                      [0.7830, 0.9254, 0.9235, 0.9601, 0.9646, 0.9868],
#                      [0.6236, 0.8381, 0.9044, 0.9688, 0.9531, 0.9655]])
# rkp = np.array([0, 0.001, 0.002, 0.003, 0.004, 0.005])
# rp = np.flip(np.array([0.1, 0.2, 0.3]))
# title = 'Performance under high pollution rates'

# Linear mid pollution
res = np.array([[0.9736, 0.9862, 0.9900],
                [0.9590, 0.9766, 0.9918],
                [0.9599, 0.9674, 0.9714],
                [0.9331, 0.9489, 0.9788],
                [0.9453, 0.9369, 0.9682]])

base = np.array([[0.8962, 0.9491, 0.9598],
                [0.8977, 0.9470, 0.9693],
                [0.9155, 0.9030, 0.9749],
                [0.8683, 0.9142, 0.9387],
                [0.8545, 0.8939, 0.9311]])

piad_all = np.array([[0.9479, 0.9692, 0.9691],
                [0.9227, 0.9566, 0.9741],
                [0.9458, 0.9514, 0.9804],
                [0.8937, 0.9491, 0.9724],
                [0.9254, 0.9235, 0.9601]])

rkp = np.array([0.001, 0.002, 0.003,])
rp = np.flip(np.array([0.12, 0.14, 0.16, 0.18, 0.2]))
title = 'Performance under high pollution rates'

# Sin attack low pollution
# res = np.array([[0.9394, 0.9673, 0.9660, 0.9610],
#                 [0.8928, 0.9438, 0.9645, 0.9582],
#                 [0.8499, 0.8944, 0.9565, 0.9509],
#                 [0.8265, 0.8955, 0.9398, 0.9264],
#                 [0.7204, 0.8541, 0.9024, 0.9178]])

# base = np.array([[0.9222, 0.9322, 0.9366, 0.9388],
#                  [0.8661, 0.8893, 0.8986, 0.9140],
#                  [0.7906, 0.7848, 0.8495, 0.8764],
#                  [0.7210, 0.7366, 0.8207, 0.8179],
#                  [0.6500, 0.7768, 0.7270, 0.8304]])

# piad_all = np.array([[0.9251, 0.9522, 0.9547, 0.9561],
#                      [0.8820, 0.9095, 0.9265, 0.9374],
#                      [0.8301, 0.8654, 0.8992, 0.9203],
#                      [0.7672, 0.8715, 0.8787, 0.9019],
#                      [0.6861, 0.7566, 0.8350, 0.8839]])

# rkp = np.array([0, 0.001, 0.002, 0.003])
# rp = np.flip(np.array([0, 0.02, 0.04, 0.06, 0.08]))
# title = 'Performance with sin attack and low pollution rates'

for i in range(len(rp)):
    print('all:%.3f'%(np.mean(piad_all[i, :]-base[i, :])*100))
    print('res:%.3f'%(np.mean(res[i, :]-base[i, :])*100))
    print()

# for j in range(len(rkp)):
#     print('all:%.3f'%(np.mean(piad_all[:, j]-base[:, j])*100))
#     print('res:%.3f'%(np.mean(res[:, j]-base[:, j])*100))
#     print()

# fig, ax = plt.subplots(1, 3)

# im0 = ax[0].imshow(np.flipud(base), vmin=np.min(base), vmax=1)
# ax[0].set_xticks(range(len(rkp)), labels=rkp)
# ax[0].set_yticks(range(len(rp)), labels=rp)
# ax[0].set_title('Performance of the baseline model.')
# divider0 = make_axes_locatable(ax[0])
# cax0 = divider0.append_axes("right", size="5%", pad=0.05)
# cbar0 = fig.colorbar(im0, cax=cax0)


# gap1 = np.flipud(res - base)
# mag1 = np.max(np.abs(gap1))
# print(mag1)
# im1 = ax[1].imshow(gap1, cmap='seismic', vmin=-mag1, vmax=mag1)
# ax[1].set_xticks(range(len(rkp)), labels=rkp)
# ax[1].set_yticks(range(len(rp)), labels=rp)
# ax[1].set_title('Performance gap between PIPDres and the baseline')
# divider1 = make_axes_locatable(ax[1])
# cax1 = divider1.append_axes("right", size="5%", pad=0.05)
# cbar1 = fig.colorbar(im1, cax=cax1)

# gap2 = np.flipud(piad_all - base)
# mag2 = np.max(np.abs(gap2))
# print(mag2)
# im2 = ax[2].imshow(gap2, cmap='seismic', vmin=-mag2, vmax=mag2)
# ax[2].set_xticks(range(len(rkp)), labels=rkp)
# ax[2].set_yticks(range(len(rp)), labels=rp)
# ax[2].set_title('Performance gap between the PIPDall model and the baseline')
# divider2 = make_axes_locatable(ax[2])
# cax2 = divider2.append_axes("right", size="5%", pad=0.05)
# # Now create the colorbar inside that axis
# cbar2 = fig.colorbar(im2, cax=cax2)

# fig.suptitle(title, fontsize=16)
# # fig.tight_layout()
# plt.show()