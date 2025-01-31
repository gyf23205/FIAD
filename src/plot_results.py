import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
rp = np.array([0.1, 0.2, 0.3])
rko = np.linspace(0, 0.01, 11)

# Baseline
best_auc_seed4_nopretrain = np.array([[0.5611, 0.7347, 0.7443, 0.6598, 0.7341, 0.7313, 0.6694, 0.8418, 0.9550, 0.9436, 0.9467],
                     [0.8687, 0.6151, 0.9564, 0.9086, 0.8053, 0.7650, 0.8870, 0.8052, 0.9321, 0.8889, 0.9252],
                     [0.6731, 0.8834, 0.7826, 0.7205, 0.6911, 0.7285, 0.7765, 0.8386, 0.8712, 0.9598, 0.8250]])

best_auc_seed6_nopretrain = np.array([[0.9061, 0.7347, 0.7841, 0.8144, 0.9041, 0.8619, 0.9000, 0.8812, 0.7900, 0.8965, 0.9041],
                           [0.7882, 0.8614, 0.8111, 0.8072, 0.7839, 0.9480, 0.8878, 0.8519, 0.9313, 0.9475, 0.9546],
                           [0.8853, 0.8740, 0.8982, 0.7858, 0.8116, 0.8798, 0.8253, 0.9620, 0.7923, 0.8517, 0.8038]])

best_auc_seed8_nopretrain = np.array([[0.7585, 0.6771, 0.8481, 0.8606, 0.6899, 0.7674, 0.8150, 0.8964, 0.8080, 0.9847, 0.9082],
                           [0.8512, 0.8844, 0.8105, 0.7768, 0.7901, 0.7643, 0.8219, 0.8158, 0.9576, 0.7244, 0.9271],
                           [0.5535, 0.6426, 0.8935, 0.8309, 0.6900, 0.8239, 0.8489, 0.8869, 0.8328, 0.8589, 0.8676]])

best_auc_seed10000_nopretrain = np.array([[0.9152, 0.4357, 0.4627, 0.9062, 0.6927, 0.9024, 0.8887, 0.9510, 0.8639, 0.8673, 0.7796],
                           [0.8428, 0.8156, 0.7583, 0.7783, 0.8101, 0.8673, 0.9216, 0.9856, 0.8518, 0.8755, 0.8535],
                           [0.6843, 0.6619, 0.7848, 0.6918, 0.7014, 0.7341, 0.8814, 0.8326, 0.7494, 0.8096, 0.8264]])

best_auc_seed10086_nopretrain = np.array([[0.8586, 0.7993, 0.8562, 0.9187, 0.7912, 0.8651, 0.9144, 0.8468, 0.8881, 0.9359, 0.9053 ],
                           [0.6343, 0.8481, 0.9235, 0.7259, 0.8984, 0.8745, 0.9527, 0.9526, 0.8830, 0.8636, 0.8967],
                           [0.9453, 0.7891, 0.8765, 0.8304, 0.7450, 0.8676, 0.8712, 0.7356, 0.8279, 0.9154, 0.8121]])

best_auc = np.array([best_auc_seed4_nopretrain, best_auc_seed6_nopretrain, best_auc_seed8_nopretrain, best_auc_seed10000_nopretrain, best_auc_seed10086_nopretrain])
best_auc_mean = np.mean(best_auc, axis=0)
best_auc_std = np.std(best_auc, axis=0)
print(best_auc_mean)
# print(best_auc_std)

# Physics-informed-parallel
best_auc_physical_seed4_nopretrain = np.array([[0.4514, 0.7363, 0.6424, 0.8286, 0.7710, 0.8083, 0.8644, 0.9033, 0.9238, 0.9019, 0.8776],
                     [0.9347, 0.8866, 0.9623, 0.7789, 0.9770, 0.9732, 0.8774, 0.8151, 0.8897, 0.9026, 0.8492],
                     [0.7097, 0.8353, 0.6985, 0.7380, 0.7994, 0.8804, 0.8255, 0.9451, 0.8404, 0.9395, 0.7810]])

best_auc_physical_seed6_nopretrain = np.array([[0.4514, 0.7363, 0.6424, 0.8286, 0.7710, 0.8083, 0.8644, 0.9033, 0.9238, 0.9019, 0.8776 ],
                     [0.9347, 0.8866, 0.9623, 0.7789, 0.9770, 0.9732, 0.8774, 0.8151, 0.8897, 0.9026, 0.8492],
                     [0.7097, 0.8353, 0.6985, 0.7380, 0.7994, 0.8804, 0.8255, 0.9451, 0.8404, 0.9395, 0.7810]])

best_auc_physical_seed8_nopretrain = np.array([[0.9621, 0.9148, 0.7321, 0.8470, 0.8726, 0.8370, 0.9186, 0.9525, 0.9441, 0.9447, 0.8811],
                                    [0.7383, 0.7985, 0.7562, 0.7212, 0.6092, 0.6451, 0.7470, 0.7073, 0.9229, 0.9223, 0.8846],
                                    [0.8545, 0.9466, 0.7932, 0.9293, 0.7457, 0.8031, 0.8530, 0.8347, 0.8271, 0.9354, 0.9387]])

best_auc_physical_seed10000_nopretrain = np.array([[0.8569, 0.8161, 0.9107, 0.8244, 0.9627, 0.9154, 0.9522, 0.9322, 0.9349, 0.9202, 0.9385],
                                        [0.7868, 0.8814, 0.8943, 0.7575, 0.8735, 0.8403, 0.8380, 0.8579, 0.8860, 0.9232, 0.9574],
                                        [0.7465, 0.8577, 0.7739, 0.7994, 0.7653, 0.7465, 0.8233, 0.8757, 0.7799, 0.7781, 0.8508]])

best_auc_physical_seed10086_nopretrain = np.array([[0.9904, 0.7672, 0.5057, 0.6960, 0.8933, 0.6778, 0.8728, 0.8661, 0.9732, 0.9241, 0.7927],
                                        [0.5851, 0.5898, 0.9150, 0.7449, 0.7544, 0.8930, 0.9001, 0.7836, 0.9202, 0.7513, 0.9627],
                                        [0.7908, 0.6176, 0.8538, 0.7144, 0.7524, 0.7338, 0.7979, 0.8727, 0.8055, 0.7641, 0.7763]])

best_auc_physical = np.array([best_auc_physical_seed4_nopretrain, best_auc_physical_seed6_nopretrain, best_auc_physical_seed8_nopretrain, best_auc_physical_seed10000_nopretrain, best_auc_physical_seed10086_nopretrain])
best_auc_physical_mean = np.mean(best_auc_physical, axis=0)
best_auc_physical_std = np.std(best_auc_physical, axis=0)

# Physics-informed-init
best_auc_physical_init_seed4 = np.array([[0.5466, 0.8322, 0.7412, 0.6639, 0.8498, 0.8468, 0.8125, 0.8305, 0.7357, 0.8043, 0.7815],
                                         [0.8252, 0.7839, 0.7605, 0.9130, 0.9230, 0.9210, 0.8081, 0.8529, 0.8450, 0.8502, 0.7399],
                                         [0.9341, 0.8853, 0.9539, 0.9533, 0.9285, 0.9379, 0.9614, 0.9322, 0.9051, 0.8918, 0.9073]])

best_auc_physical_init_seed6 = np.array([[0.7721, 0.8611, 0.7934, 0.8125, 0.8684, 0.8516, 0.9567, 0.9316, 0.8854, 0.6589, 0.8776],
                                         [0.9157, 0.8764, 0.7803, 0.9109, 0.8984, 0.8561, 0.8671, 0.7914, 0.8996, 0.8317, 0.9511],
                                         [0.9543, 0.8932, 0.8671, 0.8897, 0.8144, 0.9095, 0.8724, 0.6819, 0.8861, 0.8626, 0.9243]])

best_auc_physical_init_seed8 = np.array([[0.7782, 0.8571, 0.7815, 0.8005, 0.8057, 0.7788, 0.8186, 0.8637, 0.6886, 0.7910, 0.9005],
                                         [0.8860, 0.7237, 0.7918, 0.8230, 0.6628, 0.9944, 0.9166, 0.8855, 0.8564, 0.9151, 0.8969],
                                         [0.9417, 0.8689, 0.9650, 0.8785, 0.8631, 0.9621, 0.8934, 0.8735, 0.8684, 0.8302, 0.8810]])

best_auc_physical_init_seed10000 = np.array([[0.8157, 0.9012, 0.8819, 0.7970, 0.7989, 0.7856, 0.9107, 0.9016, 0.8510, 0.8421, 0.7128],
                                             [0.8954, 0.9074, 0.8735, 0.9216, 0.8681, 0.8991, 0.8823, 0.8718, 0.8090, 0.8414, 0.9426],
                                             [0.9242, 0.7689, 0.7965, 0.9449, 0.8778, 0.8866, 0.8562, 0.8544, 0.8773, 0.8982, 0.9283]])

best_auc_physical_init_seed10086 = np.array([[0.6658, 0.7280, 0.6354, 0.8579, 0.7956, 0.8136, 0.6550, 0.7780, 0.8628, 0.8280, 0.8544],
                                             [0.8435, 0.8439, 0.8266, 0.8179, 0.8669, 0.7546, 0.7716, 0.8240, 0.8778, 0.8618, 0.8543],
                                             [0.9154, 0.8178, 0.9232, 0.9521, 0.7679, 0.8717, 0.7871, 0.8728, 0.9465, 0.8697, 0.9133]])
best_auc_physical_init = np.array([best_auc_physical_init_seed4, best_auc_physical_init_seed6, best_auc_physical_init_seed8, best_auc_physical_init_seed10000, best_auc_physical_init_seed10086])
best_auc_physical_init_mean = np.mean(best_auc_physical_init, axis=0)
# best_auc_physical_init_std = np.std(best_auc_physical, axis=0)
print(best_auc_physical_init_mean)
# print(best_auc_physical_std)
# rp = np.array([0.1, 0.2, 0.3])
# rko = np.linspace(0, 0.007, 8)
# best_auc = np.array([[0.5611, 0.7347, 0.7443, 0.6598, 0.7341, 0.7313, 0.6694, 0.8418 ],
#                      [0.8687, 0.6151, 0.9564, 0.9086, 0.8053, 0.7650, 0.8870, 0.8052],
#                      [0.6731, 0.8834, 0.7826, 0.7205, 0.6911, 0.7285, 0.7765, 0.8386]])

# best_auc_physical = np.array([[0.4514, 0.7363, 0.6424, 0.8286, 0.7710, 0.8083, 0.8644, 0.9033],
#                               [0.9347, 0.8866, 0.9623, 0.7789, 0.9770, 0.9732, 0.8774, 0.8151],
#                               [0.7097, 0.8353, 0.6985, 0.7380, 0.7994, 0.8804, 0.8255, 0.9451]])

diff = best_auc_physical_init-best_auc
diff_mean = np.mean(diff, axis=0)
diff_std = np.std(diff, axis=0)
print(diff_mean)
print(diff_std)
# print(diff) # Number of experiment where physics-informed approach performs better
# print(diff)
# # print(np.where())
# x, y = np.meshgrid(rko, rp)

# # Set up plot
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

# ls = LightSource(270, 45)
# # To use a custom hillshading mode, override the built-in shading and pass
# # in the rgb colors of the shaded surface calculated from "shade".
# rgb = ls.shade(best_auc, cmap=cm.plasma, vert_exag=0.1, blend_mode='soft')

# surf = ax.plot_surface(x, y, best_auc, rstride=1, cstride=1,
#                        linewidth=0, antialiased=False, shade=False)

# surf1 = ax.plot_surface(x, y, best_auc_physical, rstride=1, cstride=1,
#                        linewidth=0, antialiased=False, shade=False)
# # ax.legend([surf, surf1], ['Baseline', 'Ours'])

# ax.set_xlabel('rko')
# ax.set_ylabel('rp')

# plt.show()