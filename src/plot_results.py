import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
# rp = np.array([0.1, 0.2, 0.3])
# rko = np.linspace(0, 0.01, 11)
# best_auc = np.array([[0.5611, 0.7347, 0.7443, 0.6598, 0.7341, 0.7313, 0.6694, 0.8418, 0.9550, 0.9436, 0.9467 ],
#                      [0.8687, 0.6151, 0.9564, 0.9086, 0.8053, 0.7650, 0.8870, 0.8052, 0.9321, 0.8889, 0.9252],
#                      [0.6731, 0.8834, 0.7826, 0.7205, 0.6911, 0.7285, 0.7765, 0.8386, 0.8712, 0.9598, 0.8250]])

# best_auc_physical = np.array([[0.4514, 0.7363, 0.6424, 0.8286, 0.7710, 0.8083, 0.8644, 0.9033, 0.9238, 0.9019, 0.8776],
#                      [0.9347, 0.8866, 0.9623, 0.7789, 0.9770, 0.9732, 0.8774, 0.8151, 0.8897, 0.9026, 0.8492],
#                      [0.7097, 0.8353, 0.6985, 0.7380, 0.7994, 0.8804, 0.8255, 0.9451, 0.8404, 0.9395, 0.7810]])

rp = np.array([0.1, 0.2, 0.3])
rko = np.linspace(0, 0.007, 8)
best_auc = np.array([[0.5611, 0.7347, 0.7443, 0.6598, 0.7341, 0.7313, 0.6694, 0.8418 ],
                     [0.8687, 0.6151, 0.9564, 0.9086, 0.8053, 0.7650, 0.8870, 0.8052],
                     [0.6731, 0.8834, 0.7826, 0.7205, 0.6911, 0.7285, 0.7765, 0.8386]])

best_auc_physical = np.array([[0.4514, 0.7363, 0.6424, 0.8286, 0.7710, 0.8083, 0.8644, 0.9033],
                              [0.9347, 0.8866, 0.9623, 0.7789, 0.9770, 0.9732, 0.8774, 0.8151],
                              [0.7097, 0.8353, 0.6985, 0.7380, 0.7994, 0.8804, 0.8255, 0.9451]])

diff = best_auc_physical-best_auc
print(np.sum(diff)/(len(rp)*len(rko))) # Number of experiment where physics-informed approach performs better
# print(np.where())
x, y = np.meshgrid(rko, rp)

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(best_auc, cmap=cm.plasma, vert_exag=0.1, blend_mode='soft')

surf = ax.plot_surface(x, y, best_auc, rstride=1, cstride=1,
                       linewidth=0, antialiased=False, shade=False)

surf = ax.plot_surface(x, y, best_auc_physical, rstride=1, cstride=1,
                       linewidth=0, antialiased=False, shade=False)

ax.set_xlabel('rko')
ax.set_ylabel('rp')

plt.show()