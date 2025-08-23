import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FuncFormatter
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 6))

# Make data.
lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
X = Y = lams
X, Y = np.meshgrid(X, Y)
# Z = np.load('land_vanilla.npy')
Z = np.load('land_physical_res.npy')
Z = Z/1e3
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel('v2', labelpad=13, size=19)
ax.set_ylabel('v1', labelpad=13, size=19)
ax.set_zlabel('Loss value ($\\times 10^3$)', labelpad=13, size=19)

ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
# ax.set_title('Loss landscape of the baseline', fontsize=20)

# Add a color bar which maps values to colors.
cbar = fig.colorbar(surf, shrink=0.5, aspect=5, location='left', pad=0.02)
cbar.ax.tick_params(labelsize=16)
plt.savefig('land_PIPDres.pdf', bbox_inches='tight', dpi=300)
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# lams = np.load('lams_temp.npy')
# loss_list = np.load('loss_list_temp.npy')
# plt.plot(lams, loss_list)
# plt.ylabel('Loss')
# plt.xlabel('Perturbation')
# plt.title('Loss landscape perturbed based on top Hessian eigenvector')
# plt.show()