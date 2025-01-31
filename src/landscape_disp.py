import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FuncFormatter

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
X = Y = lams
X, Y = np.meshgrid(X, Y)
# Z = np.load('land_vanilla.npy')
Z = np.load('land_physical.npy')
Z = Z/1e5
# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
ax.set_xlabel('v2', labelpad=13, size=20)
ax.set_ylabel('v1', labelpad=13, size=20)
ax.set_zlabel('Loss value ($\\times 10^5$)', labelpad=13, size=20)

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

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