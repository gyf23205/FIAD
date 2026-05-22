import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FuncFormatter
from matplotlib import rcParams
rcParams['text.usetex'] = True

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(13, 8))
method = 0
# Make data.
lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
X = Y = lams
X, Y = np.meshgrid(X, Y)
if method == 0:
    Z = np.load('land_vanilla.npy')
elif method == 1:
    Z = np.load('land_physical.npy')
elif method==2:
    Z = np.load('land_physical_res.npy')
else:
    pass
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
# ax.tick_params(axis='z', pad=10)

# Add a color bar which maps values to colors.
cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.02, location='left')
cbar.ax.tick_params(labelsize=18)
if method == 0:
   plt.savefig('land_base.pdf', dpi=300)
elif method == 1:
    plt.savefig('land_PIPDall.pdf', dpi=300)
elif method==2:
    plt.savefig('land_PIPDres.pdf', dpi=300)
else:
    pass

plt.show()