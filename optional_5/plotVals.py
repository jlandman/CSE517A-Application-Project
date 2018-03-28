import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

with open('svm_vals.txt') as f:
    svm_vals = np.loadtxt(f)

with open('lr_vals.txt') as f:
    lr_vals = np.loadtxt(f)

C = np.array([2**i for i in xrange(-3, 3)])
gamma = np.array([2**i for i in xrange(-8,-1)])

alphas = 10.0**np.arange(-6, 5, 0.05)

fig = plt.figure()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
"""
plt.semilogx(alphas, lr_vals)
plt.grid(True)
plt.xlabel(r'\alpha')
plt.ylabel(r'Validation Error')
plt.title(r'Validation Error as a Function of \alpha')
"""

ax = fig.gca(projection='3d')


X, Y = np.meshgrid(C, gamma)
surf = ax.plot_surface(
            X, Y, svm_vals.T,
            rstride=1, cstride=1,
            cmap='viridis', edgecolor='none',
            linewidth=0, antialiased=True
        )

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel(r'C')
ax.set_ylabel(r'\gamma')
ax.set_zlabel(r'Validation Error')
ax.set_title(r'10-Fold CV Error of SVM with RBF Kernel')
# """
plt.show()