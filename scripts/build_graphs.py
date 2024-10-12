import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def graph_1d():
    # 1D case
    x = np.linspace(-35, 35, 3000)
    y = np.zeros_like(x)

    a = 4
    c = 5
    d = 10

    for i in range(x.shape[0]):
        n_xij_d = np.abs(x[i]) - d
        fx = a*(1 - np.exp(-n_xij_d/c))*(x[i]/(1 + np.abs(x[i])))
        y[i] = np.abs(fx)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, linewidth=2)

    # Xlim Ylim
    ax.set_xlim([0, 32])
    ax.set_ylim([-2, 12])
    # ax.set_xticks([-10, 0, 10])

    # Ticks
    # ax.set_xticks([-10, 0, 10])
    # ax.tick_params(pad=10)
    ax.set_yticks([-2,
                   0, a, a + int((10 - a)*0.5), 10])

    # Annotation
    ax.axhline(color='black', linewidth=1)
    ax.axvline(color='black', linewidth=1)
    ax.axhline(y=a, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=-a, color="gray", linestyle="--", linewidth=1)
    ax.axvline(x=d, color="gray", linestyle=":", linewidth=1)
    ax.axvline(x=-d, color="gray", linestyle=":", linewidth=1)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Force")

    ax.annotate("a = 4", [29, a+0.25])
    ax.plot([d, -d], [0, 0], 'o')
    ax.annotate("d = 10", xy=[d, 0], xytext=[d+1, -1])

    plt.show()
    # plt.savefig('force_func.pdf')


def graph_2d():
    a = 4
    c = 5
    d = 10

    def fun(x, y):
        xij = np.array([x, y])
        xij_norm = np.linalg.norm(xij)

        n_xij_d = xij_norm - d
        # fx = a*(1 - np.exp(-n_xij_d/c))

        # # Potential function
        # px = -fx * (xij/(1 + xij_norm))
        # fx = c*np.log(np.cosh(n_xij_d/c))
        fx = a*(1 - np.exp(-n_xij_d/c))*(xij/(1 + xij_norm))
        px = fx
        return np.linalg.norm(px)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    # Create a meshgrid for x and z values
    y = np.linspace(-20, 20, 100)
    z = np.linspace(0, 12, 100)
    Y, Z = np.meshgrid(y, z)
    # Set y-values to a constant value of 0
    X = np.zeros_like(Y)
    ax.plot_surface(X, Y, Z, alpha=0.75, zorder=2)

    x = y = np.arange(-20.0, 20.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, zorder=1, alpha=1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_zticks([0, 12])
    # ax.set_zticks([-2,
    #             0, a, a + int((10 - a)*0.5), 10])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Force Magnitude')
    ax.set_aspect('equal')
    ax.annotate("d = 10", xy=[d, 0], xytext=[d+1, -1])

    plt.show()


graph_1d()
