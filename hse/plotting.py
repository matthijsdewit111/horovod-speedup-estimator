from copy import copy

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_surface(ax, X, Y, Z, label):
    surf = ax.plot_surface(X, Y, Z, alpha=0.5, label=label)
    # hack to plot legend in 3d plot
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d


def plot_prediction(function, xdata, popt, X, Y, means):
    Z = function(xdata, *popt).reshape(means.shape)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    plot_surface(ax, X, Y, Z, 'prediction')
    plot_surface(ax, X, Y, means, 'real data')

    ax.set_xlabel('#MPI processes')
    ax.set_ylabel('num parameters')
    ax.set_zlabel('communication time')
    ax.legend()
    plt.show()


def plot_predicted_speedup(speedup, x_values, y_values, save_as=None, cmap_name='plasma', title=''):
    cmap = copy(cm.get_cmap(cmap_name))
    cmap.set_bad(color='black')

    plt.imshow(speedup, cmap=cmap, origin='lower', vmin=1, aspect='auto',
               extent=(min(x_values) - .5, max(x_values) + .5, min(y_values) - .5, max(y_values) + .5))
    plt.colorbar(label='speedup')
    plt.title(title)
    plt.xlabel('batch size')
    plt.ylabel('#MPI processes')
    # plt.xticks(np.arange(min(x_values), max(x_values) + 1, 5))
    # plt.yticks(np.arange(min(y_values), max(y_values) + 1, 4))
    plt.grid(linewidth=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()
