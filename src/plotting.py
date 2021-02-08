import numpy as np
import matplotlib.pyplot as plt


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(len(np.unique(targets))):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.25)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(targets)


def plot_decision_boundaries(
    model, X, y, n_activations=None, xlim=None, ylim=None
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(*list(zip(*X)), c=y, cmap='RdBu', alpha=0.25)

    # plot the decision function
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    XY = np.vstack([XX.ravel(), YY.ravel()]).T

    # plot linear region boundaries
    activations = model.activations(XY)
    colors = ['red', 'blue', 'black']
    for i in range(len(activations[:n_activations])):
        acts = np.hstack(activations[:len(activations)-i])
        Z = np.unique(acts, axis=0, return_inverse=True)[1].reshape(XX.shape)
        ax.contour(XX, YY, Z, colors=colors[i], alpha=0.5)

    plt.show()
