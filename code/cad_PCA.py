"""
(Helper) code for the PCA exercise in ../notebooks/PCA_exercise.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


def reshape_and_rescale(x):
    x = x.reshape((24, 24, 3))
    return ((x - x.min()) * (255 / (x.max() - x.min()))).astype('uint8')


def reconstruction_demo(X, pca, stochastic=True):
    n_samples, n_features = X.shape
    mu = np.mean(X, axis=0)  # Needed later for image reconstruction
    explained_r = np.cumsum(pca.explained_variance_ratio_)

    # Select 5 random images
    ix = np.random.randint(0, n_samples, 5) if stochastic else np.arange(0, 5)

    # Make plotting architecture
    fig, axes = plt.subplots(
            nrows=3, ncols=5, figsize=(12, 12),
            gridspec_kw={'height_ratios': [2, 1, 1]})

    # Merge top axes into one big plot
    gs = axes[0, 0].get_gridspec()
    for ax in axes[0, :5]:
        ax.remove()
    axbig = fig.add_subplot(gs[0, :5], xlim=(1, 200), ylim=(0, 1))
    axbig.set_xlabel("Number of principle components", fontweight='bold', fontsize=13)
    axbig.set_ylabel("Retained variance (%)", fontweight='bold', fontsize=13)

    x_big = np.arange(1, n_features + 1)
    r = np.full((n_features, 1), np.nan)
    r_curve, = axbig.plot(x_big, r)

    # Show original images
    original_ims = [im.reshape((24, 24, 3)).astype('uint8') for im in X[ix, :]]
    for ax, im in zip(axes[2, :], original_ims):
        ax.imshow(im)
        ax.axis('off')
    axes[2, 2].text(0.5, -0.2, "Original images",
                    fontdict={'weight': 'bold', 'size': 13},
                    ha='center',
                    transform=axes[2, 2].transAxes)

    # Show reconstructed images
    reconstructed_ims = [np.full((24, 24, 3), np.nan) for _ in range(5)]
    image_plots = []
    for ax, im in zip(axes[1, :], reconstructed_ims):
        im = ax.imshow(im)
        ax.axis('off')
        image_plots.append(im)
    axes[1, 3].text(0.5, -0.2, "Reconstructed images",
                    fontdict={'weight': 'bold', 'size': 13},
                    ha='center',
                    transform=axes[1, 2].transAxes)

    # Reconstruct images with variable amount of PCs
    pcs = 1
    X_pca = pca.transform(X)
    for i in range(200):

        # Reconstruct images
        X_rec = X_pca[:, :pcs].dot(pca.components_[:pcs, :])
        X_rec += mu

        clear_output(wait=True)

        # Update reconstructed images
        ims = X_rec[ix, :]
        reconstructed_ims = [reshape_and_rescale(im) for im in ims]
        for plot, im in zip(image_plots, reconstructed_ims):
            plot.set_data(im)

        # Update r_curve
        r[i] = explained_r[i]
        r_curve.set_ydata(r)

        pcs += 1
        display(fig)
