import jax.numpy as jnp
from matplotlib import pyplot as plt


def display_grid(grid: jnp.ndarray):
    fig, ax = plt.subplots(figsize=(12, 12))

    im = ax.imshow(grid, cmap="binary", interpolation="nearest")

    plt.colorbar(im, ax=ax, label="Boolean value")

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()
