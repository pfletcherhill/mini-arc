from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

COLORS = [
    "#252525",  # black
    "#0074D9",  # blue
    "#FF4136",  # red
    "#37D449",  # 2ECC40', # green
    "#FFDC00",  # yellow
    "#E6E6E6",  # grey
    "#F012BE",  # pink
    "#FF871E",  # orange
    "#54D2EB",  # 7FDBFF', # light blue
    "#8D1D2C",  # 870C25', # brown
    "#FFFFFF",
]


def visualize_grids(
    train_grids: list[dict],
    input_grid: list[list[int]],
    output_grid: Optional[list[list[int]]] = None,
):
    # Create a colormap from the list of colors
    cmap = mcolors.ListedColormap(COLORS)

    num_rows = len(train_grids) + 1

    fig, axes = plt.subplots(num_rows, 2, figsize=(8, 4 * num_rows))

    # Ensure axes is always 2D, even with only one row
    if num_rows == 1:
        axes = axes.reshape(1, 2)

    # Plot training pairs
    for i, pair in enumerate(train_grids):
        axes[i, 0].imshow(pair["input"], cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
        axes[i, 1].imshow(pair["output"], cmap=cmap, vmin=0, vmax=len(COLORS) - 1)

        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Output")

    # Plot test input grid
    axes[-1, 0].imshow(input_grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    axes[-1, 0].axis("off")
    axes[-1, 0].set_title("Test Input")

    # Plot test output grid if provided
    if output_grid is not None:
        axes[-1, 1].imshow(output_grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
        axes[-1, 1].set_title("Test Output")
    else:
        axes[-1, 1].text(0.5, 0.5, "Output Not Provided", ha="center", va="center")
        axes[-1, 1].set_title("Expected Output")
    axes[-1, 1].axis("off")

    plt.tight_layout()
    plt.show()
