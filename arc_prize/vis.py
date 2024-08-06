from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from arc_prize.train import EpochState

COLORS = [
    "#DADADA",  # padding grey
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


def visualize_epochs(epochs: dict[str, list[EpochState]]):
    plt.figure(figsize=(10, 5))
    for k, v in epochs.items():
        train_loss = []
        eval_loss = []
        for epoch in v:
            train_loss.append(epoch.train_loss)
            eval_loss.append(epoch.val_loss)
        # plt.plot(train_loss, label=f"{k} Training Loss")
        plt.plot(eval_loss, label=f"{k} Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Evaluation Losses")
    plt.show()


def visualize_grids(
    train_grids: list[dict],
    input_grid: list[list[int]],
    output_grid: Optional[list[list[int]]] = None,
):
    # Create a colormap from the list of colors
    cmap = mcolors.ListedColormap(COLORS)

    num_rows = len(train_grids) + 1
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4 * num_rows))

    # Plot training pairs
    for i, pair in enumerate(train_grids):
        row = (i * 2) // num_cols
        col = (i * 2) % num_cols
        axes[row, col].imshow(pair["input"], cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
        axes[row, col + 1].imshow(
            pair["output"], cmap=cmap, vmin=0, vmax=len(COLORS) - 1
        )

        axes[row,].axis("off")

    # Plot test input grid
    axes[num_rows - 1, 0].imshow(input_grid, cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    axes[num_rows - 1,].axis("off")
    axes[num_rows - 1, 0].set_title("Test Input")

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


def visualize_mask(mask, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask.cpu().numpy(), cmap="YlGnBu", cbar=True)
    plt.title(title)
    plt.show()


def visualize_tensors(
    grids: torch.Tensor,
    output_grid: torch.Tensor,
    prediction: torch.Tensor,
):
    # Create a colormap from the list of colors
    cmap = mcolors.ListedColormap(COLORS)

    # Determine the number of input/output pairs
    num_pairs = (grids.shape[0] - 1) // 2  # Subtract 1 for the test input

    # Calculate the number of rows needed
    num_cols = 4
    num_rows = (num_pairs // (num_cols // 2)) + 1  # +1 for the test row

    # Create a figure with the appropriate number of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))

    def plot_grid(ax, grid):
        ax.imshow(grid.cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.5)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )

    # Plot input/output training pairs
    for i in range(num_pairs):
        row = (i * 2) // num_cols
        col = (i * 2) % num_cols

        plot_grid(axes[row, col], grids[2 * i])
        plot_grid(axes[row, col + 1], grids[2 * i + 1])
        # axes[row, col].imshow(
        #     grids[2 * i].cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1
        # )
        # axes[row, col + 1].imshow(
        #     grids[2 * i + 1].cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1
        # )

        if i == 0:
            axes[row, 0].set_title("Input")
            axes[row, 1].set_title("Output")

    # Plot test input grid
    plot_grid(axes[-1, 0], grids[-1])
    # axes[-1, 0].imshow(grids[-1].cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    axes[-1, 0].set_title("Test Input")

    # Plot test output grid and prediction side by side

    # test_output_ax.imshow(output_grid.cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    plot_grid(axes[-1, 1], output_grid)
    axes[-1, 1].set_title("Expected Output")

    # Add prediction as a small subplot
    # pred_ax = fig.add_axes([0.75, 0.125, 0.2, 0.2])  # [left, bottom, width, height]
    # pred_ax.imshow(prediction.cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    plot_grid(axes[-1, 2], prediction)
    axes[-1, 2].set_title("Prediction")

    plt.tight_layout()
    plt.show()


def visualize_output_query(output_query):
    # Convert to numpy for easy handling with matplotlib
    output_query_np = output_query.detach().cpu().numpy()

    # Plot the mean value of each dimension
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(output_query_np, axis=1).squeeze())
    plt.title("Mean Value of Output Query Dimensions")
    plt.xlabel("Dimension")
    plt.ylabel("Mean Value")
    plt.show()

    print(output_query_np.shape)

    # Plot the values as a heatmap (if grid_dim is small enough)
    # if output_query_np.shape[1] <= 64:  # Adjust threshold as needed
    plt.figure(figsize=(10, 10))
    plt.imshow(output_query_np.squeeze(), cmap="viridis")
    plt.title("Output Query Heatmap")
    plt.colorbar()
    plt.show()
