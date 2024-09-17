from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from arc_prize.train import EpochState

COLORS = [
    "#c2c0c0",  # padding grey
    "#111111",  # black
    "#1E93FF",  # blue
    "#F93C31",  # red
    "#4FCC30",  # green
    "#FFDC00",  # yellow
    "#E6E6E6",  # grey
    "#E53AA3",  # magenta
    "#FF851B",  # orange
    "#87D8F1",  # light blue
    "#921231",  # maroon
    "#FFFFFF",
]


def visualize_epochs(epochs: dict[str, list[EpochState]]):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

    for k, v in epochs.items():
        train_loss = []
        eval_loss = []
        train_accuracy = []
        eval_accuracy = []
        lr = []
        grad_norm = []
        for epoch in v:
            train_loss.append(epoch.train_loss)
            eval_loss.append(epoch.val_loss)
            train_accuracy.append(epoch.train_accuracy)
            eval_accuracy.append(epoch.val_accuracy)
            lr.append(epoch.lr)
            grad_norm.append(epoch.grad_norm)

        # Plot training and evaluation loss
        ax1.plot(train_loss, label=f"{k} Training Loss", linestyle="-")
        ax1.plot(eval_loss, label=f"{k} Evaluation Loss", linestyle="--")

        ax2.plot(train_accuracy, label=f"{k} Training Accuracy", linestyle="-")
        ax2.plot(eval_accuracy, label=f"{k} Evaluation Accuracy", linestyle="--")

        # Plot grad_norm
        ax3.plot(grad_norm, label=f"{k} grad_norm", linestyle="-")

        # Plot learning rate
        ax4.plot(lr, label=f"{k} lr", linestyle=":")

    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Evaluation Losses")
    ax1.legend()

    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Evaluation Accuracies")
    ax2.legend()

    ax3.set_ylabel("Gradient Norm")
    ax3.set_title("Gradient Norm")
    ax3.legend()

    ax4.set_ylabel("Learning Rate")
    ax4.set_title("Learning Rate")
    ax4.legend()

    # Add gridlines to all subplots
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)

    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.show()


# def visualize_epochs(epochs: dict[str, list[EpochState]]):
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     ax2 = ax1.twinx()

#     for k, v in epochs.items():
#         train_loss = []
#         eval_loss = []
#         lr = []
#         weight_decay = []
#         beta1 = []
#         beta2 = []
#         epsilon = []
#         grad_norm = []
#         param_norm = []
#         for epoch in v:
#             train_loss.append(epoch.train_loss)
#             eval_loss.append(epoch.val_loss)
#             lr.append(epoch.lr)
#             weight_decay.append(epoch.weight_decay)
#             beta1.append(epoch.beta1)
#             beta2.append(epoch.beta2)
#             epsilon.append(epoch.epsilon)
#             grad_norm.append(epoch.grad_norm)
#             # param_norm.append(epoch.param_norm)

#         ax1.plot(train_loss, label=f"{k} Training Loss", linestyle="-")
#         ax1.plot(eval_loss, label=f"{k} Evaluation Loss", linestyle="--")
#         # ax1.plot(grad_norm, label=f"{k} grad_norm", linestyle="-")
#         # ax1.plot(param_norm, label=f"{k} param_norm", linestyle="-")

#         ax2.plot(lr, label=f"{k} lr", linestyle=":")
#         # ax2.plot(weight_decay, label=f"{k} weight_decay", linestyle="--")
#         # ax2.plot(beta1, label=f"{k} beta1")
#         # ax2.plot(beta2, label=f"{k} beta2")
#         # ax2.plot(epsilon, label=f"{k} epsilon")

#     # Combine legends from both axes
#     lines1, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()

#     # Place legend outside the plot
#     fig.legend(
#         lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(1, 0.5)
#     )

#     plt.xlabel("Epoch")
#     plt.title("Training and Evaluation Losses")
#     plt.show()


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
    output_grid: Optional[torch.Tensor],
    prediction: Optional[torch.Tensor],
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
    if output_grid is not None:
        plot_grid(axes[-1, 1], output_grid)
        axes[-1, 1].set_title("Expected Output")

    # Add prediction as a small subplot
    # pred_ax = fig.add_axes([0.75, 0.125, 0.2, 0.2])  # [left, bottom, width, height]
    # pred_ax.imshow(prediction.cpu(), cmap=cmap, vmin=0, vmax=len(COLORS) - 1)
    if prediction is not None:
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


def visualize_all_heads(attention_weights, title: str):
    # Get the number of heads
    num_heads = attention_weights.shape[0]
    print("num_heads", num_heads)

    # Create a grid of subplots
    fig, axes = plt.subplots(2, max(2, num_heads // 2), figsize=(20, 10))
    fig.suptitle(title)

    for head in range(num_heads):
        print("head", head)
        ax = axes[head // (num_heads // 2), head % (num_heads // 2)]
        attn = attention_weights[head].cpu().numpy()
        print("attn", attn.shape)
        sns.heatmap(attn, ax=ax, cmap="viridis")
        ax.set_title(f"Head {head}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_mean_mha_attention(attention_weights, num_grids=9, grid_size=10):
    # Reshape the attention weights
    # From [4, 100, 900] to [4, 100, 9, 10, 10]
    reshaped_attention = attention_weights.view(4, 100, num_grids, grid_size, grid_size)

    # Calculate mean attention across the target sequence (dim=1)
    mean_attention = reshaped_attention.mean(dim=1)  # Shape: [4, 9, 10, 10]

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(4, 9, figsize=(20, 10))

    for head in range(4):
        # Create a 3x3 grid of heatmaps
        for i in range(9):
            grid_attention = mean_attention[head, i]

            # Add subplot within the head's subplot
            # sub_ax = ax.inset_axes([1/9])
            ax = axes[head, i]
            im = ax.imshow(grid_attention, cmap="viridis", interpolation="nearest")
            ax.axis("off")

            if i == 0:
                ax.set_ylabel(f"Head {head + 1}", rotation=0, ha="right", va="center")

            # Add colorbar for each grid
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Remove ticks from the main subplot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def visualize_mean_sa_attention(attn_weights: torch.Tensor, grid_dim: int = 10):
    num_heads = attn_weights.shape[0]
    # Reshape the attention weights
    # From [4, 100, 900] to [4, 100, 9, 10, 10]
    reshaped_attention = attn_weights.view(num_heads, grid_dim**2, grid_dim, grid_dim)

    # Calculate mean attention across the target sequence (dim=1)
    mean_attention = reshaped_attention.mean(dim=1)

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(num_heads, 1, figsize=(20, 10))

    for head in range(num_heads):
        grid_attention = mean_attention[head]

        ax = axes[head]
        im = ax.imshow(grid_attention, cmap="viridis", interpolation="nearest")
        ax.axis("off")

        if i == 0:
            ax.set_ylabel(f"Head {head + 1}", rotation=0, ha="right", va="center")

        # Add colorbar for each grid
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Remove ticks from the main subplot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def visualize_mean_attention(attention_weights, num_grids=9, grid_size=10):
    # Reshape the attention weights
    # From [4, 100, 900] to [4, 100, 9, 10, 10]
    num_heads = attention_weights.shape[0]
    reshaped_attention = attention_weights.view(
        num_heads, -1, num_grids, grid_size, grid_size
    )

    # Calculate mean attention across the target sequence (dim=1)
    # mean_attention = reshaped_attention.mean(dim=1)  # Shape: [4, 9, 10, 10]
    mean_attention = reshaped_attention

    # Create a figure with subplots for each head
    fig, axes = plt.subplots(4, 9, figsize=(20, 10))

    for head in range(4):
        # Create a 3x3 grid of heatmaps
        for i in range(9):
            grid_attention = mean_attention[head, i]

            # Add subplot within the head's subplot
            # sub_ax = ax.inset_axes([1/9])
            ax = axes[head, i]
            im = ax.imshow(grid_attention, cmap="viridis", interpolation="nearest")
            ax.axis("off")

            if i == 0:
                ax.set_ylabel(f"Head {head + 1}", rotation=0, ha="right", va="center")

            # Add colorbar for each grid
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Remove ticks from the main subplot
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
