import random
import string

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from arc_prize.data import ARCDataset
from arc_prize.env import modal_app
from arc_prize.model import ARCTransformer


def masked_cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor, mask):
    B, H, W, C = predictions.shape
    predictions = predictions.contiguous().view(B * H * W, C)
    targets = targets.long().contiguous().view(B * H * W)
    mask = mask.contiguous().view(B * H * W)

    loss = nn.CrossEntropyLoss(reduction="none")(predictions, targets)
    masked_loss = (loss * mask.float()).sum() / mask.float().sum()
    return masked_loss


# TODO: combined masked and weighted, making the weight a param
def weighted_cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor, mask):
    B, H, W, C = predictions.shape
    predictions = predictions.contiguous().view(B * H * W, C)
    targets = targets.long().contiguous().view(B * H * W)
    mask = mask.contiguous().view(B * H * W)

    # Assign a lower weight to padded cells (e.g., 0.1) and full weight to non-padded cells
    weights = torch.where(mask == 1, torch.tensor(1.0), torch.tensor(0.1))

    loss = nn.CrossEntropyLoss(reduction="none")(predictions, targets)
    weighted_loss = (loss * weights).mean()
    return weighted_loss


def unmasked_cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor):
    B, H, W, C = predictions.shape
    predictions = predictions.contiguous().view(B * H * W, C)
    targets = targets.long().contiguous().view(B * H * W)

    loss = nn.CrossEntropyLoss()(predictions, targets)
    return loss


@modal_app.function(gpu="t4")
def train_arc_transformer(
    model: ARCTransformer,
    train_loader: DataLoader[ARCDataset],
    val_loader: DataLoader[ARCDataset],
    num_epochs: int,
    learning_rate: float,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, batch in enumerate(train_loader):
            grids, grid_masks, output_grid = [item.to(device) for item in batch]

            optimizer.zero_grad()

            predictions = model(grids, grid_masks, output_grid)

            loss = unmasked_cross_entropy_loss(predictions, output_grid)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # if (i + 1) % 10 == 0:  # Print every 10 batches
            print(
                f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
            )

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                grids, grid_masks, output_grid = [item.to(device) for item in batch]

                predictions = model(grids, grid_masks)

                loss = unmasked_cross_entropy_loss(predictions, output_grid)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    model_file_name = f"tmp_model_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}.pth"
    print(model_file_name)
    torch.save(model.state_dict(), model_file_name)

    return model, history