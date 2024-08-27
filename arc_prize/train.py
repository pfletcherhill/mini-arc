from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from arc_prize.data import (
    ARCDatasetParams,
    make_data_loaders,
    make_re_arc_data_loaders,
)
from arc_prize.model import (
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
    ARCVisionEncoderDecoder,
)


@dataclass(frozen=True)
class ARCTrainParams:
    batch_size: int
    learning_rate: float
    weight_decay: float
    dataset_dir: list[str]
    loss_class_weights: Optional[dict[int, float]] = None


@dataclass(frozen=True)
class EpochState:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    grad_norm: float
    param_norm: float


@dataclass
class ARCModelState:
    model_params: ARCTransformerEncoderDecoderParams
    model_state_dict: Optional[dict]
    train_params: ARCTrainParams
    optimizer_state_dict: Optional[dict]
    epochs: list[EpochState]
    best_val_loss: float
    # encoder_attn_weights: Optional[list] = None # Too large to keep


def calculate_grad_norm(model: ARCTransformerEncoderDecoder):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def calculate_param_norm(model: ARCTransformerEncoderDecoder):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def train_arc_transformer(
    model_filename: str,
    num_epochs: int,
    patience: int = 10,
    train_params: Optional[ARCTrainParams] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.mha.set_fastpath_enabled(False)

    checkpoint_dict = torch.load(model_filename, weights_only=False)
    checkpoint = ARCModelState(**checkpoint_dict)

    # model = ARCTransformerEncoderDecoder(checkpoint.model_params)
    model = ARCVisionEncoderDecoder(checkpoint.model_params)
    if checkpoint.model_state_dict is not None:
        model.load_state_dict(checkpoint.model_state_dict)

    parallel_model = nn.DataParallel(model)

    print(f"Using {torch.cuda.device_count()} GPUs in parallel")

    parallel_model = parallel_model.to(device)

    dataset_params = ARCDatasetParams(
        max_grid_size=parallel_model.module.grid_dim,
        max_train_grids=parallel_model.module.num_train_pairs,
        color_offset=1,
    )

    train_loader, val_loader = make_data_loaders(
        checkpoint.train_params.dataset_dir,
        checkpoint.train_params.batch_size,
        dataset_params,
    )
    # train_loader, val_loader = make_re_arc_data_loaders(
    #     checkpoint.train_params.dataset_dir,
    #     checkpoint.train_params.batch_size,
    #     dataset_params,
    # )

    print(
        f"Starting training run with dataset of {len(train_loader.dataset)} training items and {len(val_loader.dataset)} evaluation items"
    )

    train_params = train_params or checkpoint.train_params

    class_weights = torch.ones(parallel_model.module.num_classes).to(device)
    if train_params.loss_class_weights is not None:
        for cls, weight in train_params.loss_class_weights.items():
            class_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        parallel_model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )
    if checkpoint.optimizer_state_dict is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    def save_checkpoint():
        torch.save(checkpoint.__dict__, model_filename)
        print("Saved checkpoint", model_filename)

    total_epochs = len(checkpoint.epochs) + num_epochs
    epochs_without_improvement = 0

    for epoch in range(len(checkpoint.epochs), total_epochs):
        parallel_model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for batch in train_loader:
            # grids, masks, target_grid = batch
            grids, masks, target_grid = [item.to(device) for item in batch]

            optimizer.zero_grad()
            output = parallel_model.forward(grids, masks)[0]

            loss = criterion(
                output.view(-1, parallel_model.module.num_classes),
                target_grid.view(-1).long(),
            )
            loss.backward()

            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(output, dim=-1)
            train_accuracy += (predictions == target_grid).float().mean().item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        # Validation
        parallel_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # grids, masks, target_grid = batch
                grids, masks, target_grid = [item.to(device) for item in batch]

                output = parallel_model.forward(grids, masks)[0]
                loss = criterion(
                    output.view(-1, parallel_model.module.num_classes),
                    target_grid.view(-1).long(),
                )

                val_loss += loss.item()

                predictions = torch.argmax(output, dim=-1)
                val_accuracy += (predictions == target_grid).float().mean().item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        # Learning rate scheduling
        scheduler.step(val_loss)

        param_group = optimizer.param_groups[0]
        beta1, beta2 = param_group["betas"]

        checkpoint.epochs.append(
            EpochState(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                lr=param_group["lr"],
                beta1=beta1,
                beta2=beta2,
                epsilon=param_group["eps"],
                weight_decay=param_group["weight_decay"],
                grad_norm=calculate_grad_norm(parallel_model),
                param_norm=calculate_param_norm(parallel_model),
            )
        )

        # for batch in encoder_attn_weights:
        #     for layer in batch:
        #         visualize_all_heads(layer)

        print(f"Epoch {epoch+1}/{total_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_loss < checkpoint.best_val_loss:
            checkpoint.best_val_loss = val_loss
            checkpoint.model_state_dict = parallel_model.module.state_dict()
            # checkpoint.encoder_attn_weights = encoder_attn_weights
            epochs_without_improvement = 0
            print("New best val loss", val_loss)
        else:
            epochs_without_improvement += 1

        checkpoint.optimizer_state_dict = optimizer.state_dict()
        save_checkpoint()

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs without improvement"
            )
            break

    print("Training completed")


def train_on_mac(
    model_name: str,
    num_epochs: int,
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[ARCTrainParams] = None,
):
    model_filename = f"models/{model_name}.pth"

    if model_params is not None and train_params is not None:
        print("Starting new model", model_name)
        model_state = ARCModelState(
            model_state_dict=None,
            model_params=model_params,
            train_params=train_params,
            optimizer_state_dict=None,
            epochs=[],
            best_val_loss=float("inf"),
        )
        torch.save(model_state.__dict__, model_filename)

    return train_arc_transformer(model_filename, num_epochs, train_params=train_params)
