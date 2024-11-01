import copy
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from arc_prize.data import (
    ARCDatasetParams,
    DistributedRandomSampler,
    FinetuneDataset,
    collate_arc_fn,
    make_datasets,
)
from arc_prize.model import (
    ARCTransformerEncoder,
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
    meta_batch_size: Optional[int] = None
    meta_learning_rate: Optional[float] = None
    meta_weight_decay: Optional[float] = None
    meta_num_epochs: Optional[int] = None
    train_steps_per_epoch: Optional[int] = None
    eval_steps_per_epoch: Optional[int] = None


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
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ARCModelState:
    model_params: ARCTransformerEncoderDecoderParams
    model_type: Optional[str]
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


def fine_tune_transformer(
    model: nn.Module, train_params: ARCTrainParams, dataset: FinetuneDataset
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.mha.set_fastpath_enabled(False)

    model = copy.deepcopy(model)
    model = model.to(device)

    batch_size = 4

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    print(f"Starting fine-tuning run with dataset of {len(dataset)} training items")
    print(f"Using batch size of {batch_size}")

    class_weights = torch.ones(model.num_classes).to(device)
    if train_params.loss_class_weights is not None:
        for cls, weight in train_params.loss_class_weights.items():
            class_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )

    scaler = GradScaler(device.type)

    num_epochs = 10

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        start_time = time.time()

        for batch in data_loader:
            grids, masks, target_grid = [item.to(device) for item in batch]

            optimizer.zero_grad()

            with autocast(device.type):
                output = model.forward(grids, masks)[0]
                loss = criterion(
                    output.view(-1, model.num_classes),
                    target_grid.view(-1).long(),
                )

            # Use the scaler for backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            predictions = torch.argmax(output, dim=-1)
            train_accuracy += (predictions == target_grid).float().mean().item()

        train_loss /= len(data_loader)
        train_accuracy /= len(data_loader)

        end_time = time.time()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        duration = end_time - start_time
        print(f"Epoch duration: {duration:.2f}s ({(duration / 60):.2f}m)")

    print("Fine-tuning completed")
    return model


def train_arc_transformer(
    model_filename: str,
    num_epochs: int,
    patience: int = 10,
    train_params: Optional[ARCTrainParams] = None,
    force_compile: bool = False,
    local_rank: Optional[int] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_compilation = force_compile or device.type == "cuda"
    # use_compilation = False  # Fix later

    use_distributed = torch.cuda.device_count() > 0 and local_rank is not None
    local_rank = local_rank if use_distributed else 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.backends.mha.set_fastpath_enabled(False)

    checkpoint_dict = torch.load(model_filename, weights_only=False)
    checkpoint = ARCModelState(**checkpoint_dict)

    if checkpoint.model_type == "vision":
        model = ARCVisionEncoderDecoder(checkpoint.model_params)
    elif checkpoint.model_type == "encoder":
        model = ARCTransformerEncoder(checkpoint.model_params)
    else:
        model = ARCTransformerEncoderDecoder(checkpoint.model_params)

    if checkpoint.model_state_dict is not None:
        model.load_state_dict(checkpoint.model_state_dict)

    model = model.to(device)

    if use_compilation:
        try:
            model = torch.compile(model, mode="default", dynamic=True, fullgraph=True)
            print("Successfully compiled model")
        except Exception as e:
            print(
                f"Warning: Model compilation failed, falling back to eager mode. Error: {str(e)}"
            )

    if use_distributed:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[local_rank])
        print(f"Using DistributedDataParallel on GPU {local_rank}", flush=True)
    else:
        print(f"Using single {'GPU' if device.type == 'cuda' else 'CPU'}")

    compiled_model = model

    base_model = model.module if hasattr(model, "module") else model
    shapes = {
        "grid_dim": base_model.grid_dim,
        "num_train_pairs": base_model.num_train_pairs,
        "num_classes": base_model.num_classes,
    }

    dataset_params = ARCDatasetParams(
        max_grid_size=shapes["grid_dim"],
        max_train_grids=shapes["num_train_pairs"],
        color_offset=1,
    )

    train_params = train_params or checkpoint.train_params

    train_dataset, val_dataset = make_datasets(train_params.dataset_dir, dataset_params)

    dataset_dir_names = ", ".join(train_params.dataset_dir)

    print(
        f"Starting training run with dataset of {len(train_dataset)} training items and {len(val_dataset)} evaluation items: {dataset_dir_names}",
        flush=True,
    )
    print(f"Using batch size of {train_params.batch_size}", flush=True)

    num_replicas = max(torch.cuda.device_count(), 1)
    train_sampler = DistributedRandomSampler(
        train_dataset,
        num_samples=(
            (train_params.train_steps_per_epoch * train_params.batch_size)
            if train_params.train_steps_per_epoch is not None
            else None
        ),
        num_replicas=num_replicas,
        rank=local_rank,
    )
    val_sampler = DistributedRandomSampler(
        val_dataset,
        num_samples=(
            (train_params.eval_steps_per_epoch * train_params.batch_size)
            if train_params.eval_steps_per_epoch is not None
            else None
        ),
        num_replicas=num_replicas,
        rank=local_rank,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params.batch_size,
        sampler=train_sampler,
        num_workers=4,
        collate_fn=collate_arc_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params.batch_size,
        sampler=val_sampler,
        num_workers=4,
        collate_fn=collate_arc_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    class_weights = torch.ones(shapes["num_classes"]).to(device)
    if train_params.loss_class_weights is not None:
        for cls, weight in train_params.loss_class_weights.items():
            class_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        compiled_model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )
    if checkpoint.optimizer_state_dict is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    scaler = GradScaler(device.type)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    def forward_pass(model: nn.Module, grids: torch.Tensor, masks: torch.Tensor):
        output = model(grids, masks)
        return output[0] if isinstance(output, tuple) else output

    def training_step(
        model: nn.Module,
        grids: torch.Tensor,
        masks: torch.Tensor,
        target_grid: torch.Tensor,
    ):
        with autocast(device.type):
            output = forward_pass(model, grids, masks)
            loss = criterion(
                output.view(-1, shapes["num_classes"]),
                target_grid.view(-1).long(),
            )
        predictions = torch.argmax(output, dim=-1)
        accuracy = (predictions == target_grid).float().mean()
        return output, loss, accuracy

    def validation_step(
        model: nn.Module,
        grids: torch.Tensor,
        masks: torch.Tensor,
        target_grid: torch.Tensor,
    ):
        with autocast(device.type):
            output = forward_pass(model, grids, masks)
            loss = criterion(
                output.view(-1, shapes["num_classes"]),
                target_grid.view(-1).long(),
            )
        predictions = torch.argmax(output, dim=-1)
        accuracy = (predictions == target_grid).float().mean()
        return output, loss, accuracy

    def save_checkpoint():
        torch.save(checkpoint.__dict__, model_filename)
        print("Saved checkpoint", model_filename, flush=True)

    total_epochs = len(checkpoint.epochs) + num_epochs
    epochs_without_improvement = 0

    for epoch in range(len(checkpoint.epochs), total_epochs):
        train_sampler.set_epoch(epoch)
        compiled_model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        start_time = time.time()

        for batch in train_loader:
            grids, masks, target_grid = [item.to(device) for item in batch]

            optimizer.zero_grad()

            _, loss, accuracy = training_step(compiled_model, grids, masks, target_grid)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_accuracy += accuracy.item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        if use_distributed:
            train_metrics = torch.tensor([train_loss, train_accuracy], device=device)
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
            train_metrics /= dist.get_world_size()
            train_loss, train_accuracy = train_metrics.tolist()

        val_sampler.set_epoch(epoch)
        compiled_model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for batch in val_loader:
                grids, masks, target_grid = [item.to(device) for item in batch]

                _, loss, accuracy = validation_step(
                    compiled_model, grids, masks, target_grid
                )

                val_loss += loss.item()
                val_accuracy += accuracy.item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        if use_distributed:
            val_metrics = torch.tensor([val_loss, val_accuracy], device=device)
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
            val_metrics /= dist.get_world_size()
            val_loss, val_accuracy = val_metrics.tolist()

        # Learning rate scheduling
        scheduler.step(val_loss)

        param_group = optimizer.param_groups[0]
        beta1, beta2 = param_group["betas"]

        end_time = time.time()

        if local_rank == 0:
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
                    grad_norm=calculate_grad_norm(compiled_model),
                    param_norm=calculate_param_norm(compiled_model),
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            print(f"Epoch {epoch+1}/{total_epochs} for {model_filename}:", flush=True)
            print(
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Steps: {len(train_loader)}",
                flush=True,
            )
            print(
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Steps: {len(val_loader)}",
                flush=True,
            )

            duration = end_time - start_time
            print(
                f"Epoch duration: {duration:.2f}s ({(duration / 60):.2f}m)", flush=True
            )

            base_model = (
                compiled_model.module
                if hasattr(compiled_model, "module")
                else compiled_model
            )

            # Save the best model
            if val_loss < checkpoint.best_val_loss:
                checkpoint.best_val_loss = val_loss
                checkpoint.model_state_dict = base_model.state_dict()
                # checkpoint.encoder_attn_weights = encoder_attn_weights
                epochs_without_improvement = 0
                print("New best val loss", val_loss, flush=True)
            else:
                epochs_without_improvement += 1

            checkpoint.optimizer_state_dict = optimizer.state_dict()

            save_checkpoint()

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} epochs without improvement",
                flush=True,
            )
            break

    print("Training completed")


def train_on_mac(
    model_name: str,
    num_epochs: int,
    model_type: Optional[str] = "normal",
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[ARCTrainParams] = None,
):
    model_filename = f"models/{model_name}.pth"

    if model_params is not None and train_params is not None:
        print("Starting new model", model_name)
        model_state = ARCModelState(
            model_type=model_type,
            model_state_dict=None,
            model_params=model_params,
            train_params=train_params,
            optimizer_state_dict=None,
            epochs=[],
            best_val_loss=float("inf"),
        )
        torch.save(model_state.__dict__, model_filename)

    return train_arc_transformer(model_filename, num_epochs, train_params=train_params)
