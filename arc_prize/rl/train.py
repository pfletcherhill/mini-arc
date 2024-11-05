import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from arc_prize.data import (
    ARCDatasetParams,
    DistributedRandomSampler,
    collate_arc_fn,
    make_datasets,
)
from arc_prize.model import ARCVisionEncoder
from arc_prize.rl.model import ValueNetwork, ValueNetworkParams
from arc_prize.train import ARCTransformer, load_model_from_checkpoint


@dataclass
class SearchParams:
    temperatures: list[float] = [0.1, 0.3, 0.6, 0.9]
    beam_width: int = 3
    num_samples_per_temperature: int = 3
    max_depth: int = 5


@dataclass(frozen=True)
class ValueNetworkTrainParams:
    batch_size: int
    learning_rate: float
    weight_decay: float
    model_filename: str
    dataset_dirs: list[str]
    search_params: SearchParams
    loss_class_weights: Optional[dict[int, float]] = None
    train_steps_per_epoch: Optional[int] = None
    eval_steps_per_epoch: Optional[int] = None
    warmup_epochs: Optional[int] = None


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
class ValueNetworkTrainState:
    model_params: ValueNetworkParams
    model_state_dict: Optional[dict]
    train_params: ValueNetworkTrainParams
    optimizer_state_dict: Optional[dict]
    epochs: list[EpochState]
    best_val_loss: float


@dataclass
class TrajectoryCandidate:
    predictions: torch.Tensor
    values: torch.Tensor
    accuracy: Optional[torch.Tensor] = None


@dataclass
class Trajectory:
    steps: list[TrajectoryCandidate]


def generate_candidates(
    model: ARCVisionEncoder,
    value_network: ValueNetwork,
    grids: torch.Tensor,
    masks: torch.Tensor,
    target_grids: torch.Tensor,
    temperatures: list[float],
    output: Optional[torch.Tensor] = None,
    num_samples_per_temperature: int = 3,
) -> list[TrajectoryCandidate]:
    candidates: list[TrajectoryCandidate] = []
    for temp in temperatures:
        with torch.no_grad():
            logits, _ = model.forward(grids, masks, tgt=output, temperature=temp)

        probs = torch.softmax(logits, dim=-1)

        samples = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=num_samples_per_temperature,
            replacement=True,
        ).view(-1, *probs.size()[:-1])

        for predictions in samples:
            seq_embedded, mask_embedded = model.embed(grids, masks, output)

            value_logits = value_network.forward(seq_embedded, mask_embedded)

            accuracy = (predictions == target_grids).float().mean()
            candidate = TrajectoryCandidate(
                predictions=predictions, values=value_logits, accuracy=accuracy
            )
            candidates.append(candidate)

    return candidates


def calculate_trajectory_score(trajectory: Trajectory) -> float:
    # TODO: update with momentum
    length_penalty = len(trajectory.steps) * 0.02
    last_value = trajectory.steps[-1].values.item()
    return last_value - length_penalty


def search_and_predict(
    model: ARCVisionEncoder,
    value_network: ValueNetwork,
    grids: torch.Tensor,
    masks: torch.Tensor,
    target_grids: torch.Tensor,
    temperatures: list[float],
    output: Optional[torch.Tensor] = None,
    beam_width: int = 3,
    max_depth: int = 5,
    num_samples_per_temperature: int = 3,
):
    # Get initial candidates
    candidates = generate_candidates(
        model,
        value_network,
        grids,
        masks,
        target_grids,
        temperatures,
        output=output,
        num_samples_per_temperature=num_samples_per_temperature,
    )

    best_candidate = max(candidates, key=lambda x: x.values.item())

    active_trajectories = [Trajectory(steps=[candidate]) for candidate in candidates]

    # Prune to top beam_width candidates based on value predictions
    active_trajectories = sorted(
        active_trajectories,
        key=lambda x: calculate_trajectory_score(x),
        reverse=True,
    )[:beam_width]

    # Iterative refinement
    for _ in range(max_depth):
        new_trajectories = []

        # Generate candidates from each active trajectory
        for trajectory in active_trajectories:
            candidates = generate_candidates(
                model,
                value_network,
                grids,
                masks,
                target_grids,
                temperatures,
                output=trajectory.steps[-1].predictions,
                num_samples_per_temperature=num_samples_per_temperature,
            )
            for candidate in candidates:
                if candidate.values.item() > best_candidate.values.item():
                    best_candidate = candidate
                new_trajectory = Trajectory(steps=[*trajectory.steps, candidate])
                new_trajectories.append(new_trajectory)

        # If no new candidates, stop
        if not new_trajectories:
            break

        # Prune to top beam_width candidates
        active_trajectories = sorted(
            new_trajectories,
            key=lambda x: calculate_trajectory_score(x),
            reverse=True,
        )[:beam_width]

    return best_candidate.predictions


def load_value_network_from_checkpoint(
    filename: str,
) -> tuple[ValueNetwork, ValueNetworkTrainState]:
    checkpoint_dict = torch.load(filename, weights_only=False)
    checkpoint = ValueNetworkTrainState(**checkpoint_dict)

    value_network = ValueNetwork(checkpoint.model_params)

    if checkpoint.model_state_dict is not None:
        value_network.load_state_dict(checkpoint.model_state_dict)

    return value_network, checkpoint


def train_arc_rl(
    value_network_filename: str,
    num_epochs: int,
    patience: int = 10,
    train_params: Optional[ValueNetworkTrainParams] = None,
    local_rank: Optional[int] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_distributed = torch.cuda.device_count() > 0 and local_rank is not None
    local_rank = local_rank if use_distributed else 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.backends.mha.set_fastpath_enabled(False)

    value_network, checkpoint = load_value_network_from_checkpoint(
        value_network_filename
    )

    value_network = value_network.to(device)

    if use_distributed:
        from torch.nn.parallel import DistributedDataParallel

        value_network = DistributedDataParallel(
            value_network, device_ids=[local_rank], find_unused_parameters=True
        )
        print(f"Using DistributedDataParallel on GPU {local_rank}", flush=True)
    else:
        print(f"Using single {'GPU' if device.type == 'cuda' else 'CPU'}")

    base_value_network = (
        value_network.module if hasattr(value_network, "module") else value_network
    )

    train_params = train_params or checkpoint.train_params

    model, model_checkpoint = load_model_from_checkpoint(train_params.model_filename)
    model = model.to(device)

    dataset_params = ARCDatasetParams(
        max_grid_size=model.grid_dim,
        max_train_grids=model.num_train_pairs,
        color_offset=1,
    )

    train_dataset, val_dataset = make_datasets(
        train_params.dataset_dirs, dataset_params
    )

    dataset_dir_names = ", ".join(train_params.dataset_dirs)

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

    # class_weights = torch.ones(model.num_classes).to(device)
    # if train_params.loss_class_weights is not None:
    #     for cls, weight in train_params.loss_class_weights.items():
    #         class_weights[cls] = weight
    # model_criterion = nn.CrossEntropyLoss(weight=class_weights)

    value_network_criterion = nn.MSELoss()

    epoch = len(checkpoint.epochs)

    optimizer = optim.AdamW(
        value_network.parameters(),
        lr=train_params.learning_rate,
        weight_decay=train_params.weight_decay,
    )
    if checkpoint.optimizer_state_dict is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    scaler = GradScaler(device.type)

    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    if train_params.warmup_epochs is not None:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=train_params.warmup_epochs,
            last_epoch=(epoch - 1),
        )
    else:
        warmup_scheduler = None

    def update_scheduler(epoch: int, loss: float):
        if (
            train_params.warmup_epochs is not None
            and warmup_scheduler is not None
            and epoch <= train_params.warmup_epochs
        ):
            warmup_scheduler.step()

        else:
            plateau_scheduler.step(loss)

    def training_step(
        grids: torch.Tensor,
        masks: torch.Tensor,
        target_grid: torch.Tensor,
    ):
        output = search_and_predict(
            model,
            value_network,
            grids,
            masks,
            target_grid,
            train_params.search_params.temperatures,
            output=None,
            beam_width=train_params.search_params.beam_width,
            max_depth=train_params.search_params.max_depth,
            num_samples_per_temperature=train_params.search_params.num_samples_per_temperature,
        )
        with autocast(device.type):
            output = forward_pass(model, grids, masks, tgt=tgt)
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
        do_refinement = (
            train_params.refinement_ratio is not None
            and random.random() < train_params.refinement_ratio
        )
        if do_refinement:
            refinement_noise_ratio = random.uniform(0.0, 0.6)
            tgt = create_noisy_tgt(
                target_grid, refinement_noise_ratio, shapes["num_classes"]
            )
        else:
            tgt = None
        with autocast(device.type):
            output = forward_pass(model, grids, masks, tgt=tgt)
            loss = criterion(
                output.view(-1, shapes["num_classes"]),
                target_grid.view(-1).long(),
            )
        predictions = torch.argmax(output, dim=-1)
        accuracy = (predictions == target_grid).float().mean()
        return output, loss, accuracy

    def save_checkpoint():
        torch.save(checkpoint.__dict__, value_network_filename)
        print("Saved checkpoint", value_network_filename, flush=True)

    total_epochs = epoch + num_epochs
    epochs_without_improvement = 0

    for epoch in range(epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        value_network.train()
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
            print(f"LR: {param_group['lr']}", flush=True)
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

        # Learning rate scheduling
        update_scheduler(epoch, val_loss)

    print("Training completed")


def train_rl_local(
    model_filename,
    num_epochs: int,
    value_network_name: str,
    model_params: Optional[ValueNetworkParams] = None,
    train_params: Optional[ValueNetworkTrainParams] = None,
    model_dir: str = "models/value_network",
):
    model, checkpoint = load_model_from_checkpoint(model_filename)
    value_network_filename = f"{model_dir}/{value_network_name}.pth"

    if model_params is not None and train_params is not None:
        print("Starting new value network", value_network_name)
        value_network_state = ValueNetworkTrainState(
            model_params=model_params,
            model_state_dict=None,
            train_params=train_params,
            optimizer_state_dict=None,
            epochs=[],
            best_val_loss=float("inf"),
        )
        torch.save(value_network_state.__dict__, value_network_filename)

    return train_arc_rl(value_network_filename, num_epochs, train_params=train_params)


# def train_on_mac(
#     model_name: str,
#     num_epochs: int,
#     model_type: Optional[str] = "normal",
#     model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
#     train_params: Optional[ARCTrainParams] = None,
# ):
#     model_filename = f"models/{model_name}.pth"

#     if model_params is not None and train_params is not None:
#         print("Starting new model", model_name)
#         model_state = ARCModelState(
#             model_type=model_type,
#             model_state_dict=None,
#             model_params=model_params,
#             train_params=train_params,
#             optimizer_state_dict=None,
#             epochs=[],
#             best_val_loss=float("inf"),
#         )
#         torch.save(model_state.__dict__, model_filename)

#     return train_arc_transformer(model_filename, num_epochs, train_params=train_params)
