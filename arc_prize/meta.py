import time
from typing import Optional, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from arc_prize.data import (
    ARCDataset,
    ARCDatasetParams,
    FinetuneDataset,
    collate_arc_fn,
    make_data_loaders,
    make_finetune_dataset,
)
from arc_prize.model import (
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
    ARCVisionEncoderDecoder,
)
from arc_prize.train import (
    ARCModelState,
    ARCTrainParams,
    EpochState,
    calculate_grad_norm,
    calculate_param_norm,
)


def meta_fine_tune_transformer(
    model: nn.Module, train_params: ARCTrainParams, dataset: FinetuneDataset
) -> dict[str, torch.Tensor]:
    if (
        train_params.meta_batch_size is None
        or train_params.meta_learning_rate is None
        or train_params.meta_weight_decay is None
    ):
        raise Exception(
            "meta_batch_size, meta_learning_rate, and meta_weight_decay must all be defined"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        dataset,
        batch_size=train_params.meta_batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    print(f"Starting inner loop with {len(data_loader.dataset)} tasks")

    class_weights = torch.ones(model.num_classes).to(device)
    if train_params.loss_class_weights is not None:
        for cls, weight in train_params.loss_class_weights.items():
            class_weights[cls] = weight

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = train_params.meta_num_epochs or 10

    print("Inner Loop parameters")
    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    # adapted_params = {name: param for name, param in model.named_parameters()}
    # adapted_params = OrderedDict(model.named_parameters())
    params = model.named_parameters()
    adapted_params = OrderedDict(
        {name: param.to(device=device) for name, param in params if param.requires_grad}
    )

    def get_adapted_params():
        for param in adapted_params.values():
            yield param

    optimizer = optim.SGD(get_adapted_params(), lr=train_params.meta_learning_rate)

    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        for batch in data_loader:
            grids, masks, target_grid = [item.to(device) for item in batch]

            with autocast(device.type):
                output = torch.func.functional_call(
                    model, adapted_params, (grids, masks), {}
                )[0]
                loss = criterion(
                    output.view(-1, model.num_classes),
                    target_grid.view(-1).long(),
                )

            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=True
            )

            for param, grad in zip(adapted_params.values(), grads):
                param.grad = grad

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            predictions = torch.argmax(output, dim=-1)
            train_accuracy += (predictions == target_grid).float().mean().item()

        train_loss /= len(data_loader)
        train_accuracy /= len(data_loader)

    end_time = time.time()
    duration = end_time - start_time

    print(
        f"Fine-tuning finished - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Total duration: {duration:.2f}s ({(duration / 60):.2f}m)"
    )

    return adapted_params


def meta_train_arc_transformer(
    model_filename: str,
    num_epochs: int,
    patience: int = 10,
    train_params: Optional[ARCTrainParams] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.mha.set_fastpath_enabled(False)

    checkpoint_dict = torch.load(model_filename, weights_only=False)
    checkpoint = ARCModelState(**checkpoint_dict)

    if checkpoint.model_type == "vision":
        model = ARCVisionEncoderDecoder(checkpoint.model_params)
    else:
        model = ARCTransformerEncoderDecoder(checkpoint.model_params)

    if checkpoint.model_state_dict is not None:
        model.load_state_dict(checkpoint.model_state_dict)

    model = model.to(device)

    parallel_model = nn.DataParallel(model)

    print(f"Using {torch.cuda.device_count()} GPUs in parallel")

    print("Model", parallel_model.module.state_dict().keys())

    dataset_params = ARCDatasetParams(
        max_grid_size=parallel_model.module.grid_dim,
        max_train_grids=parallel_model.module.num_train_pairs,
        color_offset=1,
    )

    train_params = train_params or checkpoint.train_params

    train_loader, val_loader = make_data_loaders(
        train_params.dataset_dir,
        train_params.batch_size,
        dataset_params,
    )

    dataset_dir_names = ", ".join(train_params.dataset_dir)

    print(
        f"Starting training run with dataset of {len(train_loader.dataset)} training items and {len(val_loader.dataset)} evaluation items: {dataset_dir_names}"
    )
    print(f"Using batch size of {train_params.batch_size}")

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

    scaler = GradScaler(device.type)

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
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Starting outer loop batch {i + 1}/{len(train_loader)}")
            grids, masks, target_grid = [item.to(device) for item in batch]

            optimizer.zero_grad()

            finetune_dataset = make_finetune_dataset(grids, dataset_params)

            adapted_params = meta_fine_tune_transformer(
                parallel_model.module, train_params, finetune_dataset
            )

            # with torch.no_grad():
            #     for name, param in parallel_model.module.named_parameters():
            #         param.copy_(adapted_params[name])

            with autocast(device.type):
                # output = parallel_model.forward(grids, masks)[0]
                output = torch.func.functional_call(
                    parallel_model.module, adapted_params, (grids, masks)
                )[0]
                loss = criterion(
                    output.view(-1, parallel_model.module.num_classes),
                    target_grid.view(-1).long(),
                )

            # Use the scaler for backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

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

                finetune_dataset = make_finetune_dataset(grids, dataset_params)

                adapted_params = meta_fine_tune_transformer(
                    parallel_model.module, train_params, finetune_dataset
                )

                with autocast(device.type):
                    output = torch.func.functional_call(
                        parallel_model.module, adapted_params, (grids, masks), {}
                    )[0]
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

        end_time = time.time()

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
                start_time=start_time,
                end_time=end_time,
            )
        )

        # for batch in encoder_attn_weights:
        #     for layer in batch:
        #         visualize_all_heads(layer)

        print(f"Epoch {epoch+1}/{total_epochs} for {model_filename}:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        duration = end_time - start_time
        print(f"Epoch duration: {duration:.2f}s ({(duration / 60):.2f}m)")

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


def meta_train_on_mac(
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

    return meta_train_arc_transformer(
        model_filename, num_epochs, train_params=train_params
    )
