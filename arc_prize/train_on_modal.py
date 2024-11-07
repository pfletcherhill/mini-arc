import os
from typing import Optional

import modal
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.utils.data import ConcatDataset

from arc_prize.data import (
    ARCDataset,
    ARCDatasetParams,
    make_data_loaders,
    make_finetune_dataset,
)
from arc_prize.model import (
    ARCTransformerEncoder,
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
    ARCVisionEncoder,
    ARCVisionEncoderDecoder,
)
from arc_prize.train import (
    ARCModelState,
    ARCTrainParams,
    fine_tune_transformer,
    load_model_from_checkpoint,
    train_arc_transformer,
)

modal_image = modal.Image.debian_slim().pip_install(["torch", "psutil"])
modal_app = modal.App(name="arc-prize", image=modal_image)


models_volume = modal.Volume.from_name("arc-model-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("arc-data")


def train_process(
    rank: int,
    world_size: int,
    model_filename: str,
    num_epochs: int,
    train_params: Optional[ARCTrainParams] = None,
):
    print(f"Starting process rank={rank}, world_size={world_size}", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        train_arc_transformer(
            model_filename=model_filename,
            num_epochs=num_epochs,
            train_params=train_params,
            local_rank=rank,
        )
    finally:
        print(f"Shutting down process rank={rank}", flush=True)
        dist.destroy_process_group()


@modal_app.function(
    gpu="a100-80gb:4",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60 * 24),
)
def train(
    model_name: str,
    num_epochs: int,
    model_type: Optional[str] = "normal",
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[ARCTrainParams] = None,
):
    model_filename = f"/vol/models/{model_name}.pth"

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

    world_size = torch.cuda.device_count()
    print(f"Starting distributed training across {world_size} GPUs")

    port = 12355
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    print(f"Using port {port} for distributed training")

    return spawn(
        train_process,
        args=(world_size, model_filename, num_epochs, train_params),
        nprocs=world_size,
        join=True,
    )


@modal_app.function(volumes={"/vol/models": models_volume})
def get_model(model_name: str):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(model_filename, map_location=device)
    keys = ["model_params", "model_type", "train_params", "epochs", "best_val_loss"]
    return {k: checkpoint_dict[k] for k in keys}


@modal_app.function(
    gpu="a10g",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def evaluate_model(
    model_name: str,
    dataset_dir: list[str],
    need_attn_weights: bool = False,
    temperature: list[float] = [0.0, 0.0],
    num_tasks: Optional[int] = None,
    chunked: bool = False,
):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(model_filename)
    model = model.to(device)

    dataset_params = ARCDatasetParams(
        max_grid_size=model.grid_dim,
        max_train_grids=model.num_train_pairs,
        color_offset=1,
    )

    _, val_loader = make_data_loaders(
        dataset_dir, batch_size=1, params=dataset_params, chunked=chunked
    )

    model.eval()

    output = []

    for i, batch in enumerate(val_loader):
        if num_tasks is not None and i > num_tasks:
            break

        grids, grid_masks, output_grid = [item.to(device) for item in batch]
        predictions = model.generate(
            grids, grid_masks, tgt=None, temperature=temperature[0]
        )[0]

        refined_predictions = model.generate(
            grids, grid_masks, tgt=predictions, temperature=temperature[1]
        )[0]
        # if encoder_attn_weights is not None:
        #     print("encoder_attn_weights", encoder_attn_weights.shape)
        # if decoder_sa_weights is not None:
        #     print("decoder_sa_weights", decoder_sa_weights.shape)
        # if decoder_mha_weights is not None:
        #     print("decoder_mha_weights", decoder_mha_weights.shape)
        output.append(
            {
                "grids": grids.cpu().numpy(),
                "output_grid": output_grid.cpu().numpy(),
                "predictions": predictions.cpu().numpy(),
                "refined_predictions": refined_predictions.cpu().numpy(),
                # "encoder_attn_weights": encoder_attn_weights.mean(dim=-2).cpu().numpy()
                # if encoder_attn_weights is not None
                # else None,
                # "decoder_sa_attn_weights": decoder_sa_weights.mean(dim=-2).cpu().numpy()
                # if decoder_sa_weights is not None
                # else None,
                # "decoder_mha_attn_weights": decoder_mha_weights.mean(dim=-2)
                # .cpu()
                # .numpy()
                # if decoder_mha_weights is not None
                # else None,
            }
        )

    return output


@modal_app.function(
    gpu="A100",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def finetune_and_predict(
    model_name: str,
    dataset_dir: list[str],
    num_finetune_epochs: int = 2,
    num_tasks: Optional[int] = None,
):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(model_filename, map_location=device)
    checkpoint_dict["model_type"] = checkpoint_dict.get("model_type", None) or "normal"
    checkpoint = ARCModelState(**checkpoint_dict)

    train_params = checkpoint.train_params

    if checkpoint.model_type == "vision":
        model = ARCVisionEncoderDecoder(checkpoint.model_params).to(device)
    elif checkpoint.model_type == "encoder":
        model = ARCTransformerEncoder(checkpoint.model_params).to(device)
    elif checkpoint.model_type == "vision_encoder":
        model = ARCVisionEncoder(checkpoint.model_params).to(device)
    else:
        model = ARCTransformerEncoderDecoder(checkpoint.model_params).to(device)

    if checkpoint.model_state_dict is not None:
        model.load_state_dict(checkpoint.model_state_dict)

    dataset_params = ARCDatasetParams(
        max_grid_size=model.grid_dim,
        max_train_grids=model.num_train_pairs,
        color_offset=1,
    )

    train_datasets = []

    for dir in dataset_dir:
        train_dataset = ARCDataset(
            f"{dir}/training_challenges.json",
            solutions_file=f"{dir}/training_solutions.json",
            config=dataset_params,
        )
        train_datasets.append(train_dataset)

    train_dataset = ConcatDataset(train_datasets)

    model.eval()

    output = []

    for task in train_dataset:
        print("Starting", task["task_id"])
        # Predict
        predictions = model.generate(
            task["grids"].to(device).unsqueeze(0),
            task["masks"].to(device).unsqueeze(0),
            need_weights=False,
        )[0][0]

        finetune_dataset = make_finetune_dataset(
            task["grids"].to(device).unsqueeze(0), dataset_params
        )

        finetune_model = fine_tune_transformer(
            model, train_params, finetune_dataset, num_finetune_epochs
        )

        finetune_predictions = finetune_model.generate(
            task["grids"].to(device).unsqueeze(0),
            task["masks"].to(device).unsqueeze(0),
            need_weights=False,
        )[0][0]

        output.append(
            {
                "grids": task["grids"].detach().cpu().numpy(),
                "output_grid": task["output"].detach().cpu().numpy(),
                "predictions": predictions.detach().cpu().numpy(),
                "finetune_predictions": finetune_predictions.detach().cpu().numpy(),
            }
        )

        del finetune_model, predictions, finetune_predictions

        if num_tasks is not None and len(output) >= num_tasks:
            break

    return output
