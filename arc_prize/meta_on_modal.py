import copy
import itertools
from typing import Optional

import modal
import torch
from torch.utils.data import ConcatDataset

from arc_prize.data import ARCDataset, ARCDatasetParams, FinetuneDataset, unpad_grid
from arc_prize.meta import meta_train_arc_transformer
from arc_prize.model import (
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
    ARCVisionEncoderDecoder,
)
from arc_prize.train import ARCModelState, ARCTrainParams, fine_tune_transformer

modal_image = modal.Image.debian_slim().pip_install("torch")
modal_app = modal.App(name="arc-prize-meta", image=modal_image)


models_volume = modal.Volume.from_name("arc-model-vol")
data_volume = modal.Volume.from_name("arc-data")


@modal_app.function(
    gpu="A100:2",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60 * 24),
)
def meta_train(
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

    return meta_train_arc_transformer(
        model_filename, num_epochs, train_params=train_params
    )


@modal_app.function(
    gpu="a10g",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def finetune_and_predict(
    model_name: str, dataset_dir: list[str], num_tasks: Optional[int] = None
):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(model_filename, map_location=device)
    checkpoint_dict["model_type"] = checkpoint_dict.get("model_type", None) or "normal"
    checkpoint = ARCModelState(**checkpoint_dict)

    if checkpoint.model_type == "vision":
        model = ARCVisionEncoderDecoder(checkpoint.model_params).to(device)
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

        # Fine-tune model
        pairs = task["grids"][:-1].reshape(
            dataset_params.max_train_grids,
            2,
            dataset_params.max_grid_size,
            dataset_params.max_grid_size,
        )
        finetune_pairs: list[list[list[list[int]]]] = []
        for pair in pairs:
            finetune_pairs.append([unpad_grid(grid) for grid in pair])

        finetune_permutations = []
        for length in range(2, len(finetune_pairs) + 1):
            for combination in itertools.combinations(finetune_pairs, length):
                for permutation in itertools.permutations(combination):
                    finetune_permutations.append(list(permutation))

        finetune_dataset = FinetuneDataset(
            tasks=finetune_permutations, config=dataset_params
        )
        finetune_model = copy.deepcopy(model)

        train_params = ARCTrainParams(
            batch_size=2,
            learning_rate=1e-5,
            weight_decay=1e-5,
            dataset_dir=[],
            loss_class_weights={0: 0.2},
        )

        finetune_model = fine_tune_transformer(
            finetune_model, train_params, finetune_dataset
        )

        # Predict
        finetune_model.eval()
        finetune_predictions = finetune_model.generate(
            task["grids"].to(device).unsqueeze(0),
            task["masks"].to(device).unsqueeze(0),
            need_weights=False,
        )[0][0]
        output.append(
            {
                "grids": task["grids"].cpu().numpy(),
                "output_grid": task["output"].cpu().numpy(),
                "predictions": predictions.cpu().numpy(),
                "finetune_predictions": finetune_predictions.cpu().numpy(),
            }
        )

        if num_tasks is not None and len(output) >= num_tasks:
            break

    return output
