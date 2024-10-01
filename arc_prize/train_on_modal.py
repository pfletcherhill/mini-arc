from typing import Optional

import modal
import torch

from arc_prize.data import ARCDatasetParams, make_data_loaders
from arc_prize.model import (
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
    ARCVisionEncoderDecoder,
)
from arc_prize.train import ARCModelState, ARCTrainParams, train_arc_transformer

modal_image = modal.Image.debian_slim().pip_install("torch")
modal_app = modal.App(name="arc-prize", image=modal_image)


models_volume = modal.Volume.from_name("arc-model-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("arc-data")


@modal_app.function(
    gpu="A100:2",
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

    return train_arc_transformer(model_filename, num_epochs, train_params=train_params)


@modal_app.function(volumes={"/vol/models": models_volume})
def get_model(model_name: str):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_filename, map_location=device)
    return checkpoint


@modal_app.function(
    gpu="a10g",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def evaluate_model(
    model_name: str,
    dataset_dir: list[str],
    need_attn_weights: bool = False,
    num_tasks: Optional[int] = None,
):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = ARCModelState(**torch.load(model_filename, map_location=device))

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
    # dataset_dir = f"/vol/data/{checkpoint.train_params.dataset_name}"
    _, val_loader = make_data_loaders(
        dataset_dir,
        batch_size=1,
        params=dataset_params,
    )
    # _, val_loader = make_re_arc_data_loaders(
    #     dataset_dir,
    #     batch_size=1,
    #     params=dataset_params,
    # )

    model.eval()

    output = []

    for i, batch in enumerate(val_loader):
        if num_tasks is not None and i > num_tasks:
            break

        grids, grid_masks, output_grid = [item.to(device) for item in batch]
        (
            predictions,
            encoder_attn_weights,
            decoder_sa_weights,
            decoder_mha_weights,
        ) = model.generate(grids, grid_masks, need_attn_weights)
        if encoder_attn_weights is not None:
            print("encoder_attn_weights", encoder_attn_weights.shape)
        if decoder_sa_weights is not None:
            print("decoder_sa_weights", decoder_sa_weights.shape)
        if decoder_mha_weights is not None:
            print("decoder_mha_weights", decoder_mha_weights.shape)
        output.append(
            {
                "grids": grids.cpu().numpy(),
                "output_grid": output_grid.cpu().numpy(),
                "predictions": predictions.cpu().numpy(),
                "encoder_attn_weights": encoder_attn_weights.mean(dim=-2).cpu().numpy()
                if encoder_attn_weights is not None
                else None,
                "decoder_sa_attn_weights": decoder_sa_weights.mean(dim=-2).cpu().numpy()
                if decoder_sa_weights is not None
                else None,
                "decoder_mha_attn_weights": decoder_mha_weights.mean(dim=-2)
                .cpu()
                .numpy()
                if decoder_mha_weights is not None
                else None,
            }
        )

    return output
