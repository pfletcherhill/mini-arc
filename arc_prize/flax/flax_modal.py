import os
from typing import Optional

import modal

from arc_prize.flax.models import ARCTransformerEncoderDecoderParams
from arc_prize.flax.train import (
    TrainParams,
    get_config_params,
    save_config_params,
    train_and_evaluate,
)
from arc_prize.flax.train import predict as flax_predict

modal_image = modal.Image.debian_slim().pip_install("torch", "jax[cuda12]", "flax")
modal_app = modal.App(name="arc-jax", image=modal_image)


models_volume = modal.Volume.from_name("arc-model-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("arc-data")


@modal_app.function(
    gpu="A100",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60 * 24),
)
def train(
    model_dir: str,
    model_params: ARCTransformerEncoderDecoderParams,
    train_params: TrainParams,
    num_epochs: int,
):
    os.makedirs(model_dir, exist_ok=True)
    model_params, train_params = get_config_params(
        model_dir, model_params, train_params
    )
    save_config_params(model_dir, model_params, train_params)
    return train_and_evaluate(model_dir, model_params, train_params, num_epochs)


@modal_app.function(
    gpu="a10g",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def predict(
    model_dir: str,
    dataset_dir: str,
    num_steps: Optional[int] = None,
):
    model_params, _ = get_config_params(model_dir)
    return flax_predict(model_dir, model_params, dataset_dir, num_steps)
