import os

import modal

from arc_prize.flax.models import ARCTransformerEncoderDecoderParams
from arc_prize.flax.train import TrainParams, train_and_evaluate

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
    train_params: TrainParams,
    model_params: ARCTransformerEncoderDecoderParams,
    num_epochs: int,
):
    os.makedirs(model_dir, exist_ok=True)
    return train_and_evaluate(model_dir, model_params, train_params, num_epochs)
