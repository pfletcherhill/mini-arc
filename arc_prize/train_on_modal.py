from typing import Optional

# import matplotlib.pyplot as plt
import modal
import torch

from arc_prize.model import ARCTransformerEncoderDecoderParams
from arc_prize.train import ARCModelState, ARCTrainParams, train_arc_transformer

modal_image = modal.Image.debian_slim().pip_install("torch")
modal_app = modal.App(name="arc-prize", image=modal_image)


models_volume = modal.Volume.from_name("arc-model-vol", create_if_missing=True)
data_volume = modal.Volume.from_name("arc-data")


@modal_app.function(
    gpu="t4",
    volumes={"/vol/models": models_volume, "/vol/data": data_volume},
    timeout=(60 * 60),
)
def train(
    model_name: str,
    num_epochs: int,
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[ARCTrainParams] = None,
):
    model_filename = f"/vol/models/{model_name}.pth"

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

    return train_arc_transformer(model_filename, num_epochs)


@modal_app.function(volumes={"/vol/models": models_volume})
def get_model(model_name: str):
    model_filename = f"/vol/models/{model_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_filename, map_location=device)
    return checkpoint

    # plt.figure(figsize=(10, 5))
    # plt.plot(checkpoint, label='Training Loss')
    # plt.plot(eval_losses, label='Evaluation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Evaluation Losses')
    # plt.show()
