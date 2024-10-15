import json
import os
import random
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx, struct
from flax.training.train_state import TrainState as FlaxTrainState
from torch.utils.data import DataLoader, Dataset, Subset

from arc_prize.data import (
    ARCDatasetParams,
    make_datasets,
)
from arc_prize.flax.models import (
    ARCTransformerEncoderDecoder,
    ARCTransformerEncoderDecoderParams,
)


class TrainState(FlaxTrainState):
    graphdef: nnx.GraphDef[ARCTransformerEncoderDecoder]
    other_state: nnx.State


@struct.dataclass
class TrainParams:
    batch_size: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    dataset_dirs: list[str]
    train_steps_per_epoch: Optional[int] = None
    eval_steps_per_epoch: Optional[int] = None
    loss_class_weights: Optional[dict[int, float]] = None


def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
):
    """Applies a reverse square-root schedule.

    The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

    Args:
      init_value: Base learning rate (before applying the rsqrt schedule).
      shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
        schedule makes it less steep in the beginning (close to 0).

    Returns:
      A schedule that applies the reverse square root.
    """

    def schedule(count):
        return init_value * (count + shift) ** -0.5 * shift**0.5

    return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
    """Creates a rsqrt schedule with linear warmup."""
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
            rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
        ],
        boundaries=[warmup_steps],
    )


def loss_fn(
    model: ARCTransformerEncoderDecoder,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    grids, masks, targets = batch
    logits = model(grids, masks)

    # Compute weighted cross-entropy loss
    weighted_ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    weights = class_weights[targets]
    weighted_ce_loss = weighted_ce_loss * weights

    # Normalize the loss by the sum of weights
    total_weight = jnp.sum(weights)
    loss = jnp.sum(weighted_ce_loss) / total_weight

    # Calculate accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = predictions == targets
    accuracy = jnp.mean(correct_predictions)

    return (loss, accuracy)


@jax.jit
def train_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    model = nnx.merge(state.graphdef, state.params, state.other_state)
    model.train(decode=False)

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, batch, class_weights
    )

    _, _, other_state = nnx.split(model, nnx.Param, ...)

    new_state = state.apply_gradients(grads=grads, other_state=other_state)
    return (new_state, loss, accuracy)


@jax.jit
def eval_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    model = nnx.merge(state.graphdef, state.params, state.other_state)
    model.eval(decode=False)
    loss, accuracy = loss_fn(model, batch, class_weights)
    return (loss, accuracy)


def collate_flax_arc_fn(
    batch: list[dict],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Convert each item in the batch to a JAX array
    grids = jnp.stack([jnp.array(item["grids"]) for item in batch])
    masks = jnp.stack([jnp.array(item["masks"]) for item in batch])
    output = jnp.stack([jnp.array(item["output"]) for item in batch])

    return (grids, masks, output)


def get_epoch_data_loader(
    dataset: Dataset, batch_size: int, num_steps: Optional[int] = None
):
    total_samples = len(dataset)
    subset = None

    if num_steps is not None:
        num_samples = num_steps * batch_size

        assert num_samples <= total_samples

        indices = random.sample(range(total_samples), num_samples)

        subset = Subset(dataset, indices)

    loader = DataLoader(
        subset or dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_flax_arc_fn,
        num_workers=0,
    )

    return loader


def setup_train_state(
    model: ARCTransformerEncoderDecoder, tx: optax.GradientTransformation
) -> TrainState:
    graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

    def _fix_random_key(x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        return x

    other_state = jax.tree.map(_fix_random_key, other_state)

    state = TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
        graphdef=graphdef,
        other_state=other_state,
    )

    return state


def train_and_evaluate(
    model_dir: str,
    model_params: ARCTransformerEncoderDecoderParams,
    train_params: TrainParams,
    num_epochs: int,
):
    checkpoint_options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(f"{model_dir}/checkpoints"),
        options=checkpoint_options,
    ) as checkpoint_mngr:
        # os.makedirs(workdir, exist_ok=True)
        print("GPU devices", jax.devices())

        dataset_params = ARCDatasetParams(
            max_grid_size=model_params.grid_dim,
            max_train_grids=model_params.num_train_pairs,
            color_offset=1,
        )

        train_dataset, val_dataset = make_datasets(
            train_params.dataset_dirs,
            dataset_params,
        )

        dataset_dir_names = ", ".join(train_params.dataset_dirs)

        print(
            f"Starting training run with dataset of {len(train_dataset)} training items and {len(val_dataset)} evaluation items: {dataset_dir_names}"
        )
        print(f"Using batch size of {train_params.batch_size}")

        rngs = nnx.Rngs(0)

        model = ARCTransformerEncoderDecoder(params=model_params, rngs=rngs)

        learning_rate_fn = create_learning_rate_schedule(
            learning_rate=train_params.learning_rate,
            warmup_steps=train_params.warmup_steps,
        )

        optimizer = optax.adamw(
            learning_rate_fn,
            weight_decay=train_params.weight_decay,
        )

        class_weights = jnp.ones(model.num_classes)
        if train_params.loss_class_weights is not None:
            for cls, weight in train_params.loss_class_weights.items():
                class_weights = class_weights.at[int(cls)].set(weight)

        print("Saving checkpoint")

        state = setup_train_state(model, optimizer)

        checkpoint_mngr.save(0, args=ocp.args.StandardSave(state))
        checkpoint_mngr.wait_until_finished()

        print(checkpoint_mngr.all_steps())

        checkpoint_mngr.restore(checkpoint_mngr.latest_step(), state)
        # print(restored_state_dict.keys())
        # restored_train_state = TrainState.create(
        #     tx=optimizer,
        #     apply_fn=restored_state_dict["graphdef"]["apply"],
        #     **restored_state_dict,
        # )
        # print(restored_train_state)
        # nnx.update(model, restored_state.params)

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            start_time = time.perf_counter()
            # model.set_attributes(deterministic=False, decode=False)
            train_loss = 0.0
            train_accuracy = 0.0
            train_data_loader = get_epoch_data_loader(
                train_dataset,
                train_params.batch_size,
                num_steps=train_params.train_steps_per_epoch,
            )
            for batch in train_data_loader:
                state, loss, accuracy = train_step(state, batch, class_weights)
                train_loss += loss.item()
                train_accuracy += accuracy.item()
            train_loss /= len(train_data_loader)
            train_accuracy /= len(train_data_loader)

            train_time = time.perf_counter()
            print(f"Train loop completed in {train_time - start_time}")
            print(f"Train loss: {train_loss}, accuracy: {train_accuracy}")

            # model.set_attributes(deterministic=True, decode=False)
            val_losses = []
            val_accuracy = []
            val_data_loader = get_epoch_data_loader(
                val_dataset,
                train_params.batch_size,
                num_steps=train_params.eval_steps_per_epoch,
            )
            for batch in val_data_loader:
                loss, accuracy = eval_step(state, batch, class_weights)
                val_losses.append(np.asarray(loss))
                val_accuracy.append(np.asarray(accuracy))

            time_diff = time.perf_counter() - train_time
            print(f"Eval loop completed in {time_diff}")
            print(
                f"Eval loss: {np.mean(val_losses)}, accuracy: {np.mean(val_accuracy)}"
            )

            checkpoint_mngr.save(0, args=ocp.args.PyTreeSave(state))
            checkpoint_mngr.wait_until_finished()


def train_and_evaluate_local(
    model_dir: str,
    num_epochs: int,
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[TrainParams] = None,
):
    os.makedirs(model_dir, exist_ok=True)
    config_path = f"{model_dir}/config.json"

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)

    model_params = model_params or ARCTransformerEncoderDecoderParams(
        **config_dict["model"]
    )
    train_params = train_params or TrainParams(**config_dict["train"])

    assert model_params is not None and train_params is not None

    with open(f"{model_dir}/config.json", "w") as f:
        config_dict = {"model": model_params.__dict__, "train": train_params.__dict__}
        json.dump(config_dict, f)
    return train_and_evaluate(model_dir, model_params, train_params, num_epochs)
