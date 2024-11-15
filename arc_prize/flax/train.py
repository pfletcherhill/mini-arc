import json
import os
import random
import time
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx, struct
from flax.training.train_state import TrainState as FlaxTrainState
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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


# @struct.dataclass
# class TrainState:
#     params: nnx.GraphState
#     other_state: nnx.GraphState
#     opt_state: optax.OptState
#     step: int


@struct.dataclass
class ModelState:
    params: nnx.GraphState
    other_state: nnx.GraphState


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


@partial(jax.jit, static_argnames=["graphdef", "tx"])
def train_step(
    graphdef: nnx.GraphDef[ARCTransformerEncoderDecoder],
    model_state: ModelState,
    tx: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[ModelState, optax.OptState, jnp.ndarray, jnp.ndarray]:
    model = nnx.merge(graphdef, model_state.params, model_state.other_state)
    model.train(decode=False)

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, batch, class_weights
    )

    updates, new_opt_state = tx.update(grads, opt_state, model_state.params)
    new_params = optax.apply_updates(model_state.params, updates)

    _, _, new_other_state = nnx.split(model, nnx.Param, ...)

    new_model_state = ModelState(params=new_params, other_state=new_other_state)

    return (new_model_state, new_opt_state, loss, accuracy)


@partial(jax.jit, static_argnames=["graphdef"])
def eval_step(
    # state: TrainState,
    graphdef: nnx.GraphDef[ARCTransformerEncoderDecoder],
    model_state: ModelState,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    class_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    model = nnx.merge(graphdef, model_state.params, model_state.other_state)
    model.eval(decode=False)
    loss, accuracy = loss_fn(model, batch, class_weights)
    return (loss, accuracy)


def predict_step(
    graphdef: nnx.GraphDef,
    params: nnx.GraphState,
    other_state: nnx.GraphState,
    grids: jnp.ndarray,
    masks: jnp.ndarray,
) -> jnp.ndarray:
    model = nnx.merge(graphdef, params, other_state)
    print("model", model)
    model.eval(decode=False)
    logits = model(grids, masks)
    return logits


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


def get_model_state(
    model: ARCTransformerEncoderDecoder,
) -> tuple[nnx.GraphDef[ARCTransformerEncoderDecoder], ModelState]:
    graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

    def _fix_random_key(x):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x = jax.random.key_data(x)
        return x

    other_state = jax.tree.map(_fix_random_key, other_state)

    model_state = ModelState(params=params, other_state=other_state)

    return (graphdef, model_state)


def train_and_evaluate(
    model_dir: str,
    model_params: ARCTransformerEncoderDecoderParams,
    train_params: TrainParams,
    num_epochs: int,
    force_restart: Optional[bool] = False,
):
    checkpoint_dir = f"{model_dir}/checkpoints"

    if force_restart is True:
        ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    checkpoint_options = ocp.CheckpointManagerOptions()

    checkpoint_mngr = ocp.CheckpointManager(checkpoint_dir, options=checkpoint_options)

    epoch = checkpoint_mngr.latest_step()
    print("Latest step", epoch)

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
    graphdef, model_state = get_model_state(model)

    warmup_lr_schedule = optax.schedules.warmup_constant_schedule(
        0, train_params.learning_rate, warmup_steps=train_params.warmup_steps
    )
    # plateau_lr_reducer = optax.contrib.reduce_on_plateau(factor=0.1, patience=5)

    optimizer = optax.chain(
        optax.adamw(
            learning_rate=warmup_lr_schedule,
            weight_decay=train_params.weight_decay,
        )
    )

    opt_state = optimizer.init(model_state.params)

    if epoch is not None:
        restored_args = checkpoint_mngr.restore(
            epoch,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardRestore(item=model_state),
                opt_state=ocp.args.StandardRestore(item=opt_state),
            ),
        )
        model_state: ModelState = restored_args["model_state"]
        opt_state: optax.OptState = restored_args["opt_state"]

    class_weights = jnp.ones(model.num_classes)
    if train_params.loss_class_weights is not None:
        for cls, weight in train_params.loss_class_weights.items():
            class_weights = class_weights.at[int(cls)].set(weight)

    total_epochs = (epoch or 0) + num_epochs
    for epoch in range(epoch or 0, total_epochs):
        print(f"Starting epoch {epoch + 1}/{total_epochs}")
        start_time = time.perf_counter()

        train_loss = 0.0
        train_accuracy = 0.0
        train_data_loader = get_epoch_data_loader(
            train_dataset,
            train_params.batch_size,
            num_steps=train_params.train_steps_per_epoch,
        )
        for batch in train_data_loader:
            model_state, opt_state, loss, accuracy = train_step(
                graphdef, model_state, optimizer, opt_state, batch, class_weights
            )
            train_loss += loss.item()
            train_accuracy += accuracy.item()
        train_loss /= len(train_data_loader)
        train_accuracy /= len(train_data_loader)

        train_time = time.perf_counter()

        print(
            f"Train loss (completed in {(train_time - start_time):.2f}s): {train_loss:.4f}, accuracy: {train_accuracy:.4f}"
        )

        eval_loss = 0.0
        eval_accuracy = 0.0
        eval_data_loader = get_epoch_data_loader(
            val_dataset,
            train_params.batch_size,
            num_steps=train_params.eval_steps_per_epoch,
        )
        for batch in eval_data_loader:
            loss, accuracy = eval_step(graphdef, model_state, batch, class_weights)
            eval_loss += loss.item()
            eval_accuracy += accuracy.item()
        eval_loss /= len(eval_data_loader)
        eval_accuracy /= len(eval_data_loader)

        time_diff = time.perf_counter() - train_time

        print(
            f"Eval loss (competed in {time_diff:.2f}s): {eval_loss:.4f}, accuracy: {eval_accuracy:.4f}"
        )

        checkpoint_mngr.save(
            epoch,
            args=ocp.args.Composite(
                model_state=ocp.args.StandardSave(item=model_state),
                opt_state=ocp.args.StandardSave(item=opt_state),
            ),
        )
        checkpoint_mngr.wait_until_finished()

    checkpoint_mngr.close()


def predict(
    model_dir: str,
    model_params: ARCTransformerEncoderDecoderParams,
    dataset_dir: str,
    num_steps: Optional[int] = None,
):
    checkpoint_options = ocp.CheckpointManagerOptions(read_only=True)
    checkpoint_mngr = ocp.CheckpointManager(
        f"{model_dir}/checkpoints",
        options=checkpoint_options,
    )
    # os.makedirs(workdir, exist_ok=True)
    print("GPU devices", jax.devices())

    dataset_params = ARCDatasetParams(
        max_grid_size=model_params.grid_dim,
        max_train_grids=model_params.num_train_pairs,
        color_offset=1,
    )

    train_dataset, val_dataset = make_datasets(
        [dataset_dir],
        dataset_params,
    )
    dataset = ConcatDataset([train_dataset, val_dataset])

    rngs = nnx.Rngs(0)

    model = ARCTransformerEncoderDecoder(params=model_params, rngs=rngs)

    print("latest step", checkpoint_mngr.latest_step())
    checkpoint_mngr.restore(
        checkpoint_mngr.latest_step(), args=ocp.args.PyTreeRestore()
    )

    print("graphdef", restored_state["graphdef"])
    print("keys", restored_state.keys())

    data_loader = get_epoch_data_loader(dataset, 1, num_steps)

    output = []

    for batch in data_loader:
        grids, masks, targets = batch
        logits = predict_step(
            restored_state["graphdef"],
            restored_state["params"],
            restored_state["other_state"],
            grids,
            masks,
        )
        predictions = jnp.argmax(logits, axis=-1)
        print("logits", logits, predictions)
        output.append({"grids": grids, "targets": targets, "predictions": predictions})

    checkpoint_mngr.close()
    return output


def get_config_params(
    model_dir: str,
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[TrainParams] = None,
) -> tuple[ARCTransformerEncoderDecoderParams, TrainParams]:
    config_path = f"{model_dir}/config.json"

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)

    model_params = model_params or ARCTransformerEncoderDecoderParams(
        **config_dict["model"]
    )
    train_params = train_params or TrainParams(**config_dict["train"])

    assert model_params is not None and train_params is not None

    return (model_params, train_params)


def save_config_params(
    model_dir: str,
    model_params: ARCTransformerEncoderDecoderParams,
    train_params: TrainParams,
) -> None:
    with open(f"{model_dir}/config.json", "w") as f:
        config_dict = {"model": model_params.__dict__, "train": train_params.__dict__}
        json.dump(config_dict, f)


def train_and_evaluate_local(
    model_dir: str,
    num_epochs: int,
    model_params: Optional[ARCTransformerEncoderDecoderParams] = None,
    train_params: Optional[TrainParams] = None,
    force_restart: Optional[bool] = False,
):
    os.makedirs(model_dir, exist_ok=True)

    model_params, train_params = get_config_params(
        model_dir, model_params, train_params
    )

    save_config_params(model_dir, model_params, train_params)

    return train_and_evaluate(
        model_dir, model_params, train_params, num_epochs, force_restart
    )
