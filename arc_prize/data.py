import itertools
import json
import math
import random
from dataclasses import dataclass
from operator import itemgetter
from typing import Callable, Iterator, Optional

import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    Subset,
    random_split,
)


@dataclass(frozen=True)
class ARCDatasetParams:
    max_grid_size: int = 30
    max_train_grids: int = 10
    color_offset: int = 1


def pad_and_mask_grid(
    grid: list[list[int]], config: ARCDatasetParams
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = len(grid), len(grid[0])
    if h > config.max_grid_size or w > config.max_grid_size:
        raise Exception("grid size too large")

    padded = torch.zeros((config.max_grid_size, config.max_grid_size), dtype=torch.int)
    mask = torch.zeros((config.max_grid_size, config.max_grid_size), dtype=torch.bool)

    # Calculate padding
    pad_h = (config.max_grid_size - h) // 2
    pad_w = (config.max_grid_size - w) // 2

    # Place the grid in the center
    padded[pad_h : pad_h + h, pad_w : pad_w + w] = (
        torch.tensor(grid, dtype=torch.int) + config.color_offset
    )
    mask[pad_h : pad_h + h, pad_w : pad_w + w] = True

    return (padded, mask)


def unpad_grid(grid: torch.Tensor) -> list[list[int]]:
    grid = grid - 1
    filtered_rows: list[list] = []
    for row in grid:
        filtered_row = row[row != -1]
        if len(filtered_row) > 0:
            filtered_rows.append(filtered_row.tolist())
    # Hack to ensure there's always at least one value
    if len(filtered_rows) == 0:
        filtered_rows.append([0])
    max_length = max(len(row) for row in filtered_rows)
    padded_rows = [(row + [0] * (max_length - len(row))) for row in filtered_rows]
    return padded_rows


class FinetuneDataset(Dataset):
    tasks: list[list[list[list[list[int]]]]]
    config: ARCDatasetParams

    def __init__(
        self,
        tasks: list[list[list[list[list[int]]]]],
        config: ARCDatasetParams,
    ):
        self.tasks = tasks
        self.config = config

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict:
        task = self.tasks[idx]

        grids = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.int,
        )
        masks = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.bool,
        )

        for i, pair in enumerate(task[:-1]):
            if i >= self.config.max_train_grids:
                print("Training pairs exceed max", i, self.config.max_train_grids)
                break

            input_grid, input_mask = pad_and_mask_grid(pair[0], self.config)
            output_grid, output_mask = pad_and_mask_grid(pair[1], self.config)
            grids[2 * i] = input_grid
            masks[2 * i] = input_mask
            grids[2 * i + 1] = output_grid
            masks[2 * i + 1] = output_mask

        test_input_grid, test_input_mask = pad_and_mask_grid(task[-1][0], self.config)
        grids[-1] = test_input_grid
        masks[-1] = test_input_mask

        test_output_grid = pad_and_mask_grid(task[-1][1], self.config)[0]

        return {
            "grids": grids,
            "masks": masks,
            "output": test_output_grid,
        }


def make_finetune_dataset(
    grids: torch.Tensor, config: ARCDatasetParams
) -> FinetuneDataset:
    if len(grids.shape) == 3:
        grids = grids.unsqueeze(0)
    if len(grids.shape) != 4:
        raise Exception("incorrect grids dimension")
    tasks = []
    for task in grids:
        pairs = task[:-1].reshape(
            config.max_train_grids,
            2,
            config.max_grid_size,
            config.max_grid_size,
        )
        finetune_pairs: list[list[list[list[int]]]] = []
        for pair in pairs:
            finetune_pairs.append([unpad_grid(grid) for grid in pair])

        for length in range(2, len(finetune_pairs) + 1):
            for combination in itertools.combinations(finetune_pairs, length):
                for permutation in itertools.permutations(combination):
                    tasks.append(list(permutation))

    return FinetuneDataset(tasks=tasks, config=config)


class ARCDataset(Dataset):
    challenges: dict[str, dict]
    solutions: dict[str, list]
    task_ids: list[str]
    config: ARCDatasetParams

    def __init__(
        self,
        challenges_file: str,
        config: ARCDatasetParams,
        solutions_file: Optional[str] = None,
    ):
        with open(challenges_file, "r") as f:
            self.challenges = json.load(f)
            self.task_ids = list(self.challenges.keys())
        if solutions_file is not None:
            with open(solutions_file, "r") as f:
                self.solutions = json.load(f)
        else:
            self.solutions = {}
        self.config = config

    def __len__(self) -> int:
        return len(self.task_ids)

    def __getitem__(self, idx: int) -> dict:
        task_id = self.task_ids[idx]
        challenge = self.challenges[task_id]
        solution = self.solutions.get(task_id, None)

        grids = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.int,
        )
        masks = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.bool,
        )

        for i, pair in enumerate(challenge["train"]):
            if i >= self.config.max_train_grids:
                raise Exception(
                    "Training pairs exceed max", i, self.config.max_train_grids
                )
            input_grid, input_mask = pad_and_mask_grid(pair["input"], self.config)
            grids[2 * i] = input_grid
            masks[2 * i] = input_mask
            output_grid, output_mask = pad_and_mask_grid(pair["output"], self.config)
            grids[2 * i + 1] = output_grid
            masks[2 * i + 1] = output_mask

        test_input_grid, test_input_mask = pad_and_mask_grid(
            challenge["test"][0]["input"], self.config
        )
        grids[-1] = test_input_grid
        masks[-1] = test_input_mask
        test_output_grid = (
            pad_and_mask_grid(solution[0], self.config)[0]
            if solution is not None
            else None
        )

        return {
            "task_id": task_id,
            "grids": grids,
            "masks": masks,
            "output": test_output_grid,
        }


class ARCKaggleDataset(Dataset):
    challenges: dict[str, dict]
    task_ids: list[str]
    config: ARCDatasetParams

    def __init__(
        self,
        challenges_file: str,
        config: ARCDatasetParams,
    ):
        with open(challenges_file, "r") as f:
            self.challenges = json.load(f)
            self.task_ids = list(self.challenges.keys())
        self.config = config

    def __len__(self) -> int:
        return len(self.task_ids)

    def __getitem__(self, idx: int) -> dict:
        task_id = self.task_ids[idx]
        challenge = self.challenges[task_id]

        grids = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.int,
        )
        masks = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.bool,
        )

        for i, pair in enumerate(challenge["train"]):
            if i >= self.config.max_train_grids:
                print(
                    "Training pairs exceed max", task_id, i, self.config.max_train_grids
                )
                break

            try:
                input_grid, input_mask = pad_and_mask_grid(pair["input"], self.config)
                output_grid, output_mask = pad_and_mask_grid(
                    pair["output"], self.config
                )
                grids[2 * i] = input_grid
                masks[2 * i] = input_mask
                grids[2 * i + 1] = output_grid
                masks[2 * i + 1] = output_mask
            except Exception as e:
                print("Got exception for training pair", task_id, i, e)

        try:
            test_input_grid, test_input_mask = pad_and_mask_grid(
                challenge["test"][0]["input"], self.config
            )
            grids[-1] = test_input_grid
            masks[-1] = test_input_mask
        except Exception as e:
            print("Got exception on test input", task_id, e)

        return {"task_id": task_id, "grids": grids, "masks": masks}


class ReARCDataset(Dataset):
    tasks: list
    config: ARCDatasetParams

    def __init__(self, tasks_file: str, config: ARCDatasetParams):
        with open(tasks_file, "r") as f:
            self.tasks = json.load(f)
        self.config = config

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx) -> dict:
        num_train_pairs = random.randint(1, self.config.max_train_grids)
        train_task_idxs = set()
        while len(train_task_idxs) < num_train_pairs:
            task_idx = random.randint(0, len(self.tasks) - 1)
            if task_idx != idx:
                train_task_idxs.add(task_idx)
        train_tasks = [self.tasks[task_idx] for task_idx in train_task_idxs]

        grids = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.int,
        )
        masks = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.bool,
        )

        for i, pair in enumerate(train_tasks):
            if i >= self.config.max_train_grids:
                raise Exception(
                    "Training pairs exceed max", i, self.config.max_train_grids
                )
            input_grid, input_mask = pad_and_mask_grid(pair["input"], self.config)
            grids[2 * i] = input_grid
            masks[2 * i] = input_mask
            output_grid, output_mask = pad_and_mask_grid(pair["output"], self.config)
            grids[2 * i + 1] = output_grid
            masks[2 * i + 1] = output_mask

        test_task = self.tasks[idx]

        test_input_grid, test_input_mask = pad_and_mask_grid(
            test_task["input"], self.config
        )
        grids[-1] = test_input_grid
        masks[-1] = test_input_mask
        test_output_grid = pad_and_mask_grid(test_task["output"], self.config)[0]

        return {
            "task_id": f"{idx}",
            "grids": grids,
            "masks": masks,
            "output": test_output_grid,
        }


def collate_arc_fn(
    batch: list[dict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grids = torch.stack([item["grids"] for item in batch])
    masks = torch.stack([item["masks"] for item in batch])
    output = torch.stack([item["output"] for item in batch])

    return (grids, masks, output)


def make_datasets(
    dataset_dir: list[str], params: ARCDatasetParams
) -> tuple[Dataset, Dataset]:
    train_datasets = []
    val_datasets = []

    for dir in dataset_dir:
        train_dataset = ARCDataset(
            f"{dir}/training_challenges.json",
            solutions_file=f"{dir}/training_solutions.json",
            config=params,
        )
        train_datasets.append(train_dataset)
        val_dataset = ARCDataset(
            f"{dir}/evaluation_challenges.json",
            solutions_file=f"{dir}/evaluation_solutions.json",
            config=params,
        )
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    return (train_dataset, val_dataset)


def make_data_loaders(
    dataset_dir: list[str], batch_size: int, params: ARCDatasetParams
) -> tuple[DataLoader[ARCDataset], DataLoader[ARCDataset]]:
    train_dataset, val_dataset = make_datasets(dataset_dir, params)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    return (train_loader, val_loader)


def make_re_arc_data_loaders(
    dataset_files: list[str], batch_size: int, params: ARCDatasetParams
) -> tuple[DataLoader[ARCDataset], DataLoader[ARCDataset]]:
    datasets: list[Dataset] = []

    for file_path in dataset_files:
        dataset = ReARCDataset(
            file_path,
            config=params,
        )
        datasets.append(dataset)

    val_ratio = 0.25
    train_ratio = 1 - val_ratio
    train_dataset, val_dataset = random_split(
        ConcatDataset(datasets), lengths=[train_ratio, val_ratio]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arc_fn,
        num_workers=0,
    )

    return (train_loader, val_loader)


def make_epoch_data_loader(
    dataset: Dataset,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    num_steps: Optional[int] = None,
    sampler: Optional[DistributedSampler] = None,
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
        collate_fn=(collate_fn or collate_arc_fn),
        num_workers=0,
        sampler=sampler,
    )

    return loader


class DistributedRandomSampler(DistributedSampler):
    """
    Combines DistributedSampler and RandomSampler functionality.
    Allows sampling a specific number of samples randomly while maintaining distributed properties.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to draw (per replica). If None, samples whole dataset
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: If True, sampler will shuffle the indices
            seed: Random seed for reproducibility
            drop_last: If True, drops the last non-full batch
        """
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        # Calculate number of samples per replica
        if num_samples is not None:
            self.num_samples = num_samples
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        n = len(self.dataset)

        # Without replacement: shuffle and repeat if necessary
        if self.shuffle:
            # Generate permutation of all indices
            indices = []
            for _ in range(math.ceil(self.total_size / n)):
                indices.extend(torch.randperm(n, generator=g).tolist())
            indices = indices[: self.total_size]
        else:
            # Use sequential indices and repeat if necessary
            indices = list(range(n))
            if self.total_size > n:
                indices = indices * math.ceil(self.total_size / n)
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # Distribute indices across replicas
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
