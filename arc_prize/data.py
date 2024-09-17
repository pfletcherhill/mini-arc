import json
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


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


def make_data_loaders(
    dataset_dir: list[str], batch_size: int, params: ARCDatasetParams
) -> tuple[DataLoader[ARCDataset], DataLoader[ARCDataset]]:
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
