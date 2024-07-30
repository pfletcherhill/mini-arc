import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ARCDatasetConfig:
    max_grid_size: int = 30
    max_train_grids: int = 10
    color_offset: int = 1


def get_task_from_file(file_name: str, task_id: str) -> dict:
    with open(file_name, "r") as f:
        tasks = json.load(f)
        return tasks[task_id]


def pad_and_mask_grid(
    grid: list[list[int]], config: ARCDatasetConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = len(grid), len(grid[0])
    if h > config.max_grid_size or w > config.max_grid_size:
        print("too large", h, w)
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
    challenges_file: str
    solutions_file: str
    task_ids: list[str]
    config: ARCDatasetConfig

    def __init__(
        self, challenges_file: str, solutions_file: str, config: ARCDatasetConfig
    ):
        self.challenges_file = challenges_file
        self.solutions_file = solutions_file
        with open(challenges_file, "r") as f:
            challenges = json.load(f)
            self.task_ids = list(challenges.keys())
        self.config = config

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx) -> dict:
        task_id = self.task_ids[idx]
        challenge = get_task_from_file(self.challenges_file, task_id)
        solution = get_task_from_file(self.solutions_file, task_id)

        grids = torch.zeros(
            2 * self.config.max_train_grids + 1,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.long,
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
        test_output_grid = pad_and_mask_grid(solution[0], self.config)[0]

        return {
            "task_id": task_id,
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
