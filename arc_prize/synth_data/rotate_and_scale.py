import random
from typing import List, Tuple

import numpy as np

from arc_prize.synth_data.utils import DatasetInterface, GridPair


class RotateScaleDataset(DatasetInterface):
    def generate_pair(self, rotation: int, scale: int) -> GridPair:
        input_dim = random.randint(2, self.grid_dim // scale)
        input_grid = np.random.randint(
            0, self.num_colors - 1, size=(input_dim, input_dim)
        )

        # First, rotate the input grid
        rotated_grid = np.rot90(input_grid, k=rotation)

        # Then, scale the rotated grid
        output_grid = np.repeat(np.repeat(rotated_grid, scale, axis=0), scale, axis=1)

        # Ensure the output grid fits within the maximum dimensions
        output_grid = output_grid[: self.grid_dim, : self.grid_dim]

        return GridPair(input=input_grid.tolist(), output=output_grid.tolist())

    def generate_task(self) -> Tuple[List[GridPair], GridPair]:
        num_train_pairs = random.randint(1, self.max_train_pairs)
        rotation = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        scale = random.randint(2, 4)  # Scale factor between 2 and 4
        train_pairs = [
            self.generate_pair(rotation, scale) for _ in range(num_train_pairs)
        ]
        test_pair = self.generate_pair(rotation, scale)
        return train_pairs, test_pair
