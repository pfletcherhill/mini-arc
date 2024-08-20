import random
from typing import List, Tuple

import numpy as np

from arc_prize.synth_data.utils import DatasetInterface, GridPair


class RotateDataset(DatasetInterface):
    def generate_pair(self, rotation: int) -> GridPair:
        input_dim = random.randint(2, self.grid_dim)
        input_grid = np.random.randint(
            0, self.num_colors - 1, size=(input_dim, input_dim)
        )
        output_grid = np.rot90(input_grid, k=rotation)

        return GridPair(input=input_grid.tolist(), output=output_grid.tolist())

    def generate_task(self) -> Tuple[List[GridPair], GridPair]:
        num_train_pairs = random.randint(1, self.max_train_pairs)
        rotation = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        train_pairs = [self.generate_pair(rotation) for _ in range(num_train_pairs)]
        test_pair = self.generate_pair(rotation)
        return train_pairs, test_pair
