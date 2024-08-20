import random

import numpy as np

from arc_prize.synth_data.utils import (
    DatasetInterface,
    GridPair,
)


class ScaleDataset(DatasetInterface):
    def generate_pair(self, zoom: int) -> GridPair:
        input_dim = random.randint(2, self.grid_dim // zoom)
        input_grid = np.random.randint(
            0, self.num_colors - 1, size=(input_dim, input_dim)
        )
        output_grid = np.repeat(np.repeat(input_grid, zoom, axis=0), zoom, axis=1)

        return GridPair(input=input_grid.tolist(), output=output_grid.tolist())

    def generate_task(self) -> tuple[list[GridPair], GridPair]:
        num_train_pairs = random.randint(1, 4)
        zoom = random.randint(2, 4)
        train_pairs = [self.generate_pair(zoom) for _ in range(num_train_pairs)]
        test_pair = self.generate_pair(zoom)
        return train_pairs, test_pair
