import random
from typing import List, Tuple

import numpy as np

from arc_prize.synth_data.utils import (
    DatasetInterface,
    GridPair,
    generate_grid,
    random_color,
)


class ARCecaa0ec1Dataset(DatasetInterface):
    def generate_grid_pair(
        self,
        indicator_color: int,
        align_indicator_offset: int,
        dir_indicator_offset: int,
    ) -> GridPair:
        num_rows = random.randint(9, 12)
        num_cols = random.randint(9, 12)
        input_grid = generate_grid(num_rows, num_cols)
        square_dim = random.randint(2, 3)
        square = np.random.randint(0, self.num_colors - 1, size=(input_dim, input_dim))
        output_grid = np.rot90(input_grid, k=rotation)

    def generate_task(self) -> Tuple[List[GridPair], GridPair]:
        num_train_pairs = random.randint(1, self.max_train_pairs)

        # Task-wide params
        indicator_color = random_color()
        align_indicator_offset = random.randint(1, 2)
        dir_indicator_offset = random.randint(2, 3)

        train_pairs = [
            self.generate_grid_pair(indicator_color) for _ in range(num_train_pairs)
        ]
        test_pair = self.generate_grid_pair(indicator_color)

        return train_pairs, test_pair
