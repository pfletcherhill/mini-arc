import copy
import random

import numpy as np

from arc_prize.synth_data.utils import DatasetInterface, GridPair, generate_grid


class MoveDiagonalScaleDataset(DatasetInterface):
    def _add_random_square(self, grid: list) -> list:
        max_shape_size = min(3, len(grid), len(grid[0]))
        shape_size = random.randint(1, max_shape_size)
        color = random.randint(1, self.num_colors - 1)
        start_row = random.randint(0, len(grid) - shape_size)
        start_col = random.randint(0, len(grid[0]) - shape_size)

        for i in range(shape_size):
            for j in range(shape_size):
                grid[start_row + i][start_col + j] = color

        return grid

    def _shift_grid(
        self, grid: list, vertical_shift: int, horizontal_shift: int
    ) -> list:
        new_grid = copy.deepcopy(grid)
        width = len(new_grid[0])

        for _ in range(abs(vertical_shift)):
            if vertical_shift < 0:
                new_grid.pop(0)
                new_grid.append([0] * width)
            elif vertical_shift > 0:
                new_grid.pop()
                new_grid.insert(0, [0] * width)

        for _ in range(abs(horizontal_shift)):
            if horizontal_shift < 0:
                for row in new_grid:
                    row.pop(0)
                    row.append(0)
            elif horizontal_shift > 0:
                for row in new_grid:
                    row.pop()
                    row.insert(0, 0)

        return new_grid

    def _scale_grid(self, grid: list, scale: int) -> list:
        np_grid = np.array(grid)
        scaled_grid = np.repeat(np.repeat(np_grid, scale, axis=0), scale, axis=1)
        return scaled_grid.tolist()

    def generate_pair(
        self, vertical_shift: int, horizontal_shift: int, scale: int
    ) -> GridPair:
        input_dim = max(4, self.grid_dim // scale)  # Ensure input_dim is at least 4
        input_grid = generate_grid(input_dim, input_dim)
        num_shapes = random.randint(1, max(1, min(8, input_dim * input_dim // 4)))
        for _ in range(num_shapes):
            input_grid = self._add_random_square(input_grid)

        scaled_grid = self._scale_grid(input_grid, scale)
        output_grid = self._shift_grid(scaled_grid, vertical_shift, horizontal_shift)

        # Ensure the output grid fits within the maximum dimensions
        output_grid = [row[: self.grid_dim] for row in output_grid[: self.grid_dim]]

        return GridPair(input=input_grid, output=output_grid)

    def generate_task(self) -> tuple[list[GridPair], GridPair]:
        num_train_pairs = random.randint(1, 4)
        vertical_shift = random.randint(-3, 3)
        horizontal_shift = random.randint(-3, 3)
        scale = 2  # for now
        train_pairs = [
            self.generate_pair(vertical_shift, horizontal_shift, scale)
            for _ in range(num_train_pairs)
        ]
        test_pair = self.generate_pair(vertical_shift, horizontal_shift, scale)
        return train_pairs, test_pair
