# 00dbd492 Description
# - Puzzle-wide params:
#   - Border color
#   - Padding-to-color map, excluding the border color and black
# - Input grid:
#   - The grid should be between 7x7 and 30x30 or whatever the max dimension of the dataset is based on the config
#   - Draw 1-6 squares (or as many fit) that do not overlap or touch each other.
#   - The size of the squares should be odd (ie 5x5, 7x7, etc), with a point in the middle, 1-8 layers of black, then a 1 cell wide border.
#   - The minimum size for a square is 5x5.
#   - The center point and the border should be the border color.
# - Output grid:
#   - The output grid should be the same as the input grid except that instead of having black inside each square, the interior of the square should be filled in with a color determined by the padding-to-color map.
#   - For instance, if the padding-to-color map has {1 -> blue, 2 -> red}, then fill in the squares with 1 cell of padding with blue and the squares with 2 cells of padding with red.
#   - The center point and the border should still be the border color and unchanged.

import random
from typing import Optional

from arc_prize.synth_data.utils import DatasetInterface, GridPair


class ARCEval00dbd492Dataset(DatasetInterface):
    def draw_square(
        self,
        grid: list[list[int]],
        x: int,
        y: int,
        padding: int,
        border_color: int,
        fill_color: Optional[int],
    ):
        size = 3 + padding * 2

        # Draw border
        for i in range(size):
            grid[x + i][y] = grid[x + i][y + size - 1] = border_color
            grid[x][y + i] = grid[x + size - 1][y + i] = border_color

        if fill_color is not None:
            for i in range(1, size - 1):
                for j in range(1, size - 1):
                    grid[x + i][y + j] = fill_color

        # Draw center point
        grid[x + padding + 1][y + padding + 1] = border_color

        return grid

    def generate_task(self) -> tuple[list[GridPair], GridPair]:
        # Define puzzle-wide parameters
        border_color = random.randint(1, self.num_colors - 1)  # Exclude black (0)
        available_colors = [c for c in range(1, self.num_colors) if c != border_color]
        random.shuffle(available_colors)

        # Generate 2 to 5 pairs
        num_pairs = random.randint(2, 5)
        pairs = []

        for _ in range(num_pairs):
            # Generate input grid
            grid_size = random.randint(7, min(30, self.grid_dim))
            input_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
            output_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

            # Generate 1-6 squares
            num_squares = random.randint(1, 6)
            squares = []
            for _ in range(num_squares):
                max_attempts = 100
                for _ in range(max_attempts):
                    padding = random.randint(1, len(available_colors))
                    size = 3 + padding * 2
                    if size > grid_size:
                        continue
                    x = random.randint(0, grid_size - size)
                    y = random.randint(0, grid_size - size)

                    # Check if the square overlaps or touches existing squares
                    if all(
                        abs(x - sx) > size or abs(y - sy) > size
                        for sx, sy, ss in squares
                    ):
                        squares.append((x, y, padding))
                        break
                if len(squares) == num_squares:
                    break

            # Draw squares on the input grid
            for x, y, padding in squares:
                input_grid = self.draw_square(
                    input_grid, x, y, padding, border_color, None
                )
                fill_color = available_colors[padding - 1]
                output_grid = self.draw_square(
                    output_grid, x, y, padding, border_color, fill_color
                )

            pairs.append(GridPair(input_grid, output_grid))

        # Split pairs into train and test sets
        train_size = random.randint(1, len(pairs) - 1)
        return pairs[:train_size], pairs[train_size]
