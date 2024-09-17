import random
from itertools import combinations
from typing import List, Set, Tuple

from arc_prize.synth_data.utils import DatasetInterface, GridPair


class ARCEval009d5c81Dataset(DatasetInterface):
    def _generate_shape_commands(self) -> List[Set[Tuple[int, int]]]:
        all_cells = [(i, j) for i in range(3) for j in range(3)]
        commands = []
        for i in range(1, 9):  # Generate commands for 1 to 8 filled cells
            commands.extend(set(combo) for combo in combinations(all_cells, i))
        return commands

    def _generate_connected_shape(
        self, color: int, max_size: int = 10
    ) -> Set[Tuple[int, int]]:
        shape = {(0, 0)}
        size = random.randint(3, max_size)
        while len(shape) < size:
            x, y = random.choice(list(shape))
            new_cell = random.choice([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
            shape.add(new_cell)
        return shape

    def _draw_shape(
        self,
        grid: List[List[int]],
        shape: Set[Tuple[int, int]],
        color: int,
        offset: Tuple[int, int] = (0, 0),
    ) -> None:
        for x, y in shape:
            grid[x + offset[0]][y + offset[1]] = color

    def generate_task(self) -> Tuple[List[GridPair], GridPair]:
        shape_commands = self._generate_shape_commands()
        base_color = random.randint(0, self.num_colors - 1)
        command_color = random.choice(
            [c for c in range(self.num_colors) if c != base_color]
        )

        connected_shape = self._generate_connected_shape(base_color)

        # Generate 2 to 4 pairs
        num_pairs = random.randint(2, 4)
        pairs = []

        for _ in range(num_pairs):
            input_grid = [
                [0 for _ in range(self.grid_dim)] for _ in range(self.grid_dim)
            ]

            # Draw connected shape
            self._draw_shape(input_grid, connected_shape, base_color)

            # Select and draw command shape
            command_shape = random.choice(shape_commands)
            command_offset = (
                random.randint(0, self.grid_dim - 3),
                random.randint(0, self.grid_dim - 3),
            )
            self._draw_shape(input_grid, command_shape, command_color, command_offset)

            # Determine new color based on command shape
            new_color = random.choice(
                [
                    c
                    for c in range(self.num_colors)
                    if c not in [base_color, command_color]
                ]
            )

            # Create output grid
            output_grid = [row[:] for row in input_grid]

            # Remove command shape
            for x, y in command_shape:
                output_grid[x + command_offset[0]][y + command_offset[1]] = 0

            # Recolor connected shape
            for x, y in connected_shape:
                output_grid[x][y] = new_color

            pairs.append(GridPair(input=input_grid, output=output_grid))

        # Use the last pair as the test pair
        test_pair = pairs.pop()
        train_pairs = pairs

        return train_pairs, test_pair
