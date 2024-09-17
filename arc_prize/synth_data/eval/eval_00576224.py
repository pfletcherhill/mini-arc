from random import choice
from typing import List, Tuple

from arc_prize.synth_data.dsl_utils import unifint
from arc_prize.synth_data.utils import (
    DatasetInterface,
    GridPair,
)


def generate_00576224() -> dict:
    # Define possible colors
    colors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Generate output tile size (how many times the input grid is repeated)
    output_tile_size = unifint(0, 1, (2, 3))

    # Generate number of examples (1 to 4)
    num_examples = unifint(0, 1, (2, 5))

    examples = []
    for _ in range(num_examples):
        # Generate input grid size for this example
        input_height = unifint(0, 1, (2, 4))
        input_width = unifint(0, 1, (2, 4))

        # Create input grid with random colors
        input_grid = tuple(
            tuple(choice(colors) for _ in range(input_width))
            for _ in range(input_height)
        )

        # Create output grid by tiling the input grid
        output_grid = tuple(
            tuple(
                input_grid[i % input_height][j % input_width]
                for j in range(input_width * output_tile_size)
            )
            for i in range(input_height * output_tile_size)
        )

        examples.append({"input": input_grid, "output": output_grid})

    # Split examples into train and test sets
    train_size = unifint(
        0, 1, (1, max(1, num_examples - 1))
    )  # At least one example in train set
    return {"train": examples[:train_size], "test": examples[train_size:]}


class ARCEval00576224Dataset(DatasetInterface):
    def __init__(
        self, grid_dim: int = 12, max_train_pairs: int = 4, num_colors: int = 10
    ) -> None:
        super().__init__(grid_dim, max_train_pairs, num_colors)

    def generate_task(self) -> Tuple[List[GridPair], GridPair]:
        puzzle = generate_00576224()

        train_pairs = [
            GridPair(input=pair["input"], output=pair["output"])
            for pair in puzzle["train"]
        ]

        test_pair = (
            GridPair(
                input=puzzle["test"][0]["input"], output=puzzle["test"][0]["output"]
            )
            if puzzle["test"]
            else None
        )

        # If there's no test pair, use the last train pair as the test pair
        if test_pair is None:
            test_pair = train_pairs.pop()

        return train_pairs, test_pair
