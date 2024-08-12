import random

from arc_prize.synth_data.utils import generate_grid, random_color


def add_random_square(grid):
    shape_size = random.randint(1, 4)
    color = random_color()
    start_row = random.randint(0, 9 - shape_size)
    start_col = random.randint(0, 9 - shape_size)

    for i in range(shape_size):
        for j in range(shape_size):
            grid[start_row + i][start_col + j] = color

    return grid


def flip_grid(grid, axis):
    if axis == "horizontal":
        return [row[::-1] for row in grid]

    elif axis == "vertical":
        return grid[::-1]


def generate_example(axis: str):
    input_grid = generate_grid(10, 10)
    num_shapes = random.randint(3, 8)
    for _ in range(num_shapes):
        input_grid = add_random_square(input_grid)

    output_grid = flip_grid(input_grid, axis)

    return {"input": input_grid, "output": output_grid, "axis": axis}


def generate_dataset(num_tasks: int) -> tuple:
    challenges = {}
    solutions = {}
    for task_id in range(num_tasks):
        task_key = f"flip_{task_id}"
        num_train_pairs = random.randint(1, 4)
        axis = random.choice(["vertical", "horizontal"])
        train_pairs = [generate_example(axis) for _ in range(num_train_pairs)]
        test_pair = generate_example(axis)

        challenges[task_key] = {
            "train": train_pairs,
            "test": [{"input": test_pair["input"]}],
        }
        solutions[task_key] = [test_pair["output"]]

    return challenges, solutions
