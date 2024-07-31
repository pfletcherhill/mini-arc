import copy
import random

# TODO: update this file to handle many other patterns and sizes


def generate_grid():
    return [[0 for _ in range(10)] for _ in range(10)]


def add_random_shape(grid):
    shape_size = random.randint(1, 3)
    color = random.randint(1, 9)
    start_row = random.randint(0, 9 - shape_size)
    start_col = random.randint(0, 8 - shape_size)  # Ensure space to move right

    for i in range(shape_size):
        for j in range(shape_size):
            grid[start_row + i][start_col + j] = color


def move_right(grid):
    new_grid = copy.deepcopy(grid)
    for row in range(10):
        new_grid[row] = [0] + new_grid[row][:-1]
    return new_grid


def generate_example():
    input_grid = generate_grid()
    num_shapes = random.randint(1, 3)
    for _ in range(num_shapes):
        add_random_shape(input_grid)
    output_grid = move_right(input_grid)
    return {"input": input_grid, "output": output_grid}


def generate_dataset(num_tasks: int) -> tuple:
    challenges = {}
    solutions = {}
    for task_id in range(num_tasks):
        num_train_pairs = random.randint(1, 4)
        train_pairs = [generate_example() for _ in range(num_train_pairs)]
        test_pair = generate_example()

        challenges[f"task_{task_id}"] = {
            "train": train_pairs,
            "test": [{"input": test_pair["input"]}],
        }
        solutions[f"task_{task_id}"] = [test_pair["output"]]

    return challenges, solutions
