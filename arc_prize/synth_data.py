import copy
import random


def generate_grid():
    return [[0 for _ in range(10)] for _ in range(10)]


def add_random_shape(grid):
    shape_size = random.randint(1, 3)
    color = random.randint(1, 9)
    start_row = random.randint(0, 9 - shape_size)
    start_col = random.randint(0, 9 - shape_size)

    for i in range(shape_size):
        for j in range(shape_size):
            grid[start_row + i][start_col + j] = color


def move_shapes(grid, direction, shift):
    new_grid = copy.deepcopy(grid)

    for _ in range(shift):
        if direction == "up":
            new_grid.pop(0)
            new_grid.append([0] * 10)
        elif direction == "down":
            new_grid.pop()
            new_grid.insert(0, [0] * 10)
        elif direction == "left":
            for row in new_grid:
                row.pop(0)
                row.append(0)
        elif direction == "right":
            for row in new_grid:
                row.pop()
                row.insert(0, 0)

    return new_grid


def generate_example(direction, shift):
    input_grid = generate_grid()
    num_shapes = random.randint(3, 8)
    for _ in range(num_shapes):
        add_random_shape(input_grid)

    output_grid = move_shapes(input_grid, direction, shift)

    return {
        "input": input_grid,
        "output": output_grid,
        "direction": direction,
        "shift": shift,
    }


def generate_dataset(num_tasks: int) -> tuple:
    challenges = {}
    solutions = {}
    for task_id in range(num_tasks):
        num_train_pairs = random.randint(1, 4)
        direction = random.choice(["up", "down", "left", "right"])
        shift = random.randint(1, 3)
        train_pairs = [
            generate_example(direction, shift) for _ in range(num_train_pairs)
        ]
        test_pair = generate_example(direction, shift)

        challenges[f"task_{task_id}"] = {
            "train": train_pairs,
            "test": [{"input": test_pair["input"]}],
        }
        solutions[f"task_{task_id}"] = [test_pair["output"]]

    return challenges, solutions
