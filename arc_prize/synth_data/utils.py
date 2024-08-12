import random


def generate_grid(num_rows: int, num_cols: int):
    return [[0 for _ in range(num_cols)] for _ in range(num_rows)]


def random_color():
    return random.randint(1, 9)
