import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True)
class GridInput:
    input: list[list[int]]


@dataclass_json
@dataclass(frozen=True)
class GridOutput:
    output: list[list[int]]


@dataclass_json
@dataclass(frozen=True)
class GridPair(GridInput, GridOutput):
    pass


@dataclass_json
@dataclass(frozen=True)
class ChallengeTask:
    train: list[GridPair]
    test: list[GridInput]


@dataclass_json
@dataclass(frozen=True)
class DatasetTasks:
    challenges: dict[str, ChallengeTask]
    solutions: dict[str, list[list[list[int]]]]


def generate_grid(num_rows: int, num_cols: int):
    return [[0 for _ in range(num_cols)] for _ in range(num_rows)]


def random_color():
    return random.randint(1, 9)


class DatasetInterface(ABC):
    grid_dim: int
    max_train_pairs: int
    num_colors: int

    def __init__(
        self, grid_dim: int = 10, max_train_pairs: int = 4, num_colors: int = 10
    ) -> None:
        self.grid_dim = grid_dim
        self.max_train_pairs = max_train_pairs
        self.num_colors = num_colors

    @abstractmethod
    def generate_task(self) -> tuple[list[GridPair], GridPair]:
        pass

    def generate_tasks(self, num_tasks: int) -> DatasetTasks:
        task_prefix = self.__class__.__name__.lower()
        challenges: dict[str, ChallengeTask] = {}
        solutions: dict[str, list[list]] = {}
        for task_id in range(num_tasks):
            train_pairs, test_pair = self.generate_task()
            challenges[f"{task_prefix}_{task_id}"] = ChallengeTask(
                train=train_pairs, test=[GridInput(input=test_pair.input)]
            )
            solutions[f"{task_prefix}_{task_id}"] = [test_pair.output]

        return DatasetTasks(challenges=challenges, solutions=solutions)
