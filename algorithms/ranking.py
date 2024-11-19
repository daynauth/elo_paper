from abc import ABC, abstractmethod
import numpy as np

from typing import List, Dict, TypeVar, Union
from .dataset import Dataset

T = TypeVar('T')

class Result(ABC):
    ...

class Ranking(ABC):
    def __init__(self, data: Dataset, models: List[str]):
        self._data = data
        self._models = models
        self.ranks = {}

    @property
    def data(self) -> Dataset:
        return self._data

    @property
    def models(self) -> List[str]:
        return self._models

    @abstractmethod
    def get_rank(self, model: str) -> float:
        ...

    def get_rank_score(self, model: str) -> float:
        return self.get_rank(model)

    def calculate_win(self, p1: Union[str, float], p2: Union[str, float]) -> float:
        if isinstance(p1, str):
            p1 = self.get_rank(p1)
        if isinstance(p2, str):
            p2 = self.get_rank(p2)
        return self.calculate_win_probability(p1, p2)

    @abstractmethod
    def calculate_win_probability(self, p1: T, p2: T) -> float:
        ...

    def log_loss(self, model1, model2, winner) -> float:
        p12 = self.calculate_win_probability(self.get_rank(model1), self.get_rank(model2))
        if p12 < 0:
            raise ValueError('Invalid probability')

        if p12 > 1:
            raise ValueError(f'{p12} is an Invalid probability')

        y12 = self.calculate_binary_outcome(model1, model2, winner)
        return self.loss(p12, y12)

    @staticmethod
    def loss(p, y):
        if p == 0:
            p = 1e-15
        elif p == 1:
            p = 1 - 1e-15
        return -1 * (y * np.log(p) + (1 - y) * np.log(1 - p))

    @staticmethod
    def calculate_binary_outcome(model1, model2, winner) -> float:
        if winner == model1:
            return 1
        elif winner == model2:
            return 0
        elif winner == 'tie':
            return 0.5
        else:
            raise ValueError('Invalid winner')

    def mean_log_loss(self, games: Dataset=None, include_ties=True):
        if games is None:
            games = self.data

        total = 0
        for game in games:
            if game.selected == 'tie' and not include_ties:
                continue
            total += self.log_loss(game.model1, game.model2, game.selected)
        return total / len(self.data)

    def calculate_loss(self, include_ties=True):
        return self.mean_log_loss(include_ties=include_ties)

    def calculate_ranks(self, **kwargs) -> Dict[str, float]:
        ...

    def __str__(self) -> str:
        if len(self.ranks) == 0:
            self.calculate_ranks()

        _repr = ''
        for model, rank in sorted(self.ranks.items(), key=lambda x: x[1], reverse=True):
            _repr += f'{model}: {rank:.5f}\n'

        return _repr

