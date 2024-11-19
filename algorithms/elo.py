from .ranking import Ranking, Result
from .dataset import Dataset, Game

from typing import List, Union, Dict
from dataclasses import dataclass, field

import copy
import random

class EloProbability:
    def __call__(self, a: Union[float, int], b: Union[float, int]) -> float:
        return 1 / (1 + 10 ** ((b - a) / 400))

@dataclass
class EloResult(Result):
    avg_elo: float = 0.0
    elo: List = field(default_factory=list)

    def calculate_elo(self):
        self.avg_elo = sum(self.elo) / len(self.elo)


class Elo(Ranking):
    def __init__(self, data: Dataset,  models: List[str], k: Union[int, float] = 32) -> None:
        super().__init__(data, models)
        self.results = self.elo_statistics()
        self.initial = 1500
        self.K = k
        self.probability = EloProbability()

    def elo_statistics(self) -> Dict[str, EloResult]:
        results = {}
        for model in self.models:
            results[model] = EloResult()

        return results

    def get_rank(self, model: str):
        return self.results[model].avg_elo

    def calculate_elo(self, total: int = 1) -> None:
        if type(total) != int:
            raise ValueError('Total must be an integer')

        if total < 1:
            raise ValueError('Total must be greater than 0')


        for model in self.models:
            self.results[model].elo = [self.initial for _ in range(total)]

        for i in range(total):
            for data in self.data:
                self.update_elo(data, index=i)

            if total > 1:
                random.shuffle(self.data)  # copy by reference

        for _, results in self.results.items():
            results.calculate_elo()

        return copy.deepcopy(self.results)



    def calculate_ranks(self, **kwargs):
        if 'total' not in kwargs:
            kwargs['total'] = 100

        self.calculate_elo(kwargs['total'])

        for model in self.models:
            self.ranks[model] = self.results[model].avg_elo

        return self.ranks

    def calculate_win_probability(self, elo_a, elo_b):
        return self.probability(elo_a, elo_b)

    def update_elo(self, data: Game, index: int):
        model1 = data.model1
        model2 = data.model2
        winner = data.selected

        ea = self.calculate_win_probability(self.results[model1].elo[index], self.results[model2].elo[index])
        eb = self.calculate_win_probability(self.results[model2].elo[index], self.results[model1].elo[index])

        sa = self.calculate_binary_outcome(model1, model2, winner)


        self.results[model1].elo[index] += self.K * (sa - ea)
        self.results[model2].elo[index] += self.K * (1 - sa - eb)

