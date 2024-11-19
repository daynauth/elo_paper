from abc import ABC, abstractmethod
from typing import Union


class Probability(ABC):
    @abstractmethod
    def __call__(self, a: Union[float, int], b: Union[float, int]) -> float:
        ...


class EloProbability(Probability):
    def __call__(self, a: Union[float, int], b: Union[float, int]) -> float:
        return 1 / (1 + 10 ** ((b - a) / 400))


class BradleyTerryProbability(Probability):
    def __call__(self, a: Union[float, int], b: Union[float, int]) -> float:
        return a / (a + b)
