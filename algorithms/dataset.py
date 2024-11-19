from dataclasses import dataclass, field
from typing import List, Union, Tuple

@dataclass
class Game:
    model1: str
    model2: str
    selected: str

    def __contains__(self, item) -> bool:
        return item == self.model1 or item == self.model2

    def get_looser(self) -> Union[str, None]:
        if self.selected == 'tie':
            return None
        if self.selected == self.model1:
            return self.model2
        return self.model1

    def get_winner(self) -> Union[str, None]:
        if self.selected == 'tie':
            return None
        return self.selected

    def get_winner_looser(self) -> Tuple[Union[str, None], Union[str, None]]:
        return self.get_winner(), self.get_looser()

    def get_opponent(self, model: str) -> str:
        if model == self.model1:
            return self.model2
        elif model == self.model2:
            return self.model1
        else:
            raise ValueError('Invalid model')

    def contains(self, model: str) -> bool:
        return model == self.model1 or model == self.model2

    def __str__(self) -> str:
        return f'{self.model1} vs {self.model2} - {self.selected}'


Dataset = List[Game]