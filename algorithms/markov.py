"""
@Author: Roland Daynauth
@Date: 2024-02-01
@Description: This file contains the implementation of a Markov Chain to rank
the models based on their performance.
"""
from typing import List, Dict, Union
from dataclasses import dataclass, field

from .ranking import Ranking, Result
import numpy as np
import copy

from data_loader import loader, Dataset
from probability import BradleyTerryProbability


@dataclass
class MarkovResult(Result):
    wins_against: dict = field(default_factory=dict)
    losses_against: dict = field(default_factory=dict)

    games: int = 0
    wins: int = 0
    ties: int = 0
    losses: int = 0

    def update_wins_against(self, model: Union[str, None]):
        if model is None:
            return

        self.wins_against[model] = self.wins_against.get(model, 0) + 1
        if model not in self.losses_against:
            self.losses_against[model] = 0

    def update_losses_against(self, model: Union[str, None]):
        if model is None:
            return
        self.losses_against[model] = self.losses_against.get(model, 0) + 1

        if model not in self.wins_against:
            self.wins_against[model] = 0


def markov_statistics(
    models: List[str], data: Dataset, ties_are_wins=True
) -> Dict[str, MarkovResult]:
    results: Dict[str, MarkovResult] = {}
    for model in models:
        results[model] = MarkovResult()

    for game in data:
        results[game.model1].games += 1
        results[game.model2].games += 1

        if game.selected == 'tie':
            if ties_are_wins:
                results[game.model1].wins += 1
                results[game.model2].wins += 1
            else:
                results[game.model1].ties += 1
                results[game.model2].ties += 1

        elif game.selected == game.model1:
            results[game.model1].wins += 1
            results[game.model2].losses += 1
        else:
            results[game.model2].wins += 1
            results[game.model1].losses += 1

    for game in data:
        winner = game.get_winner()
        looser = game.get_looser()

        if winner is None or looser is None:
            if ties_are_wins:
                results[game.model1].update_wins_against(game.model2)
                results[game.model2].update_wins_against(game.model1)
        else:
            results[winner].update_wins_against(looser)
            results[looser].update_losses_against(winner)

    return results


class Markov(Ranking):
    def __init__(self, data: Dataset, models: List[str], p: float = 0.9):
        super().__init__(data, models)
        self.results = markov_statistics(models, data)
        self.p = p
        self.transition_matrix = self.generate_transition_matrix()
        self.ranks = {}
        self.probability = BradleyTerryProbability()

    def get_rank(self, model: str) -> float:
        return self.ranks[model]

    def calculate_win_probability(self, p1: float, p2: float):
        return self.probability(p1, p2)

    def generate_stable_state(
        self, inital_state: np.ndarray, iterations: int = 1000
    ):
        current_state = copy.deepcopy(inital_state)
        # current_state = inital_state

        for i in range(iterations):
            next_state = np.matmul(current_state, self.transition_matrix)
            if np.allclose(next_state, current_state):
                break

            current_state = next_state

        return current_state

    def generate_transition_matrix(self):
        return np.array([self.transition_row(model) for model in self.models])

    def transition_row(self, model: str):
        return [self.transition_element(model, m) for m in self.models]

    def transition_element(self, model: str, element: str):
        result = self.results[model]

        if element == model:
            return self.transition_same(result)

        return self.transition_different(result, element)

    def transition_same(self, result: MarkovResult):
        wins_contrib = result.wins * self.p
        losses_contrib = result.losses * (1 - self.p)
        total_games = result.wins + result.losses
        return (wins_contrib + losses_contrib) / total_games

    def wins_against(self, result: MarkovResult, model: str):
        if model not in result.wins_against:
            return 0

        return result.wins_against[model]

    def losses_against(self, result: MarkovResult, model: str):
        if model not in result.losses_against:
            return 0

        return result.losses_against[model]

    def transition_different(self, result: MarkovResult, model: str):
        if result.wins + result.losses == 0:
            print(f"Model: {model} has no games played")
        wins_contrib = self.wins_against(result, model) * (1 - self.p)
        losses_contrib = self.losses_against(result, model) * self.p
        total_games = result.wins + result.losses
        return (wins_contrib + losses_contrib) / total_games

    def calculate_ranks(self, **kwargs) -> Dict[str, float]:
        initial_state = np.array([1/len(self.models) for _ in self.models])
        current_state = self.generate_stable_state(initial_state)
        rank_list = list(zip(self.models, current_state))
        for model, prob in rank_list:
            self.ranks[model] = prob

        return self.ranks

    def display_ranks(self):
        print(self.ranks)


def markov_ranking(data_version: Union[int, str] = 2, p=0.7) -> Markov:
    data, models = loader(data_version)
    return Markov(data, models, p)
