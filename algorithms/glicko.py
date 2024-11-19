from .ranking import Ranking
import numpy as np
from typing import List, Dict, Tuple, Callable
from data_loader import Dataset, Game
from dataclasses import dataclass, field


@dataclass
class Player:
    name: str
    rating: float = 1500
    rd: float = 350
    r_prime: float = field(default=0, init=False)
    rd_prime: float = field(default=0, init=False)
    mu: float = 0
    phi: float = 0
    mu_prime: float = 0
    phi_prime: float = 0

    def __post_init__(self):
        self.r_prime = self.rating
        self.rd_prime = self.rd


class Glicko(Ranking):
    def __init__(
        self,
        data: Dataset,
        models: List[str],
        players: Dict[str, Player] = {},
        rd: float = 350,
        c: float = 0.06
    ):
        super().__init__(data, models)
        self.ranks = {}
        if players:
            self.players = players
        else:
            self.players = {model: Player(model, 1500, rd) for model in models}

        self.q = 0.0057565
        self.c = c
        self._base = 400
        self.completed = False

    def game_generator(self, model: str = ""):
        for game in self.data:
            if model != "":
                if game.contains(model):
                    yield game

    def g(self, rd: float) -> float:
        return 1 / (np.sqrt(1 + 3 * self.q ** 2 * rd ** 2 / np.pi ** 2))

    def E(self, r: float, rj: float, rd: float, method='decimal') -> float:
        if method == 'euler':
            return 1 / (1 + np.exp(-self.g(rd) * (r - rj)/self._base))
        else:
            return 1 / (1 + 10**(-self.g(rd) * (r - rj)/self._base))

    def update_ranking(self, model: Player):
        pass

    def d_squared(self, model: str) -> float:
        sum = 0
        r = self.players[model].rating

        for game in self.game_generator(model):
            oponent = game.get_opponent(model)
            rdj = self.players[oponent].rd
            rj = self.players[oponent].rating
            e = self.E(r, rj, rdj)
            sum += self.g(rdj)**2*e*(1 - e)

        if sum == 0:
            raise ValueError(f'{model}: Sum is zero')

        return 1 / (self.q ** 2 * sum)

    def r_prime(self, model: str) -> float:
        sr = self._sum(model, self.get_rank)
        rd_squared_inv = 1 / self.players[model].rd ** 2
        d_squared_inv = 1 / self.d_squared(model)
        fraction = self.q / (rd_squared_inv + d_squared_inv)
        return self.players[model].rating + fraction * sr

    def rd_prime(self, model: str) -> float:
        rd_squared_inv = 1 / self.players[model].rd ** 2
        d_squared_inv = 1 / self.d_squared(model)
        return np.sqrt(1 / (rd_squared_inv + d_squared_inv))

    def _op_info(
        self,
        game: Game,
        model: str,
        f: Callable[[str], Tuple[float, float]]
    ) -> Tuple[float, float, float]:
        if not game.contains(model):
            raise ValueError('Invalid model')

        opponent = game.get_opponent(model)

        sj = self.calculate_binary_outcome(model, opponent, game.selected)
        return (sj, *f(opponent))

    def _sum(
        self,
        model: str,
        f: Callable[[str], Tuple[float, float]],
        n: int = 1,
        method='decimal'
    ) -> float:
        r, _ = f(model)
        sum = 0
        for game in self.game_generator(model):
            sj, rj, rdj = self._op_info(game, model, f)
            sum += self.g(rdj) ** n * (sj - self.E(r, rj, rdj, method=method))

        return sum

    def calculate_glicko(self):
        for model in self.models:
            self.players[model].r_prime = self.r_prime(model)
            self.players[model].rd_prime = self.rd_prime(model)

        self.completed = True

    def calculate_ranks(self, **kwargs) -> Dict[str, float]:
        self.calculate_glicko()

        for model in self.models:
            self.ranks[model] = self.players[model].r_prime

        return self.ranks

    def calculate_win_probability(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        r1, rd1 = p1
        r2, rd2 = p2

        result = 10**(-self.g(np.sqrt(rd1**2 + rd2**2)) * (r1 - r2)/self._base)
        return 1 / (1 + result)

    def get_rank(self, model: str) -> Tuple[float, float]:
        if not self.completed:
            return self.players[model].rating, self.players[model].rd
        return self.players[model].r_prime, self.players[model].rd_prime

    def get_rank_score(self, model: str) -> float:
        return self.get_rank(model)[0]

    def ci(self, model: str) -> Tuple[float, float]:
        r, rd = self.get_rank(model)
        return (r - 1.96 * rd, r + 1.96 * rd)

    def sort_ranks(self):
        return sorted(
            self.players.items(),
            key=lambda x: x[1].r_prime,
            reverse=True
        )


class GlickoRankingV2(Glicko):
    def __init__(
        self,
        data: Dataset,
        models: List[str],
        players: Dict[str, Player] = {},
        tau=0.5,
        sigma=0.06
    ):
        super().__init__(data, models, players)
        self.tau = tau
        self._sigma = sigma
        self.phi_star = 173.7178
        self.a: float = np.log(self._sigma ** 2)
        self._A: float = self.a
        self._rescale_players()

    def _rescale_players(self):
        for model in self.models:
            self.players[model].mu = (
                self.players[model].rating - 1500
            ) / self.phi_star
            self.players[model].phi = self.players[model].rd / self.phi_star

    def get_secondary_rank(self, model: str) -> Tuple[float, float]:
        return self.players[model].mu, self.players[model].phi

    def _op_info(
        self,
        game: Game,
        model: str,
        f: Callable[[str], Tuple[float, float]]
    ) -> Tuple[float, float, float]:
        if not game.contains(model):
            raise ValueError('Invalid model')

        opponent = game.get_opponent(model)
        return (1, *f(opponent))

    def _sum(
        self,
        model: str,
        f: Callable[[str], Tuple[float, float]],
        n: int = 1
    ) -> float:
        mu, _ = f(model)
        sum = 0
        for game in self.game_generator(model):
            _, muj, phij = self._op_info(game, model, f)
            e = self.E(mu, muj, phij)
            sum += (self.g(phij) ** n) * e * (1 - e)

        return sum

    def v(self, model: str) -> float:
        return 1/self._sum(model, self.get_secondary_rank, 2)

    def sigma(self, model: str) -> float:
        sum = 0
        mu, _ = self.get_secondary_rank(model)
        for game in self.game_generator(model):
            oponent = game.get_opponent(model)
            muj, phij = self.get_secondary_rank(oponent)
            e = self.E(mu, muj, phij)
            sj = self.calculate_binary_outcome(model, oponent, game.selected)
            sum += self.g(phij) * (sj - e)

        return self.v(model) * sum

    def f(self, x: float, model: str) -> float:
        _, phi = self.get_secondary_rank(model)
        e = np.exp(x)
        sigma_sq = self.sigma(model) ** 2
        phi_sq = phi ** 2
        v_model = self.v(model)
        numerator: float = e * (sigma_sq - phi_sq - v_model - e)
        denominator: float = 2 * (phi**2 + self.v(model) + e)**2
        return (numerator / denominator) - ((x - self.a) / (self.tau ** 2))

    def A(self, model: str) -> float:
        return self.a

    def B(self, model: str) -> float:
        mu, phi = self.get_secondary_rank(model)
        if self.sigma(model) ** 2 > phi ** 2 + self.v(model):
            return np.log(self.sigma(model) ** 2 - phi ** 2 - self.v(model))

        k = 1
        while self.f(self.a - k * self.tau, model) < 0:
            k += 1

        self._B = self.a - k * self.tau

        return self._B

    def calculate_sigma_prime(self, model: str) -> float:
        A = self.a
        B = self.B(model)

        error = 0.000001
        fA = self.f(A, model)
        fB = self.f(B, model)

        while np.abs(B - A) > error:
            C = A + (A - B) * fA / (fB - fA)
            fC = self.f(C, model)
            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2

            B = C
            fB = fC

        return np.exp(A / 2)

    def calculate_phi_star(self, model: str) -> float:
        mu, phi = self.get_secondary_rank(model)
        sigma_prime = self.calculate_sigma_prime(model)
        phi_star = np.sqrt(phi ** 2 + sigma_prime ** 2)
        return phi_star

    def calculate_phi_prime(self, model: str) -> float:
        phi_star = self.calculate_phi_star(model)
        phi_prime = 1 / np.sqrt(1 / phi_star ** 2 + 1 / self.v(model))
        return phi_prime

    def calculate_mu_prime(self, model: str, phi_prime: float) -> float:
        mu, _ = self.get_secondary_rank(model)

        sum = 0
        for game in self.game_generator(model):
            oponent = game.get_opponent(model)
            muj, phij = self.get_secondary_rank(oponent)
            e = self.E(mu, muj, phij)
            sj = self.calculate_binary_outcome(model, oponent, game.selected)
            sum += self.g(phij) * (sj - e)

        return mu + phi_prime ** 2 * sum

    def calculate_glicko(self) -> None:
        for model in self.models:
            phi_prime = self.calculate_phi_prime(model)
            mu_prime = self.calculate_mu_prime(model, phi_prime)

            self.players[model].r_prime = mu_prime * self.phi_star + 1500
            self.players[model].rd_prime = phi_prime * self.phi_star

        self.completed = True

    def g(self, phi: float) -> float:
        return 1 / (np.sqrt(1 + 3 * phi ** 2 / np.pi ** 2))

    def calculate_win_probability(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        r1, rd1 = p1
        r2, rd2 = p2

        g_value = super().g(np.sqrt(rd1**2 + rd2**2))
        exponent = -g_value * (r1 - r2) / self._base
        result = 10**exponent
        return 1 / (1 + result)

    def E(self, r: float, rj: float, rd: float) -> float:
        return 1 / (1 + np.exp(-self.g(rd) * (r - rj)))
