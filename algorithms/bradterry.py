from ranking import Ranking
from dataset import Dataset

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from typing import List, Union, Dict
import math

"""
Bradley-Terry Ranking
Adapted from Chatbox Arena
https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mSizG3Pzglte

"""

class BradleyTerryProbability:
    def __call__(self, a: Union[float, int], b: Union[float, int]) -> float:
        return a / (a + b)


class BradTerryRanking(Ranking):
    def __init__(self, data: Dataset, models: List[str], weighted: bool = False):
        super().__init__(data, models)
        self.base = 10
        self.initial = 1500
        self.scale = 400
        self.probability = BradleyTerryProbability()
        self.weighted = weighted

    def _calculate_bradterry(self, df: pd.DataFrame):
        models = pd.concat([df["model1"], df["model2"]]).unique()
        models = pd.Series(np.arange(len(models)), index=models)

        df = pd.concat([df, df], ignore_index=True)
        p = len(models.index)
        n = df.shape[0]

        x = np.zeros((n, p))
        x[np.arange(n), models[df["model1"]]] = +math.log(self.base)
        x[np.arange(n), models[df["model2"]]] = -math.log(self.base)

        y = np.zeros(n)
        y[df["selected"] == df["model1"]] = 1.0
        tie_idx = df["selected"] == "tie"
        tie_idx[len(tie_idx) // 2:] = False
        y[tie_idx] = 1.0

        lr = LogisticRegression(fit_intercept=False)
        lr.fit(x, y)

        return lr, models

    def _calculate_weighted_bradterry(self, df: pd.DataFrame):
        ptbl_a_win = pd.pivot_table(
            df[df["selected"] == df["model1"]],
            index="model1",
            columns="model2",
            aggfunc="size", # type: ignore
            fill_value=0,
        ) # type: ignore

        if sum(df["selected"] == "tie") == 0:
            ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
        else:
            ptbl_tie = pd.pivot_table(
                df[df["selected"] == "tie"],
                index="model1",
                columns="model2",
                aggfunc="size", # type: ignore
                fill_value=0,
            ) # type: ignore
            ptbl_tie = ptbl_tie + ptbl_tie.T


        ptbl_b_win = pd.pivot_table(
            df[df["selected"] == df["model2"]],
            index="model1",
            columns="model2",
            aggfunc="size", # type: ignore
            fill_value=0,
        ) # type: ignore

        ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

        models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)
        p = len(models)
        x = np.zeros([p * (p - 1) * 2, p])
        y = np.zeros(p * (p - 1) * 2)

        cur_row = 0
        sample_weights = []
        for m_a in ptbl_win.index:
            for m_b in ptbl_win.columns:
                if m_a == m_b:
                    continue
                # if nan skip
                if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                    continue
                x[cur_row, models[m_a]] = +math.log(self.base)
                x[cur_row, models[m_b]] = -math.log(self.base)
                y[cur_row] = 1.0
                sample_weights.append(ptbl_win.loc[m_a, m_b])

                x[cur_row + 1, models[m_a]] = math.log(self.base)
                x[cur_row + 1, models[m_b]] = -math.log(self.base)
                y[cur_row + 1] = 0.0
                sample_weights.append(ptbl_win.loc[m_b, m_a])
                cur_row += 2
        x = x[:cur_row]
        y = y[:cur_row]

        lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
        lr.fit(x, y, sample_weight=sample_weights)

        return lr, models

    def calculate_bradterry(self):
        df = pd.DataFrame(self.data)
        if self.weighted:
            lr, models = self._calculate_weighted_bradterry(df)
        else:
            lr, models = self._calculate_bradterry(df)

        scores = self.scale * lr.coef_[0] + self.initial
        self.ranks = {model: score for model, score in zip(models.index, scores)}
        return self.ranks

    def calculate_ranks(self, **kwargs) -> Dict[str, float]:
        return self.calculate_bradterry()

    def display_ranks(self):
        for model, rank in sorted(self.ranks.items(), key=lambda x: x[1], reverse=True):
            print(f"{model}: {rank:.2f}")

    def get_rank(self, model: str) -> float:
        if len(self.ranks) == 0:
            self.calculate_bradterry()

        return self.ranks[model]

    def calculate_win_probability(self, p1: float, p2: float) -> float:
        return self.probability(p1, p2)