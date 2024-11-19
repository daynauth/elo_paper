from data_loader import loader
from algorithms.elo import Elo
from algorithms.markov import Markov
from algorithms.glicko import Glicko

from typing import Union

def load_dataset(dataset):
    if dataset == "slam":
        data, models = loader(3)
    elif dataset == "arena":
        data, models = loader("arena_full")
    else:
        raise ValueError("Invalid dataset")
    return data, models

def get_ranker(
        data,
        models,
        algorithm: str,
        dataset:str="slam",
        h1: Union[int, float] = None,
        h2: Union[int, float] = None):
    if algorithm.lower() == "elo":
        if h1 is not None:
            k = h1
        else:
            k = 2.285 if dataset == "slam" else 0.825
        ranker = Elo(data, models, k=k)

    elif algorithm.lower() == "markov":
        if h1 is not None:
            p = h1
        else:
            p = 0.9125 if dataset == "slam" else 0.5434
        ranker = Markov(data, models, p=p)
    elif algorithm.lower() == 'glicko':
        if h1 is not None and h2 is not None:
            tau = h1
            rating = h2
            ranker = Glicko(data, models, rd=tau, c = rating)
        else:
            raise ValueError("Glicko requires two hyperparameters")
    else:
        raise ValueError("Invalid algorithm")

    return ranker