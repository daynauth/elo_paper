from algorithms.dataset import Dataset, Game

import jsonlines
from typing import List, Tuple, Union, Set
import numpy as np


def load_data(file: str) -> Dataset:
    data: Dataset = []

    try:
        with jsonlines.open(file) as reader:
            for obj in reader:
                data.append(Game(obj['model1'], obj['model2'], obj['selected']))
    except Exception as e:
        print(e)
        return []

    return data

def get_models(data: Dataset) -> List[str]:
    models: Set[str] = set()

    for d in data:
        models.add(d.model1)
        models.add(d.model2)
    return list(models)


def loader(data_version: Union[int, str] = 2) -> Tuple[Dataset, List[str]]:
    if data_version == 2:
        file = './data/response_set_v2.jsonl'
    elif data_version == 3:
        file = './data/response_set_v3.jsonl'
    elif data_version == "arena":
        file = './data/chatbot_arena.jsonl'
    elif data_version == "arena_full":
        file = './data/chatbot_arena_full.jsonl'
    elif data_version == "rps":
        file = './data/rps.jsonl'
    else:
        file = './data/response_set_v1.jsonl'

    data = load_data(file)
    models = get_models(data)

    return data, models

def get_win_matrix(data: Dataset, models: List[str]) -> np.ndarray:
    """
    This function will return a matrix of wins between models.
    :param data: Dataset
    :param models: List[str]
    :return: np.ndarray
    """
    n: int = len(models)
    win_matrix = np.zeros(shape=(n, n), dtype=np.int32)
    model_index = {model: i for i, model in enumerate(models)}

    for d in data:
        if d.selected == 'tie':
            continue
        i = model_index[d.model1]
        j = model_index[d.model2]
        if d.selected == d.model1:
            win_matrix[i, j] += 1
        else:
            win_matrix[j, i] += 1
    return win_matrix