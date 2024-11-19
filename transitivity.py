from algorithms.dataset import Dataset, Game
from utils import load_dataset
from algorithms.elo import Elo


import numpy as np

def get_p_matrix(data: Dataset, models: list[str]) -> np.ndarray:
    """
    Calculate the win rate matrix for the given data and models.
    :param data: Dataset
    :param models: List of model names
    :return: 2D numpy array representing the win rate matrix
    """

    p = np.zeros((len(models), len(models)))
    m = np.zeros((len(models), len(models)))

    for game in data:
        model1 = game.model1
        model2 = game.model2

        if game.selected == model1:
            m[models.index(model1), models.index(model2)] += 1
        elif game.selected == model2:
            m[models.index(model2), models.index(model1)] += 1

    for i in range(len(models)):
        for j in range(len(models)):
            if i < j:
                if m[i, j] + m[j, i] == 0:
                    p[i, j] = 0.5
                    p[j, i] = 0.5
                else:
                    p[i, j] = m[i, j] / (m[i, j] + m[j, i])
                    p[j, i] = 1 - p[i, j]

    return p


def model_wins(p: np.array, i: int, j: int) -> bool:
    """
    Check if model i wins against model j based on the probability matrix p.
    :param p: 2D numpy array representing the probability matrix
    :param i: index of the first model
    :param j: index of the second model
    :return: True if model i wins against model j, False otherwise
    """
    assert i >= 0 and j >= 0, "Indices must be greater than 0"
    assert i < p.shape[0] and j < p.shape[1], "Indices must be within the bounds of the matrix"

    return p[i, j] > 0.5

def calculate_transitive_triads(p: np.array, models: list[str], base: int) -> list[list[str]]:
    """
    Calculate transitive triads for a given base model.
    :param p: 2D numpy array representing the probability matrix
    :param models: List of model names
    :param base: index of the base model
    :return: List of 3 models that form a transitive triad
    """
    triads = []
    for i in range(len(models)):
        if i == base:
            continue

        if model_wins(p, base, i):
            for j in range(len(models)):
                if j == base or j == i:
                    continue

                if model_wins(p, i, j):
                    if model_wins(p, base, j):
                        triads.append([models[base], models[i], models[j]])

    return triads


def print_triads(triads: list[list[str]], ranks: dict, win_rate_matrix: np.array, models: list[str]) -> None:
    """
    Print the transitive triads in a readable format.
    :param triads: List of transitive triads
    :param ranks: Dictionary of model ranks
    :param win_rate_matrix: 2D numpy array representing the win rate matrix
    :param models: List of model names
    """
    for triad in triads:
        if ranks[triad[0]] > ranks[triad[1]] > ranks[triad[2]]:
            continue

        if ranks[triad[0]] > ranks[triad[1]]:
            continue

        if (ranks[triad[1]] - ranks[triad[0]]) < 100:
            continue

        if ranks[triad[1]] > ranks[triad[2]]:
            continue

        print(" > ".join(triad),
              f"Ranks: {[ranks[m] for m in triad]},"
              f" Win Rates: {[win_rate_matrix[models.index(triad[0]), models.index(m)] for m in triad]}")

def convert_score_to_rank(ranks: dict[str, float], model: str) -> float:
    """
    Convert a model's score to its rank.
    :param ranks: Dictionary of model ranks
    :param model: Model name
    :return: Rank of the model
    """

    # convert rank score to numbered position
    return sorted(ranks.values(), reverse=True).index(ranks[model]) + 1



def analyse_transitivity(algorithm: str = "elo"):
    dataset = "arena"
    data, models = load_dataset(dataset)

    # sort models
    models = sorted(models)

    p = get_p_matrix(data, models)

    # get all transitive triads for base
    triads = []
    for m in models:
        base = models.index(m)
        triads += calculate_transitive_triads(p, models, base)

    # print(triads)
    # print(len(triads))


    ranker = Elo(data, models, k=32)
    ranks = ranker.calculate_ranks(total=1)

    print_triads(triads, ranks, p, models)  # Print the triads

    model1 = 'gpt-3.5-turbo-1106'
    model2 = 'gemini-pro'
    model3 = 'llama-2-13b-chat'

    print( f"Rank of {model1}: {convert_score_to_rank(ranks, model1)}")
    print( f"Rank of {model2}: {convert_score_to_rank(ranks, model2)}")
    print( f"Rank of {model3}: {convert_score_to_rank(ranks, model3)}")

    # model 1 vs 2 win-rate
    print(f"Win rate of {model1} vs {model2}: {p[models.index(model1), models.index(model2)]}")

    # model 2 vs 3 win-rate
    print(f"Win rate of {model2} vs {model3}: {p[models.index(model2), models.index(model3)]}")

    # model 1 vs 3 win-rate
    print(f"Win rate of {model1} vs {model3}: {p[models.index(model1), models.index(model3)]}")

    # print_triads( triads)
    # df = pd.read_csv(f"./data/{dataset}_rank_comparison.csv", index_col=0)
    #
    #
    #
    # column = df[algorithm]
    #
    # transitivity = 0
    # for t in triads:
    #     if column[t[0]] > column[t[1]] > column[t[2]]:
    #         if column[t[0]] > column[t[2]]:
    #             transitivity += 1
    #
    # print(
    #     f"Transitivity for {algorithm}: "
    #     f"{transitivity/len(triads) * 100:.2f}"
    # )
