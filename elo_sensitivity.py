"""
plot for showing how elo rating changes with different parameters
"""
from algorithms.elo import Elo
from utils import load_dataset

import matplotlib.pyplot as plt

def elo_prediction(k):
    data, models = load_dataset("arena")
    ranker = Elo(data, models, k=k)
    ranks = ranker.calculate_ranks(total = 1)
    print(ranks)
    # sort ranks by elo score
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return sorted_ranks

def elo_predictions():
    k1 = 20
    k2 = 32

    ranks1 = elo_prediction(k=k1)
    ranks2 = elo_prediction(k=k2)

    print( f"Elo ranks with k={k1}:")
    for model, rank in ranks1:
        print(f"{model}: {rank:.5f}")

    print(f"\nElo ranks with k={k2}:")
    for model, rank in ranks2:
        print(f"{model}: {rank:.5f}")

    # print models side by side
    print("\nComparison of Elo ranks:")
    print(f"{'Model':<20} {'Model'}")
    for (model1, rank1), (model2, rank2) in zip(ranks1, ranks2):
        print(f"{model1:<20} {model2}")


    # print(ranks)