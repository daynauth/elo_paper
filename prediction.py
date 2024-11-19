from PIL.ImageChops import offset
from sklearn.metrics import f1_score

from algorithms.dataset import Dataset
from data_loader import load_data, get_win_matrix
from utils import load_dataset, get_ranker
from algorithms.elo import Elo


from typing import List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def plot_box(
        df: pd.DataFrame,
        ax: plt.Axes,
        xoffset: Union[int, float] = 0,
        color = 'black',
        linewidth = 2
) -> None:
    data = [df.iloc[row, 1:].values for row in range(len(df))]


    #remove the xlabels
    ax.set_xticklabels([])

    for i in range(len(data)):
        q1 = np.percentile(data[i], 25)
        q3 = np.percentile(data[i], 75)
        median = np.percentile(data[i], 50)

        #plot a dot at q1
        ax.plot(i+1 + xoffset, q1, 'o', color=color, markersize=linewidth)
        ax.plot(i+1 + xoffset, q3, 'o', color=color, markersize=linewidth)
        ax.plot([i + 1 + xoffset, i + 1 + xoffset], [q1, q3], color=color, linewidth=linewidth)


def plot_arena_f1_scores(ax, df1: pd.DataFrame, df2: pd.DataFrame = None, df3: pd.DataFrame = None, dataset: str = 'arena') -> None:
    """
    Plot the box plot of the F1 scores for each model.
    :param df1: pd.DataFrame
        DataFrame containing the F1 scores.
    :param df2: pd.DataFrame
        DataFrame containing the F1 scores for a different algorithm.
    :param df3: pd.DataFrame
        DataFrame containing the F1 scores for a different algorithm.
    :param dataset: str
        Name of the dataset.
    :return: None
    """

    # plot the box plot with the index as the model name in the x-axis,
    # and the F1 scores in the y-axis. The box plot should use the columns to
    # calculate the mean and interquartile range for the box plot

    # fig, ax = plt.subplots(figsize=(5, 3))

    models = list(df1['model'])

    # set y-axis to be between 0 and 1
    ax.set_ylim(0.6, 1)

    #set y ticks font size
    ax.tick_params(axis='y', labelsize=8)

    # label the x-axis with the model names
    df1_color = '#ea5545'
    df2_color = '#27aeef'
    df3_color = 'green'
    algorithm1 = 'Elo'
    algorithm2 = 'Markov'
    algorithm3 = 'Glicko'

    plot_box(df1, ax, xoffset=-0.15, color=df1_color)

    if df2 is not None:
        plot_box(df2, ax, xoffset = 0.15, color=df2_color)

    if df3 is not None:
        plot_box(df3, ax, xoffset = 0, color=df3_color)

    # ax.set_xticklabels(models, rotation=45, ha='right')
    # plt.xticks(range(1, len(models) + 1), models, rotation=45, ha='right', fontsize=12)
    ax.set_xticks(range(1, len(models) + 1), models, rotation=45, ha='right', fontsize=7)

    ax.set_ylabel('F1 Score', fontsize=8)


    # Add legends for both algorithms
    if dataset == 'slam':
        #set each legend to a column
        legend_elements = [
            plt.Line2D([0], [0], color=df1_color, lw=4, label=algorithm1),
            plt.Line2D([0], [0], color=df2_color, lw=4, label=algorithm2),
            plt.Line2D([0], [0], color=df3_color, lw=4, label=algorithm3)
        ]

        # ax.legend(handles=legend_elements, loc='lower right', ncol=3)
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 1.2),
                  fontsize = 8,
                  frameon = False,
                  ncol=3)


    # add x and y gridlines at alpha=0.1
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.title('F1 Scores for Each Model', fontsize=16)
    # ax.set_ylabel('F1 Score', fontsize=14)


def get_accuracy_scores(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * (precision * recall) / (precision + recall)
          if precision + recall > 0 else 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0

    return precision, recall, f1, accuracy


def get_accuracy_stats(expected, actual, total=0):
    if expected >= actual:
        tp = actual
        fp = expected - actual
        fn = 0
        tn = total - expected
    else:
        tp = expected
        fp = 0
        fn = actual - expected
        tn = total - actual

    assert tp + fp + fn + tn == total

    return tp, fp, fn, tn

def get_prediction_matrix(ranker, models):
    prediction_matrix = np.zeros((len(models), len(models)))
    for i in range(len(models)):
        for j in range(len(models)):
            if j > i:
                p12 = ranker.calculate_win(models[i], models[j])
                prediction_matrix[i, j] = p12
                prediction_matrix[j, i] = 1 - p12
            elif j == i:
                prediction_matrix[i, j] = 0.5

    return prediction_matrix

def get_f1_matrix(models, actual_win_matrix, prediction_matrix):
    f1_scores = []
    for i in range(len(models)):
        tp, fp, fn, tn = 0, 0, 0, 0
        for j in range(len(models)):
            if i != j:
                total_wins = actual_win_matrix[i, j] + actual_win_matrix[j, i]
                estimated_wins = int(prediction_matrix[i, j] * total_wins)
                _tp, _fp, _fn, _tn = get_accuracy_stats(
                    estimated_wins, actual_win_matrix[i, j],
                    int(actual_win_matrix[i, j]) + int(actual_win_matrix[j, i])
                )
                tp += _tp
                fp += _fp
                fn += _fn
                tn += _tn

        _, _, f1, _ = get_accuracy_scores(tp, fp, fn, tn)
        f1_scores.append(f1)

    return f1_scores

def get_f1_data(
        train_data: Dataset, models: List[str],
        algorithm: str,
        dataset_name: str,
        actual_win_matrix: np.ndarray,
        hyperparameters: Union[int, float] = None,
        h2 = None
) -> List[float]:
    """
    Calculate the F1 scores for each model compared to the actual win matrix.
    :param train_data: Dataset
        The training dataset used to calculate the ranks and the win probabilities.
    :param models: List[str]
        List of models in the dataset.
    :param algorithm: str
        Ranking algorithm to use.
    :param dataset_name:
        Name of the dataset used, either 'slam' or 'arena'.
    :param actual_win_matrix: np.ndarray
        The actual win matrix calculated from the test data.
    :param hyperparameters: dict
        Hyperparameters for the ranking algorithm.
    :param h2: Union[int, float]
    :return: List[float]
    """
    ranker = get_ranker(train_data, models, algorithm, dataset_name, hyperparameters, h2)
    ranks = ranker.calculate_ranks(total = 1)

    prediction_matrix = get_prediction_matrix(ranker, models)
    f1_scores = get_f1_matrix(models, actual_win_matrix, prediction_matrix)

    return f1_scores




def calculate_f1_over_interval(
        algorithm: str,
        dataset: str,
        interval: list,
        models: list, data: Dataset,
        win_matrix: np.ndarray,
        interval2: list = None
) -> pd.DataFrame:

    print(f"Calculating F1 scores for {algorithm} algorithm ... ")

    if interval2 is not None:
        assert len(interval) == len(interval2)

    f1_dict = {'model': models}
    for k in interval:
        if interval2 is not None:
            for k2 in interval2:
                key = f"{k}_{k2}"
                print(f"Calculating F1 scores for {algorithm} algorithm with k1={k} and k2={k2} ... ")
                f1_scores = get_f1_data(data, models, algorithm, dataset, win_matrix, k, k2)
                f1_dict[key] = f1_scores
                column = f"f1_{algorithm.lower()}_{k}_{k2}"
                f1_dict[column] = f1_scores

        else:
            f1_scores = get_f1_data(data, models, algorithm, dataset, win_matrix, k)
            column = f"f1_{algorithm.lower()}_{k}"
            f1_dict[column] = f1_scores

    df = pd.DataFrame(f1_dict)
    df = df.sort_values(by='model')

    return df

def slam_partial_prediction(ax) -> None:
    """
    This function will be used to predict the outcome of a games in the test set.
    :return: None
    """
    dataset = 'slam'
    train_file = f"./data/prediction/{dataset}_75_train_data.jsonl"
    test_file = f"./data/prediction/{dataset}_75_test_data.jsonl"
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    _, models = load_dataset(dataset)
    win_matrix = get_win_matrix(test_data, models)

    algorithm = 'Elo'


    df1_path = f"./data/prediction/{dataset}_elo_f1_scores.csv"

    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
    else:
        interval = np.linspace(0, 100, 100, endpoint=False)
        interval = interval.tolist()
        df1 = calculate_f1_over_interval(algorithm, dataset, interval, models, train_data, win_matrix)


        # save the DataFrame to a CSV file
        df1.to_csv(df1_path, index=False)




    df2_path = f"./data/prediction/{dataset}_markov_f1_scores.csv"

    if os.path.exists(df2_path):
        df2 = pd.read_csv(df2_path)
    else:
        # range of 100 points between 0.5 and 1 exclusive
        interval = np.linspace(0.5, 1, 100, endpoint=False)
        interval = interval.tolist()

        df2 = calculate_f1_over_interval('Markov', dataset, interval, models, train_data, win_matrix)
        df2.to_csv(df2_path, index=False)




    df3_path = f"./data/prediction/{dataset}_glicko_f1_scores.csv"

    if os.path.exists(df3_path):
        df3 = pd.read_csv(df3_path)
    else:
        interval1 = np.linspace(100, 500, 10, endpoint=False)
        interval1 = interval1.tolist()
        interval2 = np.linspace(0.01, 0.1, 10, endpoint=False)
        interval2 = interval2.tolist()

        df3 = calculate_f1_over_interval('glicko', dataset, interval1, models, train_data, win_matrix, interval2)
        df3.to_csv(df3_path, index=False)

    plot_arena_f1_scores(ax, df1, df2, df3, dataset)





def arena_partial_prediction(ax):
    dataset = 'arena'
    train_file = f"./data/prediction/{dataset}_75_train_data.jsonl"
    test_file = f"./data/prediction/{dataset}_75_test_data.jsonl"
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    _, models = load_dataset(dataset)
    win_matrix = get_win_matrix(test_data, models)


    algorithm = 'Elo'
    df1_path = f"./data/prediction/{dataset}_elo_f1_scores.csv"

    if os.path.exists(df1_path):
        df1 = pd.read_csv(df1_path)
    else:
        f1_dict = {'model': models}
        interval = np.linspace(0, 100, 100, endpoint=False).tolist()

        for k in interval:
            f1_scores = get_f1_data(train_data, models, algorithm, dataset, win_matrix, k)
            column = f"f1_{algorithm.lower()}_{k}"
            f1_dict[column] = f1_scores

        df1 = pd.DataFrame(f1_dict)
        df1 = df1.sort_values(by='model')
        df1.to_csv(df1_path, index=False)

        # df1 = df1[0:11]

    df2_path = f"./data/prediction/{dataset}_markov_f1_scores.csv"

    if os.path.exists(df2_path):
        df2 = pd.read_csv(df2_path)
    else:
        # range of 100 points between 0.5 and 1 exclusive
        interval = np.linspace(0.5, 1, 100, endpoint=False)
        interval = interval.tolist()

        df2 = calculate_f1_over_interval('Markov', dataset, interval, models, train_data, win_matrix)
        df2.to_csv(df2_path, index=False)


    df3_path = f"./data/prediction/{dataset}_glicko_f1_scores.csv"

    if os.path.exists(df3_path):
        df3 = pd.read_csv(df3_path)
    else:
        interval1 = np.linspace(100, 500, 10, endpoint=False)
        interval1 = interval1.tolist()
        interval2 = np.linspace(0.01, 0.1, 10, endpoint=False)
        interval2 = interval2.tolist()

        df3 = calculate_f1_over_interval('glicko', dataset, interval1, models, train_data, win_matrix, interval2)
        df3.to_csv(df3_path, index=False)

    # choose models with the shortest names to display
    selected_models = [
        'alpaca-13b',
        'chatglm-6b',
        'claude-1',
        'dolly-v2-12',
        'gemini-pro',
        'gpt-4-0314',
        'koala-13b',
        'llama-13',
        'openchat-3.5',
        'palm-2',
        'vicuna-7b',

    ]

    df1 = df1[df1['model'].isin(selected_models)]
    df2 = df2[df2['model'].isin(selected_models)]
    df3 = df3[df3['model'].isin(selected_models)]

    plot_arena_f1_scores(ax, df1, df2, df3, dataset)


def plot_partial_prediction():
    fig, (ax1, ax2) = plt.subplots(2)
    # fig.text(0.01, 0.5, 'F1 Score', va='center', rotation='vertical', fontsize=10)

    #set size of the figure
    fig.set_size_inches(6, 5)

    slam_partial_prediction(ax1)
    arena_partial_prediction(ax2)

    plt.tight_layout()
    plt.savefig('elo_sensitivity.png', dpi=300)

def elo_prediction(k):
    """
    Calculate the predicted wins for each model using Elo ranking.
    :param k:
    :return:
    """
    data, models = load_dataset("arena")

    #sort models alphabeically
    models.sort()

    ranker = Elo(data, models, k=k)
    ranks = ranker.calculate_ranks(total=1)


    win_matrix = get_win_matrix(data, models)
    prediction_matrix = get_prediction_matrix(ranker, models)

    # add win_matrix to transpose
    total = win_matrix + win_matrix.T

    # item wide multiplcation
    predicted_wins = prediction_matrix * total

    # round to 0 decimal places
    predicted_wins = np.round(predicted_wins)

    # change to int
    predicted_wins = predicted_wins.astype(int)

    print(win_matrix)
    print(prediction_matrix)
    print(total)
    print(predicted_wins)

    # print models
    print(models)

    actual_wins = win_matrix.sum(axis=1)
    predicted_wins = predicted_wins.sum(axis=1)
    print(actual_wins)
    print(predicted_wins)
