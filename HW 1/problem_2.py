import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

TRAIN_DATA = "problem_2_train_data.txt"
VAL_DATA = "problem_2_val_data.txt"


def load_data(input_path):
    # Loads promoter and negative sequences from the input_path.
    # DO NOT modify this function.

    promoter_sequences, negative_sequences = [], []
    with open(input_path) as f:
        for line in f:
            seq, clazz = line.strip().split()
            if clazz == "1":
                promoter_sequences.append(seq)
            elif clazz == "0":
                negative_sequences.append(seq)
            else:
                raise Exception("All class values should be either 0 or 1.")

    return promoter_sequences, negative_sequences


def generate_states(k, chars, vals):
    if k == 0:
        return vals
    else:
        t = []
        for v in vals:
            for c in chars:
                t.append(v + c)
        return generate_states(k - 1, chars, t)


def train_markov_model(sequences, k):
    # Fits a Markov model where each state is a substring of size k.
    # These states are overlapping. So, if a sequence started with "ACTGA"
    # with k = 3, the first few states would be ["ACT", "CTG", "TGA", ...].
    #
    # returns:
    #	- states: an ordered list of all possible states in the Markov model
    #	- transition_matrix: a probability matrix (2D numpy array) with size 4^k by 4^k such that
    # 	                     transition_matrix[row][col] = P(pi_{i + 1} = state[col] | pi_{i} = state[row])
    #							* in the above notation, pi_{i} denotes the ith state in the sequence
    states = generate_states(k - 1, ["A", "C", "G", "T"], ["A", "C", "G", "T"])
    transition = np.zeros((4 ** k, 4 ** k), dtype=np.float128)
    idx = {v: i for i, v in enumerate(states)}
    totals = defaultdict(lambda: 0)

    def count(a, b):
        nonlocal totals
        if len(a) == k and len(b) == k:
            transition[idx[a]][idx[b]] += 1
            totals[idx[a]] += 1

    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            count(sequences[i][j: j + k], sequences[i][j + 1: j + k + 1])

    for k in totals:
        transition[k] = transition[k] / totals[k]

    return states, transition


def get_log_odds_ratio(seq, states, k, promoter_transition_matrix, negative_transition_matrix):
    # returns: log { P(sequence | promoter sequence model) / P(sequence | negative sequence model) }
    #
    # assume that all first states are equally likely. That is, P(pi_{0} = state) = 1 / 4^k for all states
    pos = neg = 1 / (4 ** k)
    idx = {s: i for i, s in enumerate(states)}
    for i in range(len(seq)):
        a, b = seq[i: i + k], seq[i + 1: i + k + 1]
        if len(a) != k or len(b) != k:
            break
        pos *= promoter_transition_matrix[idx[a]][idx[b]]
        neg *= negative_transition_matrix[idx[a]][idx[b]]

    return np.log(pos / neg)


def get_accuracy(promoter_sequences, negative_sequences, states, k,
                 promoter_transition_matrix, negative_transition_matrix):
    # Determine our model's accuracy on the given sequences.
    # Per our model, we classify a sequence as coming from a promoter iff it has a log odds ratio > 0.

    promoter = [1 if get_log_odds_ratio(s, states, k, promoter_transition_matrix, negative_transition_matrix) > 0 else 0
                for s in promoter_sequences]
    neg = [1 if get_log_odds_ratio(s, states, k, promoter_transition_matrix, negative_transition_matrix) <= 0 else 0
           for s in negative_sequences]
    correct = sum(promoter) + sum(neg)
    total = len(promoter) + len(neg)
    return correct / total


def main():
    train_promoter_sequences, train_negative_sequences = load_data(TRAIN_DATA)
    val_promoter_sequences, val_negative_sequences = load_data(VAL_DATA)
    train, val = [], []

    for k in range(1, 6):
        states, promoter_transition_matrix = train_markov_model(train_promoter_sequences, k)
        _, negative_transition_matrix = train_markov_model(train_negative_sequences, k)

        train_accuracy = get_accuracy(train_promoter_sequences, train_negative_sequences, states, k,
                                      promoter_transition_matrix, negative_transition_matrix)
        val_accuracy = get_accuracy(val_promoter_sequences, val_negative_sequences, states, k,
                                    promoter_transition_matrix, negative_transition_matrix)
        train.append(train_accuracy)
        val.append(val_accuracy)
        print("k = {}, train accuracy = {}, val accuracy = {}".format(k, train_accuracy, val_accuracy))
    return train, val


if __name__ == '__main__':
    train, val = main()
    plot1 = plt.figure(1)
    plt.plot(list(range(1, 6)), train)
    plt.title("Training accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plot2 = plt.figure(2)
    plt.title("Validation accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.plot(list(range(1, 6)), val)
    plt.show()
