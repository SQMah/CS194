import math
import numpy as np
from scipy import stats
from scipy.special import logsumexp

### BEGIN: HMM model parameters

states_map = {
    0: "Nick",
    1: "Coach",
    2: "Winston"
}

means = {  # mean of each friend's driving speed (modeled by a Gaussian)
    0: 60.0,
    1: 65.0,
    2: 63.0
}

variances = {  # variance of each friend's driving speed (modeled by a Gaussian)
    0: 5.0,
    1: 6.0,
    2: 8.0
}

# transition_matrix[row][col] = P(pi_{i} = col | pi_{i - 1} = row)
transition_matrix = np.asarray([
    [0.90, 0.05, 0.05],
    [0.10, 0.80, 0.10],
    [0.15, 0.15, 0.70]
])


### END: HMM model parameters

def read_data():
    # Helper function to read data.
    # DO NOT modify this function.

    num_samples = 10
    observation_length = 100

    observations_matrix = np.zeros((num_samples, observation_length), dtype=np.float32)
    most_probable_states = np.zeros((num_samples, observation_length), dtype=np.int32)
    observation_probs = np.zeros((num_samples), dtype=np.float64)

    with open("problem_4_data.txt") as f:
        for i in range(num_samples):
            observations = f.readline().strip().split(',')
            viterbi_output = f.readline().strip().split(',')
            forward_output = float(f.readline().strip())
            f.readline()  # empty line

            observations_matrix[i] = np.asarray(list(map(float, observations)))
            most_probable_states[i] = np.asarray(list(map(int, viterbi_output)))
            observation_probs[i] = float(forward_output)

    return observations_matrix, most_probable_states, observation_probs


def emission_probabilites(observations):
    # P(observation | state)
    norms = [stats.norm(means[j], math.sqrt(variances[j])) for j in states_map]
    res = np.zeros((len(states_map), len(observations)), dtype=np.float64)
    for i, ob in enumerate(observations):
        for j in states_map:
            res[j][i] = norms[j].pdf(ob)
    return res


def run_viterbi(observations):
    # observations: a numpy array of driving speeds
    # returns: numpy array of states with the same length as observations
    y = observations
    A = transition_matrix
    B = emission_probabilites(observations)
    K = A.shape[0]
    Pi = np.full(K, 1 / K)
    T = len(y)
    T1 = np.zeros((K, T), dtype=np.float64)
    T2 = np.zeros((K, T), dtype=np.float64)

    T1[:, 0] = Pi * B[:, 0]
    T2[:, 0] = 0

    for i in range(1, T):
        T1[:, i] = np.max(np.float64(T1[:, i - 1]) * A.T * np.float64(B[np.newaxis, :, i].T), 1)
        T2[:, i] = np.argmax(np.float64(T1[:, i - 1] * A.T), 1)

    # Backtrack
    x = np.zeros(T, dtype=np.int32)
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    return x


def get(val):
    if np.isneginf(val):
        return 0
    else:
        return val


def run_forward_algorithm(observations):
    # observations: a numpy array of driving speeds
    # returns: floating point probability of observing these observations under the model
    e = emission_probabilites(observations)
    k = len(states_map)
    dp = np.zeros((k, len(observations)), dtype=np.float64)
    for k in states_map:
        dp[k][0] = np.float64(1 / len(states_map)) * e[k][0]
    for i in range(1, len(observations)):
        for l in states_map:
            dp[l][i] = np.float64(e[l][i] * sum([transition_matrix[k][l] * dp[k][i - 1] for k in states_map]))
    return sum([dp[k][-1] for k in states_map])


def forward_trellis(observations):
    e = emission_probabilites(observations)
    k = len(states_map)
    dp = np.zeros((k, len(observations)), dtype=np.float64)
    for k in states_map:
        dp[k][0] = np.float64(1 / len(states_map)) * e[k][0]
    for i in range(1, len(observations)):
        for l in states_map:
            dp[l][i] = np.float64(e[l][i] * sum([transition_matrix[k][l] * dp[k][i - 1] for k in states_map]))
    return dp


def backward_trellis(observations):
    e = emission_probabilites(observations)
    k = len(states_map)
    dp = np.zeros((k, len(observations)), dtype=np.float64)
    for k in states_map:
        dp[k][-1] = 1
    for i in reversed(range(len(observations) - 1)):
        for l in states_map:
            dp[l][i] = np.float64(sum([e[k][i+1] * transition_matrix[l][k] * dp[k][i + 1] for k in states_map]))
    return dp


def run_posterior_decoding(observations):
    # observations: a numpy array of driving speeds
    # returns: a 3 x m numpy array where m is the length of the observations array and
    # 		   arr[j][l] = P(pi_{l} = states[j] | observations)
    # note: No tests for this function are provided. Write your own to ensure correctness.
    res = np.zeros((len(states_map), len(observations)), dtype=np.float64)
    forward = forward_trellis(observations)
    backward = backward_trellis(observations)
    p_x = sum([forward[k][-1] for k in states_map])
    for i in states_map:
        for j in range(len(observations)):
            res[i][j] = np.float64(forward[i][j] * backward[i][j] / p_x)
    return res


def main():
    (observations, most_probable_states, observation_probs) = read_data()

    for i in range(observations.shape[0]):
        if not np.allclose(most_probable_states[i], run_viterbi(observations[i])):
            raise Exception("run_viterbi() is incorrect for example {}".format(i + 1))
        f = run_forward_algorithm(observations[i])
        b = backward_trellis(observations[i])
        e = emission_probabilites(observations[i])
        if not math.isclose(observation_probs[i], f, rel_tol=1e-03):
            print(observation_probs[i], f)
            raise Exception("run_forward_algorithm() is incorrect for example {}".format(i + 1))

        # Test that forward and backwards give the same probabilities
        if not math.isclose(f, sum([1/len(states_map) * e[l][0] * b[l][0] for l in states_map]), rel_tol=1e-03):
            print(f, sum([1/len(states_map) * e[l][0] * b[l][0] for l in states_map]))
            raise Exception("Forward and backward was not equal for example {}".format(i + 1))


if __name__ == '__main__':
    main()
