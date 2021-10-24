import numpy as np
from scipy.special import logsumexp

obs = [1, 2, 3, 4, 5, 6]

LOGZERO = "LOGZERO"


def get_data():
    with open("problem_5_data.txt") as f:
        rolls = f.readline().strip()
        return [int(r) for r in rolls]


def print_parameters(transition_matrix, emission_1, emission_2):
    print("Transition probability matrix: {}".format(transition_matrix))
    print("Emission probabilities for dice 1: {}".format(emission_1))
    print("Emission probabilities for dice 2: {}".format(emission_2))


def log(x):
    if x == 0:
        return LOGZERO
    elif x < 0:
        raise ValueError("Log needs to be >= 0")
    else:
        return np.log(x)


def log_add(x, y):
    if x == LOGZERO or y == LOGZERO:
        if x == LOGZERO:
            return y
        else:
            return x
    return max(x, y) + np.log(1 + np.exp(min(x, y) - max(x, y)))


def log_prod(x, y):
    if x == LOGZERO or y == LOGZERO:
        return LOGZERO
    else:
        return x + y


def exp(x):
    def helper(y):
        if y == LOGZERO:
            return 0
        else:
            return np.exp(y)

    return np.vectorize(helper)(x)


def forward_trellis(observations, e, k, t, pi):
    dp = np.zeros((k, len(observations)), dtype=np.float64)
    for i in range(k):
        dp[i][0] = log_prod(pi[i][0], e[observations[0] - 1][i])
    for o in range(1, len(observations)):
        for j in range(k):
            alph = LOGZERO
            for i in range(k):
                alph = log_add(t[i][j] + dp[i][o - 1], alph)
            dp[j][o] = log_prod(alph, e[observations[o] - 1][j])
    return dp


def backward_trellis(observations, e, k, t, pi):
    dp = np.zeros((k, len(observations)), dtype=np.float64)
    for i in range(k):
        dp[i][0] = 0
    for o in reversed(range(len(observations) - 1)):
        for i in range(k):
            beta = LOGZERO
            for j in range(k):
                beta = log_add(beta, log_prod(t[i][j], log_prod(e[observations[o + 1] - 1][j], dp[j][o + 1])))
            dp[i][o] = beta
    return dp


def run_baum_welch(rolls, thres):
    # return:
    # 	- 2 x 2 transition probability matrix
    #	- 6-dimensional array containing emission probabilities for dice 1
    #	- 6-dimensional array containing emission probabilities for dice 2
    states = [0, 1]  # Dice 1, dice 2
    trans = np.log(np.full((2, 2), 0.5, dtype=np.float64))
    pi_i = np.random.rand(len(states), 1)
    s = np.sum(pi_i, axis=0, dtype=np.float64)
    pi_i = pi_i / s[None, :]
    pi_i = np.vectorize(log)(pi_i)

    e = np.random.rand(len(obs), len(states))  # obs | state
    # sum over rows (all states add to 1)
    s = np.sum(e, axis=0, dtype=np.float64)
    e = e / s[None, :]
    e = np.vectorize(log)(e)

    f = forward_trellis(rolls, e, 2, trans, pi_i)
    prob = log_add(f[0][-1], f[1][-1])
    print("Log probability of: {}".format(prob))



    while prob < thres:
        b = backward_trellis(rolls, e, len(states), trans, pi_i)
        gamma = np.zeros((len(rolls), len(states)), dtype=np.float64)
        for t in range(len(rolls)):
            norm = LOGZERO
            for i in states:
                gamma[t][i] = log_prod(f[i][t], b[i][t])
                norm = log_add(norm, gamma[t][i])
            for i in states:
                gamma[t][i] = log_prod(gamma[t][i], -norm)

        eps = np.zeros((len(rolls) - 1, len(states), len(states)), dtype=np.float64)
        for t in range(len(rolls) - 1):
            norm = LOGZERO
            for i in states:
                for j in states:
                    eps[t][i][j] = log_prod(f[j][t],
                                            log_prod(trans[i][j], log_prod(e[rolls[t + 1] - 1][j], b[j][t + 1])))
                    norm = log_add(norm, eps[t][i][j])
            for i in states:
                for j in states:
                    eps[t][i][j] = log_prod(eps[t][i][j], -norm)

        for s in states:
            pi_i[s] = gamma[0][s]

        for i in states:
            for j in states:
                top = bottom = LOGZERO
                for t in range(len(rolls) - 1):
                    top = log_add(top, eps[t][i][j])
                    bottom = log_add(bottom, gamma[t][i])
                trans[i][j] = log_prod(top, -bottom)

        for j in states:
            for ob in obs:
                top = bottom = LOGZERO
                for t in range(len(rolls)):
                    if rolls[t] == ob:
                        top = log_add(top, gamma[t][j])
                    bottom = log_add(bottom, gamma[t][j])
                e[ob - 1][j] = log_prod(top, -bottom)

        # E_kb = np.zeros((len(states), len(obs)), dtype=np.float64)
        # for i, ob in enumerate(obs):
        #     f_b = f[:, args[i]]
        #     b_b = b[:, args[i]]
        #     E_kb[:, ob - 1] = logsumexp(f_b + b_b, axis=1) - prob
        # t = np.zeros((len(states), len(states)), dtype=np.float64)
        # for k in states:
        #     for l in states:
        #         t[k][l] = logsumexp(
        #             [f[k][i] + trans[k][l] + e[rolls[i + 1] - 1][l] + b[l][i + 1] for i in range(len(rolls) - 1)]
        #         ) - prob
        # e = (E_kb - logsumexp(E_kb, axis=1)[:, None]).T
        # trans = t
        # print(e)
        # print("TRANS")
        # print(trans)
        # print(pi_i)
        f = forward_trellis(rolls, e, len(states), trans, pi_i)
        prob = log_add(f[0][-1], f[1][-1])
        print("Log probability of: {}".format(prob))
        print(exp(trans))
        print(exp(e.T))
        print("==============")
    return exp(trans), exp(e.T[0]), exp(e.T[1])

# TODO: write this function


def main():
    rolls = get_data()

    # Part A
    print("Part A")
    transition_matrix_A, emission_1_A, emission_2_A = run_baum_welch(rolls, -178400)
    print_parameters(transition_matrix_A, emission_1_A, emission_2_A)
    print("")

    # Part B
    print("Part B")
    transition_matrix_B, emission_1_B, emission_2_B = run_baum_welch(rolls[: 1000], -1772)
    print_parameters(transition_matrix_B, emission_1_B, emission_2_B)


if __name__ == '__main__':
    main()
