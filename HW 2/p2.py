import heapq

import numpy as np


def apply_bonferroni_correction(pvalues, alpha):
    """
	Args:
		pvalues: numpy array of p-values
		alpha: desired familywise error rate (FWER)

	Returns:
		rejects: a numpy array of booleans indicating whether the null hypothesis should be rejected
				 at the given alpha
	"""

    return np.array([v <= alpha / len(pvalues) for v in pvalues])


def apply_benjamini_hochberg_correction(pvalues, alpha):
    """
	Args:
		pvalues: numpy array of independent p-values
		alpha: desired false discovery rate (FDR)

	Returns:
		rejects: a numpy array of booleans indicating whether the null hypothesis should be rejected
				 at the given alpha
	"""
    h = []
    for i, v in enumerate(pvalues):
        heapq.heappush(h, (v, i))
    v_s = set()
    for rank, v_i in enumerate(h):
        p_val, orig_i = v_i
        if (((rank + 1) / len(pvalues)) * alpha) < p_val:
            break
        v_s.add(orig_i)
    return np.array([i in v_s for i, v in enumerate(pvalues)])


def main():
    pass


if __name__ == '__main__':
    main()
