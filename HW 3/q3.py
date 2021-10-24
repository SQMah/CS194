import numpy as np
from matplotlib import pyplot as plt

probs = np.arange(0.01, 1, 0.01)

error_rate = [1 - max(v, 1-v) for v in probs]
gini = [2*v*(1-v) for v in probs]
entropy = [-v*np.log2(v) - (1-v)*np.log2(1-v) for v in probs]

plt.plot(probs, error_rate, label='Error Rate')
plt.plot(probs, gini, label='Gini Index')
plt.plot(probs, entropy, label='Entropy')
plt.xlabel('p_1')
plt.ylabel('Rates')
plt.legend()
plt.savefig('q3.png')
