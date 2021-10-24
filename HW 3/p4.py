import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
import time


DATA_CSV = "pulsar_data.csv"

def read_data():
	# DO NOT modify this function

	data = np.genfromtxt(DATA_CSV, skip_header=1, delimiter=',')
	X, Y = data[:, : 8], data[:, 8]
	X = np.nan_to_num(X)

	permutation = np.random.permutation(X.shape[0])
	train_idxs = permutation[: 1_000]
	test_idxs = permutation[1_000: 2_000]

	X_train, Y_train = X[train_idxs], Y[train_idxs]
	X_test, Y_test = X[test_idxs], Y[test_idxs]
	return X_train, Y_train, X_test, Y_test

def k_fold_cross_validation(X, Y, k):
	p = np.random.permutation(X.shape[0])
	X, Y = X[p], Y[p]
	for i in range(k):
		start = i * X.shape[0] // k
		end = (i + 1) * X.shape[0] // k
		X_train, Y_train = X[:start] + X[end:], Y[:start] + Y[end:]
		X_val, Y_val = X[start:end], Y[start:end]
		model = SVC(kernel='linear', C=1e5)

def plot_roc(model_type, roc, C):
	plt.figure()
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel("False positive rate")
	plt.ylabel("True positive rate")
	plt.title("{} ROC, C: {}, AUC: {}".format(model_type, C, metrics.auc(roc[0], roc[1])))
	plt.plot(roc[0], roc[1])
	plt.savefig("{}_roc.png".format(model_type))

def main():
	X_train, Y_train, X_test, Y_test = read_data()
	print("Finished reading data")
	p = np.random.permutation(X_train.shape[0])
	X, Y = X_train[p], Y_train[p]
	k = 5
	res_linear = []
	res_poly_2 = []
	res_poly_5 = []
	res_rbf = []
	runtime_linear = []
	runtime_poly_2 = []
	runtime_poly_5 = []
	runtime_rbf = []
	
	for exponent in range(-1, 3, 1):
		C = 10 ** exponent
		print("Training models for C = {}".format(C))
		res_linear.append([])
		res_poly_2.append([])
		res_poly_5.append([])
		res_rbf.append([])
		runtime_linear.append(0)
		runtime_poly_2.append(0)
		runtime_poly_5.append(0)
		runtime_rbf.append(0)	

		for i in range(k):
			start = i * X.shape[0] // k
			end = (i + 1) * X.shape[0] // k

			X_train_split, Y_train_split = np.vstack((X[:start, :], X[end:, :])), np.hstack((Y[:start], Y[end:]))
			X_val, Y_val = X[start:end, :], Y[start:end]
			model_linear = SVC(kernel='linear', C=C)
			model_poly_2 = SVC(kernel='poly', C=C, degree=2)
			model_poly_5 = SVC(kernel='poly', C=C, degree=5)
			model_rbf = SVC(kernel='rbf', C=C)
			start = time.time()
			model_linear.fit(X_train_split, Y_train_split)
			runtime_linear[-1] += time.time() - start
			start = time.time()
			model_poly_2.fit(X_train_split, Y_train_split)
			start = time.time()
			runtime_poly_2[-1] += time.time() - start
			start = time.time()
			model_poly_5.fit(X_train_split, Y_train_split)
			runtime_poly_5[-1] += time.time() - start
			start = time.time()
			model_rbf.fit(X_train_split, Y_train_split)
			runtime_rbf[-1] += time.time() - start
			res_linear[-1].append(model_linear.score(X_val, Y_val))
			res_poly_2[-1].append(model_poly_2.score(X_val, Y_val))
			res_poly_5[-1].append(model_poly_5.score(X_val, Y_val))
			res_rbf[-1].append(model_rbf.score(X_val, Y_val))
	
	mean_acc_linear = np.mean(res_linear, axis=1)
	mean_acc_poly_2 = np.mean(res_poly_2, axis=1)
	mean_acc_poly_5 = np.mean(res_poly_5, axis=1)
	mean_acc_rbf = np.mean(res_rbf, axis=1)


	best_C_linear_idx = np.argmax(mean_acc_linear)
	best_C_linear = 10.0 ** (best_C_linear_idx - 1)

	best_C_poly_2_idx = np.argmax(mean_acc_poly_2)
	best_C_poly_2 = 10.0 ** (best_C_poly_2_idx - 1)

	best_C_poly_5_idx = np.argmax(mean_acc_poly_5)
	best_C_poly_5 = 10.0 ** (best_C_poly_5_idx - 1)

	best_C_rbf_idx = np.argmax(mean_acc_rbf)
	best_C_rbf = 10.0 ** (np.argmax(best_C_rbf_idx) - 1)

	full_model_linear = SVC(kernel='linear', C=best_C_linear)
	full_model_linear.fit(X_train, Y_train)
	full_model_poly_2 = SVC(kernel='poly', C=best_C_poly_2, degree=2)
	full_model_poly_2.fit(X_train, Y_train)
	full_model_poly_5 = SVC(kernel='poly', C=best_C_poly_5, degree=5)
	full_model_poly_5.fit(X_train, Y_train)
	full_mode_rbf = SVC(kernel='rbf', C=best_C_rbf)
	full_mode_rbf.fit(X_train, Y_train)

	data = {'category': ['linear', 'poly_2', 'poly_5', 'rbf'],
			'C': [best_C_linear, best_C_poly_2, best_C_poly_5, best_C_rbf],
			'cross_validation_acc': [max(mean_acc_linear), max(mean_acc_poly_2), max(mean_acc_poly_5), max(mean_acc_rbf)],
			'test_acc': [full_model_linear.score(X_test, Y_test), full_model_poly_2.score(X_test, Y_test), full_model_poly_5.score(X_test, Y_test), full_mode_rbf.score(X_test, Y_test)],
			'runtime': [runtime_linear[best_C_linear_idx], runtime_poly_2[best_C_poly_2_idx], runtime_poly_5[best_C_poly_5_idx], runtime_rbf[best_C_rbf_idx]]
		}

	df = pd.DataFrame(data)
	print(df)
	"""
		category      C  cross_validation_acc  test_acc    runtime
	0   linear   10.0                 0.974     0.964  36.281712
	1   poly_2    1.0                 0.972     0.952   0.000015
	2   poly_5  100.0                 0.963     0.937   1.113672
	3      rbf    0.1                 0.974     0.937   0.051417
	"""

	roc_linear = metrics.roc_curve(Y_test, full_model_linear.decision_function(X_test))
	plot_roc('linear', roc_linear, best_C_linear)
	roc_poly_2 = metrics.roc_curve(Y_test, full_model_poly_2.decision_function(X_test))
	plot_roc('poly_2', roc_poly_2, best_C_poly_2)
	roc_poly_5 = metrics.roc_curve(Y_test, full_model_poly_5.decision_function(X_test))
	plot_roc('poly_5', roc_poly_5, best_C_poly_5)
	roc_rbf = metrics.roc_curve(Y_test, full_mode_rbf.decision_function(X_test))
	plot_roc('rbf', roc_rbf, best_C_rbf)

	"""
	This is because with increasing values of C, it is harder for a linear
	model to find a hyperplane that correctly separates the data as
	it has less room to 'wiggle' around the incorrectly classified datapoints, hence 
	it takes longer for the model to converge compared to models with more
	complex kernels.
	"""



if __name__ == '__main__':
	main()