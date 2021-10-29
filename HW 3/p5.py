import re
import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import cPickle as pickle
import pyBigWig
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pickle
from p4 import plot_roc

np.random.seed(123)

def get_vals(lines, gene_map, default=50):
    value_res = np.zeros(len(lines))
    value_res.fill(default)
    for i, line in enumerate(lines):
        tokens = re.split(r'\t+', line.rstrip())
        infos = tokens[7].split(";")
        for info in infos:
            if info.startswith("GENES"):
                genes = info.split("=")[1].split(",")
                value = 0
                for gene in genes:
                    if gene in gene_map:
                        value += gene_map[gene]
                    else:
                        value += default
                if len(genes):
                    value_res[i] = value / len(genes)
    return value_res

def get_phastcons(line, bw_f):
    tokens = re.split(r'\t+', line.rstrip())
    chrom = tokens[0]
    pos = int(tokens[1])
    ref = tokens[3]
    value = bw_f.stats("chr" + chrom, pos, pos+len(ref))[0]
    if value is None:
        return 0
    return value

def main():
    benign = [] # CLASS LABEL: 0
    pathogenic = [] # CLASS LABEL: 1

    if not os.path.exists("cache"):
        os.makedirs("cache")

    def process_line(line):
        if not line.startswith("#"):
            tokens = re.split(r'\t+', line.rstrip())
            sample_id = tokens[2]
            ref = tokens[3]
            alt_str = tokens[4]
            alts = alt_str.split(",")
            infos = tokens[7].split(";")
            for info in infos:
                if info == "CLNSIG=Benign":
                    benign.append(line)
                elif info == "CLNSIG=Pathogenic":
                    pathogenic.append(line)

    with open("clinvar_missense.vcf", 'r') as f:
        for l in f:
            process_line(l)

    print("Number of benign variants:", len(benign))
    print("Number of pathogenic variants:", len(pathogenic))

    # Part (b) Shuffle and 80/20 split.
    y_benign = np.zeros(len(benign))
    y_pathogenic = np.ones(len(pathogenic))
    y = np.concatenate((y_benign, y_pathogenic))
    x = np.concatenate((benign, pathogenic))
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    x_train, x_val = x[:int(0.8*len(x))], x[int(0.8 * len(x)):]
    y_train, y_val = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
    print("Number of benign in training: {}, number of pathogenic in training {}"
        .format(len(y_train) - np.sum(y_train), np.sum(y_train)))
    print("Number of benign in validation: {}, number of pathogenic in validation {}"
        .format(len(y_val) - np.sum(y_val), np.sum(y_val)))

    """
    Number of benign variants: 24346
    Number of pathogenic variants: 20596
    Number of benign in training: 19416.0, number of pathogenic in training 16537.0
    Number of benign in validation: 4930.0, number of pathogenic in validation 4059.0
    """
    
    # Part (c) Construct features from variants for RVIS
    rvis_map = dict()
    with open("rvis.txt", 'r') as f:
        for i, l in enumerate(f):
            if i != 0:
                tokens = l.split("\t")
                rvis_map[tokens[0]] = float(tokens[3])
    
    if os.path.exists("cache/rvis_train.npy"):
        rvis_train = np.load("cache/rvis_train.npy")
        rvis_val = np.load("cache/rvis_val.npy")
        rvis_benign = np.load("cache/rvis_benign.npy")
        rvis_pathogenic = np.load("cache/rvis_pathogenic.npy")
        rvis_total = np.concatenate((rvis_train, rvis_val))
    else:
        rvis_train = get_vals(x_train, rvis_map)
        rvis_val = get_vals(x_val, rvis_map)
        rvis_total = np.concatenate((rvis_train, rvis_val))
        mask_benign = np.where(y == 0)[0]
        mask_pathogenic = np.where(y == 1)[0]
        rvis_benign = rvis_total[mask_benign]
        rvis_pathogenic = rvis_total[mask_pathogenic]
        np.save("cache/rvis_train.npy", rvis_train)
        np.save("cache/rvis_val.npy", rvis_val)
        np.save("cache/rvis_benign.npy", rvis_benign)
        np.save("cache/rvis_pathogenic.npy", rvis_pathogenic)
    
        bins = np.linspace(min(rvis_total), max(rvis_total), 50)
        fig, axs = plt.subplots(2,sharex=True, sharey=True)
        axs[0].hist(rvis_benign, bins, alpha=0.5, label="Benign")
        axs[1].hist(rvis_pathogenic, bins, alpha=0.5, label="Pathogenic")
        axs[0].set_title("RVIS distribution benign")
        axs[1].set_title("RVIS distribution pathogenic")
        plt.savefig("rvis_distribution.png")
    """
    It does not look like a particularly good feature. They share the
    same peak bin, and the peak bin is not very wide, even with a higher
    bin size. Most of them are at 50%, and the rest is spread out 
    fairy uniformly. However, we also know that any genes with RVIS > 50%
    is also more likely to be benign based on the histogram.
    """

    # Part (d) get o/e ratios
    o_e_map = dict()
    total = num = 0
    with open("gnomad.v2.1.1.lof_metrics.by_gene.txt", "r") as f:
        for i, l in enumerate(f):
            tokens = l.split("\t")
            if i != 0:
                if tokens[34] != "NA":
                    o_e_map[tokens[0]] = float(tokens[34])
                    total += float(tokens[34])
                    num += 1
    avg_o_e = total/num

    if os.path.exists("cache/o_e_train.npy"):
        o_e_train = np.load("cache/o_e_train.npy")
        o_e_val = np.load("cache/o_e_val.npy")
        o_e_benign = np.load("cache/o_e_benign.npy")
        o_e_pathogenic = np.load("cache/o_e_pathogenic.npy")
        o_e_total = np.concatenate((o_e_train, o_e_val))
    else:
        o_e_train = get_vals(x_train, o_e_map, default=avg_o_e)
        o_e_val = get_vals(x_val, o_e_map, default=avg_o_e)
        o_e_total = np.concatenate((o_e_train, o_e_val))
        mask_benign = np.where(y == 0)[0]
        mask_pathogenic = np.where(y == 1)[0]
        o_e_benign = o_e_total[mask_benign]
        o_e_pathogenic = o_e_total[mask_pathogenic]
        np.save("cache/o_e_train.npy", o_e_train)
        np.save("cache/o_e_val.npy", o_e_val)
        np.save("cache/o_e_benign.npy", o_e_benign)
        np.save("cache/o_e_pathogenic.npy", o_e_pathogenic)
    
        bins = np.linspace(min(o_e_total), max(o_e_total), 50)
        fig, axs = plt.subplots(2,sharex=True, sharey=True)
        axs[0].hist(o_e_benign, bins, alpha=0.5, label="Benign")
        axs[1].hist(o_e_pathogenic, bins, alpha=0.5, label="Pathogenic")
        axs[0].set_title("O/E scores benign")
        axs[1].set_title("O/E scores pathogenic")
        plt.savefig("o_e_scores.png")

    """
    Based on the plot, O/E scores are a good feature. Pathogenic
    variants are more likely to have O/E scores < 2500, and benign
    variants have a higher frequency for scores > 12500.
    """

    # Part (e) get PhastCons scores
    bw = pyBigWig.open("hg38.phastCons100way.bw")
    if os.path.exists("cache/phastcons_train.npy"):
        phastcons_train = np.load("cache/phastcons_train.npy")
        phastcons_val = np.load("cache/phastcons_val.npy")
        phastcons_benign = np.load("cache/phastcons_benign.npy")
        phastcons_pathogenic = np.load("cache/phastcons_pathogenic.npy")
        phastcons_total = np.concatenate((phastcons_train, phastcons_val))
    else:
        phastcons_train = [get_phastcons(x, bw) for x in x_train]
        phastcons_val = [get_phastcons(x, bw) for x in x_val]
        phastcons_total = np.concatenate((phastcons_train, phastcons_val))
        mask_benign = np.where(y == 0)[0]
        mask_pathogenic = np.where(y == 1)[0]
        phastcons_benign = phastcons_total[mask_benign]
        phastcons_pathogenic = phastcons_total[mask_pathogenic]
        np.save("cache/phastcons_train.npy", phastcons_train)
        np.save("cache/phastcons_val.npy", phastcons_val)
        np.save("cache/phastcons_benign.npy", phastcons_benign)
        np.save("cache/phastcons_pathogenic.npy", phastcons_pathogenic)

        bins = np.linspace(min(phastcons_total), max(phastcons_total), 20)
        fig, axs = plt.subplots(2,sharex=True, sharey=True)
        axs[0].hist(phastcons_benign, bins, alpha=0.5, label="Benign")
        axs[1].hist(phastcons_pathogenic, bins, alpha=0.5, label="Pathogenic")
        axs[0].set_title("PhastCons scores benign")
        axs[1].set_title("PhastCons scores pathogenic")
        plt.savefig("phastcons_scores.png")

    """
    Based on the plot, PhastCons scores are a good feature. Benign variants
    are much more likely to have a score <0.2, whereas pathogenic variants
    are more likely to have a score > 0.8. There isn't too much
    overlap in between.
    """

    # Part (f) Create feature matrix
    X = np.concatenate((np.expand_dims(rvis_total, -1), np.expand_dims(o_e_total, -1), np.expand_dims(phastcons_total, -1)), axis=1)
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # k-cross validation
    p = np.random.permutation(len(y))
    X = X[p]
    X_norm = X_norm[p]    
    Y = y[p]
    k = 5
    model_results = {"SVM_linear": [], "SVM_POLY2": [], "SVM_POLY5": [], "SVM_RBF": [], "RF": []}
    for i in range(k):
        print("="*10)
        print("Fold {}".format(i))
        start = i * X.shape[0] // k
        end = (i + 1) * X.shape[0] // k
        a, b = X[:start], X[end:]
        if a.shape[0] and b.shape[0]:
            x_train = np.vstack([X[:start, :], X[end:, :]])
            x_norm_train = np.vstack([X_norm[:start, :], X_norm[end:, :]])
            y_train = np.hstack((Y[:start], Y[end:]))
        else:
            if a.shape[0]:
                x_train = X[:start, :]
                x_norm_train = X_norm[:start, :]
                y_train = Y[:start]
            else:
                x_train = X[end:, :]
                x_norm_train = X_norm[end:, :]
                y_train = Y[end:]

        x_val = X[start:end, :]
        x_norm_val = X_norm[start:end, :]
        y_val = Y[start:end]
        C = 10
        model_linear = SVC(kernel='linear', C=0.1)
        model_poly_2 = SVC(kernel='poly', C=C, degree=2)
        model_poly_5 = SVC(kernel='poly', C=C, degree=5)
        model_rbf = SVC(kernel='rbf', C=C)
        model_rf = RandomForestClassifier(n_estimators=100)
       
        # Found n_estimators = 100 worked the best using this code
        # for i in range(20, 120, 20):
        #     print("Estimators: {}".format(i))
        #     model_rf = RandomForestClassifier(n_estimators=i)
        #     model_rf.fit(x_train, y_train)
        #     rf_score = model_rf.score(x_val, y_val)
        #     print("Random forest train acc {}".format(model_rf.score(x_train, y_train)))
        #     print("Random forest val acc {}".format(rf_score))

        print("Training linear SVM")
        model_linear.fit(x_norm_train, y_train)
        linear_score = model_linear.score(x_norm_val, y_val)
        print("Linear SVM train acc {}".format(model_linear.score(x_norm_train, y_train)))
        print("Linear SVM val acc {}".format(linear_score))
        print("Training poly2 SVM")
        model_poly_2.fit(x_norm_train, y_train)
        poly_2_score = model_poly_2.score(x_norm_val, y_val)
        print("Poly2 SVM train acc {}".format(model_poly_2.score(x_norm_train, y_train)))
        print("Poly2 SVM val acc {}".format(poly_2_score))
        print("Training poly5 SVM")
        model_poly_5.fit(x_norm_train, y_train)
        poly_5_score = model_poly_5.score(x_norm_val, y_val)
        print("Poly5 SVM train acc {}".format(model_poly_5.score(x_norm_train, y_train)))
        print("Poly5 SVM val acc {}".format(poly_5_score))
        print("Training rbf SVM")
        model_rbf.fit(x_norm_train, y_train)
        rbf_score = model_rbf.score(x_norm_val, y_val)
        print("RBF SVM train acc {}".format(model_rbf.score(x_norm_train, y_train)))
        print("RBF SVM val acc {}".format(rbf_score))
        print("Training random forest")
        model_rf.fit(x_train, y_train)
        rf_score = model_rf.score(x_val, y_val)
        print("Random forest train acc {}".format(model_rf.score(x_train, y_train)))
        print("Random forest val acc {}".format(rf_score))

        model_results["SVM_linear"].append(linear_score)
        model_results["SVM_POLY2"].append(poly_2_score)
        model_results["SVM_POLY5"].append(poly_5_score)
        model_results["SVM_RBF"].append(rbf_score)
        model_results["RF"].append(rf_score)

        pickle.dump(model_linear, open("cache/model_linear_{}.pkl".format(k), "wb"))
        pickle.dump(model_poly_2, open("cache/model_poly_2_{}.pkl".format(k), "wb"))
        pickle.dump(model_poly_5, open("cache/model_poly_5_{}.pkl".format(k), "wb"))
        pickle.dump(model_rbf, open("cache/model_rbf_{}.pkl".format(k), "wb"))
        pickle.dump(model_rf, open("cache/model_rf_{}.pkl".format(k), "wb"))

    so_far = float("-inf")
    best = None
    for model in model_results:
        print(model, np.mean(model_results[model]))
        if np.mean(model_results[model]) > so_far:
            so_far = np.mean(model_results[model])
            best = model
    print("Best model: {}, avg val acc: {}".format(best, np.mean(model_results[best])))

    """
    SVM_linear 0.6470117707529892
    SVM_POLY2 0.6454764273775153
    SVM_POLY5 0.6399581810988588
    SVM_RBF 0.6815451293557973
    RF 0.7728406592778209
    Best model: RF, avg val acc: 0.7728406592778209
    """

    x_val = X[int(0.8*len(X)):, :]
    y_val = Y[int(0.8*len(X)):]
    random_forest = pickle.load(open('./cache/model_rf_5.pkl', 'rb'))
    # Plot ROC Curve
    svc_disp = metrics.RocCurveDisplay.from_estimator(random_forest, x_val, y_val)
    plt.savefig("ROC and AUC of Random Forest Classifier.png")



if __name__ == "__main__":
    main()