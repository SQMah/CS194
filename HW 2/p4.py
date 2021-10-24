import numpy as np
import os
import re
from statsmodels.discrete.discrete_model import Logit
import heapq
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

PHENOTYPE_DATA = "phenotypes.npz"
ALPHA = 0.05


def get_phenotype_data():
    '''
    Returns:
        probands: a numpy array (dtype: string) of proband ids
        phenotypes: a numpy array (dtype: int) of binary phenotypes
            phenotypes[i] describes the phenotype for probands[i]
    '''
    data = np.load(PHENOTYPE_DATA)
    probands = data["probands"]
    phenotypes = data["phenotypes"]
    return probands, phenotypes


def get_dosages(probands):
    p_look = {}
    dosages = [[] for _ in probands]
    for i, v in enumerate(probands):
        p_look[i] = v

    def process_line(line):
        if not line.startswith("#"):
            tokens = re.split(r'\t+', line.rstrip())
            sample_id = tokens[2]
            ref = tokens[3]
            alt_str = tokens[4]
            alts = alt_str.split(",")
            # We know that probands list and the order of #CHROM probands are the same
            for j in range(9, len(tokens)):
                i = j - 9
                a, b = tokens[j].split("|")
                dosages[i].append(int(a) + int(b))

    with open("chr22_subsampled_snps.vcf", 'r') as f:
        for l in f:
            process_line(l)
    return np.array(dosages)


def main():
    # PART A
    probands, phenotypes = get_phenotype_data()
    if not os.path.exists("dosages.npy"):
        dosages = np.save("dosages", get_dosages(probands))
    else:
        print("Loaded dosages from saved array")
        dosages = np.load("dosages.npy")
    print(dosages)

    if not os.path.exists("p_vals.npy"):
        individuals, snps = dosages.shape
        X = np.ones((2, individuals))
        p_vals = []
        for j in range(snps):
            print("Training model for SNP {}".format(j))
            snp = dosages[:, j]
            if np.all(snp == snp[0]):
                print("SNP {} all the same".format(j))
                p_vals.append(1.0)
            X[0] = snp
            model = Logit(phenotypes, X.T).fit(method="bfgs")
            p_vals.append(model.pvalues[0])
        p_vals = np.array(p_vals)
        np.save("p_vals", p_vals)
    else:
        print("Loaded p_vals from saved array")
        p_vals = np.load("p_vals.npy")
    print("P-VALUES:", p_vals)
    # PART B
    log_bonferri_thres = -np.log(ALPHA / len(p_vals))
    h = []
    for i, v in enumerate(p_vals):
        heapq.heappush(h, v)
    # To handle potential numerical instability
    prev = min(p_vals)
    up_to = 0
    for rank, v_i in enumerate(h):
        if ((rank + 1) * ALPHA / len(p_vals)) >= v_i:
            prev = v_i
            up_to = rank
        else:
            break

    log_benj_thres = -np.log(prev)
    log_p_values = -np.log(p_vals)

    plt.figure(figsize=(14, 8))
    plt.scatter(np.arange(len(p_vals)), log_p_values)
    plt.hlines(log_bonferri_thres, 0, len(p_vals) - 1, label="Bonferri", colors="C3")
    plt.hlines(log_benj_thres, 0, len(p_vals) - 1, label="Benjamini hochberg", colors="C15")
    plt.ylabel("-log(p-value)")
    plt.xlabel("SNP index")
    plt.legend()
    plt.title("Chrom 22 GWAS")
    plt.savefig("chrom_22_gwas.png")

    # PART C
    print("REJECTED BONFERRI: ", sum([1 if (val <= (ALPHA / len(p_vals))) else 0 for val in p_vals]))
    print("REJECTED BENJAMINI-HOCHBERG", up_to + 1)
    """
    REJECTED BONFERRI:  4007
    REJECTED BENJAMINI-HOCHBERG 304
    """
    # PART D
    """This is because the corrections only help reduce the false positive rate (with certain
     tradeoffs) and doesn't necessarily eliminate all false positives. In the end, it is
     is only a heuristic that depends on an alpha that is also a heuristic."""

    # PART E
    std_dosages = (dosages - np.expand_dims(np.mean(dosages, axis=0), axis=0))
    for i, v in enumerate(np.std(dosages, axis=0)):
        if v:
            std_dosages[:, i] /= v
    pca = PCA(n_components=3)
    pca.fit(std_dosages.T)
    components = pca.components_

    if not os.path.exists("p_vals_pca.npy"):
        individuals, snps = dosages.shape
        X = np.ones((5, individuals))
        pca_p_vals = []
        for j in range(snps):
            print("Training PCA augmented model for SNP {}".format(j))
            snp = dosages[:, j]
            if np.all(snp == snp[0]):
                print("SNP {} all the same".format(j))
                pca_p_vals.append(1.0)
            X[0] = snp
            X[1:4, :] = components
            model = Logit(phenotypes, X.T).fit(method="bfgs", maxiter=100)
            pca_p_vals.append(model.pvalues[0])
        pca_p_vals = np.array(pca_p_vals)
        np.save("p_vals_pca", pca_p_vals)
    else:
        print("Loaded PCA p_vals from saved array")
        pca_p_vals = np.load("p_vals_pca.npy")
    print("PCA P-VALUES:", pca_p_vals)

    log_bonferri_thres = -np.log(ALPHA / len(pca_p_vals))
    h = []
    for i, v in enumerate(pca_p_vals):
        heapq.heappush(h, v)
    # To handle potential numerical instability
    prev = min(pca_p_vals)
    up_to = 0
    for rank, v_i in enumerate(h):
        if ((rank + 1) * ALPHA / len(pca_p_vals)) >= v_i:
            prev = v_i
            up_to = rank
        else:
            break

    log_benj_thres = -np.log(prev)
    log_pca_p_values = -np.log(pca_p_vals)

    plt.figure(figsize=(14, 8))
    plt.scatter(np.arange(len(pca_p_vals)), log_pca_p_values)
    plt.hlines(log_bonferri_thres, 0, len(pca_p_vals) - 1, label="Bonferri", colors="C3")
    plt.hlines(log_benj_thres, 0, len(pca_p_vals) - 1, label="Benjamini hochberg", colors="C15")
    plt.ylabel("-log(p-value)")
    plt.xlabel("SNP index")
    plt.legend()
    plt.title("Chrom 22 GWAS PCA")
    plt.savefig("chrom_22_gwas_pca.png")

    print("REJECTED PCA BONFERRI: ", sum([1 if (val <= (ALPHA / len(p_vals))) else 0 for val in pca_p_vals]))
    print("REJECTED PCA BENJAMINI-HOCHBERG", up_to + 1)
    """
    REJECTED PCA BONFERRI:  258
    REJECTED PCA BENJAMINI-HOCHBERG 4
    """

    # PART F
    """
    This is because PCA removes some multicollinearity by picking out 
    vectors that contribute to the dosage matrix most strongly but are
    also orthogonal. Hence it gives logistic regression stronger signals
    to learn a regressive model on.
    """

    # PART G
    """
    This is because individuals of African descent could have very different
    SNPs or other variations that lead to certain phenotypes compared to
    individuals of European descent. Doing a PCA would also likely lead to
    very different principal components as well.
    """


if __name__ == '__main__':
    main()
