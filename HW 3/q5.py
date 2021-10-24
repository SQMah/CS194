import re
import numpy as np
import matplotlib.pyplot as plt
import os
from six.moves import cPickle as pickle
import pyBigWig

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

def main():


    benign = [] # CLASS LABEL: 0
    pathogenic = [] # CLASS LABEL: 1

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
    
    if os.path.exists("rvis_train.npy"):
        rvis_train = np.load("rvis_train.npy")
        rvis_val = np.load("rvis_val.npy")
        rvis_benign = np.load("rvis_benign.npy")
        rvis_pathogenic = np.load("rvis_pathogenic.npy")
        rvis_total = np.concatenate((rvis_train, rvis_val))
    else:
        rvis_train = get_vals(x_train, rvis_map)
        rvis_val = get_vals(x_val, rvis_map)
        rvis_total = np.concatenate((rvis_train, rvis_val))
        mask_benign = np.where(y_train == 0)[0]
        mask_pathogenic = np.where(y_train == 1)[0]
        rvis_benign = rvis_total[mask_benign]
        rvis_pathogenic = rvis_total[mask_pathogenic]
        np.save("rvis_train.npy", rvis_train)
        np.save("rvis_val.npy", rvis_val)
        np.save("rvis_benign.npy", rvis_benign)
        np.save("rvis_pathogenic.npy", rvis_pathogenic)
    
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

    if os.path.exists("o_e_train.npy"):
        o_e_train = np.load("o_e_train.npy")
        o_e_val = np.load("o_e_val.npy")
        o_e_benign = np.load("o_e_benign.npy")
        o_e_pathogenic = np.load("o_e_pathogenic.npy")
        o_e_total = np.concatenate((o_e_train, o_e_val))
    else:
        o_e_train = get_vals(x_train, o_e_map, default=avg_o_e)
        o_e_val = get_vals(x_val, o_e_map, default=avg_o_e)
        o_e_total = np.concatenate((o_e_train, o_e_val))
        mask_benign = np.where(y_train == 0)[0]
        mask_pathogenic = np.where(y_train == 1)[0]
        o_e_benign = o_e_total[mask_benign]
        o_e_pathogenic = o_e_total[mask_pathogenic]
        np.save("o_e_train.npy", o_e_train)
        np.save("o_e_val.npy", o_e_val)
        np.save("o_e_benign.npy", o_e_benign)
        np.save("o_e_pathogenic.npy", o_e_pathogenic)
    
    bins = np.linspace(min(o_e_total), max(o_e_total), 50)
    fig, axs = plt.subplots(2,sharex=True, sharey=True)
    axs[0].hist(o_e_benign, bins, alpha=0.5, label="Benign")
    axs[1].hist(o_e_pathogenic, bins, alpha=0.5, label="Pathogenic")
    axs[0].set_title("O/E scores benign")
    axs[1].set_title("O/E scores pathogenic")
    plt.savefig("o_e_scores.png")

    bw = pyBigWig.open("hg38.phastCons100way.bw")
    print(bw.header())



if __name__ == "__main__":
    main()