import pickle
from p5 import get_vals, get_phastcons
import pyBigWig
import numpy as np
import re

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



if __name__ == "__main__":
    random_forest = pickle.load(open('./cache/model_rf_5.pkl', 'rb'))
    gene_list = []
    
    with open("test_set.vcf", "r") as f:
        for line in f:
            if not line.startswith("#"):
                gene_list.append(line)

    rvis_map = dict()
    with open("rvis.txt", 'r') as f:
        for i, l in enumerate(f):
            if i != 0:
                tokens = l.split("\t")
                rvis_map[tokens[0]] = float(tokens[3])

    rvis = get_vals(gene_list, rvis_map)

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

    o_e = get_vals(gene_list, o_e_map, default=avg_o_e)
    

    bw = pyBigWig.open("hg38.phastCons100way.bw")
    phastcons = np.array([get_phastcons(x, bw) for x in gene_list])
    print(phastcons)
    print(np.sum(phastcons))
    X = np.concatenate((np.expand_dims(rvis, -1), np.expand_dims(o_e, -1), np.expand_dims(phastcons, -1)), axis=1)
    y = random_forest.predict_proba(X)

    with open("test_set.predictions", "w") as f:
        for val in y:
            f.write(str(val[1]) + "\n")