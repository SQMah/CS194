import re
import matplotlib.pyplot as plt
from multiprocessing import Pool


def process_line(line):
    freqs = {}
    snps = 0
    individuals = {}
    num_dbsnp = 0
    if not line.startswith("#"):
        tokens = re.split(r'\t+', line)
        sample_id = tokens[2]
        if sample_id != ".":
            print("FOUND", sample_id)
        ref = tokens[3]
        alt = tokens[4]
        alts = alt.split(",")
        for a in alts:
            if len(ref) == len(a) and ref != "." and a != "." \
                    and sum([1 for one, two in zip(ref, a) if one != two]) == 1:
                snps += 1
            if a not in freqs:
                freqs[a] = 0
            freqs[a] += 1
        infos = tokens[7].split(";")
        for info in infos:
            if "DB" in info:
                num_dbsnp += 1
                break
        for j in range(9, len(tokens)):
            if j not in individuals:
                individuals[j] = 0
            a, b = tokens[j].split("|")
            if int(a) > 0:
                individuals[j] += 1
            if int(b) > 0:
                individuals[j] += 1
    return freqs, snps, individuals, num_dbsnp


def ans(name):
    freqs_t = {}
    snps_t = 0
    individuals_t = {}
    num_dbsnp = 0
    results = []
    with open(name, 'r') as f:
        for line in f:
            results.append(process_line(line))
    for r in results:
        freqs, snps, individuals, num_dbsnps = r
        for f in freqs:
            if f not in freqs_t:
                freqs_t[f] = 0
            freqs_t[f] += freqs[f]
        snps_t += snps
        num_dbsnp += num_dbsnps
        for i in individuals:
            if i not in individuals_t:
                individuals_t[i] = 0
            individuals_t[i] += individuals[i]
    print("TOTAL SNPS:", snps_t)
    freq_sum = sum([freqs_t[v] for v in freqs_t])
    print("Less than 1%: ", len([v for v in freqs_t
                                 if freqs_t[v] < (0.01 * freq_sum)]))
    print("NUM SAMPLES:", len(individuals_t))
    individuals_sum = sum([individuals_t[v] for v in individuals_t])
    print("AVERAGE VARIANTS PER INDIVIDUAL", individuals_sum / len(individuals_t))
    print("NUM DBSNP", num_dbsnp)
    plt.figure()
    plt.hist([freqs_t[k] for k in freqs_t], 100)
    plt.savefig("freq.png")


if __name__ == '__main__':
    filename = "chr21_filtered.vcf"
    ans(name=filename)
    """
    TOTAL SNPS: 191079
    Less than 1%:  692
    NUM SAMPLES: 2504
    AVERAGE VARIANTS PER INDIVIDUAL 16042.289137380192
    """
