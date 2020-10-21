#!/usr/bin/python
import bz2
import datetime
import lzma
import math
import optparse
import os
import sys
from collections import Counter, defaultdict

import SeqIterator
"""
Gets features for machine learning from FASTQ reads.
"""


def transformPhredCharToProb(c, offset=33):
    return 10**((ord(c) - offset) / (-10.0))


def quality_features(vector):
    mean = float(sum(vector)) / len(vector)
    my_max = max(vector)
    my_min = min(vector)
    variance = float(sum([math.pow(item - mean, 2)
                          for item in vector])) / len(vector)
    # Division by zero error
    try:
        skewness = float(
            sum([
                math.pow((item - mean) / (variance + 0.0), 3)
                for item in vector
            ])) / len(vector)
    except ZeroDivisionError:
        skewness = 0
    try:
        kurt = float(
            sum([
                math.pow((item - mean) / (variance + 0.0), 4)
                for item in vector
            ])) / len(vector)
    except ZeroDivisionError:
        kurt = 9 / 5.0
    diff = []
    for i in range(1, len(vector)):
        diff.append(vector[i] - vector[i - 1])
    mean_diff = float(sum(diff)) / len(diff)
    return [mean, my_max, my_min, variance, skewness, kurt, mean_diff]


def dna_seq_features(alpha_vector, ignore=False):
    alpha_vector = alpha_vector.upper()
    dna_count = Counter(alpha_vector)
    for base in dna_count:
        dna_count[base] = dna_count[base] / (len(alpha_vector) + 0.0)
    ent = -1 * sum(
        [dna_count[base] * math.log(dna_count[base], 2) for base in dna_count])
    dust_score = dust(alpha_vector)
    bz2_score, lzma_score = compression_score(alpha_vector)
    keys = ("A", "C", "G", "T")
    freq = [dna_count.get(k, 0.0) for k in keys]
    if not ignore:
        run_stats = longest_run(alpha_vector)
        k_stats = kmer_stats(alpha_vector)
        return [ent, dust_score, bz2_score, lzma_score,
                len(alpha_vector)] + freq + run_stats + k_stats
    else:
        return [ent, dust_score, bz2_score, lzma_score] + freq


def longest_run(alpha_vector):
    run_base = alpha_vector[0]
    run_length = 1
    run_length_dist = []
    for base in alpha_vector[1:len(alpha_vector)]:
        if base == run_base:
            run_length += 1
        else:
            run_length_dist.append(run_length)
            run_base = base
            run_length = 1
    run_length_dist.append(run_length)
    max_length = max(run_length_dist)
    mean = float(sum(run_length_dist)) / len(run_length_dist)
    vector = run_length_dist
    variance = float(sum([math.pow(item - mean, 2)
                          for item in vector])) / len(vector)
    return [mean, variance, max_length]


def kmer_stats(alpha_vector):
    k_stats = []
    for k in range(2, 6):
        cnt = kmer_freq(alpha_vector, k)
        k_stats.append(dkg(cnt, k, len(alpha_vector)))
        k_stats.append(rkg(cnt, k, len(alpha_vector)))
    return k_stats


def kmer_freq(vector, k):
    cnt = Counter()
    for i in range(len(vector) - k + 1):
        cnt[vector[i:i + k]] += 1
    return cnt


def dkg(cnt, k, len_g):
    return len(list(cnt.keys())) / (len_g - k + 1.0)


def rkg(cnt, k, len_g):
    kmer_cnt = 0
    for kmer in cnt:
        if cnt[kmer] > 1:
            kmer_cnt += cnt[kmer]
    return kmer_cnt / (len_g - k + 1.0)


def dust(alpha_vector):
    n = len(alpha_vector)
    if n <= 3:
        return 0
    tri_dict = defaultdict(int)
    for i in range(n - 2):
        tri_dict[alpha_vector[i:i + 3]] += 1
    dust_score = 0.0
    for key in tri_dict:
        dust_score += tri_dict[key] * (tri_dict[key] - 1) / 2.0
    dust_score /= (n - 3)
    normalization = ((n - 2) * (n - 3) / 2.0) / (n - 3)
    return dust_score / normalization


def compression_score(alpha_vector):
    bz2_score = len(bz2.compress(alpha_vector)) / float(len(alpha_vector))
    lzma_score = len(lzma.compress(alpha_vector)) / float(len(alpha_vector))
    return (bz2_score, lzma_score)


def extractFeatureRow(fastq_record, subportions=3):
    seq_id = fastq_record[0].split(" ")[0]
    seq_seq = fastq_record[1]
    if "N" in seq_seq:
        return None
    seq_qual = fastq_record[2]
    seq_qual_prob = list(map(transformPhredCharToProb, seq_qual))
    f1 = [seq_id] + dna_seq_features(seq_seq) + quality_features(seq_qual_prob)
    sublength = len(seq_seq) / subportions
    f2 = []
    for i in range(subportions):
        if i == subportions - 1:
            finallength = len(seq_seq)
        else:
            finallength = (i + 1) * sublength
        f2 += dna_seq_features(seq_seq[i * sublength:finallength], ignore=True)
        f2 += quality_features(seq_qual_prob[i * sublength:finallength])
    return f1 + f2


def extractFeatures(FASTQ_file):
    fastq_iter = SeqIterator.SeqIterator(FASTQ_file, file_type='fastq')
    # 28 features including the seq_id
    feature_labels = [
        "seq_id", "entropy", "dust", "bz2", "lzma", "length", "freq_A",
        "freq_C", "freq_G", "freq_T", "run_mean", "run_variance",
        "run_max_length", "dkg_2", "rkg_2", "dkg_3", "rkg_3", "dkg_4", "rkg_4",
        "dkg_5", "rkg_5", "qual_mean", "qual_max", "qual_min", "qual_variance",
        "qual_skewness", "qual_kurt", "qual_mean_diff"
    ]
    # 15 features for each partition
    partition1_labels = [
        "entropy_1", "dust_1", "bz2_1", "lzma_1", "freq_A_1", "freq_C_1",
        "freq_G_1", "freq_T_1", "qual_mean_1", "qual_max_1", "qual_min_1",
        "qual_variance_1", "qual_skewness_1", "qual_kurt_1", "qual_mean_diff_1"
    ]
    partition2_labels = [
        "entropy_2", "dust_2", "bz2_2", "lzma_2", "freq_A_2", "freq_C_2",
        "freq_G_2", "freq_T_2", "qual_mean_2", "qual_max_2", "qual_min_2",
        "qual_variance_2", "qual_skewness_2", "qual_kurt_2", "qual_mean_diff_2"
    ]
    partition3_labels = [
        "entropy_3", "dust_3", "bz2_3", "lzma_3", "freq_A_3", "freq_C_3",
        "freq_G_3", "freq_T_3", "qual_mean_3", "qual_max_3", "qual_min_3",
        "qual_variance_3", "qual_skewness_3", "qual_kurt_3", "qual_mean_diff_3"
    ]
    print(feature_labels + partition1_labels + partition2_labels +
          partition3_labels)
    noNs = 0
    total = 0
    for fastq_record in fastq_iter:
        features = extractFeatureRow(fastq_record)
        total += 1
        if features:
            print(features)
            noNs += 1
    sys.stderr.write("The number without Ns, and the total "
                     "in the file:\t%d\t%d\n" % (noNs, total))


def main():
    now = datetime.datetime.now()
    usage = "usage: %prog [options] <FASTQ>"
    version = "%prog "
    description = ""
    epilog = ""
    p = optparse.OptionParser(usage=usage,
                              version=version,
                              description=description,
                              epilog=epilog)
    options, args = p.parse_args()
    if len(args) != 1:
        p.error("There must be one FASTQ files specified.  "
                "Please check the input.")
    FASTQ_file = args[0]
    if (not os.path.exists(FASTQ_file) or not os.path.exists(FASTQ_file)
            or not os.path.exists(FASTQ_file)):
        p.error("The reads file does not exist or could not be accessed.")
    sys.stderr.write("ExtractFeatures was started on " + str(now) + "\n")
    sys.stderr.flush()
    extractFeatures(FASTQ_file)
    later = datetime.datetime.now()
    sys.stderr.write("ExtractFeatures finished on " + str(later) +
                     " and took time: " + str(later - now) + "\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main()
