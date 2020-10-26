#!/usr/bin/env python
"""Computes average alignment score and average edit distance."""
import argparse
import datetime
import sys
from collections import defaultdict

import Constants
import SeqIterator


def examine_mapping(sam_file, predict_file):
    """Compute average alignment score and edit distance."""
    sam_iter = SeqIterator.SeqIterator(sam_file, file_type='sam')
    total_sam = 0
    count_as = 0
    count_ed = 0
    avg_align_score = 0
    avg_edit_distance = 0
    pred_dict = defaultdict(lambda: [0] * 3)
    if predict_file:
        with open(predict_file) as predict_fd:
            for line in predict_fd:
                line = line.split()
                pred_dict[line[0].replace("'", "")] = line[1:]
    for record in sam_iter:
        total_sam += 1
        name = record["QNAME"]
        pred = pred_dict[name]
        align_score = float(record[Constants.SAM_KEY_ALIGNMENT_SCORE])
        try:
            edit_distance = int(record["NM:i"])
        except KeyError:
            edit_distance = None
        flag = int(record["FLAG"])
        if (int(pred[0]) == 0 or int(pred[0]) == 1) and flag != 4:
            count_as += 1
            avg_align_score += align_score
            if flag < 512:
                avg_edit_distance += edit_distance
                count_ed += 1
    return count_as, count_ed, total_sam, avg_align_score / count_as, avg_edit_distance / count_ed


def main():
    """Parse arguments."""
    tic = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=(''))
    parser.add_argument("sam_file", type=str, help=("A sam file."))
    parser.add_argument(
        "--predict_file",
        "-p",
        type=str,
        help=("A prediction file.  "
              "Do not include this file to get results for no predictions."),
        default=None)
    args = parser.parse_args()
    print(args, file=sys.stderr)
    sys.stderr.flush()
    count_as, count_ed, total_sam, avg_align_score, avg_edit_distance = examine_mapping(
        args.sam_file, args.predict_file)
    toc = datetime.datetime.now()
    print(("There were {} / {} records processed for the alignment score.  "
           "There were {} / {} records processed for the edit distance.  "
           "The average alignment score was: {}, "
           "and the average edit distance was: {}.").format(
               count_as, total_sam, count_ed, total_sam, avg_align_score, avg_edit_distance),
          file=sys.stderr)
    print("The process took time: {}".format(toc - tic), file=sys.stderr)


if __name__ == '__main__':
    main()
