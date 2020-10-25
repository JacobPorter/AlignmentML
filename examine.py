#!/usr/bin/env python
"""Compares read sequences to predictions."""
import argparse
import datetime
import sys

import Constants
import SeqIterator


def examine_mapping(sam_file, predict_file):
    sam_iter = SeqIterator.SeqIterator(sam_file, file_type='sam')
    if predict_file:
        predict_fd = open(predict_file)
    count = 0
    avg_align_score = 0
    avg_edit_distance = 0
    for record in sam_iter:
        if predict_file:
            pred = predict_fd.readline().split()
        else:
            pred = [0] * 4
        align_score = float(record[Constants.SAM_KEY_ALIGNMENT_SCORE])
        edit_distance = int(record["NM:i"])
        flag = int(record["FLAG"])
        if (int(pred[1]) == 0 or int(pred[1]) == 1) and flag != 4:
            count += 1
            avg_align_score += align_score
            avg_edit_distance += edit_distance
    return count, avg_align_score / count, avg_edit_distance / count


def main():
    """Parse arguments."""
    tic = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=(''))
    parser.add_argument("sam_file", type=str, help=("A sam file."))
    parser.add_argument(
        "--predict_file",
        "-p",
        type=str,
        help=
        ("A prediction file.  Do not include this file to get results for no predictions."
         ),
        default=None)
    args = parser.parse_args()
    print(args, file=sys.stderr)
    sys.stderr.flush()
    count, avg_align_score, avg_edit_distance = examine_mapping(
        args.sam_file, args.predict_file)
    toc = datetime.datetime.now()
    print(
        "There were {} records processed.  The average alignment score was: {}, and the average edit distance was: {}."
        .format(count, avg_align_score, avg_edit_distance),
        file=sys.stderr)
    print("The process took time: {}".format(toc - tic), file=sys.stderr)


if __name__ == '__main__':
    main()
