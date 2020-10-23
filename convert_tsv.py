#!/usr/bin/env python
"""
Converts extract features file into Simple Estimator features file.

Jacob S. Porter
"""
import argparse
import datetime
import sys


def convert_line(line):
    """
    Convert a line into a list.
    """
    line = line.strip().replace("]", "").replace("[", "")
    return line.split(",")


def convert_feature_file(features_file):
    """
    Convert a file.
    """
    count = 0
    with open(features_file) as fd:
        fd.readline()
        for line in fd:
            line = convert_line(line)
            print("\t".join(map(lambda x: str(x).strip(), line)), sys.stdout)
            count += 1
    return count


def main():
    """Parse arguments."""
    tic = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=(''))
    parser.add_argument("features_file",
                        type=str,
                        help=("A features file from AlignmentTypeML."))
    args = parser.parse_args()
    print(args, file=sys.stderr)
    sys.stderr.flush()
    count = convert_feature_file(args.features_file)
    toc = datetime.datetime.now()
    print("The process took time: {}".format(toc - tic), file=sys.stderr)


if __name__ == '__main__':
    main()
