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


def convert_feature_file(features_file, id_file):
    """
    Convert a file.
    """
    count = 0
    id_list = open(id_file)
    with open(features_file) as fd:
        fd.readline()
        for line in fd:
            line = convert_line(line)
            print(line[0], file=id_list)
            print("\t".join(map(lambda x: str(x).strip(), line[1:])), file=sys.stdout)
            count += 1
    return count


def main():
    """Parse arguments."""
    tic = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=(''))
    parser.add_argument("features_file",
                        type=str,
                        help=("A features file from AlignmentTypeML."))
    parser.add_argument("-l", "--id_list", type=str, help=("The file to store sequence ids."), default="./id_list.txt")
    args = parser.parse_args()
    print(args, file=sys.stderr)
    sys.stderr.flush()
    count = convert_feature_file(args.features_file, args.id_list)
    toc = datetime.datetime.now()
    print("The process took time: {}".format(toc - tic), file=sys.stderr)


if __name__ == '__main__':
    main()
