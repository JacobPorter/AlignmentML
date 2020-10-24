#!/usr/bin/env python
"""Iterates through a file and removes reads with wildcard characters."""
import argparse
import datetime
import sys

import SeqIterator


def remove_wildcards(fastq_file):
    """Iterate through a fastq_file and remove reads that have wildcards in them. """
    fastq_iter = SeqIterator.SeqIterator(fastq_file, file_type='fastq')
    fastq_writer = SeqIterator.SeqWriter(sys.stdout, file_type='fastq')
    count = 0
    count_n = 0
    for fastq_record in fastq_iter:
        count += 1
        if "N" in fastq_record[0] or "n" in fastq_record[0]:
            count_n += 1
        else:
            fastq_writer.write(fastq_record)
    return count, count_n


def main():
    """Parse arguments."""
    tic = datetime.datetime.now()
    parser = argparse.ArgumentParser(description=(''))
    parser.add_argument("fastq_file", type=str, help=("A fastq file."))
    args = parser.parse_args()
    print(args, file=sys.stderr)
    sys.stderr.flush()
    count, count_n = remove_wildcards(args.fastq_file)
    toc = datetime.datetime.now()
    print(
        "There were {} total records processed.  There were {} records removed."
        .format(count, count_n),
        file=sys.stderr)
    print("The process took time: {}".format(toc - tic), file=sys.stderr)


if __name__ == '__main__':
    main()
