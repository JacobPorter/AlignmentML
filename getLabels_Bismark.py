#!/usr/bin/env python
"""Make the machine learning labels for Bismark"""

import argparse
import datetime
import sys

from SeqIterator import SeqIterator


def get_labels(uniq_loc, ambig_loc, unmap_loc):
    """
    Get the labels for Bismark

    Parameters
    ----------
    uniq_loc: str
        File location if the SAM file for uniquely mapped reads
    ambig_loc: str
        File location of the ambiguously mapped reads
    unmap_loc: str
        File location of the unmapped reads

    Returns
    -------
    None

    """
    uniq_iter = SeqIterator(uniq_loc, file_type="SAM")
    for record in uniq_iter:
        print("{}, 0, Unique".format(record["QNAME"].split('_')[0]))
    ambig_iter = SeqIterator(ambig_loc, file_type="fastq")
    for record in ambig_iter:
        print("{}, 1, Ambig".format(record[0].split('_')[0]))
    unmap_iter = SeqIterator(unmap_loc, file_type="fastq")
    for record in unmap_iter:
        print("{}, 2, Unmap".format(record[0].split('_')[0]))


def main():
    """Parse arguments for the program."""
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser(
        description=('Create labels for '
                     'bismark output.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("uniquely_mapped",
                        type=str,
                        help=("The location of the SAM file "
                              "for uniquely mapped reads."))
    parser.add_argument("ambiguously_mapped",
                        type=str,
                        help=("The ambiguously mapped reads."))
    parser.add_argument("unmapped_reads",
                        type=str,
                        help=('The location of the unmapped reads.'))
    args = parser.parse_args()
    get_labels(args.uniquely_mapped, args.ambiguously_mapped,
               args.unmapped_reads)
    later = datetime.datetime.now()
    print("The process took time {}.".format(later - now), file=sys.stderr)


if __name__ == "__main__":
    main()
