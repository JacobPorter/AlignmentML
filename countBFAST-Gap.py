#!/usr/bin/env python
import SeqIterator
import datetime
import optparse
import os
import sys
import Constants

logstr = 'countBFAST-GAP '

class Counts:
    """
    A class that stores the counts.
    """
    def __init__(self):
        self.__total = 0
        self.__ambig = 0
        self.__unmap = 0
        self.__filt = 0
        self.__uniq = 0
        self.__sam = 0

    def count_sam(self):
        self.__sam += 1

    def count_total(self):
        self.__total += 1

    def count_ambig(self):
        self.__ambig += 1

    def count_unmap(self):
        self.__unmap += 1

    def count_uniq(self):
        self.__uniq += 1

    def count_filt(self):
        self.__filt += 1

    def total(self):
        return self.__total

    def ambig(self):
        return self.__ambig

    def unmap(self):
        return self.__unmap

    def filt(self):
        return self.__filt

    def uniq(self):
        return self.__uniq

    def sam(self):
        return self.__sam

def isUnmapped(flag):
    """Checks if the SAM flag indicates an unmapped read."""
    return ((int(flag) >> 2) % 2) == 1

def isFiltered(SAM_record, filter_value):
    """Checks if the filter record is filtered"""
    return float(SAM_record[Constants.SAM_KEY_ALIGNMENT_SCORE]) / (len(SAM_record[Constants.SAM_KEY_SEQ]) + 0.0) <= filter_value


# def isProper(flag):
#     return ((int(flag) >> 1) % 2) == 1

def processSAM(SAM_file, gzip_switch, paired_end, filter_value, labels_file):
    if paired_end:
        raise NotImplementedError
    if gzip_switch:
        raise NotImplementedError
    filter_value = float(filter_value)
    sam_iterator = SeqIterator.SeqIterator(SAM_file, file_type = Constants.SAM, gzip_switch = gzip_switch)
    my_counter = Counts()
    record_dict = sam_iterator.convertToDict("R1", "R2")
    my_counter.__sam = sam_iterator.records_processed()
    if not paired_end:
        processSingle(record_dict, my_counter, filter_value, labels_file)
    else:
        pass
    return my_counter

def processSingle(record_dict, my_counter, filter_value, labels_file):
    if labels_file:
        labels_fd = open(labels_file, 'w')
        labels_dict = {0:"Unique", 1:"Ambig", 2:"Filt", 3:"Unmap"}
    for key in record_dict:
        SAM_record_list = record_dict[key]
        my_counter.count_total()
        if len(SAM_record_list) != 1:
            my_counter.count_ambig()
            for _ in range(len(SAM_record_list)):
                my_counter.count_sam()
            label = 1
        elif isUnmapped(SAM_record_list[0][Constants.SAM_KEY_FLAG]):
            my_counter.count_unmap()
            my_counter.count_sam()
            label = 3
        elif isFiltered(SAM_record_list[0], filter_value):
            my_counter.count_filt()
            my_counter.count_sam()
            label = 2
        else:
            my_counter.count_uniq()
            my_counter.count_sam()
            label = 0
        if labels_file:
            labels_fd.write("%s, %d, %s\n" % (key.split(" ")[0], label, labels_dict[label]))
    if labels_file:
        labels_fd.flush()
        labels_fd.close()

def report(counter, paired_end, SAM_file, now, later, filter_value):
    add_s = "paired end" if paired_end else "single end"
    sys.stdout.write("CountBFAST-Gap by Jacob Porter\n")
    sys.stdout.write("\n")
    sys.stdout.write("The %s SAM file %s had %s SAM records.  Processing began at %s taking %s time.  A filter value of %s was used.\n" % (add_s, str(SAM_file), str(counter.sam()), str(now), str(later - now), str(filter_value)))
    t = counter.total() + 0.0
    if t == 0.0:
        t = 0.000000000001
    sys.stdout.write("Reads:\t%s\n" % str(counter.total()))
    sys.stdout.write("Unique:\t%s\t%f\n" % (str(counter.uniq()), counter.uniq() / t))
    sys.stdout.write("Ambig:\t%s\t%f\n" % (str(counter.ambig()), counter.ambig() / t))
    sys.stdout.write("Unmap:\t%s\t%f\n" % (str(counter.unmap()), counter.unmap() / t))
    sys.stdout.write("Filt:\t%s\t%f\n" % (str(counter.filt()), counter.filt() / t))
    sys.stdout.write("Unmap + Filt:\t%s\t%f\n" % (str(counter.filt() + counter.unmap()), (counter.filt() + counter.unmap()) / t))


def main():
    now = datetime.datetime.now()
    usage = "usage: %prog [options] <sam_file> "
    description = ""
    p = optparse.OptionParser(usage = usage, description = description)
    #p.add_option('--gzip', '-z', help='The input file is gzip compressed, and the output file will be gzip compressed. [default: %default]', action='store_true', default=False)
    #p.add_option('--paired_end', '-p', help='Turn this on if the data is paired end data.', action='store_true', default=False)
    #p.add_option('--bound', '-b', help='The interval length above and below the correct location in the reference genome.  This determines how sensitive the calculation is to the correct location. [default: %default]', default = 3)
    p.add_option('--filter_value', '-f', help='The filter value to use to determine filtered records. [default: %default]', default='75.0')
    p.add_option('--labels', '-l', help = 'File to write labels for each read to.', default=None)
    #p.add_option('--bispin', '-b', help='Turn this on if the SAM file comes from BisPin.', action='store_true', default=False)
    options, args = p.parse_args()
    if len(args) == 0:
        p.print_help()
    if not os.path.exists(args[0]):
        p.error("The SAM file could not be found.")
    #counter = processSAM(args[0], options.gzip, options.paired_end, options.filter_value)
    counter = processSAM(args[0], None, None, options.filter_value, options.labels)
    later = datetime.datetime.now()
    #report(counter, options.paired_end, args[0], now, later)
    report(counter, None, args[0], now, later, options.filter_value)
    sys.stderr.write("%sThe process started at %s and took %s time.\n" % (logstr, str(now), str(later - now)))

if __name__ == '__main__':
    main()

