from SeqIterator import SeqIterator

fastq_file = ""
labels_file = ""

labels_dict = {}
for line in open(labels_file):
    labels_dict[line.split(",")[0]] = 1

fastq_dict = SeqIterator(open("fastq_file"), file_type='fastq').convertToDict()

for seq_id in fastq_dict:
    if seq_id not in labels_dict:
        print("{}, {}, {}".format(seq_id, 3, 'Unmap'))
