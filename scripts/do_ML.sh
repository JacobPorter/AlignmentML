#!/bin/sh
# $1 SRA number
# $2 read mapper
python /home/jsporter/Applications/AlignmentML/AlignmentTypeML3.py -n 3 -s 2500000 -t 500000 -c Random,RF,MLP,LR -b -d /scratch/jsporter/AlignmentML/$1/$2/ /scratch/jsporter/AlignmentML/$1/$1.3mil.features.txt /scratch/jsporter/AlignmentML/$1/$2/$1.3mil.$2.labels.txt 1> /scratch/jsporter/AlignmentML/$1/$2/$1.3mil.$2.ml.out 2> /scratch/jsporter/AlignmentML/$1/$2/$1.3mil.$2.ml.err
