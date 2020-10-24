#!/usr/bin/env bash
# $1 Genome location
# $2 reads location
# $3 output directory
bfast-gap match -f $1 -i 1 -r $2 -A 0 -t 1> $3/bfast-gap.bmf 2> $3/bfast.match.err
bfast-gap localalign -f $1 -m $3/bfast.bmf -x ~/BFAST-Gap/scoring_function.txt  1> $3/bfast-gap.baf 2> $3/bfast.localalign.err
bfast-gap postprocess -f $1 -i $3/bfast-gap.baf -x ~/BFAST-Gap/scoring_function.txt 1> $3/bfast-gap.sam 2> $3/bfast.postprocess.err
