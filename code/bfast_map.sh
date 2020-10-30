#!/usr/bin/env bash
# $1 Genome location
# $2 reads location
# $3 output directory
bfast match -f $1 -i 1 -r $2 -A 0 -t 1> $3/bfast.bmf 2> $3/bfast.match.err
bfast localalign -f $1 -m $3/bfast.bmf 1> $3/bfast.baf 2> $3/bfast.localalign.err
bfast postprocess -f $1 -i $3/bfast.baf 1> $3/bfast.sam 2> $3/bfast.postprocess.err
