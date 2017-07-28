#!/bin/bash
OUTFILE=/compute/nem/cluster_tree_profiler.out
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 2 24`;
do
    python cluster_tree_profiler.py 1 $(( 2**${e} )) >> ${OUTFILE}
done
