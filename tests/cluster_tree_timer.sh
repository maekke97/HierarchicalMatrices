#!/bin/bash
OUTFILE=/compute/nem/cluster_tree_timer.out
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 2 20`;
do
    python cluster_tree_timer.py 1 $(( 2**${e} )) >> ${OUTFILE}
done
