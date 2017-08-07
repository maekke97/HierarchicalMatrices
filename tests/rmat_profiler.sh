#!/bin/bash
OUTFILE=/compute/nem/rmat_profiler.out
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 4 10`;
do
    for k in `seq 1 3`;
    do
        python rmat_profiler.py ${k} $(( 2**${e} )) >> ${OUTFILE}
    done
done
