#!/bin/bash
OUTFILE=/compute/nem/rmat_profiler.out
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 4 20`;
do
    for k in `seq 1 5`;
    do
        python rmat_profiler.py $(( 2**${e} )) ${k} >> ${OUTFILE}
    done
done
