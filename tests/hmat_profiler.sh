#!/bin/bash
OUTFILE=/compute/nem/hmat_profiler.out
ADDCSV=/compute/nem/hmat_addition_profiler.csv
SUBCSV=/compute/nem/hmat_subtraction_profiler.csv
MULCSV=/compute/nem/hmat_multiplication_profiler.csv
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 12 21`;
do
    python cluster_tree_profiler.py 1 $(( 2**${e} )) >> ${OUTFILE}
done
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Addition/{ print $4, $7, $9}' ${OUTFILE} > ${ADDCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Subtraction/{ print $4, $7, $9}' ${OUTFILE} > ${SUBCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Multiplication/{ print $4, $7, $9}' ${OUTFILE} > ${MULCSV}