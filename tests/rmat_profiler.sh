#!/bin/bash
OUTFILE=/compute/nem/rmat_profiler.out
ADDCSV=/compute/nem/rmat_addition_profiler.csv
SUBCSV=/compute/nem/rmat_subtraction_profiler.csv
MULCSV=/compute/nem/rmat_multiplication_profiler.csv
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 12 21`;
do
    for k in `seq 1 10`;
    do
        python rmat_profiler.py $(( 2**${e} )) ${k} >> ${OUTFILE}
    done
done
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Addition/{ print $4, $7, $9}' ${OUTFILE} > ${ADDCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Subtraction/{ print $4, $7, $9}' ${OUTFILE} > ${SUBCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Multiplication/{ print $4, $7, $9}' ${OUTFILE} > ${MULCSV}