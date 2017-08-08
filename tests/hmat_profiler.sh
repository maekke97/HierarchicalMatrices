#!/bin/bash
OUTFILE=/compute/nem/hmat_profiler.out
ADDCSV=/compute/nem/hmat_addition_profiler.csv
SUBCSV=/compute/nem/hmat_subtraction_profiler.csv
MULCSV=/compute/nem/hmat_multiplication_profiler.csv
if [ -f ${OUTFILE} ];
then
    rm -f ${OUTFILE}
fi
for e in `seq 2 20`;
do
    python hmat_profiler.py $(( 2**${e} )) >> ${OUTFILE}
done
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Addition/{ print $4, $6}' ${OUTFILE} > ${ADDCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Subtraction/{ print $4, $6}' ${OUTFILE} > ${SUBCSV}
awk 'BEGIN{ OFS=","; FS="[ \t]+|=|" }/Multiplication/{ print $4, $6}' ${OUTFILE} > ${MULCSV}