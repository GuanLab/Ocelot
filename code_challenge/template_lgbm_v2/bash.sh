#!/bin/bash

set -e

##  D  E  F  G  H  I ##
## 18 17 16 20 22 29 ##

assay_feature=(FEATURE)
assay_target=(TARGET)
cell_test=(TEST)
cell_cv=('CXX' 'CYY')

time python train.py -f ${assay_feature[@]} -t ${assay_target[@]} -cv ${cell_cv[@]} | tee -a log_1.txt 

for the_cell in ${cell_test[@]}
do
    echo pred_$the_cell
    cell_cvt=('CXX' 'CYY')
    cell_cvt+=($the_cell)
    time python pred25bp.py -f ${assay_feature[@]} -t ${assay_target[@]} -cv ${cell_cvt[@]} &
done
wait




