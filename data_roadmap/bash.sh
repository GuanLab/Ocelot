#!/bin/bash

## 0.download roadmap data
bash bash_download.sh

## 1.subsample all
python bash_subsample.py
#real    536m37.062s

## 2.create avg ref
python calculate_ref_for_qn_all.py
#real	0m19.079s

## 3.quantile normalization
python bash_qn_all.py

## 4.create qn avg
python bash_avg_all.py

## 5.create 25bp gt for evaluation
python bash_gt_all.py

## 6.create n5cut
python bash_n5cut.py
#python bash_n5cut_old.py

## 7.create n5cut avg
python bash_avg_n5cut_all.py

## 8.prepare train-validation samples for lightgbm
python bash_prepare_sample_all.py



