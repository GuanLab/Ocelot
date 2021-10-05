#!/bin/bash

## 0. download the ENCODE Imputation Challenge training data in the bigwig folder

## 1. epigenetic features and targets ##
# quantile normalization
python subsample_for_anchor.py # all 
python create_ref_for_anchor_final.py # all
python anchor_signal_bigwig.py # all
# average feature
python create_avg_chr_final.py # all
# average baseline
python generate_baseline_avg_anchored_bigwig.py
# training target
python generate_gold_anchored_bigwig.py # for train

# orange features - number of unique values within each 25bp bin ##
# generate orange feature
bash bash_orange.sh
# convert into orange
python generate_orange_bigwig.py
# average orange
python create_avg_orange_final.py 

### 2. one-hot encoding DNA-sequence
##download GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta first
#cd hg38
#bash bash_separate_chr.sh  
#python encode_dna_sequence.py  
#python generate_dna_bigwig.py
#cd ..

## 3. prepare files for evaluation
# annotation bigwig files of promoter, gene, enhancer
python create_anno_bigwig.py
# variance of each position
python calculate_var.py




