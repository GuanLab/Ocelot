import os
import sys 
import numpy as np
import re

# 3 seconds

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
tmp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=tmp[i]

step_size=1000 # subsample ratio
sample_len=np.ceil(np.array(tmp)/step_size).astype(int)

path1='./sample_for_anchor_final/'
path2='./sample_for_anchor_final/'
os.system('mkdir -p ' + path2)

assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for the_assay in assay_all:
    ref=np.zeros(sum(sample_len))
    count=0.0
    for i in np.arange(1,52):
        the_cell='C' + '%02d' % i
        the_id = the_cell + the_assay
        if os.path.isfile(path1 + 'sample_' + the_id + '.npy'):
            print(the_id)
            ref = ref + np.load(path1 + 'sample_' + the_id + '.npy')
            count+=1.0
    ref = ref/count
    np.save(path2 + 'ref_' + the_assay, ref)


