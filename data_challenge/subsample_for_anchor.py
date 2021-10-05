import os
import sys 
import numpy as np
import pyBigWig
import re

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

step_size=1000 # subsample ratio
sample_len=np.ceil(np.array(num_bp)/step_size).astype(int)

chr_to_seed={}
i=0
for the_chr in chr_all:
    chr_to_seed[the_chr]=i
    i+=1

path1='./bigwig/'
path2='./sample_for_anchor_final/'
os.system('mkdir -p ' + path2)

# 7min per id
assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for i in np.arange(1,52):
    print(i)
    the_cell='C' + '%02d' % i
    for the_assay in assay_all:
        the_id = the_cell + the_assay
        if os.path.isfile(path1 + the_id + '.bigwig'):
            print(the_id)
            bw = pyBigWig.open(path1 + the_id + '.bigwig')
            sample=np.zeros(sum(sample_len))
            start=0
            j=0
            for the_chr in chr_all:
                signal=np.array(bw.values(the_chr,0,chr_len[the_chr]))
                signal[np.isnan(signal)]=0 ## !! raw bigwig has nan e.g. at ends
                # random index
                np.random.seed(chr_to_seed[the_chr])
                index=np.random.randint(0,chr_len[the_chr],sample_len[j])
                # subsample
                sample[start:(start+sample_len[j])]=signal[index]
                start+=sample_len[j]
                j+=1
            if np.any(np.isnan(sample)):
                print('sample contains nan!')
            sample.sort()
            np.save(path2 + 'sample_' + the_id, sample)



