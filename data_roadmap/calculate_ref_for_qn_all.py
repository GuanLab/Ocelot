import os
import sys
import numpy as np
import re

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp_grch37=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]
chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp_grch37[i]

step_size=1000 # subsample ratio
sample_len=np.ceil(np.array(num_bp_grch37)/step_size).astype(int)

path1='./sample_for_anchor/'
path2='./sample_ref_all/'
os.system('mkdir -p ' + path2)

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac','DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac','H3K23me2','H2AK9ac','H3T11ph','H4K12ac']
tier4=['methyl','RNA-seq']

path0='./roadmap/'

for the_assay in tier0+tier1+tier2+tier3+tier4:
    print(the_assay + ':')
    ref=np.zeros(sum(sample_len))
    count=0.0
    for i in np.arange(1,130):
        the_cell='E%03d' % i
        the_id = the_cell + '_' + the_assay
        if os.path.isfile(path1 + the_id + '.npy'):
            print(the_cell)
            ref = ref + np.load(path1 + the_id + '.npy')
            count+=1.0
    ref = ref/count
    np.save(path2 + 'ref_' + the_assay, ref)
    print(count)


