import pyBigWig
import numpy as np
import sys
import os
import re

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

path1='./signal_anchored_final/'
path2='./signal_anchored_final/'
os.system('mkdir -p ' + path2)

assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for the_assay in assay_all:
    print(the_assay)
    bw = pyBigWig.open(path2 + 'avg_'+ the_assay + '.bigwig', 'w')
    bw.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples
    for the_chr in chr_all:
        print(the_chr)
        ## 1. calculate avg
        avg=np.zeros(chr_len[the_chr])
        count=0.0
        for i in np.arange(1,52):
            the_cell='C' + '%02d' % i
            the_id = the_cell + the_assay
            if os.path.isfile(path1 + the_id + '.bigwig'):
                print(the_id)
                the_bw = pyBigWig.open(path1 + the_id + '.bigwig')
                tmp = np.array(the_bw.values(the_chr,0,chr_len[the_chr]))
                if np.any(np.isnan(tmp)):
                    print('signal contains nan!')
                avg = avg + tmp
                the_bw.close()
                count += 1.0
        avg = avg / count
        ## 2. save bigwig
        # pad two zeroes
        z=np.concatenate(([0],avg,[0]))
        # find boundary
        starts=np.where(np.diff(z)!=0)[0]
        ends=starts[1:]
        starts=starts[:-1]
        vals=avg[starts]
        if starts[0]!=0:
            ends=np.concatenate(([starts[0]],ends))
            starts=np.concatenate(([0],starts))
            vals=np.concatenate(([0],vals))
        if ends[-1]!=chr_len[the_chr]:
            starts=np.concatenate((starts,[ends[-1]]))
            ends=np.concatenate((ends,[chr_len[the_chr]]))
            vals=np.concatenate((vals,[0]))
        # write
        chroms = np.array([the_chr] * len(vals))
        bw.addEntries(chroms, starts, ends=ends, values=vals)
        del avg
    bw.close()



