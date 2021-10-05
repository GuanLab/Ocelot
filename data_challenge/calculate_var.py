import pyBigWig
import os
import sys
import numpy as np
import scipy.stats
import re

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895])
num_bp25=[9958257, 9687742, 7931823, 7608583, 7261531, 6832240, 6373839, 5805546, 5535789, 5351897, 5403465, 5331013, 4574574, 4281749, 4079648, 3613534, 3330298, 3214932, 2344705, 2577767, 1868400, 2032739, 6241636]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=num_bp25[i]

path1='./bigwig/'
path2='./var_final/'
os.system('mkdir -p ' + path2)

## test cell
#cell_test=['C04','C17','C20','C24','C32','C34','C46','C48','C50']
assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for the_assay in assay_all:
    bw2 = pyBigWig.open(path2 + 'var_'+ the_assay + '.bigwig', 'w')
    bw2.addHeader(list(zip(chr_all , num_bp25)), maxZooms=0)
    for the_chr in chr_all:
        print(the_chr)
        x=[]
        for i in np.arange(1,52):
            the_cell = 'C' + '%02d' % i
            the_id = the_cell + the_assay
            if os.path.isfile(path1 + the_id + '.bigwig'):
                bw1 = pyBigWig.open(path1 + the_id + '.bigwig')
                y=np.array(bw1.values(the_chr,0,chr_len[the_chr]))
                y[np.isnan(y)]=0 # fill nan with 0
                tmp=np.zeros(int(np.ceil(len(y)/25.0)*25 - len(y)))
                y=np.concatenate((y,tmp))
                y=y.reshape((-1,25)).T
                x.append(np.mean(y,axis=0))
                bw1.close()
        x=np.std(np.array(x), axis=0) ** 2
        np.save(path2 + 'var_'+ the_assay + '_' + the_chr, x)
        # pad two zeroes
        z=np.concatenate(([0],x,[0]))
        # find boundary
        starts=np.where(np.diff(z)!=0)[0]
        ends=starts[1:]
        starts=starts[:-1]
        vals=x[starts]
        if starts[0]!=0:
            ends=np.concatenate(([starts[0]],ends))
            starts=np.concatenate(([0],starts))
            vals=np.concatenate(([0],vals))
        if ends[-1]!=chr_len25[the_chr]:
            starts=np.concatenate((starts,[ends[-1]]))
            ends=np.concatenate((ends,[chr_len25[the_chr]]))
            vals=np.concatenate((vals,[0]))
        # write
        chroms = np.array([the_chr] * len(vals))
        bw2.addEntries(chroms, starts, ends=ends, values=vals)
    bw2.close()




