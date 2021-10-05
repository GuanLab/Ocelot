import pyBigWig
import numpy as np
import os
import sys

# 2hrs

path1='./'
path2='../signal_anchored_final/'
os.system('mkdir -p ' + path2)

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

base_all=['A','C','G','T']

for i in np.arange(len(base_all)):
    print(base_all[i])
    bw = pyBigWig.open(path2 + base_all[i] + '.bigwig', 'w')
    bw.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples
    for the_chr in chr_all:
        print(the_chr)
        x=np.load(the_chr + '.npy')
        # pad two zeroes
        z=np.concatenate(([0],x[i,:],[0]))
        # find boundary
        tmp1=np.where(np.diff(z)==1)[0]
        tmp2=np.where(np.diff(z)==-1)[0]
        starts=np.concatenate((tmp1, tmp2))
        starts.sort()
        ends=starts[1:]
        starts=starts[:-1]
        vals=np.zeros(len(starts))
        vals[np.arange(0,len(vals),2)]=1 # assume start with 0
        if starts[0]!=0: # if start with 1
            ends=np.concatenate(([starts[0]],ends))
            starts=np.concatenate(([0],starts))
            vals=np.concatenate(([0],vals))
        if ends[-1]!=chr_len[the_chr]: # if end with 0
            starts=np.concatenate((starts,[ends[-1]]))
            ends=np.concatenate((ends,[chr_len[the_chr]]))
            vals=np.concatenate((vals,[0]))
        # write
        chroms = np.array([the_chr] * len(vals))
        bw.addEntries(chroms, starts, ends=ends, values=vals)
    bw.close()





