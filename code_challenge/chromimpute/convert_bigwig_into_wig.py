import os
import sys
import numpy as np
import pyBigWig

chr_all=['chr' + str(i) for i in range(1,23)] + ['chrX']

num_bp_grch38=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len_grch38={}
for i in np.arange(len(chr_all)):
    chr_len_grch38[chr_all[i]]=num_bp_grch38[i]

id_all=np.loadtxt('list_train_vali.txt',dtype='str')

path0 = '../../data_challenge/bigwig/'
path1 = './CHALLENGE/CONVERTEDDATADIR/'
os.system('mkdir -p ' + path1)

list_chr=chr_all

for the_id in id_all:
    print(the_id)
    bw=pyBigWig.open(path0 + the_id + '.bigwig') 
    for the_chr in list_chr:
        x=np.array(bw.values(the_chr,0,chr_len_grch38[the_chr]))
        x[np.isnan(x)]=0
        # mean of 25bp bin
        x=x.reshape((1,len(x)))
        tmp=np.zeros((1,int(np.ceil(x.shape[1]/25.0)*25 - x.shape[1])))
        x=np.concatenate((x,tmp),axis=1)
        x=x.reshape((-1,25)).T
        x=np.mean(x,axis=0)
        np.savetxt(f'{path1}{the_chr}_{the_id}.wig', x, fmt='%.2f')
        # header for wig file
        os.system(f'sed -e "s/XXX/{the_id}/g; s/YYY/{the_chr}/g" < {path1}header.txt > tmp.wig')
        os.system(f'ex - {path1}{the_chr}_{the_id}.wig < tmp.wig') 
        # gzip othervise chromimpute input error
        os.system(f'gzip {path1}{the_chr}_{the_id}.wig')
    bw.close()

