import os
import sys
import pyBigWig
import numpy as np

# 30mins per assay

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
num_25bp=[9958257, 9687742, 7931823, 7608583, 7261531, 6832240, 6373839, 5805546, 5535789, 5351897, 5403465, 5331013, 4574574, 4281749, 4079648, 3613534, 3330298, 3214932, 2344705, 2577767, 1868400, 2032739, 6241636]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

path1='./orange_rank/'
path2='./signal_anchored_final/'
os.system('mkdir -p ' + path2)

assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for the_assay in assay_all:
    for i in np.arange(1,52):
        the_id='C' + '%02d' % i + the_assay
        if os.path.isfile(path1 + the_id + '.txt'):
            print(the_id)
            bw = pyBigWig.open(path2 + 'orange_'+ the_id + '.bigwig', 'w')
            bw.addHeader(list(zip(chr_all , num_25bp)), maxZooms=0) # zip two turples
            orange=np.loadtxt(path1 + the_id + '.txt')
            start_txt=0
            for the_chr in chr_all:
                print(the_chr)
                end_txt = chr_len25[the_chr] + start_txt
                x=orange[start_txt:end_txt]
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
#                if ends[-1]!=chr_len[the_chr]: 
#                    starts=np.concatenate((starts,[ends[-1]]))
#                    ends=np.concatenate((ends,[chr_len[the_chr]]))
#                    vals=np.concatenate((vals,[0]))
                # write
                chroms = np.array([the_chr] * len(vals))
                bw.addEntries(chroms, starts, ends=ends, values=vals)
                start_txt = end_txt
            bw.close()





