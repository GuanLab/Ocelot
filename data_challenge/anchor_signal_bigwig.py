import os
import sys 
import numpy as np
import re
import pyBigWig

def anchor (input,sample,ref): # input 1d array
    sample.sort()
    ref.sort()
    # 0. create the mapping function
    index=np.array(np.where(np.diff(sample)!=0))+1
    index=index.flatten()
    x=np.concatenate((np.zeros(1),sample[index])) # domain
    y=np.zeros(len(x)) # codomain
    for i in np.arange(0,len(index)-1,1):
        start=index[i]
        end=index[i+1]
        y[i+1]=np.mean(ref[start:end])
    i+=1
    start=index[i]
    end=len(ref)
    y[i+1]=np.mean(ref[start:end])
    # 1. interpolate
    output=np.interp(input, x, y)
    # no extrapolate - simply map to the ref max to remove extremely large values
    # 2. extrapolate
#    degree=1 # degree of the fitting polynomial
#    num=10 # number of positions for extrapolate
#    f1=np.poly1d(np.polyfit(sample[-num:],ref[-num:],degree))
#    f2=np.poly1d(np.polyfit(sample[:num],ref[:num],degree))
#    output[input>sample[-1]]=f1(input[input>sample[-1]])
#    output[input<sample[0]]=f2(input[input<sample[0]])
    return output

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

path1='./bigwig/'
path2='./signal_anchored_final/'
os.system('mkdir -p ' + path2)
path3='./sample_for_anchor_final/'

assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

for i in np.arange(1,52):
    for the_assay in assay_all:
        the_id='C' + '%02d' % i + the_assay
        if os.path.isfile(path1 + the_id + '.bigwig'):
            print(the_id)
            # output bw
            bw = pyBigWig.open(path2 + the_id + '.bigwig', 'w')
            bw.addHeader(list(zip(chr_all , num_bp)), maxZooms=0) # zip two turples
            # input bw
            bw_ori = pyBigWig.open(path1 + the_id + '.bigwig')
            # sample & ref for quantile normalization
            sample=np.load(path3 + 'sample_' + the_id + '.npy')
            ref=np.load(path3 + 'ref_' + the_assay + '.npy')
            for the_chr in chr_all:
                print(the_chr)
                x=bw_ori.intervals(the_chr)
                starts=[]
                ends=[]
                vals=[]
                for i in np.arange(len(x)):
                	starts.append(x[i][0])
                	ends.append(x[i][1])
                	vals.append(x[i][2])
                vals=anchor(vals,sample,ref)
                if ends[-1]!=chr_len[the_chr]:
                    starts=np.concatenate((starts,[ends[-1]]))
                    ends=np.concatenate((ends,[chr_len[the_chr]]))
                    vals=np.concatenate((vals,[0]))
                # write
                chroms = np.array([the_chr] * len(vals))
                bw.addEntries(chroms, starts, ends=ends, values=vals)
            bw.close()
            bw_ori.close()



