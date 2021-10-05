import pyBigWig
import os
import sys
import numpy as np

# summary: number/percentage of positions over genome 3 billion
#>>> prom
#4649680
#>>> gene
#73509863
#>>> enh
#808448
#>>> prom/3000000000
#0.0015498933333333333
#>>> gene/3000000000
#0.02450328766666667
#>>> enh/3000000000
#0.0002694826666666667


chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
num_bp25=[9958257, 9687742, 7931823, 7608583, 7261531, 6832240, 6373839, 5805546, 5535789, 5351897, 5403465, 5331013, 4574574, 4281749, 4079648, 3613534, 3330298, 3214932, 2344705, 2577767, 1868400, 2032739, 6241636]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=num_bp25[i]

## load annotation #########
window_size=25
prom_loc=80
dict_prom={}
dict_gene={}
f=open('./anno/gencode.v29.genes.gtf.bed','r')
for line in f:
    the_chr, start, end, _, _, strand = line.split()
    start = int(start) // window_size
    end = int(end) // window_size + 1
    if the_chr not in dict_gene:
        dict_prom[the_chr]=[]
        dict_gene[the_chr]=[]
    dict_gene[the_chr]+=np.arange(start,end).tolist()
    if strand == '+':
        dict_prom[the_chr]+=np.arange(start-prom_loc,start).tolist()
    else:
        dict_prom[the_chr]+=np.arange(end, end+prom_loc).tolist()
f.close()

dict_enh={}
f=open('./anno/F5.hg38.enhancers.bed','r')
for line in f:
    the_chr, start, end, _, _, _, _, _, _, _, _, _ = line.split()
    start = int(start) // window_size
    end = int(end) // window_size + 1
    if the_chr not in dict_enh:
        dict_enh[the_chr]=[]
    dict_enh[the_chr]+=np.arange(start,end).tolist()
f.close()

for the_chr in chr_all:
    dict_prom[the_chr] = np.array(dict_prom[the_chr])
    dict_gene[the_chr] = np.array(dict_gene[the_chr])
    dict_enh[the_chr] = np.array(dict_enh[the_chr])

path2='./anno/'
os.system('mkdir -p ' + path2)

bw1 = pyBigWig.open(path2 + 'prom.bigwig', 'w')
bw1.addHeader(list(zip(chr_all , num_bp25)), maxZooms=0)
for the_chr in chr_all:
    print(the_chr)
    x=np.zeros(chr_len25[the_chr])
    x[dict_prom[the_chr]]=1
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
    bw1.addEntries(chroms, starts, ends=ends, values=vals)
bw1.close()


bw1 = pyBigWig.open(path2 + 'gene.bigwig', 'w')
bw1.addHeader(list(zip(chr_all , num_bp25)), maxZooms=0)
for the_chr in chr_all:
    print(the_chr)
    x=np.zeros(chr_len25[the_chr])
    x[dict_gene[the_chr]]=1
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
    bw1.addEntries(chroms, starts, ends=ends, values=vals)
bw1.close()


bw1 = pyBigWig.open(path2 + 'enh.bigwig', 'w')
bw1.addHeader(list(zip(chr_all , num_bp25)), maxZooms=0)
for the_chr in chr_all:
    print(the_chr)
    x=np.zeros(chr_len25[the_chr])
    x[dict_enh[the_chr]]=1
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
    bw1.addEntries(chroms, starts, ends=ends, values=vals)
bw1.close()



