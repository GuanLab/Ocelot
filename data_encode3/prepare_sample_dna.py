import pyBigWig
import argparse
import os
import sys
import numpy as np

def calculate_mmm(input3d, axis=2): # input 3d output 3d; calculate max, min, mean along axis=2
    output3d=np.zeros((input3d.shape[0], input3d.shape[1], 3), dtype='float32')
    output3d[:,:,0]= np.max(input3d, axis=axis)
    output3d[:,:,1]= np.min(input3d, axis=axis)
    output3d[:,:,2]= np.mean(input3d, axis=axis)
    return output3d

def convert_feature_tensor(input2d, reso=25): # e.g. input2d d0 * (25*n) -> output2d (d0*25) * n 
    d0 = input2d.shape[0]
    output2d=input2d.reshape((d0, -1, reso))
    output2d=np.swapaxes(output2d,1,2)
    output2d=output2d.reshape((d0 * reso,-1))
    return output2d

reso=25
size_unet=128
tmp_num_sample=50000
flank=5
neighbor=2*flank+1
flank_dna=1
neighbor_dna=2*flank_dna+1

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
# grch38
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
#num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/reso))

freq=np.rint(np.array(num_bp)/sum(num_bp)*tmp_num_sample).astype('int')
num_sample = np.sum(freq) # num_sample may shift e.g. 50,000 -> 50,002
print(freq)

def get_args():
    parser = argparse.ArgumentParser(description="prepare sample")
    parser.add_argument('-i', '--input', default='E017_H3K4me1.bigwig', type=str, help='input bigwig')
    parser.add_argument('-s', '--seed', default='449', type=int, help='sampling seed')
    args = parser.parse_args()
    return args

args=get_args()
print(args)

the_id=args.input
the_seed=args.seed
np.random.seed(the_seed)

#the_cell, the_assay = the_id.split('_')

path0='./bigwig_all/'
#path1='./sample_all_unet_seed' + str(the_seed) + '/'
#os.system('mkdir -p ' + path1)
path2='./sample_all_lgbm_seed' + str(the_seed) + '/'
os.system('mkdir -p ' + path2)

# final output
#list_dna=['A','C','G','T']
#image_unet_dna=np.zeros((size_unet*reso, num_sample), dtype='int32')
# before converting
image_lgbm_dna=np.zeros((1, num_sample*25*neighbor_dna), dtype='int32')

bw_dna=pyBigWig.open(path0 + the_id + '.bigwig')

index_sample=0
start_label=0
for i in np.arange(len(chr_all)):
    the_chr = chr_all[i]
    print('sample data from ' + the_chr)
#    print(index_sample, start_label)
    image_raw = np.array(bw_dna.values(the_chr, 0, chr_len[the_chr]), dtype='int32') 
    # index
    end_label = start_label + freq[i]
    start_dna = start_label*25*neighbor_dna
    end_dna = end_label*25*neighbor_dna
    ## 1.unet
    start=np.random.randint(0, chr_len_bin[the_chr]-size_unet-1, freq[i])
#    print(start)
    end=start + size_unet
#    for j in np.arange(len(start)):
#        image_unet_dna[:,index_sample] = image_raw[start[j]*reso:end[j]*reso]
#        index_sample += 1
    ###################### 
    ## 2.lgbm
    start=np.random.randint(0+flank, chr_len_bin[the_chr]-1-flank, freq[i])
    print(start)
    # 2.2 feature dna at 1bp reso
    the_index=np.zeros((25*neighbor_dna,freq[i]))
    for aaa in np.arange(25*neighbor_dna):
        the_index[aaa,:]=start*25 - flank_dna*25 + aaa
    the_index=the_index.T.flatten().astype('int')
    image_lgbm_dna[:,start_dna:end_dna]=image_raw[the_index]
    # update
    start_label = end_label

# converting
image_lgbm_dna=convert_feature_tensor(image_lgbm_dna, reso=25*neighbor_dna) # 2d - d0 * n

# save
#np.save(path1 + 'image_' + the_id + '.npy', image_unet_dna)
np.save(path2 + 'image_' + the_id + '.npy', image_lgbm_dna)

bw_dna.close()

## feature order:
#e.g. flank=5, neighbor=11
#f-feature; n-neighbor; m-mmm(max,min,mean)
#(n1-upstream; n6-center; n11-downstream )
#
#f1-n1-m1
#f1-n1-m2
#f1-n1-m3
#--
#f1-n2-m1
#f1-n2-m2
#f1-n2-m3
#--
#f1-n3-m1
#f1-n3-m2
#f1-n3-m3
#.
#.
#.
#f1-n11-m1
#f1-n11-m2
#f1-n11-m3

 
