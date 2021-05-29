import argparse
import pyBigWig
import os
import sys
import numpy as np
import scipy.stats
import re
import lightgbm as lgb
import pickle

def calculate_mmm(input3d, axis=2): # input 3d output 3d; calculate max, min, mean along axis=2
    output3d=np.zeros((input3d.shape[0], input3d.shape[1], 3))
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

num_epoch=1
size_batch=100000 # 0.1m; must be 25*n
flank=5
neighbor=2*flank+1
flank_dna=1
neighbor_dna=2*flank_dna+1

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
#num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len_cut={}
for i in np.arange(len(chr_all)):
    chr_len_cut[chr_all[i]]=int(np.floor(num_bp[i]/25.0)*25) # HERE I cut tails

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

path0='../../data/bigwig_all/'

## test cell
exclude_all=[]
list_dna=['A','C','G','T']

# argv
def get_args():
    parser = argparse.ArgumentParser(description="globe prediction")
    parser.add_argument('-f', '--feature', default='H3K4me1', nargs='+', type=str,
        help='feature assay')
    parser.add_argument('-t', '--target', default='H3K9ac',type=str,
        help='target assay')
    parser.add_argument('-c', '--cell', default='E002', nargs='+',type=str,
        help='cell lines as common and specific features')
    parser.add_argument('-s', '--seed', default='0', type=int,
        help='seed for common - specific partition')
    parser.add_argument('-ct', '--cell_test', default='E002', type=str,
        help='the testing cell line')
    args = parser.parse_args()
    return args

args=get_args()


assay_target = args.target
assay_feature = args.feature
cell_all = args.cell
seed_partition = args.seed
cell_test = args.cell_test
list_label_test=[cell_test + '_' + assay_target]

## parition common & specific cell types
np.random.seed(seed_partition)
np.random.shuffle(cell_all)
ratio=[0.5,0.5]
num = int(len(cell_all)*ratio[0])
cell_common = cell_all[:num]
cell_tv = cell_all[num:]

cell_all.sort()
cell_common.sort()
cell_tv.sort()

## partition train-vali cell types
np.random.seed(0) #TODO 
np.random.shuffle(cell_tv)
ratio=[0.75,0.25]
num = int(len(cell_tv)*ratio[0])
cell_train = cell_tv[:num]
cell_vali = cell_tv[num:]

cell_tv.sort()
cell_train.sort()
cell_vali.sort()

# load pyBigWig
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_feature={}
dict_n5cut={}
for the_cell in cell_common:
    the_id = the_cell + '_' + assay_target
    dict_feature[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
    dict_n5cut[the_id]=pyBigWig.open(path0 + 'n5cut_' + the_id + '.bigwig')
for the_cell in [cell_test]:
    for the_assay in assay_feature:
        the_id = the_cell + '_' + the_assay
        dict_feature[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
        dict_n5cut[the_id]=pyBigWig.open(path0 + 'n5cut_' + the_id + '.bigwig')
dict_avg={}
dict_avg_n5cut={}
dict_avg[assay_target]=pyBigWig.open(path0 + 'avg_' + assay_target + '.bigwig')
dict_avg_n5cut[assay_target]=pyBigWig.open(path0 + 'avg_n5cut_' + assay_target + '.bigwig')
for the_assay in assay_feature:
    dict_avg[the_assay]=pyBigWig.open(path0 + 'avg_' + the_assay + '.bigwig')
    dict_avg_n5cut[the_assay]=pyBigWig.open(path0 + 'avg_n5cut_' + the_assay + '.bigwig')

print('assay_target', assay_target)
print('assay_feature', len(assay_feature), assay_feature)
print('cell_common', len(cell_common), cell_common)
print('cell_tv', len(cell_tv), cell_tv)
print('cell_train', len(cell_train), cell_train)
print('cell_vali', len(cell_vali), cell_vali)
print('cell_test', len(cell_test), cell_test)

gbm=pickle.load(open('./model/lgbm_' + str(seed_partition) + '.model', 'rb'))

## prediction
list_chr=chr_all
#list_chr=['chr21'] # TODO

path_pred='./pred/'
os.system('mkdir -p ' + path_pred)
bw_output = pyBigWig.open(path_pred + 'pred_' + list_label_test[0] + '_' + str(seed_partition) + '.bigwig','w')
bw_output.addHeader(list(zip(chr_all , np.ceil(np.array(num_bp)/25).astype('int').tolist())), maxZooms=0)

num_dna=4
num_feature=len(cell_common)*2 + len(assay_feature)*2
num_n5cut=len(cell_common)*2 + len(assay_feature)*2

for the_chr in list_chr:
    print(the_chr)
    pred=np.zeros(chr_len_bin[the_chr])
    start=0
    while start < chr_len_cut[the_chr] - 25*flank:
#        print(start, start/chr_len_cut[the_chr])
        end = min(start + size_batch, chr_len_cut[the_chr])
        image=np.zeros((num_dna + num_feature, end-start), dtype='float32')
        # 2.1 dna
        num=0
        for j in np.arange(len(list_dna)):
            the_id=list_dna[j]
            image[num,:] = dict_dna[the_id].values(the_chr,start,end)
            num+=1
        # 2.2 feature & diff
        the_avg=dict_avg[assay_target].values(the_chr,start,end)
        for the_cell in cell_common:
            the_id = the_cell + '_' + assay_target
            # feature
            image[num,:] = dict_feature[the_id].values(the_chr,start,end)
            # diff
            image[num+1,:]=image[num,:] - the_avg
            num+=2
        for the_assay in assay_feature:
            the_id = cell_test + '_' + the_assay
            # feature
            image[num,:] = dict_feature[the_id].values(the_chr,start,end)
            # diff
            image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,start,end)
            num+=2

        start_n5cut=int(start/25)
        end_n5cut=int(end/25)
        n5cut=np.zeros((num_n5cut, end_n5cut-start_n5cut), dtype='float32')
        # 2.3 n5cut & diff
        num=0
        the_avg=dict_avg_n5cut[assay_target].values(the_chr,start_n5cut,end_n5cut)
        for the_cell in cell_common:
            the_id = the_cell + '_' + assay_target
            # feature
            n5cut[num,:] = dict_n5cut[the_id].values(the_chr,start_n5cut,end_n5cut)
            # diff
            n5cut[num+1,:]=n5cut[num,:] - the_avg
            num+=2
        for the_assay in assay_feature:
            the_id = cell_test + '_' + the_assay
            # feature
            n5cut[num,:] = dict_n5cut[the_id].values(the_chr,start_n5cut,end_n5cut)
            # diff
            n5cut[num+1,:]=n5cut[num,:] - dict_avg_n5cut[the_assay].values(the_chr,start_n5cut,end_n5cut)
            num+=2

        # convert to 25bp features
        # dna - 25bp one hot
        image1=convert_feature_tensor(image[:num_dna,:], reso=25) # this reso is different from train.py
        # assay - mmm features
        image2=image[4:,:].reshape((num_feature, -1, 25)) # 3d - d0 * n * 25
        image2=calculate_mmm(image2) # 3d - d0 * n * 3
        image2=image2.reshape((num_feature, -1)) # 2d - d0 * (n*3)
        image2=convert_feature_tensor(image2, reso=3) # 2d - (d0*3) * n

        # convert to largespace; f1 neighbors -> f2 neighbors -> ...
        largespace=np.zeros((num_dna*25*neighbor_dna+num_feature*3*neighbor+num_n5cut*neighbor, \
            int((end-start)/25)-2*flank))
        # dna - 25bp one hot
        for n1 in np.arange(num_dna):
            for n2 in np.arange(neighbor_dna):
                tmp1 = n1*25*neighbor_dna + n2*25
                tmp2 = tmp1 + 25
                tmp = flank-flank_dna # neighbor_dna short so we need to shift it
                largespace[tmp1:tmp2,:]=image1[n1*25:(n1+1)*25, \
                    (n2+tmp):(int((end-start)/25)-2*flank+n2+tmp)]
        # assay - mmm features
        for n1 in np.arange(num_feature):
            for n2 in np.arange(neighbor):
                tmp1 = n1*3*neighbor + n2*3 + num_dna*25*neighbor_dna
                tmp2 = tmp1 + 3
                largespace[tmp1:tmp2,:]=image2[n1*3:(n1+1)*3, n2:(int((end-start)/25)-2*flank+n2)]
        # n5cut
        for n1 in np.arange(num_n5cut):
            for n2 in np.arange(neighbor):
                tmp1 = n1*neighbor + n2*1 + num_feature*3*neighbor + num_dna*25*neighbor_dna
                tmp2 = tmp1 + 1
                largespace[tmp1:tmp2,:]=n5cut[n1:(n1+1), n2:(int((end-start)/25)-2*flank+n2)]

        start_pred=int(start/25) + flank # skip both ends of a chromosome for now            
        end_pred=int(end/25) - flank
        pred[start_pred:end_pred]=gbm.predict(largespace.T)
        del image
        del n5cut
        del image1
        del image2
        del largespace

        start = end - 25*flank

    #np.save('pred25bp_' + list_label_test[0] + '_' + cell_train + '_' + cell_vali + '_' + the_chr, pred)
    x=pred
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
    if ends[-1]!=chr_len_bin[the_chr]:
        starts=np.concatenate((starts,[ends[-1]]))
        ends=np.concatenate((ends,[chr_len_bin[the_chr]]))
        vals=np.concatenate((vals,[0]))
    # write 
    chroms = np.array([the_chr] * len(vals))
    bw_output.addEntries(chroms, starts, ends=ends, values=vals)

bw_output.close()

## close bw
for the_id in dict_dna.keys():
    dict_dna[the_id].close()
for the_id in dict_feature.keys():
    dict_feature[the_id].close()
for the_id in dict_n5cut.keys():
    dict_n5cut[the_id].close()
for the_id in dict_avg.keys():
    dict_avg[the_id].close()
for the_id in dict_avg_n5cut.keys():
    dict_avg_n5cut[the_id].close()




