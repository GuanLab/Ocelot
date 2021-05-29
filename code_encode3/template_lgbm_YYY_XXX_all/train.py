import pyBigWig
import argparse
import os
import sys
import numpy as np
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

size=25
tmp_num_sample=50000
num_epoch=1
num_early_stop=20
flank=5
neighbor=2*flank+1
flank_dna=1
neighbor_dna=2*flank_dna+1

## for lgbm ############
num_boost=500
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 50,
    'min_data_in_leaf': 20,
    #'learning_rate': 0.05,
    'verbose': 0,
    'lambda_l2': 2.0,
    'bagging_freq': 1,
    'bagging_fraction': 0.7,
}
#########################

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
# grch38
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
#num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

path1='../../data_encode3/sample_all_lgbm_seed0/'
path2='../../data_encode3/sample_all_lgbm_seed1/' 

## test cell
list_dna=['A','C','G','T']

# argv
def get_args():
    parser = argparse.ArgumentParser(description="globe train")
    parser.add_argument('-f', '--feature', default='H3K4me1', nargs='+', type=str,
        help='feature assay')
    parser.add_argument('-t', '--target', default='H3K9ac',type=str,
        help='target assay')
    parser.add_argument('-c', '--cell', default='E002', nargs='+',type=str,
        help='cell lines as common and specific features')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for common - specific partition')
    args = parser.parse_args()
    return args

args=get_args()
print(args)

assay_target = args.target
assay_feature = args.feature
cell_all = args.cell
seed_partition = args.seed

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

print('assay_target', assay_target)
print('assay_feature', len(assay_feature), assay_feature)
print('cell_common', len(cell_common), cell_common)
print('cell_tv', len(cell_tv), cell_tv)
print('cell_train', len(cell_train), cell_train)
print('cell_vali', len(cell_vali), cell_vali)

## load pre-sampled data ######################################
# sample freq for each chr
freq=np.rint(np.array(num_bp)/sum(num_bp)*tmp_num_sample).astype('int')
num_sample = np.sum(freq) # num_sample may shift e.g. 50,000 -> 50,001

# number of features for each type
num_dna=4
num_feature=len(cell_common)*2 + len(assay_feature)*2
num_n5cut=len(cell_common)*2 + len(assay_feature)*2

## 1.label
label_train=np.zeros(0)
for the_cell in cell_train:
    label_train = np.concatenate((label_train, np.load(path1 + 'label_' + the_cell + '_' + assay_target + '.npy')))
label_vali=np.zeros(0)
for the_cell in cell_vali:
    label_vali = np.concatenate((label_vali, np.load(path2 + 'label_' + the_cell + '_' + assay_target + '.npy')))

## 2.image
image_train_final=np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_n5cut*neighbor,\
    num_sample * len(cell_train)), dtype='float32')
image_vali_final=np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_n5cut*neighbor, \
    num_sample * len(cell_vali)), dtype='float32')

for k in np.arange(len(cell_train)):
    cell_target = cell_train[k]
    image_train = np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_n5cut*neighbor,\
        num_sample), dtype='float32')
    num=0
    ## dna
    for j in np.arange(len(list_dna)):
        the_id=list_dna[j]
        print(the_id)
        image_train[num:(num+25*neighbor_dna), :] = np.load(path1 + 'image_' + the_id  + '.npy')
        num += 25*neighbor_dna
    ## mmm & diff
    for the_cell in cell_common:
        the_id = the_cell + '_' + assay_target
        print(the_id)
        image_train[num:(num+3*neighbor*2), :] = np.load(path1 + 'image_' + the_id  + '.npy')
        num += 3*neighbor*2
    for the_assay in assay_feature:
        the_id = cell_target + '_' + the_assay
        print(the_id)
        image_train[num:(num+3*neighbor*2), :] = np.load(path1 + 'image_' + the_id  + '.npy')
        num += 3*neighbor*2
    ## n5cut & diff
    for the_cell in cell_common:
        the_id = the_cell + '_' + assay_target
        print(the_id)
        image_train[num:(num+neighbor*2), :] = np.load(path1 + 'image_n5cut_' + the_id  + '.npy')
        num += neighbor*2
    for the_assay in assay_feature:
        the_id = cell_target + '_' + the_assay
        print(the_id)
        image_train[num:(num+neighbor*2), :] = np.load(path1 + 'image_n5cut_' + the_id  + '.npy')
        num += neighbor*2
    image_train_final[:, (k*num_sample) : ((k+1)*num_sample)] = image_train

for k in np.arange(len(cell_vali)):
    cell_target = cell_vali[k]
    image_vali = np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_n5cut*neighbor,\
        num_sample), dtype='float32')
    num=0
    ## dna
    for j in np.arange(len(list_dna)):
        the_id=list_dna[j]
        print(the_id)
        image_vali[num:(num+25*neighbor_dna), :] = np.load(path2 + 'image_' + the_id  + '.npy')
        num += 25*neighbor_dna
    ## mmm & diff
    for the_cell in cell_common:
        the_id = the_cell + '_' + assay_target
        print(the_id)
        image_vali[num:(num+3*neighbor*2), :] = np.load(path2 + 'image_' + the_id  + '.npy')
        num += 3*neighbor*2
    for the_assay in assay_feature:
        the_id = cell_target + '_' + the_assay
        print(the_id)
        image_vali[num:(num+3*neighbor*2), :] = np.load(path2 + 'image_' + the_id  + '.npy')
        num += 3*neighbor*2
    ## n5cut & diff
    for the_cell in cell_common:
        the_id = the_cell + '_' + assay_target
        print(the_id)
        image_vali[num:(num+neighbor*2), :] = np.load(path2 + 'image_n5cut_' + the_id  + '.npy')
        num += neighbor*2
    for the_assay in assay_feature:
        the_id = cell_target + '_' + the_assay
        print(the_id)
        image_vali[num:(num+neighbor*2), :] = np.load(path2 + 'image_n5cut_' + the_id  + '.npy')
        num += neighbor*2
    image_vali_final[:, (k*num_sample) : ((k+1)*num_sample)] = image_vali

## randomly shuffle
the_index=np.arange(len(label_train))
np.random.shuffle(the_index)
image_train_final=image_train_final[:,the_index]
label_train=label_train[the_index]
the_index=np.arange(len(label_vali))
np.random.shuffle(the_index)
image_vali_final=image_vali_final[:,the_index]
label_vali=label_vali[the_index]

## train & save
os.system('mkdir -p model')
data_train=lgb.Dataset(image_train_final.T, label=label_train, free_raw_data=False)
data_vali=lgb.Dataset(image_vali_final.T, label=label_vali, free_raw_data=False)
gbm = lgb.train(params, data_train, num_boost_round=num_boost, valid_sets=data_vali, early_stopping_rounds=num_early_stop)
pickle.dump(gbm, open('./model/lgbm_' + str(seed_partition) + '.model', 'wb'))
gbm.save_model('./model/lgbm_' + str(seed_partition) + '.txt', num_iteration=None)



