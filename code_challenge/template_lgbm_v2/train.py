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
tmp_num_sample=80000
num_epoch=1
#num_early_stop=20
flank=8
neighbor=2*flank+1
flank_dna=1
neighbor_dna=0 #2*flank_dna+1

## for lgbm ############
num_boost=50
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 50,
    'min_data_in_leaf': 20,
    #'learning_rate': 0.05,
    'verbose': 0,
    'lambda_l2': 2.0,
#    'bagging_freq': 1,
#    'bagging_fraction': 0.7,
}
#########################

path1='../../data_challenge/signal_anchored_final/'

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

## test cell
#exclude_all=['C04','C17','C20','C24','C32','C34','C46','C48','C50']
exclude_all=[]
list_dna=['A','C','G','T']

# argv
def get_args():
    parser = argparse.ArgumentParser(description="globe train")
    parser.add_argument('-f', '--feature', default='M02', nargs='+', type=str,
        help='feature assay')
    parser.add_argument('-t', '--target', default='M18', nargs='+',type=str,
        help='target assay')
    parser.add_argument('-cv', '--crossvalidation', nargs='+', type=str,
        help='crossvalidation cell lines')
    args = parser.parse_args()
    return args

args=get_args()

exclude_all.extend(args.crossvalidation)
cell_train=args.crossvalidation[0]
cell_vali=args.crossvalidation[1]

## target
list_label_train=[]
list_label_vali=[]
for the_assay in args.target:
    list_label_train.append(cell_train + the_assay)
    list_label_vali.append(cell_vali + the_assay)

## row feature
list_feature_train=[]
list_feature_vali=[]
for the_assay in args.feature:
    list_feature_train.append(cell_train + the_assay)
    list_feature_vali.append(cell_vali + the_assay)

## col feature
list_feature_common=[]
for the_assay in args.target:
    for i in np.arange(1,52):
        the_cell = 'C' + '%02d' % i
        the_id = the_cell + the_assay
        if (the_cell not in exclude_all) and os.path.isfile(path1 + the_id + '.bigwig'):
            list_feature_common.append(the_id)

# load pyBigWig
dict_label_train={}
for the_id in list_label_train:
    dict_label_train[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
dict_label_vali={}
for the_id in list_label_vali:
    dict_label_vali[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
#dict_dna={}
#for the_id in list_dna:
#    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_common={}
dict_orange_common={}
for the_id in list_feature_common:
    dict_feature_common[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_orange_common[the_id]=pyBigWig.open(path1 + 'orange_' + the_id + '.bigwig')
dict_feature_train={}
dict_orange_train={}
for the_id in list_feature_train:
    dict_feature_train[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_orange_train[the_id]=pyBigWig.open(path1 + 'orange_' + the_id + '.bigwig')
dict_feature_vali={}
dict_orange_vali={}
for the_id in list_feature_vali:
    dict_feature_vali[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_orange_vali[the_id]=pyBigWig.open(path1 + 'orange_' + the_id + '.bigwig')
dict_avg={}
dict_avg_orange={}
for the_assay in args.feature:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
    dict_avg_orange[the_assay]=pyBigWig.open(path1 + 'avg_orange_' + the_assay + '.bigwig')
for the_assay in args.target:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
    dict_avg_orange[the_assay]=pyBigWig.open(path1 + 'avg_orange_' + the_assay + '.bigwig')

print('label_train', list_label_train)
print('label_vali', list_label_vali)
print('feature_common', list_feature_common)
print('feature_train', list_feature_train)
print('feature_vali', list_feature_vali)

## prepare data ######################################
# sample freq for each chr
freq=np.rint(np.array(num_bp)/sum(num_bp)*tmp_num_sample).astype('int')
num_sample = np.sum(freq) # num_sample may shift e.g. 50,000 -> 50,001
# number of features for each type
num_dna=4
num_feature=len(list_feature_common)*2 + len(list_feature_train)*2
num_orange=len(list_feature_common)*2 + len(list_feature_train)*2
# image_raw is the original 25bp -> 1bp; need to convert it later
label_train=np.zeros(num_sample, dtype='float32')
label_vali=np.zeros(num_sample, dtype='float32')
dna_train=np.zeros((num_dna, num_sample*25*neighbor_dna))
dna_vali=np.zeros((num_dna, num_sample*25*neighbor_dna))
image_train=np.zeros((num_feature, num_sample*25*neighbor), dtype='float32')
image_vali=np.zeros((num_feature, num_sample*25*neighbor), dtype='float32')
orange_train=np.zeros((num_orange, num_sample*neighbor), dtype='float32')
orange_vali=np.zeros((num_orange, num_sample*neighbor), dtype='float32')
# index for final image; for each chr, step freq[i]
start_label=0    
for i in np.arange(len(chr_all)):
    the_chr=chr_all[i]
    end_label = start_label + freq[i]
    start_dna = start_label*25*neighbor_dna
    end_dna = end_label*25*neighbor_dna
    start_image = start_label*25*neighbor
    end_image = end_label*25*neighbor
    start_orange = start_label*neighbor
    end_orange = end_label*neighbor
    # index for train & vali
    start1=np.random.randint(0 + flank, \
        int(np.ceil(chr_len[the_chr]/float(size))-1) - flank, freq[i])
    start2=np.random.randint(0 + flank, \
        int(np.ceil(chr_len[the_chr]/float(size))-1) - flank, freq[i])
    ## 1. label ##########
    for j in np.arange(len(list_label_train)): # single-task
        the_id=list_label_train[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:]=dict_label_train[the_id].values(the_chr, 0, chr_len25[the_chr])
        label_train[start_label:end_label]=tmp[start1]
    for j in np.arange(len(list_label_vali)): # single-task
        the_id=list_label_vali[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:]=dict_label_vali[the_id].values(the_chr, 0, chr_len25[the_chr])
        label_vali[start_label:end_label]=tmp[start2]
    ## 2. image ##########
#    # 2.1.0 dna index for train & vali
#    the_index1=np.zeros((25*neighbor_dna,freq[i]))
#    the_index2=np.zeros((25*neighbor_dna,freq[i]))
#    for aaa in np.arange(25*neighbor_dna):
#        the_index1[aaa,:]=start1*25 - flank_dna*25 + aaa 
#        the_index2[aaa,:]=start2*25 - flank_dna*25 + aaa
#    the_index1=the_index1.T.flatten().astype('int')
#    the_index2=the_index2.T.flatten().astype('int')
#    # 2.1 dna
#    num=0
#    for j in np.arange(len(list_dna)):
#        the_id=list_dna[j]
#        tmp = np.zeros(chr_len[the_chr])
#        tmp[:] = dict_dna[the_id].values(the_chr, 0, chr_len[the_chr])
#        dna_train[num, start_dna:end_dna] = tmp[the_index1] 
#        dna_vali[num, start_dna:end_dna] = tmp[the_index2]
#        num+=1
    # 2.2.0 image index for train & vali
    the_index1=np.zeros((25*neighbor,freq[i]))
    the_index2=np.zeros((25*neighbor,freq[i]))
    for aaa in np.arange(25*neighbor):
        the_index1[aaa,:]=start1*25 - flank*25 + aaa 
        the_index2[aaa,:]=start2*25 - flank*25 + aaa
    the_index1=the_index1.T.flatten().astype('int')
    the_index2=the_index2.T.flatten().astype('int')
    # 2.2 feature & diff
    the_assay=re.sub('C[0-9][0-9]','',list_feature_common[0])
    tmp_avg = np.zeros(chr_len[the_chr])
    tmp_avg[:] = dict_avg[the_assay].values(the_chr, 0, chr_len[the_chr])
    num=0
    for j in np.arange(len(list_feature_common)):
        the_id=list_feature_common[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_feature_common[the_id].values(the_chr, 0, chr_len[the_chr])
        image_train[num, start_image:end_image] = tmp[the_index1] 
        image_train[num+1, start_image:end_image] = tmp[the_index1] - tmp_avg[the_index1] 
        image_vali[num, start_image:end_image] = tmp[the_index2]
        image_vali[num+1, start_image:end_image] = tmp[the_index2] - tmp_avg[the_index2]
        num+=2
    for j in np.arange(len(list_feature_train)):
        the_id=list_feature_train[j]
        the_assay=re.sub('C[0-9][0-9]','',the_id)
        tmp_avg = np.zeros(chr_len[the_chr])
        tmp_avg[:] = dict_avg[the_assay].values(the_chr, 0, chr_len[the_chr])
        # train
        the_id=list_feature_train[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_feature_train[the_id].values(the_chr, 0, chr_len[the_chr])
        image_train[num, start_image:end_image] = tmp[the_index1] 
        image_train[num+1, start_image:end_image] = tmp[the_index1] - tmp_avg[the_index1] 
        # vali
        the_id=list_feature_vali[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_feature_vali[the_id].values(the_chr, 0, chr_len[the_chr])
        image_vali[num, start_image:end_image] = tmp[the_index2]
        image_vali[num+1, start_image:end_image] = tmp[the_index2] - tmp_avg[the_index2]
        num+=2
    # 2.3.0 orange index for train & vali
    the_index1=np.zeros((neighbor,freq[i]))
    the_index2=np.zeros((neighbor,freq[i]))
    for aaa in np.arange(neighbor):
        the_index1[aaa,:]=start1 - flank + aaa 
        the_index2[aaa,:]=start2 - flank + aaa
    the_index1=the_index1.T.flatten().astype('int')
    the_index2=the_index2.T.flatten().astype('int')
    # 2.3 orange & diff
    the_assay=re.sub('C[0-9][0-9]','',list_feature_common[0])
    tmp_avg = np.zeros(chr_len25[the_chr])
    tmp_avg[:] = dict_avg_orange[the_assay].values(the_chr, 0, chr_len25[the_chr])
    num=0
    for j in np.arange(len(list_feature_common)):
        the_id=list_feature_common[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:] = dict_orange_common[the_id].values(the_chr, 0, chr_len25[the_chr])
        orange_train[num, start_orange:end_orange] = tmp[the_index1] 
        orange_train[num+1, start_orange:end_orange] = tmp[the_index1] - tmp_avg[the_index1] 
        orange_vali[num, start_orange:end_orange] = tmp[the_index2]
        orange_vali[num+1, start_orange:end_orange] = tmp[the_index2] - tmp_avg[the_index2] 
        num+=2
    for j in np.arange(len(list_feature_train)):
        the_id=list_feature_train[j]
        the_assay=re.sub('C[0-9][0-9]','',the_id)
        tmp_avg = np.zeros(chr_len25[the_chr])
        tmp_avg[:] = dict_avg_orange[the_assay].values(the_chr, 0, chr_len25[the_chr])
        # train
        the_id=list_feature_train[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:] = dict_orange_train[the_id].values(the_chr, 0, chr_len25[the_chr])
        orange_train[num, start_orange:end_orange] = tmp[the_index1] 
        orange_train[num+1, start_orange:end_orange] = tmp[the_index1] - tmp_avg[the_index1] 
        # vali
        the_id=list_feature_vali[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:] = dict_orange_vali[the_id].values(the_chr, 0, chr_len25[the_chr])
        orange_vali[num, start_orange:end_orange] = tmp[the_index2] 
        orange_vali[num+1, start_orange:end_orange] = tmp[the_index2] - tmp_avg[the_index2] 
        num+=2
    start_label = end_label

# convert to 25bp features
image_train_final=np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_orange*neighbor,\
    num_sample), dtype='float32')
image_vali_final=np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_orange*neighbor, \
    num_sample), dtype='float32')

# dna - 25bp one hot
#image_train1=convert_feature_tensor(dna_train, reso=25*neighbor_dna)
# assay - mmm features
image_train2=image_train.reshape((num_feature, -1, 25)) # 3d - d0 * n * 25
image_train2=calculate_mmm(image_train2) # 3d - d0 * n * 3
image_train2=image_train2.reshape((num_feature, -1)) # 2d - d0 * (n*3)
image_train2=convert_feature_tensor(image_train2, reso=3*neighbor) # 2d - (d0*3) * n
# orange
image_train3=convert_feature_tensor(orange_train, reso=neighbor)
tmp1=num_dna*25*neighbor_dna; tmp2=tmp1+num_feature*3*neighbor
#image_train_final[:tmp1,:]=image_train1
image_train_final[tmp1:tmp2,:]=image_train2
image_train_final[tmp2:,:]=image_train3

# dna - 25bp one hot
#image_vali1=convert_feature_tensor(dna_vali, reso=25*neighbor_dna)
# assay - mmm features
image_vali2=image_vali.reshape((num_feature, -1, 25)) # 3d - d0 * n * 25
image_vali2=calculate_mmm(image_vali2) # 3d - d0 * n * 3 
image_vali2=image_vali2.reshape((num_feature, -1)) # 2d - d0 * (n*3)
image_vali2=convert_feature_tensor(image_vali2, reso=3*neighbor) # 2d - (d0*3) * n 
# orange
image_vali3=convert_feature_tensor(orange_vali, reso=neighbor)
tmp1=num_dna*25*neighbor_dna; tmp2=tmp1+num_feature*3*neighbor
#image_vali_final[:tmp1,:]=image_vali1
image_vali_final[tmp1:tmp2,:]=image_vali2
image_vali_final[tmp2:,:]=image_vali3

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
id_train_vali=list_label_train[0] + '_' + list_label_vali[0] 
data_train=lgb.Dataset(image_train_final.T, label=label_train, free_raw_data=False)
data_vali=lgb.Dataset(image_vali_final.T, label=label_vali, free_raw_data=False)
#gbm = lgb.train(params, data_train, num_boost_round=num_boost, valid_sets=data_vali, early_stopping_rounds=num_early_stop)
gbm = lgb.train(params, data_train, num_boost_round=num_boost, valid_sets=data_vali)
pickle.dump(gbm, open('./model/' + id_train_vali + '.model', 'wb'))
gbm.save_model('./model/' + id_train_vali + '.txt', num_iteration=None)

del tmp
del tmp_avg
del image_train
del image_vali
del label_train
del label_vali
del data_train
del data_vali
del gbm

## close bw
for the_id in dict_label_train.keys():
    dict_label_train[the_id].close()
for the_id in dict_label_vali.keys():
    dict_label_vali[the_id].close()
#for the_id in dict_dna.keys():
#    dict_dna[the_id].close()
for the_id in dict_feature_common.keys():
    dict_feature_common[the_id].close()
for the_id in dict_feature_train.keys():
    dict_feature_train[the_id].close()
for the_id in dict_feature_vali.keys():
    dict_feature_vali[the_id].close()
for the_id in dict_avg.keys():
    dict_avg[the_id].close()
for the_id in dict_orange_common.keys():
    dict_orange_common[the_id].close()
for the_id in dict_orange_train.keys():
    dict_orange_train[the_id].close()
for the_id in dict_orange_vali.keys():
    dict_orange_vali[the_id].close()
for the_id in dict_avg_orange.keys():
    dict_avg_orange[the_id].close()



