import argparse
import pyBigWig
import os
import sys
import numpy as np
import scipy.stats
import re
import lightgbm as lgb
import pickle
import shap

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

path1='../../data/signal_anchored_par3/'
path2='./shap_hg/'
os.system('mkdir -p ' + path2)

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len_cut={}
for i in np.arange(len(chr_all)):
    chr_len_cut[chr_all[i]]=int(np.floor(num_bp[i]/25.0)*25) # HERE I cut tails

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

## test cell
test_all=['C04','C17','C20','C24','C32','C34','C46','C48','C50']
list_dna=['A','C','G','T']

# argv
def get_args():
    parser = argparse.ArgumentParser(description="lightgbm train")
    parser.add_argument('-f', '--feature', default='M02', nargs='+', type=str,
        help='feature assay')
    parser.add_argument('-t', '--target', default='M18', nargs='+',type=str,
        help='target assay')
    parser.add_argument('-cv', '--crossvalidation', nargs='+', type=str,
        help='crossvalidation cell lines')
    args = parser.parse_args()
    return args

args=get_args()

test_all.extend(args.crossvalidation)
cell_train=args.crossvalidation[0]
cell_vali=args.crossvalidation[1]
cell_test=args.crossvalidation[2]

## target
list_label_train=[]
list_label_vali=[]
list_label_test=[]
for the_assay in args.target:
    list_label_train.append(cell_train + the_assay)
    list_label_vali.append(cell_vali + the_assay)
    list_label_test.append(cell_test + the_assay)

## row feature
list_feature_train=[]
list_feature_vali=[]
list_feature_test=[]
for the_assay in args.feature:
    list_feature_train.append(cell_train + the_assay)
    list_feature_vali.append(cell_vali + the_assay)
    list_feature_test.append(cell_test + the_assay)

## col feature
list_feature_common=[]
for the_assay in args.target:
    for i in np.arange(1,52):
        the_cell = 'C' + '%02d' % i
        the_id = the_cell + the_assay
        if (the_cell not in test_all) and os.path.isfile(path1 + the_id + '.bigwig'):
            list_feature_common.append(the_id)

# load pyBigWig
dict_label_test={}
for the_id in list_label_test:
    dict_label_test[the_id]=pyBigWig.open(path1 + 'gold_' + the_id + '.bigwig')
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_common={}
dict_orange_common={}
for the_id in list_feature_common:
    dict_feature_common[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_orange_common[the_id]=pyBigWig.open(path1 + 'orange_' + the_id + '.bigwig')
dict_feature_test={}
dict_orange_test={}
for the_id in list_feature_test:
    dict_feature_test[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
    dict_orange_test[the_id]=pyBigWig.open(path1 + 'orange_' + the_id + '.bigwig')
dict_avg={}
dict_avg_orange={}
for the_assay in args.feature:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
    dict_avg_orange[the_assay]=pyBigWig.open(path1 + 'avg_orange_' + the_assay + '.bigwig')
for the_assay in args.target:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
    dict_avg_orange[the_assay]=pyBigWig.open(path1 + 'avg_orange_' + the_assay + '.bigwig')

print('label_test',dict_label_test.keys())
print('feature_common',dict_feature_common.keys())
print('feature_test',dict_feature_test.keys())

k=0
print('epoch' + '%02d' % k)
id_train_vali=list_label_train[0] + '_' + list_label_vali[0] + '_' + '%02d' % k
filename='./model/' + id_train_vali + '.model'
gbm=pickle.load(open(filename, 'rb'))

## prediction
#list_chr=['chr11']
list_chr=chr_all
#list_chr=['chr1','chr8','chr21']
ratio_sample=0.0001 # top x% peaks

dict_chr_sample={}
num_sample=0
for the_chr in list_chr:
    dict_chr_sample[the_chr] = int(chr_len25[the_chr] * ratio_sample)
    num_sample += dict_chr_sample[the_chr]

# number of features for each type
num_dna=4
num_feature=len(list_feature_common)*2 + len(list_feature_test)*2
num_orange=len(list_feature_common)*2 + len(list_feature_test)*2
# image_raw is the original 25bp -> 1bp; need to convert it later
label_test=np.zeros(num_sample, dtype='float32')
dna_test=np.zeros((num_dna, num_sample*25*neighbor_dna))
image_test=np.zeros((num_feature, num_sample*25*neighbor), dtype='float32')
orange_test=np.zeros((num_orange, num_sample*neighbor), dtype='float32')

np.random.seed(449)

# index for final image; for each chr, step top x%
start_label=0
for the_chr in list_chr:
    print(the_chr)
    end_label = start_label + dict_chr_sample[the_chr]
    start_dna = start_label*25*neighbor_dna
    end_dna = end_label*25*neighbor_dna
    start_image = start_label*25*neighbor
    end_image = end_label*25*neighbor
    start_orange = start_label*neighbor
    end_orange = end_label*neighbor
    start1=np.random.randint(0 + flank, chr_len25[the_chr] - flank - 1, dict_chr_sample[the_chr])
    ## 1. label ##########
    for j in np.arange(len(list_label_test)): # single-task
        the_id=list_label_test[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:]=dict_label_test[the_id].values(the_chr, 0, chr_len25[the_chr])
        label_test[start_label:end_label]=tmp[start1]
    ## 2. image ##########
    # 2.1.0 dna index 
    the_index1=np.zeros((25*neighbor_dna,len(start1)))
    for aaa in np.arange(25*neighbor_dna):
        the_index1[aaa,:]=start1*25 - flank_dna*25 + aaa
    the_index1=the_index1.T.flatten().astype('int')
    # 2.1 dna
    num=0
    for j in np.arange(len(list_dna)):
        the_id=list_dna[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_dna[the_id].values(the_chr, 0, chr_len[the_chr])
        dna_test[num, start_dna:end_dna] = tmp[the_index1]
        num+=1
    # 2.2.0 image index 
    the_index1=np.zeros((25*neighbor,len(start1)))
    for aaa in np.arange(25*neighbor):
        the_index1[aaa,:]=start1*25 - flank*25 + aaa
    the_index1=the_index1.T.flatten().astype('int')
    # 2.2 feature & diff
    the_assay=re.sub('C[0-9][0-9]','',list_feature_common[0])
    tmp_avg = np.zeros(chr_len[the_chr])
    tmp_avg[:] = dict_avg[the_assay].values(the_chr, 0, chr_len[the_chr])
    num=0
    for j in np.arange(len(list_feature_common)):
        the_id=list_feature_common[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_feature_common[the_id].values(the_chr, 0, chr_len[the_chr])
        image_test[num, start_image:end_image] = tmp[the_index1]
        image_test[num+1, start_image:end_image] = tmp[the_index1] - tmp_avg[the_index1]
        num+=2
    for j in np.arange(len(list_feature_test)):
        the_id=list_feature_test[j]
        the_assay=re.sub('C[0-9][0-9]','',the_id)
        tmp_avg = np.zeros(chr_len[the_chr])
        tmp_avg[:] = dict_avg[the_assay].values(the_chr, 0, chr_len[the_chr])
        # train
        the_id=list_feature_test[j]
        tmp = np.zeros(chr_len[the_chr])
        tmp[:] = dict_feature_test[the_id].values(the_chr, 0, chr_len[the_chr])
        image_test[num, start_image:end_image] = tmp[the_index1]
        image_test[num+1, start_image:end_image] = tmp[the_index1] - tmp_avg[the_index1]
        num+=2
    # 2.3.0 orange index 
    the_index1=np.zeros((neighbor,len(start1)))
    for aaa in np.arange(neighbor):
        the_index1[aaa,:]=start1 - flank + aaa
    the_index1=the_index1.T.flatten().astype('int')
    # 2.3 orange & diff
    the_assay=re.sub('C[0-9][0-9]','',list_feature_common[0])
    tmp_avg = np.zeros(chr_len25[the_chr])
    tmp_avg[:] = dict_avg_orange[the_assay].values(the_chr, 0, chr_len25[the_chr])
    num=0
    for j in np.arange(len(list_feature_common)):
        the_id=list_feature_common[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:] = dict_orange_common[the_id].values(the_chr, 0, chr_len25[the_chr])
        orange_test[num, start_orange:end_orange] = tmp[the_index1]
        orange_test[num+1, start_orange:end_orange] = tmp[the_index1] - tmp_avg[the_index1]
        num+=2
    for j in np.arange(len(list_feature_test)):
        the_id=list_feature_test[j]
        the_assay=re.sub('C[0-9][0-9]','',the_id)
        tmp_avg = np.zeros(chr_len25[the_chr])
        tmp_avg[:] = dict_avg_orange[the_assay].values(the_chr, 0, chr_len25[the_chr])
        # train
        the_id=list_feature_test[j]
        tmp = np.zeros(chr_len25[the_chr])
        tmp[:] = dict_orange_test[the_id].values(the_chr, 0, chr_len25[the_chr])
        orange_test[num, start_orange:end_orange] = tmp[the_index1]
        orange_test[num+1, start_orange:end_orange] = tmp[the_index1] - tmp_avg[the_index1]
        num+=2
    start_label = end_label

# convert to 25bp features
image_test_final=np.zeros((num_dna*25*neighbor_dna + num_feature*3*neighbor + num_orange*neighbor,\
    num_sample), dtype='float32')
# dna - 25bp one hot
image_test1=convert_feature_tensor(dna_test, reso=25*neighbor_dna)
# assay - mmm features
image_test2=image_test.reshape((num_feature, -1, 25)) # 3d - d0 * n * 25
image_test2=calculate_mmm(image_test2) # 3d - d0 * n * 3
image_test2=image_test2.reshape((num_feature, -1)) # 2d - d0 * (n*3)
image_test2=convert_feature_tensor(image_test2, reso=3*neighbor) # 2d - (d0*3) * n
# orange
image_test3=convert_feature_tensor(orange_test, reso=neighbor)
tmp1=num_dna*25*neighbor_dna; tmp2=tmp1+num_feature*3*neighbor
# assemble three types of features
image_test_final[:tmp1,:]=image_test1
image_test_final[tmp1:tmp2,:]=image_test2
image_test_final[tmp2:,:]=image_test3

## shap analysis
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(image_test_final.T)
#np.mean(shap_values,axis=0)
pred = gbm.predict(image_test_final.T)

# matrix: sample * feature
np.save(path2 + 'shap_' + list_label_test[0] + '_feature', image_test_final.T)
np.save(path2 + 'shap_' + list_label_test[0] + '_shap', shap_values)
#np.save(path2 + 'shap_' + list_label_test[0] + '_avg', np.mean(shap_values,axis=0))
np.save(path2 + 'shap_' + list_label_test[0] + '_label', label_test)
np.save(path2 + 'shap_' + list_label_test[0] + '_index', start1)
np.save(path2 + 'shap_' + list_label_test[0] + '_pred', pred)


## close bw
for the_id in dict_label_test.keys():
    dict_label_test[the_id].close()
for the_id in dict_dna.keys():
    dict_dna[the_id].close()
for the_id in dict_feature_common.keys():
    dict_feature_common[the_id].close()
for the_id in dict_feature_test.keys():
    dict_feature_test[the_id].close()
for the_id in dict_avg.keys():
    dict_avg[the_id].close()
for the_id in dict_orange_common.keys():
    dict_orange_common[the_id].close()
for the_id in dict_orange_test.keys():
    dict_orange_test[the_id].close()
for the_id in dict_avg_orange.keys():
    dict_avg_orange[the_id].close()




