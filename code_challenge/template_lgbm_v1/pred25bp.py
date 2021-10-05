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
neighbor_dna=0 #2*flank_dna+1

path1='../../data_challenge/signal_anchored_final/'

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
#exclude_all=['C04','C17','C20','C24','C32','C34','C46','C48','C50']
exclude_all=[]
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

exclude_all.extend(args.crossvalidation)
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
        if (the_cell not in exclude_all) and os.path.isfile(path1 + the_id + '.bigwig'):
            list_feature_common.append(the_id)

# load pyBigWig
#dict_label_test={}
#for the_id in list_label_test:
#    dict_label_test[the_id]=pyBigWig.open(path1 + 'gold_' + the_id + '.bigwig')
#dict_dna={}
#for the_id in list_dna:
#    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
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

print('label_test',list_label_test)
print('feature_common',list_feature_common)
print('feature_test',list_feature_test)

id_train_vali=list_label_train[0] + '_' + list_label_vali[0] 
filename='./model/' + id_train_vali + '.model'
gbm=pickle.load(open(filename, 'rb'))

## prediction
list_chr=chr_all

num_dna=0
num_feature=len(list_feature_common)*2 + len(list_feature_test)*2
num_orange=len(list_feature_common)*2 + len(list_feature_test)*2

for the_chr in list_chr:
    print(the_chr)
    pred=np.zeros(chr_len25[the_chr])
    start=0
    while start < chr_len_cut[the_chr] - 25*flank:
        end = min(start + size_batch, chr_len_cut[the_chr])
        image=np.zeros((num_dna + num_feature, end-start), dtype='float32')
        # 2.1 dna
        num=0
#        for j in np.arange(len(list_dna)):
#            the_id=list_dna[j]
#            image[num,:] = dict_dna[the_id].values(the_chr,start,end)
#            num+=1
        # 2.2 feature & diff
        for j in np.arange(len(list_feature_common)):
            the_id=list_feature_common[j]
            # feature
            image[num,:] = dict_feature_common[the_id].values(the_chr,start,end)
            # diff
            the_assay=re.sub('C[0-9][0-9]','',the_id)
            image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,start,end)
            num+=2
        for j in np.arange(len(list_feature_test)):
            the_id=list_feature_test[j]
            # feature
            image[num,:] = dict_feature_test[the_id].values(the_chr,start,end)
            # diff
            the_assay=re.sub('C[0-9][0-9]','',the_id)
            image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,start,end)
            num+=2

        start_orange=int(start/25)
        end_orange=int(end/25)
        orange=np.zeros((num_orange, end_orange-start_orange), dtype='float32')
        # 2.3 orange & diff
        num=0
        for j in np.arange(len(list_feature_common)):
            the_id=list_feature_common[j]
            # feature
            orange[num,:] = dict_orange_common[the_id].values(the_chr,start_orange,end_orange)
            # diff
            the_assay=re.sub('C[0-9][0-9]','',the_id)
            orange[num+1,:]=orange[num,:] - dict_avg_orange[the_assay].values(the_chr,start_orange,end_orange)
            num+=2
        for j in np.arange(len(list_feature_test)):
            the_id=list_feature_test[j]
            # feature
            orange[num,:] = dict_orange_test[the_id].values(the_chr,start_orange,end_orange)
            # diff
            the_assay=re.sub('C[0-9][0-9]','',the_id)
            orange[num+1,:]=orange[num,:] - dict_avg_orange[the_assay].values(the_chr,start_orange,end_orange)
            num+=2

        # convert to 25bp features
        # dna - 25bp one hot
#        image1=convert_feature_tensor(image[:num_dna,:], reso=25) # this reso is different from train.py
        # assay - mmm features
        image2=image[num_dna:,:].reshape((num_feature, -1, 25)) # 3d - d0 * n * 25
        image2=calculate_mmm(image2) # 3d - d0 * n * 3
        image2=image2.reshape((num_feature, -1)) # 2d - d0 * (n*3)
        image2=convert_feature_tensor(image2, reso=3) # 2d - (d0*3) * n

        # convert to largespace; f1 neighbors -> f2 neighbors -> ...
        largespace=np.zeros((num_dna*25*neighbor_dna+num_feature*3*neighbor+num_orange*neighbor, \
            int((end-start)/25)-2*flank))
        # dna - 25bp one hot
#        for n1 in np.arange(num_dna):
#            for n2 in np.arange(neighbor_dna):
#                tmp1 = n1*25*neighbor_dna + n2*25
#                tmp2 = tmp1 + 25
#                tmp = flank-flank_dna # neighbor_dna short so we need to shift it
#                largespace[tmp1:tmp2,:]=image1[n1*25:(n1+1)*25, \
#                    (n2+tmp):(int((end-start)/25)-2*flank+n2+tmp)]
        # assay - mmm features
        for n1 in np.arange(num_feature):
            for n2 in np.arange(neighbor):
                tmp1 = n1*3*neighbor + n2*3 + num_dna*25*neighbor_dna
                tmp2 = tmp1 + 3
                largespace[tmp1:tmp2,:]=image2[n1*3:(n1+1)*3, n2:(int((end-start)/25)-2*flank+n2)]
        # orange
        for n1 in np.arange(num_orange):
            for n2 in np.arange(neighbor):
                tmp1 = n1*neighbor + n2*1 + num_feature*3*neighbor + num_dna*25*neighbor_dna
                tmp2 = tmp1 + 1
                largespace[tmp1:tmp2,:]=orange[n1:(n1+1), n2:(int((end-start)/25)-2*flank+n2)]

        start_pred=int(start/25) + flank # skip both ends of a chromosome for now            
        end_pred=int(end/25) - flank
        pred[start_pred:end_pred]=gbm.predict(largespace.T)
        del image
        del orange
#        del image1
        del image2
        del largespace

        start = end - 25*flank

    np.save('pred25bp_' + list_label_test[0] + '_' + the_chr, pred)

## close bw
#for the_id in dict_label_test.keys():
#    dict_label_test[the_id].close()
#for the_id in dict_dna.keys():
#    dict_dna[the_id].close()
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




