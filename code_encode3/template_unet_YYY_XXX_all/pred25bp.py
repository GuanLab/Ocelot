#!/usr/bin/env python
import argparse
import os
import sys
import logging
import numpy as np
import re
import time
import scipy.io
import glob
import unet
import tensorflow as tf
import keras
from keras import backend as K
import scipy.stats
import pyBigWig
print('tf-' + tf.__version__, 'keras-' + keras.__version__)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def score(x1,x2):
    mse=((x1-x2) ** 2.).mean()
    pearson=np.corrcoef((x1,x2))[0,1]
    y1=scipy.stats.rankdata(x1)
    y2=scipy.stats.rankdata(x2)
    spearman=np.corrcoef((y1,y2))[0,1]
    return mse,pearson,spearman

###### PARAMETER ###############

size25=128
size=size25*25
write_pred=True # whether generate .vec prediction file
size_edge=int(10) # chunk edges to be excluded
batch=200

path0='../../data_encode3/bigwig_all/'
#path1='../../data/bigwig_cv2/'

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
# grch38
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
#num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len_cut={}
for i in np.arange(len(chr_all)):
    chr_len_cut[chr_all[i]]=int(np.floor(num_bp[i]/25.0)*25) # HERE I cut tails

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

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
    parser.add_argument('-m', '--model', default='epoch01/weights_1.h5', type=str,
        help='model')
    args = parser.parse_args()
    return args

args=get_args()
print(args)

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

# model
num_class = 1
num_channel = 4 + len(assay_feature)*2 + len(cell_common)*2
model1 = unet.get_unet(num_class=num_class,num_channel=num_channel,size=size)
model1.load_weights(args.model)
#model1.summary()

# load pyBigWig
#dict_label={}
#for the_cell in cell_tv:
#    the_id = the_cell + '_' + assay_target
#    dict_label[the_id]=pyBigWig.open(path0 + 'gt_' + the_id + '.bigwig')
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_feature_common={}
for the_cell in cell_common:
    the_id = the_cell + '_' + assay_target
    dict_feature_common[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_feature_specific={}
for the_cell in [cell_test]:
    dict_feature_specific[the_cell]={}
    for the_assay in assay_feature:
        the_id = the_cell + '_' + the_assay
        dict_feature_specific[the_cell][the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_avg={}
dict_avg[assay_target]=pyBigWig.open(path0 + 'avg_' + assay_target + '.bigwig')
for the_assay in assay_feature:
    dict_avg[the_assay]=pyBigWig.open(path0 + 'avg_' + the_assay + '.bigwig')

print('assay_target', assay_target)
print('assay_feature', len(assay_feature), assay_feature)
print('cell_common', len(cell_common), cell_common)
print('cell_tv', len(cell_tv), cell_tv)
print('cell_train', len(cell_train), cell_train)
print('cell_vali', len(cell_vali), cell_vali)
print('cell_test', len(cell_test), cell_test)

list_chr=chr_all
#list_chr=['chr21'] # TODO
    
path_pred='/'.join(args.model.split('/')[:-1]) + '/'
id_label_test = cell_test + '_' + assay_target

bw_output = pyBigWig.open(path_pred + 'pred_' + id_label_test + '_seed' + str(seed_partition) + '.bigwig','w')
bw_output.addHeader(list(zip(chr_all , np.ceil(np.array(num_bp)/25).astype('int').tolist())), maxZooms=0)

#file_score=open(path_pred + 'score_' + id_label_test + '_' + cell_train + '_' + cell_vali + '.txt','w')
#mse_all=[]
#pearson_all=[]
#spearman_all=[]

for the_chr in list_chr:
    print(the_chr)
    output_all=np.zeros((len(list_label_test), int(chr_len_cut[the_chr]/25.0)))
    count_all=np.zeros((len(list_label_test), int(chr_len_cut[the_chr]/25.0)))

    ## 2. batch prediction ##########################
    phase=0.5
    tmp=int(size*batch)
    index=np.arange(0,int(np.floor(chr_len_cut[the_chr]/tmp))*tmp, tmp)
    index=np.concatenate((index,np.arange(int(size*phase),int(np.floor(chr_len_cut[the_chr]/tmp))*tmp, tmp)))
    index=np.concatenate((index,np.array([chr_len_cut[the_chr]-tmp])))
    index=np.concatenate((index,np.array([chr_len_cut[the_chr]-tmp-int(size*phase)])))
    index.sort()
    print(index)

    for i in index: # all start positions
        start = i
        end = i + size*batch
        ## 2. image
        image=np.zeros((num_channel, size*batch), dtype='float32')
        # 2.1 dna
        num=0
        for j in np.arange(len(list_dna)):
            the_id=list_dna[j]
            image[num,:] = dict_dna[the_id].values(the_chr,start,end)
            num+=1
        # 2.2 feature & diff
        the_avg=dict_avg[assay_target].values(the_chr,start,end)
        for j in np.arange(len(cell_common)):
            the_id=cell_common[j] + '_' + assay_target
            # feature
            image[num,:] = dict_feature_common[the_id].values(the_chr,start,end)
            # diff
            image[num+1,:]=image[num,:] - the_avg
            num+=2
        for j in np.arange(len(assay_feature)):
            the_assay = assay_feature[j]
            the_id=cell_test + '_' + the_assay
            # feature
            image[num,:] = dict_feature_specific[cell_test][the_id].values(the_chr,start,end)
            # diff
            the_assay=the_id.split('_')[1]
            image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,start,end)
            num+=2

        ## make predictions ################
        input_pred=np.reshape(image.T,(batch,size,num_channel))
        
        # size25 start from output 
        output1 = model1.predict(input_pred)
        output1=np.reshape(output1,(size25*batch, len(list_label_test))).T

        output_new=output1

        start25=int(start/25)
        end25=int(end/25)    
        i_batch=0
        while (i_batch<batch):
            i_start = start25 + i_batch*size25
            i_end = i_start + size25
            if (i_start==0): 
                start_new = i_start
                end_new = i_end - size_edge
                start_tmp = 0 + i_batch*size25
                end_tmp = size25 - size_edge + i_batch*size25
            elif (i_end==int(chr_len_cut[the_chr]/25)):
                start_new = i_start + size_edge
                end_new = i_end
                start_tmp = size_edge + i_batch*size25
                end_tmp = size25 + i_batch*size25
            else:
                start_new = i_start + size_edge
                end_new = i_end - size_edge
                start_tmp = size_edge + i_batch*size25
                end_tmp = size25 - size_edge + i_batch*size25
            output_all[:,start_new:end_new]=output_all[:,start_new:end_new]+output_new[:,start_tmp:end_tmp]
            count_all[:,start_new:end_new]=count_all[:,start_new:end_new]+1
            i_batch += 1

    del output1
    del output_new
    del image
    del input_pred        

    ######################################################################
    
    # 2. save bigwig
    output_all=np.divide(output_all,count_all)
    # add tail
    output_all=np.hstack((output_all, np.zeros((len(list_label_test),1))))

    for j in np.arange(len(list_label_test)):
        id_label_test=list_label_test[j]
        x=output_all[j,:]
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

# close bw
#for the_id in dict_label.keys():
#    dict_label[the_id].close()
for the_id in dict_dna.keys():
    dict_dna[the_id].close()
for the_id in dict_feature_common.keys():
    dict_feature_common[the_id].close()
for the_cell in dict_feature_specific.keys():
    for the_id in dict_feature_specific[the_cell].keys():
        dict_feature_specific[the_cell][the_id].close()
for the_id in dict_avg.keys():
    dict_avg[the_id].close()

#        if write_pred:
#            np.save(path_pred + 'pred25bp_' + id_label_test + '_' + cell_train + '_' + cell_vali + '_' + the_chr, the_output)
#
#        # gold
#        bw_label=pyBigWig.open(path0 + 'gt_' + id_label_test + '.bigwig')
#        label=np.array(bw_label.values(the_chr, 0, int(chr_len_cut[the_chr]/25.0)+1))
#        label=np.nan_to_num(label)
#   
#        the_mse, the_pearson, the_spearman = score(label, the_output)
#        # for final score 
#        mse_all.append(the_mse)
#        pearson_all.append(the_pearson)
#        spearman_all.append(the_spearman)
#   
#        file_score.write('%s\t' % id_label_test) 
#        file_score.write('%s\t' % the_chr)
#        file_score.write('%.4f\t' % the_mse)
#        file_score.write('%.4f\t' % the_pearson)
#        file_score.write('%.4f\n' % the_spearman)
#        file_score.flush()
#    
#        print(the_mse)    
#        print(the_pearson)   
#        print(the_spearman) 
#
#file_score.close()    


#    # final performance
#    mse_all=np.array(mse_all)
#    pearson_all=np.array(pearson_all)
#    spearman_all=np.array(spearman_all)
#
#    chr_length=np.array([248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895])
#
#    chr_length=chr_length[:len(mse_all)]
#    weight=chr_length/float(np.sum(chr_length))
#
#    mse.write('%s\t' % 'avg')
#    mse.write('%.4f\t' % np.sum(weight * mse_all))
#    mse.write('%.4f\t' % np.sum(weight * pearson_all))
#    mse.write('%.4f\n' % np.sum(weight * spearman_all))
#    mse.close()


