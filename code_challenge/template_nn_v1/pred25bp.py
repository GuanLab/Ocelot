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
config.gpu_options.per_process_gpu_memory_fraction = 0.21 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

###### PARAMETER ###############

size=128*25
size25=int(size/25)
write_pred=True # whether generate .vec prediction file
size_edge=int(10) # chunk edges to be excluded
ss=10 # for sorenson dice
batch=200
#length_chunk=2**20 # 1m

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
    parser = argparse.ArgumentParser(description="globe prediction")
    parser.add_argument('-f', '--feature', default='M02', nargs='+', type=str,
        help='feature assay')
    parser.add_argument('-t', '--target', default='M18', nargs='+',type=str,
        help='target assay')
    parser.add_argument('-cv', '--crossvalidation', nargs='+', type=str,
        help='crossvalidation cell lines')
    parser.add_argument('-m', '--model', default='epoch01/weights_1.h5', type=str,
        help='model')
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

# model
num_class = len(args.target)
num_channel = len(list_feature_test)*2 + len(list_feature_common)*2
model1 = unet.get_unet(num_class=num_class,num_channel=num_channel,size=size)
model1.load_weights(args.model)
#model1.summary()

# load pyBigWig
#dict_label_train={}
#for the_id in list_label_train:
#    dict_label_train[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
#dict_label_vali={}
#for the_id in list_label_vali:
#    dict_label_vali[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
#dict_label_test={}
#for the_id in list_label_test:
#    dict_label_test[the_id]=pyBigWig.open(path1 + 'gold_' + the_id + '.bigwig')
#dict_dna={}
#for the_id in list_dna:
#    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_common={}
for the_id in list_feature_common:
    dict_feature_common[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
#dict_feature_train={}
#for the_id in list_feature_train:
#    dict_feature_train[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
#dict_feature_vali={}
#for the_id in list_feature_vali:
#    dict_feature_vali[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_test={}
for the_id in list_feature_test:
    dict_feature_test[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_avg={}
for the_assay in args.feature:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
for the_assay in args.target:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')

print('label_test',list_label_test)
print('feature_common',list_feature_common)
print('feature_test',list_feature_test)

if __name__ == '__main__':

#    path3='./sample_' + id_label_test + '/'
#    os.system('mkdir -p ' + path3)

#    try:
#        sys.argv[3]
#    except IndexError:
#        list_chr=chr_all
#    else:
#        list_chr=sys.argv[3:]        
    list_chr=chr_all
        
#    mse=open('mse_' + cell_test + '.txt','w')
#    mse_all=[]
#    pearson_all=[]
#    spearman_all=[]

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

        for i in index: # all start positions
            start = i
            end = i + size*batch
            ## 2. image
            image=np.zeros((len(dict_feature_common)*2 + len(dict_feature_test)*2 , size*batch), dtype='float32')
            # 2.1 dna
            num=0
#            for j in np.arange(len(list_dna)):
#                the_id=list_dna[j]
#                image[num,:] = dict_dna[the_id].values(the_chr,start,end)
#                num+=1
            # 2.2 feature & diff
            the_assay=re.sub('C[0-9][0-9]','',list_feature_common[0])
            the_avg=dict_avg[the_assay].values(the_chr,start,end)
            for j in np.arange(len(list_feature_common)):
                the_id=list_feature_common[j]
                # feature
                image[num,:] = dict_feature_common[the_id].values(the_chr,start,end)
                # diff
                image[num+1,:]=image[num,:] - the_avg
                num+=2
            for j in np.arange(len(list_feature_test)):
                the_id=list_feature_test[j]
                # feature
                image[num,:] = dict_feature_test[the_id].values(the_chr,start,end)
                # diff
                the_assay=re.sub('C[0-9][0-9]','',the_id)
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
        
        # 2. scoring
        output_all=np.divide(output_all,count_all)
        # add tail
        output_all=np.hstack((output_all, np.zeros((len(list_label_test),1))))

        for j in np.arange(len(list_label_test)):
            id_label_test=list_label_test[j]
            the_output=output_all[j,:]

            if write_pred:
                np.save('pred25bp_' + id_label_test + '_' + the_chr, the_output)
    
#            # gold
#            bw_label=pyBigWig.open(path1 + 'gold_' + id_label_test + '.bigwig')
#            label=np.array(bw_label.values(the_chr, 0, int(chr_len_cut[the_chr]/25.0)+1))
#       
#            the_mse = np.mean((label - the_output)**2)
#            the_pearson = np.corrcoef(label, the_output)[0,1]
#            tmp1 = scipy.stats.rankdata(label)
#            tmp2 = scipy.stats.rankdata(the_output)
#            the_spearman = np.corrcoef(tmp1, tmp2)[0,1]
#        
#            # for final score 
#            mse_all.append(the_mse)
#            pearson_all.append(the_pearson)
#            spearman_all.append(the_spearman)
#       
#            mse.write('%s\t' % id_label_test) 
#            mse.write('%s\t' % the_chr)
#            mse.write('%.4f\t' % the_mse)
#            mse.write('%.4f\t' % the_pearson)
#            mse.write('%.4f\n' % the_spearman)
#            mse.flush()
#        
#            print(the_mse)    
#            print(the_pearson)   
#            print(the_spearman) 
#
#    mse.close()    
    # close bw
#    for the_id in dict_label_test.keys():
#        dict_label_test[the_id].close()
#    for the_id in dict_dna.keys():
#        dict_dna[the_id].close()
    for the_id in dict_feature_common.keys():
        dict_feature_common[the_id].close()
    for the_id in dict_feature_test.keys():
        dict_feature_test[the_id].close()
    for the_id in dict_avg.keys():
        dict_avg[the_id].close()


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

