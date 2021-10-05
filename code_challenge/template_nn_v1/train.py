import argparse
import os
import sys
import numpy as np
import re
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.45 
#set_session(tf.Session(config=config))
import random
from datetime import datetime
import unet
import pyBigWig

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size=128*25 # 3200
size25=int(size/25.0)
batch_size=50
num_sample=10000
ss = 10

path1='../../data_challenge/signal_anchored_final/'

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

## sample index for chunks ###########
index_chr=np.array([])
freq=np.rint(np.array(num_bp)/sum(num_bp)*1000).astype('int')
for i in np.arange(len(chr_all)):
    index_chr = np.hstack((index_chr, np.array([chr_all[i]] * freq[i])))
np.random.shuffle(index_chr)
#############################################

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

# model
num_class = len(args.target)
num_channel = len(list_feature_train)*2 + len(list_feature_common)*2
name_model='weights_1.h5'
model = unet.get_unet(the_lr=1e-3,num_class=num_class,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

# load pyBigWig
dict_label_train={}
for the_id in list_label_train:
    dict_label_train[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
dict_label_vali={}
for the_id in list_label_vali:
    dict_label_vali[the_id]=pyBigWig.open(path1 + 'gold_anchored_' + the_id + '.bigwig')
#dict_label_test={}
#for the_id in list_label_test:
#    dict_label_test[the_id]=pyBigWig.open(path1 + 'gold_' + the_id + '.bigwig')
#dict_dna={}
#for the_id in list_dna:
#    dict_dna[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_common={}
for the_id in list_feature_common:
    dict_feature_common[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_train={}
for the_id in list_feature_train:
    dict_feature_train[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_feature_vali={}
for the_id in list_feature_vali:
    dict_feature_vali[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
#dict_feature_test={}
#for the_id in list_feature_test:
#    dict_feature_test[the_id]=pyBigWig.open(path1 + the_id + '.bigwig')
dict_avg={}
for the_assay in args.feature:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')
for the_assay in args.target:
    dict_avg[the_assay]=pyBigWig.open(path1 + 'avg_' + the_assay + '.bigwig')

print('label_train', list_label_train)
print('label_vali', list_label_vali)
print('feature_common', list_feature_common)
print('feature_train', list_feature_train)
print('feature_vali', list_feature_vali)

##### augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
#if_mag=True
if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=False
####################################

def generate_data(batch_size, if_train):

    global index_chr

    i=0
    while True:
        b = 0
        image_batch = []
        label_batch = []

        while b < batch_size:
            if (if_train==1):
                dict_label=dict_label_train
                dict_feature=dict_feature_train
                list_label=list_label_train
                list_feature=list_feature_train
            else:
                dict_label=dict_label_vali
                dict_feature=dict_feature_vali
                list_label=list_label_vali
                list_feature=list_feature_vali

            if i == len(index_chr):
                i=0
                np.random.shuffle(index_chr)

            the_chr=index_chr[i]
            start=np.random.randint(0,int(np.ceil(chr_len[the_chr]/25.0)-size/25),1)
            end=start + int(size/25)
            ## 1. label
            label=np.zeros((len(list_label) , size25), dtype='float32')
            num=0
            #for the_id in dict_label.keys():
            for j in np.arange(len(list_label)):
                the_id=list_label[j]
                label[num,:]=np.array(dict_label[the_id].values(the_chr, int(start), int(end))).reshape((1,-1))
                num+=1
            ## 2. image
            image=np.zeros((len(list_feature_common)*2 + len(list_feature)*2, size), dtype='float32')
            # 2.1 dna
            num=0
#            #for the_id in dict_dna.keys():
#            for j in np.arange(len(list_dna)):
#                the_id=list_dna[j]
#                image[num,:] = dict_dna[the_id].values(the_chr,int(start*25),int(end*25))
#                num+=1
            # 2.2 feature & diff
            #for the_id in dict_feature_common.keys():
            for j in np.arange(len(list_feature_common)):
                the_id=list_feature_common[j]
                # feature
                image[num,:] = dict_feature_common[the_id].values(the_chr,int(start*25),int(end*25))
                # diff
                the_assay=re.sub('C[0-9][0-9]','',the_id)
                image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,int(start*25),int(end*25))
                num+=2
            #for the_id in dict_feature.keys():
            for j in np.arange(len(list_feature)):
                the_id=list_feature[j]
                # feature
                image[num,:] = dict_feature[the_id].values(the_chr,int(start*25),int(end*25))
                # diff
                the_assay=re.sub('C[0-9][0-9]','',the_id)
                image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,int(start*25),int(end*25))
                num+=2
                
            image_batch.append(image.T)
            label_batch.append(label.T)

            i+=1
            b+=1

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
        yield image_batch, label_batch

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(batch_size,True),
    steps_per_epoch=int(num_sample // (batch_size)), nb_epoch=1,
    validation_data=generate_data(batch_size,False),
    validation_steps=int(num_sample // (batch_size)),callbacks=callbacks,verbose=1)

# close bw
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


