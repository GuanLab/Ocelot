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
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
set_session(tf.Session(config=config))
import random
from datetime import datetime
import unet
import pyBigWig

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size25=128
size=size25*25
batch_size=50
num_sample=10000

path0='../../data/bigwig_all/'
#path1='../../data/bigwig_cv2/'

def arcsinh(x):
    y = np.log(x + (1 + x**2)**0.5)
    return y

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
#num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

## sample index for chunks ###########
index_chr=np.array([])
freq=np.rint(np.array(num_bp)/sum(num_bp)*1000).astype('int')
for i in np.arange(len(chr_all)):
    index_chr = np.hstack((index_chr, np.array([chr_all[i]] * freq[i])))
np.random.shuffle(index_chr)
#############################################

## test cell
exclude_all=[]
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
#    parser.add_argument('-com', '--common', default='E002', nargs='+',type=str,
#        help='cell lines as common / assay specific features')
#    parser.add_argument('-spe', '--specific', default='E003', nargs='+',type=str,
#        help='cell lines as specific / cell-line specific features')
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

# model
num_class = 1
num_channel = 4 + len(assay_feature)*2 + len(cell_common)*2
name_model='weights_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-4,num_class=num_class,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

# load pyBigWig
dict_label={}
for the_cell in cell_tv:
    the_id = the_cell + '_' + assay_target
    dict_label[the_id]=pyBigWig.open(path0 + 'gt_' + the_id + '.bigwig')
dict_dna={}
for the_id in list_dna:
    dict_dna[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_feature_common={}
for the_cell in cell_common:
    the_id = the_cell + '_' + assay_target
    dict_feature_common[the_id]=pyBigWig.open(path0 + the_id + '.bigwig')
dict_feature_specific={}
for the_cell in cell_tv:
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

np.random.seed(datetime.now().microsecond)
np.random.shuffle(cell_train)
np.random.shuffle(cell_vali)

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
                list_cell = cell_train.copy()
            else:
                list_cell = cell_vali.copy()

            if i == len(index_chr):
                i=0
                np.random.shuffle(index_chr)

            the_chr=index_chr[i]
            start=np.random.randint(0,chr_len_bin[the_chr] - size25, 1)[0]
            end=start + size25
            start_image=start*25
            end_image=end*25
            ## randomly select a cell type
            cell_target = list_cell[np.random.randint(0, len(list_cell), 1)[0]] 
            ## 1. label
            the_id = cell_target + '_' + assay_target
            label=np.zeros((1, size25), dtype='float32')
            label[0,:]=np.array(dict_label[the_id].values(the_chr, start, end),dtype='float32')
            ## 2. image
            image=np.zeros((num_channel, size), dtype='float32')
            # 2.1 dna
            num=0
            for j in np.arange(len(list_dna)):
                the_id=list_dna[j]
                image[num,:] = dict_dna[the_id].values(the_chr,start_image,end_image)
                num+=1
            # 2.2 feature & diff
            the_avg=dict_avg[assay_target].values(the_chr,start_image,end_image)
            for j in np.arange(len(cell_common)):
                the_id=cell_common[j] + '_' + assay_target
                # feature
                image[num,:] = dict_feature_common[the_id].values(the_chr,start_image,end_image)
                # diff
                image[num+1,:]=image[num,:] - the_avg
                num+=2
            for j in np.arange(len(assay_feature)):
                the_assay = assay_feature[j]
                the_id=cell_target + '_' + the_assay
                # feature
                image[num,:] = dict_feature_specific[cell_target][the_id].values(the_chr,start_image,end_image)
                # diff
                the_assay=the_id.split('_')[1]
                image[num+1,:]=image[num,:] - dict_avg[the_assay].values(the_chr,start_image,end_image)
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
for the_id in dict_label.keys():
    dict_label[the_id].close()
for the_id in dict_dna.keys():
    dict_dna[the_id].close()
for the_id in dict_feature_common.keys():
    dict_feature_common[the_id].close()
for the_cell in dict_feature_specific.keys():
    for the_id in dict_feature_specific[the_cell].keys():
        dict_feature_specific[the_cell][the_id].close()
for the_id in dict_avg.keys():
    dict_avg[the_id].close()


