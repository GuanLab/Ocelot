import os
import sys
import numpy as np
import pyBigWig
import time

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
#num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

id_all=np.loadtxt('../data_encode3/id_all.txt','str')

path0='./pred_ensemble_25bp/'
os.system('mkdir -p ' + path0)
path1='./pred_ensemble/'
os.system('mkdir -p ' + path1)

pathl='./'
pathu='./'

# YYY is the row feature and XXX is the target
dict_map={}
dict_map['H3K4me1']=['H3K4me3','H3K36me3','H3K27ac']
dict_map['H3K4me3']=['H3K4me1','H3K36me3','H3K27ac']
dict_map['H3K36me3']=['H3K4me3','H3K27me3','H3K27ac']
dict_map['H3K27me3']=['H3K4me3','H3K36me3','H3K27ac']
dict_map['H3K9me3']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

dict_map['H3K27ac']=['H3K4me3','H3K36me3','H3K27me3']
dict_map['H3K9ac']=['H3K4me3','H3K27me3','H3K27ac']

dict_map['H3K4me2']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H2AFZ']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H3K79me2']=['H3K4me3','H3K36me3','H3K27ac']
dict_map['H4K20me1']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

dict_map['H3F3A']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H3K9me2']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

the_template='template_unet_YYY_XXX_all'

reso = 25

for target_assay in tier0+tier1+tier2+tier3:
    bw_avg = pyBigWig.open('../data_encode3/bigwig_all/avg_' + target_assay + '.bigwig')
    dict_avg = {}
    for the_chr in chr_all:
        x = np.array(bw_avg.values(the_chr, 0, chr_len[the_chr]))
        x[np.isnan(x)]=0
        # convert into low resolution
        tmp=np.zeros(int(chr_len_bin[the_chr]*reso - len(x)))
        x=np.concatenate((x,tmp))
        x=np.mean(x.reshape((-1,reso)),axis=1)
        dict_avg[the_chr] = x
    bw_avg.close()
    for i in np.arange(1,80):
        the_cell='C%03d' % i
        the_id = the_cell + '_' + target_assay
        if the_id not in id_all:
            bw_output0 = pyBigWig.open(path0 + the_id + '.bigwig','w')
            bw_output0.addHeader(list(zip(chr_all , np.ceil(np.array(num_bp)/25).astype('int').tolist())), maxZooms=0)
            bw_output1 = pyBigWig.open(path1 + the_id + '.bigwig','w')
            bw_output1.addHeader(list(zip(chr_all , num_bp)), maxZooms=0)
            for feature_assay in dict_map[target_assay]:
                the_file = pathl + 'lgbm_' + feature_assay + '_' + target_assay + '_all/pred/pred_' + the_id + '_0.bigwig'
                if not os.path.isfile(the_file):
                    continue
                else:
                    bw_lgbm = pyBigWig.open(pathl + 'lgbm_' + feature_assay + '_' + target_assay + '_all/pred/pred_' + the_id + '_0.bigwig')
                    bw_unet = pyBigWig.open(pathu + 'unet_' + feature_assay + '_' + target_assay + '_all/epoch03/pred_' + the_id + '_seed0.bigwig')
                    for the_chr in chr_all:
                        print(the_id, the_chr)
                        pred = np.zeros(chr_len_bin[the_chr])
                        pred += dict_avg[the_chr]
                        pred += np.array(bw_lgbm.values(the_chr, 0, chr_len_bin[the_chr])) * 4.0
                        pred += bw_unet.values(the_chr, 0, chr_len_bin[the_chr])
                        pred = pred / 6.0
                        pred[pred<0] = 0
                        # convert into bigwig format
                        ## 0. short bigwig
                        x = pred
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
                        bw_output0.addEntries(chroms, starts, ends=ends, values=vals)
                        ## 1. full length bigwig
                        starts = np.arange(0,chr_len[the_chr],25)
                        ends = np.concatenate((starts[1:],[chr_len[the_chr]]))
                        vals = pred
                        chroms = np.array([the_chr] * len(vals))
                        bw_output1.addEntries(chroms, starts, ends=ends, values=vals)
                    bw_output0.close()
                    bw_output1.close()
                    bw_lgbm.close()
                    bw_unet.close()
                    break




