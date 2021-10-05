import pyBigWig
import os
import sys
import numpy as np
import glob

def anchor (ref, ori): # input 1d np array
    ref_new=ref.copy()
    ref_new.sort()
    ori_new=ori.copy()
    ori_new[np.argsort(ori)]=ref_new[:]
    return ori_new

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=np.array([248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895])
num_bp25=[9958257, 9687742, 7931823, 7608583, 7261531, 6832240, 6373839, 5805546, 5535789, 5351897, 5403465, 5331013, 4574574, 4281749, 4079648, 3613534, 3330298, 3214932, 2344705, 2577767, 1868400, 2032739, 6241636]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=num_bp25[i]

# number of cells used to calculate avg
assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']
tmp=[4,37,25,19,25,21,33,20]
dict_assay_count={}
for i in np.arange(len(assay_all)):
    dict_assay_count[assay_all[i]]=tmp[i]

# number of models
model_all=['C_D','C_E','C_F','C_G','C_H','C_I', \
    'CH_D','CH_E','CH_F','CH_G','CH_I', \
    'CDEH_G','CDEH_I','DEFGHI_C', \
    'DGH_C','DGH_F','DGH_I', \
    'DGI_C','DGI_E','DGI_F','DGI_H', \
    'F_C','F_D','F_E','F_G','F_H','F_I', \
    'DGHKLMN_F','DGHKLMN_I','DGHK_F','DGHK_I','DGIK_E','DGIK_F','DGIK_H']
tmp=[15,11,15,11,22,12, \
    15,11,14,11,12, \
    9,9,9, \
    11,18,17, \
    11,15,16,17, \
    15,20,16,18,20,17, \
    7,6,11,11,11,10,11]
dict_model_count={}
for i in np.arange(len(model_all)):
    dict_model_count[model_all[i]]=tmp[i]

path0='../data_challenge/baseline_avg_final/'
os.system('mkdir -p npy')

print(sys.argv)
model_all=sys.argv[1:]

for name_model in model_all:
    ids=glob.glob(name_model + '_lgbm_v1/' + 'pred25bp_C*chr1.npy')
    ids.sort()

    for the_id in ids:
        the_id = the_id.split('/')[-1].split('_')[1]
        print(the_id)
        the_assay=the_id[3:]
        the_cell=the_id[:3]
        bw=pyBigWig.open(path0 + 'gold_anchored_' + the_assay + '.bigwig')
        w1 = 1.0; w2 = 1.0; w3 = 1.0 # HERE weights for avg, lgbm, nn
        for the_chr in chr_all:
            print(the_chr)
            ## 1. stack
            # 1.1 avg
            avg = np.array(bw.values(the_chr, 0, chr_len25[the_chr]))
            # 1.2 lgbm 1+2
            pred1 = np.load(name_model + '_lgbm_v1/' + \
                'pred25bp_' + the_id + '_' + the_chr + '.npy')
            pred1 = pred1 + np.load(name_model + '_lgbm_v2/' + \
                'pred25bp_' + the_id + '_' + the_chr + '.npy')
            pred1 = pred1 / 2.0
            # 1.3 nn 
            pred2 = np.load(name_model + '_nn_v1/' + \
                'pred25bp_' + the_id + '_' + the_chr + '.npy')
            # 1.4 stack
            w11 = dict_assay_count[the_assay] * w1
            w22 = dict_model_count[name_model] * w2
            w33 = dict_model_count[name_model] * w3
            pred_stack = (avg * w11 + pred1 * w22 + pred2 * w33) / (w11 + w22 + w33)
            ###################
            ## 2. anchor bottom 90%
            cutoff=np.percentile(pred_stack,90)
            ind = pred_stack < cutoff
            pred_anchored = pred_stack.copy()
            pred_anchored[ind] = anchor(pred_stack[ind], pred1[ind]) 
            ###################
#            ## 3. max adj
#            tmp=~df_max.loc[the_cell,df_max.columns!=the_assay].isnull()
#            m1=np.nanmean(df_max.loc[:,df_max.columns!=the_assay].loc[:,tmp])
#            m2=np.nanmean(df_max.loc[the_cell,df_max.columns!=the_assay])
#            pred_final = pred_anchored * m2 / m1
            ###################
            pred_final = pred_anchored
            ## 3.1 save npy for sanity check
            np.save('./npy/pred25bp_' + the_id + '_' + the_chr, pred_final)
            ###################
        bw.close()

