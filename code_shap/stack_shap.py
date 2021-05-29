import os
import sys
import numpy as np
import glob
import re
import scipy.stats

def gwcorr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def gwspear(y_true, y_pred):
    tmp1 = scipy.stats.rankdata(y_true)
    tmp2 = scipy.stats.rankdata(y_pred)
    return np.corrcoef(tmp1, tmp2)[0,1]

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))


print(sys.argv)
name_model=sys.argv[1]
path1 = './lgbm_' + name_model + '_v1/'
os.system('mkdir -p ' + path1 + 'shap_hg/')

dirs=glob.glob('./lgbm_' + name_model + '_v1_*')
dirs.sort()
print(dirs)

ids=glob.glob('./lgbm_' + name_model + '_v1_01/shap_hg/shap_*_shap.npy')
ids.sort()

## id level
for the_id in ids:
    the_id = the_id.split('/')[-1].split('_')[1]
    print(the_id)
    shap_val = np.load(dirs[0] + '/shap_hg/shap_' + the_id + '_shap.npy')
    feature = np.load(dirs[0] + '/shap_hg/shap_' + the_id + '_feature.npy')
    pred = np.load(dirs[0] + '/shap_hg/shap_' + the_id + '_pred.npy')
    for the_dir in dirs[1:]:
        shap_val += np.load(the_dir + '/shap_hg/shap_' + the_id + '_shap.npy')
        feature += np.load(the_dir + '/shap_hg/shap_' + the_id + '_feature.npy')
        pred += np.load(the_dir + '/shap_hg/shap_' + the_id + '_pred.npy')
    shap_val = shap_val / float(len(dirs))
    feature = feature / float(len(dirs))
    pred = pred / float(len(dirs))
    np.save(path1 + 'shap_hg/shap_' + the_id + '_shap', shap_val)
    np.save(path1 + 'shap_hg/shap_' + the_id + '_feature', feature)
    np.save(path1 + 'shap_hg/shap_' + the_id + '_pred', pred)
    label = np.load(dirs[0] + '/shap_hg/shap_' + the_id + '_label.npy')
    np.save(path1 + 'shap_hg/shap_' + the_id + '_label', label)
 
