import os
import sys
import numpy as np
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

id_all=np.loadtxt('../../data_encode3/id_all.txt','str')

target_assay='XXX'
feature_assay=['YYY']
feature_done='ZZZ'.split('_')

cell_train_vali=[]
for i in np.arange(1,80):
    the_cell='C%03d' % i
    the_id = the_cell + '_' + target_assay
    if the_id not in id_all:
        continue;
    count=0
    for the_assay in feature_assay:
        the_id = the_cell + '_' + the_assay
        if the_id in id_all:
            count += 1
    if count != len(feature_assay):
        continue;
    cell_train_vali.append(the_cell)

print(cell_train_vali)

cell_done=[]
for the_feature in feature_done:
    feature_subset = [the_feature]
    for i in np.arange(1,80):
        the_cell='C%03d' % i
        the_id = the_cell + '_' + target_assay
        if the_id in id_all:
            continue;
        count=0
        for the_assay in feature_subset:
            the_id = the_cell + '_' + the_assay
            if the_id in id_all:
                count += 1
        if count != len(feature_subset):
            continue;
        cell_done.append(the_cell)

cell_done.sort()
print(cell_done)

cell_test_all=[]
for i in np.arange(1,80):
    the_cell='C%03d' % i
    the_id = the_cell + '_' + target_assay
    if the_id in id_all:
        continue;
    if the_cell in cell_done:
        continue;
    count=0
    for the_assay in feature_assay:
        the_id = the_cell + '_' + the_assay
        if the_id in id_all:
            count += 1
    if count != len(feature_assay):
        continue;
    cell_test_all.append(the_cell)

print(cell_test_all)


#for the_seed in np.arange(5):
#    print('seed =',the_seed)
the_seed = 0

num_test = len(cell_test_all)
num_parallel = 10
for aaa in np.arange(int(np.ceil(num_test/num_parallel))):
    start=num_parallel*aaa
    end=np.min(( num_test, num_parallel*(aaa+1) ))
    the_command=''
    for k in np.arange(start, end):
        cell_test = cell_test_all[k]
        the_command += 'time python pred25bp.py -t ' + target_assay + \
            ' -ct ' + cell_test + \
            ' -f ' + ' '.join(feature_assay) + \
            ' -c ' + ' '.join(cell_train_vali) + \
            ' -s ' + str(the_seed) + \
            ' | tee -a log_pred_seed' + str(the_seed) + '.txt &\n'
    the_command += 'wait'
    print(the_command)
    os.system(the_command)



