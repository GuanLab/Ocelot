import os
import sys
import numpy as np
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

id_all=np.loadtxt('../../data_encode3/id_all.txt','str')
for i in np.arange(1,4):
    os.system('mkdir -p epoch%02d' % i )
#sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g; s/the_lr=1e-3/the_lr=1e-4/g; s/model.summary()/#model.summary()/g' train.py > continue_train.py

target_assay='XXX'
feature_assay=['YYY']

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

for the_seed in np.arange(1):
    print('seed =',the_seed)
    k = 1
    the_command = 'time python train.py -t ' + target_assay + \
        ' -f ' + ' '.join(feature_assay) + \
        ' -c ' + ' '.join(cell_train_vali) + \
        ' -s ' + str(the_seed) + \
        ' | tee -a log_seed' + str(the_seed) + '.txt'
    print(the_command)
    os.system(the_command)
    os.system('cp weights_seed' + str(the_seed) + '.h5 epoch%02d' % k)
    for k in np.arange(2,4):
        the_command = 'time python continue_train.py -t ' + target_assay + \
            ' -f ' + ' '.join(feature_assay) + \
            ' -c ' + ' '.join(cell_train_vali) + \
            ' -s ' + str(the_seed) + \
            ' | tee -a log_seed' + str(the_seed) + '.txt'
        print(the_command)
        os.system(the_command)
        os.system('cp weights_seed' + str(the_seed) + '.h5 epoch%02d' % k)



