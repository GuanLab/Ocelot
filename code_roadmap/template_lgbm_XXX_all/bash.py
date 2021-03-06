import os
import sys
import numpy as np
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac','DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac','H3K23me2','H2AK9ac','H3T11ph','H4K12ac']
tier4=['methyl','RNA-seq']

id_all=np.loadtxt('../../data_roadmap/id_all.txt','str')

target_assay='XXX'
feature_assay=[]
for the_assay in tier0:
    if the_assay != target_assay:
        feature_assay.append(the_assay)

cell_train_vali=[]
for i in np.arange(1,130):
    the_cell='E%03d' % i
    the_id = the_cell + '_' + target_assay
    if the_id not in id_all:
        continue;
    count=0
    for the_assay in feature_assay:
        the_id = the_cell + '_' + target_assay
        if the_id in id_all:
            count += 1
    if count != len(feature_assay):
        continue;
    cell_train_vali.append(the_cell)

print(cell_train_vali)

for the_seed in np.arange(5):
    print('seed =',the_seed)
    the_command = 'time python train.py -t ' + target_assay + \
        ' -f ' + ' '.join(feature_assay) + \
        ' -c ' + ' '.join(cell_train_vali) + \
        ' -s ' + str(the_seed) + \
        ' | tee -a log_seed' + str(the_seed) + '.txt'
    print(the_command)
    os.system(the_command)



