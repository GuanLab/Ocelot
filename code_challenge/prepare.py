import os
import sys
import numpy as np
import time

##  D  E  F  G  H  I ##
## 18 17 16 20 22 29 ##

dict_char_assay={
    'C':'M02',
    'D':'M18',
    'E':'M17',
    'F':'M16',
    'G':'M20',
    'H':'M22',
    'I':'M29',
    'J':'M01'}

name_model = sys.argv[1]
name_template = sys.argv[2]
feature_assay,target_assay = name_model.split('_')
target_assay = dict_char_assay[target_assay]
feature_assay = [dict_char_assay[x] for x in list(feature_assay)]

id_all=np.loadtxt('../data_challenge/list_train_vali.txt','str')

cell_train_vali=[]
for i in np.arange(1,52):
    the_cell='C%02d' % i
    the_id = the_cell + target_assay
    if the_id not in id_all:
        continue;
    count=0
    for the_assay in feature_assay:
        the_id = the_cell + the_assay
        if the_id in id_all:
            count += 1
    if count != len(feature_assay):
        continue;
    cell_train_vali.append(the_cell)

dict_test={
    'C_D':['C19','C28','C39','C40'],
    'C_E':['C19','C28','C39','C40'],
    'C_F':['C19','C39','C40'],
    'C_G':['C19','C39','C40'],
    'C_H':['C19','C28','C39','C40'],
    'C_I':['C19','C28','C39','C40'],
    'CH_D':['C05','C06','C51'],
    'CH_E':['C05','C06','C22','C51'],
    'CH_F':['C06','C22','C51'],
    'CH_G':['C05','C51'],
    'CH_I':['C05','C51'],
    'CDEH_G':['C07'],
    'CDEH_I':['C07'],
    'DEFGHI_C':['C12'],
    'DGH_C':['C31'],
    'DGH_F':['C31'],
    'DGH_I':['C31'],
    'F_C':['C38'],
    'F_D':['C38'],
    'F_E':['C38'],
    'F_G':['C38'],
    'F_H':['C38'],
    'F_I':['C38'],
    'DFGHI_J':['C12'],
    'DGH_J':['C31'],
    'F_J':['C38']}

cell_test=dict_test[name_model]
#cell_test=[]
#for i in np.array([5,6,7,12,19,22,28,31,38,39,40,51]):
#    the_cell='C%02d' % i
#    the_id = the_cell + target_assay
#    if the_id in id_all:
#        continue;
#    count=0
#    for the_assay in feature_assay:
#        the_id = the_cell + the_assay
#        if the_id in id_all:
#            count += 1
#    if count != len(feature_assay):
#        continue;
#    cell_test.append(the_cell)

print('train-vali cell lines')
print(cell_train_vali)
print('test cell lines')
print(cell_test)

feature="'" + ("' '").join(feature_assay) + "'"
target="'" + target_assay + "'"
test="'" + ("' '").join(cell_test) + "'"

num=len(cell_train_vali)
for i in range(num):
    the_name = name_model + '_' + name_template + '_%02d' % (i+1)
    os.system(f'cp -r template_{name_template} {the_name}')
    cell1 = cell_train_vali[i % num]
    cell2 = cell_train_vali[(i+1) % num]
    os.system(f'sed -e "s/CXX/{cell1}/g; s/CYY/{cell2}/g; s/FEATURE/{feature}/g; s/TARGET/{target}/g; s/TEST/{test}/g" < template_{name_template}/bash.sh > {the_name}/bash.sh')
    os.system(f'chmod +x {the_name}/bash.sh')

os.system(f'sed -e "s/MODEL/{name_model}/g; s/TEMPLATE/{name_template}/g; s/NUM/{num}/g" < train_pred_template.sh >> train_pred.sh')


