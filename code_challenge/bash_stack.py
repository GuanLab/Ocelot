import os
import sys
import numpy as np
import glob

chr_all = ['chr' + str(i) for i in range(1, 23)] + ['chrX']

path1='./'

model_all=['C_D','C_E','C_F','C_G','C_H','C_I', \
    'CH_D','CH_E','CH_F','CH_G','CH_I', \
    'CDEH_G','CDEH_I',\
    'DEFGHI_C', \
    'DGH_C','DGH_F','DGH_I', \
    'F_C','F_D','F_E','F_G','F_H','F_I']

command = ''
for name_model in model_all:
    command += 'python stack.py ' + name_model + '_lgbm_v1 &\n'
    command += 'python stack.py ' + name_model + '_lgbm_v2 &\n'
    command += 'python stack.py ' + name_model + '_nn_v1 &\n'

command += 'wait'
print(command)
os.system(command)


