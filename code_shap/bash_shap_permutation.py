import os
import sys
import numpy as np

##  D  E  F  G  H  I ##
## 18 17 16 20 22 29 ##

assay_all=['C','D','E','F','G','H','I']

for assay1 in assay_all:
    for assay2 in assay_all:
        if assay1 != assay2:
            the_command = 'python shap_permutation.py ' + assay1 + '_' + assay2 + ' &'
            print(the_command)
            os.system('sleep 20s')
            os.system(the_command)

