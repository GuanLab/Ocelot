import os
import sys
import numpy as np

assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']

# prepare a wig header file
# 1. prepare the converted wig files
os.systme('python convert_bigwig_into_wig.py')

# 2. Global distance between datasets
os.system('java -mx4000M -jar ChromImpute.jar ComputeGlobalDist CHALLENGE/CONVERTEDDATADIR table_train_vali.txt grch38sizes.txt CHALLENGE/DISTANCEDIR')

## with a=10, b=20
# 3. Generate the features
the_command=''
for the_assay in assay_all:
    the_command += 'java -mx4000M -jar ChromImpute.jar GenerateTrainData -a 10 -b 20 -d 0 CHALLENGE/CONVERTEDDATADIR CHALLENGE/DISTANCEDIR table_train_vali.txt grch38sizes.txt CHALLENGE/TRAINDATA1 ' + the_assay + ' &\n' 
the_command += 'wait'
os.system(the_command)

id_all=np.loadtxt('list_final.txt',dtype='str')

# 4. Generate the trained predictors for a specific mark in a specific sample type to be predicted
num_test = len(id_all)
num_parallel = 17
for aaa in np.arange(int(np.ceil(num_test/num_parallel))):
    start=num_parallel*aaa
    end=np.min(( num_test, num_parallel*(aaa+1) ))
    the_command=''
    for k in np.arange(start, end):
        the_id = id_all[k]
        the_cell = the_id[:3]
        the_assay = the_id[3:]
        the_command += 'java -mx4000M -jar ChromImpute.jar Train -a 10 -b 20 CHALLENGE/TRAINDATA1 table_train_vali.txt CHALLENGE/PREDICTORDIR1 ' + the_cell + ' ' + the_assay + ' &\n'
    the_command += 'wait'
    print(the_command)
    os.system(the_command)

# 5. Generate the imputed signal track
num_test = len(id_all)
num_parallel = 17
for aaa in np.arange(int(np.ceil(num_test/num_parallel))):
    start=num_parallel*aaa
    end=np.min(( num_test, num_parallel*(aaa+1) ))
    the_command=''
    for k in np.arange(start, end):
        the_id = id_all[k]
        the_cell = the_id[:3] 
        the_assay = the_id[3:]    
        the_command += 'java -mx4000M -jar ChromImpute.jar Apply CHALLENGE/CONVERTEDDATADIR CHALLENGE/DISTANCEDIR CHALLENGE/PREDICTORDIR1 table_train_vali.txt grch38sizes.txt CHALLENGE/OUTPUTDATA1 ' + \
        the_cell + ' ' + the_assay + ' &\n'
    the_command += 'wait'
    print(the_command)
    os.system(the_command)

