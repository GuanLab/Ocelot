import os
import sys
import numpy as np
import pandas as pd


path1='./bigwig_encode3/'
os.system('mkdir -p ' + path1)

mat=pd.read_csv('encode3_download.csv')

for j in range(1,mat.shape[1]):
    for i in range(mat.shape[0]):
        accession = mat.iloc[i,j]
        if accession != '---':
            print(accession)
            the_id = mat.iloc[i,0] + '_' + mat.columns[j]
            command = 'wget https://www.encodeproject.org/files/' + accession + '/@@download/' + accession + '.bigWig'
            os.system(command)
            command = 'mv ' + accession + '.bigWig ' + path1 + the_id + '.bigwig'
            os.system(command)






