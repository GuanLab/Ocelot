import os
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

path0='./bigwig_encode3/'

for the_assay in tier0+tier1+tier2+tier3:
    for i in range(1,80):
        the_cell='C%03d' % i
        the_file=path0 + the_cell + '_' + the_assay + '.bigwig'
        if os.path.isfile(the_file):
            print('subsample: ' + the_file)
            os.system('time python subsample_for_qn.py -i ' + the_file + ' -o ' + the_cell + '_' + the_assay + '.npy -rg grch38 &')
            time.sleep(30)
        else:
            print('empty entry: ' + the_file)


