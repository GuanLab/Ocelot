import os
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

# for training
the_seed=0

path0='./bigwig_all/'

for the_assay in tier0+tier1+tier2+tier3:
    for i in range(1,80):
        the_cell='C%03d' % i
        the_id = the_cell + '_' + the_assay
        the_file=path0 + the_id + '.bigwig'
        if os.path.isfile(the_file):
            print('prepare sample: ' + the_file)
            os.system('time python prepare_sample_all.py -i ' + the_id + ' -s ' + str(the_seed) + ' &')
            time.sleep(60)
        else:
            print('empty entry: ' + the_file)

# one-hot DNA sequence
os.system('time python prepare_sample_dna.py -i A -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i C -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i G -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i T -s ' + str(the_seed) + ' &')


# for validation
the_seed=1

path0='./bigwig_all/'

for the_assay in tier0+tier1+tier2+tier3:
    for i in range(1,80):
        the_cell='C%03d' % i
        the_id = the_cell + '_' + the_assay
        the_file=path0 + the_id + '.bigwig'
        if os.path.isfile(the_file):
            print('prepare sample: ' + the_file)
            os.system('time python prepare_sample_all.py -i ' + the_id + ' -s ' + str(the_seed) + ' &')
            time.sleep(60)
        else:
            print('empty entry: ' + the_file)

# one-hot DNA sequence
os.system('time python prepare_sample_dna.py -i A -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i C -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i G -s ' + str(the_seed) + ' &')
os.system('time python prepare_sample_dna.py -i T -s ' + str(the_seed) + ' &')


