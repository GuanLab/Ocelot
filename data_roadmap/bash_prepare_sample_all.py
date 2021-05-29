import os
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac','DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac','H3K23me2','H2AK9ac','H3T11ph','H4K12ac']
tier4=['methyl','RNA-seq']

# for training
the_seed=0

path0='./bigwig_all/'
for the_assay in tier0+tier1+tier2+tier3+tier4[:1]:
    for i in range(1,130):
        the_cell='E%03d' % i
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
for the_assay in tier0+tier1+tier2+tier3+tier4[:1]:
    for i in range(1,130):
        the_cell='E%03d' % i
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


