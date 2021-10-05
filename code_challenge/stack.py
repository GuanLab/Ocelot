import os
import sys
import numpy as np
import glob

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len25={}
for i in np.arange(len(chr_all)):
    chr_len25[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))


print(sys.argv)
name_model=sys.argv[1]
path1 = './' + name_model + '/'
os.system('mkdir -p ' + path1)

dirs=glob.glob('./' + name_model + '_*')
dirs.sort()
print(dirs)

ids=glob.glob('./' + name_model + '_01/pred25bp_C*chr1.npy')
ids.sort()

for the_id in ids:
    the_id = the_id.split('/')[-1].split('_')[1]
    print(the_id)
    for the_chr in chr_all:
        pred = np.zeros(chr_len25[the_chr])
        for the_dir in dirs:
            pred = pred + np.load(the_dir + '/' + 'pred25bp_' + the_id + '_' + the_chr + '.npy')
        pred = pred / float(len(dirs))
        np.save(path1 + 'pred25bp_' + the_id + '_' + the_chr, pred)
 





