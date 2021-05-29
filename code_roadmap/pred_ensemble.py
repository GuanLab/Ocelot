import os
import sys
import numpy as np
import pyBigWig
import time

chr_all=['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
#num_bp=[248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895]
num_bp=[249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560]

chr_len={}
for i in np.arange(len(chr_all)):
    chr_len[chr_all[i]]=num_bp[i]

chr_len_bin={}
for i in np.arange(len(chr_all)):
    chr_len_bin[chr_all[i]]=int(np.ceil(num_bp[i]/25.0))

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac','DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac']
#,'H3K23me2','H2AK9ac','H3T11ph','H4K12ac']
tier4=['methyl','RNA-seq']

path0='./pred_ensemble/'
os.system('mkdir -p ' + path0)

id_all=np.loadtxt('../data/id_all.txt','str')

for the_assay in tier0+tier1+tier2+tier3:
    for i in np.concatenate((np.arange(1,60), np.arange(61,64), np.arange(65,130))):
        the_cell='E%03d' % i
        the_id = the_cell + '_' + the_assay
        if the_id not in id_all:
            bw_output = pyBigWig.open(path0 + the_id + '.bigwig','w')
            bw_output.addHeader(list(zip(chr_all , np.ceil(np.array(num_bp)/25).astype('int').tolist())), maxZooms=0)
            bw_avg = pyBigWig.open('../data/bigwig_all/avg_' + the_assay + '.bigwig')
            bw_lgbm = pyBigWig.open('./lgbm_' + the_assay + '_all/pred/pred_' + the_id + '_0.bigwig')
            bw_unet = pyBigWig.open('./unet_' + the_assay + '_all/epoch03/pred_' + the_id + '_seed0.bigwig')
            for the_chr in chr_all:
                print(the_id, the_chr)
                pred = np.zeros(chr_len_bin[the_chr])
                pred += bw_avg.values(the_chr, 0, chr_len_bin[the_chr])
                pred += np.array(bw_lgbm.values(the_chr, 0, chr_len_bin[the_chr])) * 2.0
                pred += bw_unet.values(the_chr, 0, chr_len_bin[the_chr])
                pred = pred / 4.0
                # convert into bigwig format
                x = pred
                # pad two zeroes
                z=np.concatenate(([0],x,[0]))
                # find boundary
                starts=np.where(np.diff(z)!=0)[0]
                ends=starts[1:]
                starts=starts[:-1]
                vals=x[starts]
                if starts[0]!=0:
                    ends=np.concatenate(([starts[0]],ends))
                    starts=np.concatenate(([0],starts))
                    vals=np.concatenate(([0],vals))
                if ends[-1]!=chr_len_bin[the_chr]:
                    starts=np.concatenate((starts,[ends[-1]]))
                    ends=np.concatenate((ends,[chr_len_bin[the_chr]]))
                    vals=np.concatenate((vals,[0]))
                # write 
                chroms = np.array([the_chr] * len(vals))
                bw_output.addEntries(chroms, starts, ends=ends, values=vals)
            bw_output.close()
            bw_avg.close()
            bw_lgbm.close()
            bw_unet.close()




