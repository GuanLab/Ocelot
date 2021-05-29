import pyBigWig
import os
import sys
import numpy as np
import scipy.stats

def score(x1,x2):
    mse=((x1-x2) ** 2.).mean()
    pearson=np.corrcoef((x1,x2))[0,1]
    y1=scipy.stats.rankdata(x1)
    y2=scipy.stats.rankdata(x2)
    spearman=np.corrcoef((y1,y2))[0,1]
    return mse,pearson,spearman


bw=pyBigWig.open('../../data/bigwig_all/gt_E017_H3K27ac.bigwig')
y=np.array(bw.values('chr21',0,1925196))
y=np.nan_to_num(y)

bw2=pyBigWig.open('/local/disk7/hyangl/2019/encode_imputation/data/chromimpute/E017-H3K27ac.imputed.pval.signal.bigwig')
tmp=np.array(bw2.intervals('chr21'))
x2=tmp[:,-1]
score(x2,y)
#(9.104614068852293, 0.7848514899208829, 0.2069477632153042)


bw0=pyBigWig.open('../../data/bigwig_cv2/gt_avg_H3K27ac.bigwig')
x0=np.array(bw0.values('chr21',0,1925196))
score(x0,y)
#(14.498669683035255, 0.586792976827146, 0.06696090346160606)



tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac','DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac','H3K23me2','H2AK9ac','H3T11ph','H4K12ac']
tier4=['methyl','RNA-seq']

id_all=np.loadtxt('../../data/id_all.txt','str')
os.system('mkdir -p log')

cell_test=[]
target_assay='H3K27ac'
feature_assay=[]
for the_assay in tier0:
    if the_assay != target_assay:
        feature_assay.append(the_assay)

cell_train_vali=[]
for i in range(2,130,2):
    the_cell='E%03d' % i
    if the_cell in cell_test:
        continue;
    the_id = the_cell + '_' + target_assay
    if the_id not in id_all:
        continue;
    count=0
    for the_assay in feature_assay:
        the_id = the_cell + '_' + target_assay
        if the_id in id_all:
            count += 1
    if count != len(feature_assay):
        continue;
    cell_train_vali.append(the_cell)

cell_test = 'E017'
num = len(cell_train_vali)
for i in range(num):
    j = (i+1) % num
    cell_train = cell_train_vali[i]
    cell_vali = cell_train_vali[j]
    if i == 0:
        x1 = np.load('pred25bp_' + cell_test + '_' + target_assay + '_' + cell_train + '_' + cell_vali + '_chr21.npy')
    else:    
        x1 += np.load('pred25bp_' + cell_test + '_' + target_assay + '_' + cell_train + '_' + cell_vali + '_chr21.npy')

x1=x1/float(num)

score(x1,y)
#(13.03954776473298, 0.687108229509144, 0.3338384004126672)

x=x1.copy()
index=x1>np.percentile(x1,99.5)
x[index]=x[index]*2
#(11.619153311684117, 0.682406826217157, 0.3338384004126672)


