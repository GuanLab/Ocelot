import sys
import numpy as np
import pyBigWig
import scipy.stats

def mse(x1,x2):
    return ((x1-x2) ** 2.).mean()

def masked_mse(x1,x2,mask): #binary mask for e.g. gene regions
    return ((x1-x2) ** 2.).dot(mask)/mask.sum()

chr_all=['chr' + str(i) for i in range(1,23)] + ['chrX']
num_bp=np.array([248956422,242193529,198295559,190214555,181538259,170805979,159345973,145138636,138394717,133797422,135086622,133275309,114364328,107043718,101991189,90338345,83257441,80373285,58617616,64444167,46709983,50818468,156040895])
num_25bp=np.ceil(num_bp/25.0).astype('int')

chr_len=dict(zip(chr_all,num_bp.tolist()))
chr_len25=dict(zip(chr_all,num_25bp.tolist()))

weight=num_25bp/float(np.sum(num_25bp))

# masked mse for promoter, gene, and enhancer regions
dict_mask1={}
dict_mask2={}
dict_mask3={}
for the_chr in chr_all[20:21]:
    bw=pyBigWig.open('../data_challenge/anno/prom.bigwig')
    dict_mask1[the_chr]=np.array(bw.values(the_chr,0,chr_len25[the_chr]))
    bw.close()
    bw=pyBigWig.open('../data_challenge/anno/gene.bigwig')
    dict_mask2[the_chr]=np.array(bw.values(the_chr,0,chr_len25[the_chr]))
    bw.close()
    bw=pyBigWig.open('../data_challenge/anno/enh.bigwig')
    dict_mask3[the_chr]=np.array(bw.values(the_chr,0,chr_len25[the_chr]))
    bw.close()


the_cell='51'
the_assay='M29'
the_id = 'C' + the_cell + the_assay
the_chr='chr21'

dict_var={}
dict_var[the_chr]=np.load('example/var_' + the_assay + '_' + the_chr + '.npy')

## load data
mat=np.zeros((2, chr_len25[the_chr]))
# 0.gt
mat[0,:]=np.load('example/C51M29_chr21.npy')
# 1.ocelot prediction
mat[1,:]=np.load('example/pred25bp_C51M29_chr21.npy')

## order for extra metrics
mat_argsort=np.zeros((2, chr_len25[the_chr]),dtype=int)
mat_rank=np.zeros((2, chr_len25[the_chr]))
for i in range(mat.shape[0]):
    mat_argsort[i,:] = np.argsort(mat[i,:])
    mat_rank[i,:] = scipy.stats.rankdata(mat[i,:])

## score
cutoff1 = int(chr_len25[the_chr] * 0.01)
cutoff5 = int(chr_len25[the_chr] * 0.05)
for i in range(1,mat.shape[0]):
    # 01.mse
    mseglobal = mse(mat[0,:],mat[i,:])
    # 02.pearson
    pear = np.corrcoef((mat[0,:],mat[i,:]))[0,1]
    # 03.spearman
    spear = np.corrcoef((mat_rank[0,:],mat_rank[i,:]))[0,1]
    # 04.mse prom
    mseprom = masked_mse(mat[0,:],mat[i,:],dict_mask1[the_chr])
    # 05.mse gene
    msegene = masked_mse(mat[0,:],mat[i,:],dict_mask2[the_chr])
    # 06.mse enh
    mseenh = masked_mse(mat[0,:],mat[i,:],dict_mask3[the_chr])
    # 07.msevar
    msevar = ((mat[0,:] - mat[i,:]) ** 2).dot(dict_var[the_chr])/dict_var[the_chr].sum()
    # 08.mse1obs
    index = mat_argsort[0,-cutoff1:]
    mse1obs = mse(mat[0,index], mat[i,index])
    # 09.mse1imp
    index = mat_argsort[i,-cutoff1:]
    mse1imp = mse(mat[0,index], mat[i,index])
    ## extra metrics based on chromimpute paper
    # 10.match1 overlapping top 1% obs vs 1% imp
    match1 = np.intersect1d(mat_argsort[0,-cutoff1:], mat_argsort[i,-cutoff1:]).shape[0] / cutoff1 
    # 11.catch1obs overlapping top 1% obs vs 5% imp
    catch1obs = np.intersect1d(mat_argsort[0,-cutoff1:], mat_argsort[i,-cutoff5:]).shape[0] / cutoff1
    # 12.catch1imp overlapping top 5% obs vs 1% imp
    catch1imp = np.intersect1d(mat_argsort[0,-cutoff5:], mat_argsort[i,-cutoff1:]).shape[0] / cutoff1
    print('scores for Ocelot prediction of C51M29 on chromosome 21:')
    print("Pearson's correlation = %.4f" % pear)
    print("Spearson's correlation = %.4f" % spear)
    print('MSEglobal = %.4f' % mseglobal)
    print('MSEProm = %.4f' % mseprom)
    print('MSEGene = %.4f' % msegene)
    print('MSEEnh = %.4f' % mseenh)
    print('MSEvar = %.4f' % msevar)
    print('MSE1obs = %.4f' % mse1obs)
    print('MSE1imp = %.4f' % mse1imp)
    print('Match1 = %.4f' % match1)
    print('Catch1obs = %.4f' % catch1obs)
    print('Catch1imp = %.4f' % catch1imp)


