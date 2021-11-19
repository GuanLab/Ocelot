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

## challenge test
assay_all=['M01','M02','M16','M17','M18','M20','M22','M29']
dict_assay_cell={}
dict_assay_cell['M01']=['12','31','38']
dict_assay_cell['M02']=['12','31','38']
dict_assay_cell['M16']=['06','19','22','31','39','40','51']
dict_assay_cell['M17']=['05','06','19','22','28','38','39','40','51']
dict_assay_cell['M18']=['05','06','19','28','38','39','40','51']
dict_assay_cell['M20']=['05','07','19','38','39','40','51']
dict_assay_cell['M22']=['19','28','38','39','40']
dict_assay_cell['M29']=['05','07','19','28','31','38','39','40','51']

# download bigwig files of ground truth, avocado submission, ocelot submission 
# download gt: http://mitra.stanford.edu/kundaje/ic/blind/
path0='../data_challenge/final_gt/'
# download avocado: http://mitra.stanford.edu/kundaje/ic/avocado/
path1='./avocado/'
# chromimpute
path2='./chromimpute/CHALLENGE/OUTPUTDATA0/'
# download ocelot submission: http://mitra.stanford.edu/kundaje/ic/round2/3393417/
path3='./ocelot/'

list_out=[]
list_out.append(open('score_avocado.txt','w'))
list_out.append(open('score_chromimpute.txt','w'))
list_out.append(open('score_ocelot.txt','w'))

for i in range(len(list_out)):
    list_out[i].write('id\tmse\tpear\tspear\tmseprom\tmsegene\tmseenh\tmsevar\tmse1obs\tmse1imp\tmatch1\tcatch1obs\tcatch1imp\n')

# masked mse
mask1_hg=np.zeros(0)
mask2_hg=np.zeros(0)
mask3_hg=np.zeros(0)
for the_chr in chr_all:
    bw=pyBigWig.open('../data_challenge/anno/prom.bigwig')
    mask1_hg = np.concatenate((mask1_hg, np.array(bw.values(the_chr,0,chr_len25[the_chr]))))
    bw.close()
    bw=pyBigWig.open('../data_challenge/anno/gene.bigwig')
    mask2_hg = np.concatenate((mask2_hg, np.array(bw.values(the_chr,0,chr_len25[the_chr]))))
    bw.close()
    bw=pyBigWig.open('../data_challenge/anno/enh.bigwig')
    mask3_hg = np.concatenate((mask3_hg, np.array(bw.values(the_chr,0,chr_len25[the_chr]))))
    bw.close()

num_metric = 12
num_test = 51

for the_assay in assay_all:
    var_hg=np.zeros(0)
    for the_chr in chr_all:
        var_hg=np.concatenate((var_hg, np.load('../data_challenge/var_final/var_' + the_assay + '_' + the_chr + '.npy')))
    for the_cell in dict_assay_cell[the_assay]:
        the_id = 'C' + the_cell + the_assay
        print(the_id)
        bw0 = pyBigWig.open(path0 + the_id + '.bigwig')
        bw1 = pyBigWig.open(path1 + the_id + '.bigwig')
        bw3 = pyBigWig.open(path3 + the_id + '.bigwig')
        ## load data
        mat=np.zeros((len(list_out)+1, np.sum(num_25bp)))
        start=0
        for k,the_chr in enumerate(chr_all):
            print(the_chr)
            end = start + chr_len25[the_chr]
            # 0.gt
            x=np.array(bw0.values(the_chr,0,chr_len[the_chr]))
            x[np.isnan(x)]=0 # fill nan with 0
            tmp=np.zeros(int(np.ceil(len(x)/25.0)*25 - len(x)))
            x=np.concatenate((x,tmp))
            x=x.reshape((-1,25)).T
            mat[0,start:end]=np.mean(x,axis=0)
            # 1.avocado - challenge submission
            x=np.array(bw1.values(the_chr,0,chr_len[the_chr]))
            x[np.isnan(x)]=0
            tmp=np.zeros(int(np.ceil(len(x)/25.0)*25 - len(x)))
            x=np.concatenate((x,tmp))
            x=x.reshape((-1,25)).T
            mat[1,start:end]=np.mean(x,axis=0)
            # 2.chromimpute
            mat[2,start:end]=np.loadtxt(path2 + the_chr + '_impute_' + 'C' + the_cell + '_' + the_assay + '.wig', skiprows=2)
            # 3.ocelot - challenge submission 
            x=np.array(bw3.values(the_chr,0,chr_len[the_chr]))
            x[np.isnan(x)]=0
            tmp=np.zeros(int(np.ceil(len(x)/25.0)*25 - len(x)))
            x=np.concatenate((x,tmp))
            x=x.reshape((-1,25)).T
            mat[3,start:end]=np.mean(x,axis=0)
            start += chr_len25[the_chr]

        ## order for extra metrics
        mat_argsort=np.zeros((len(list_out)+1, np.sum(num_25bp)),dtype=int)
        mat_rank=np.zeros((len(list_out)+1, np.sum(num_25bp)))
        for i in range(mat.shape[0]):
            mat_argsort[i,:] = np.argsort(mat[i,:])
            mat_rank[i,:] = scipy.stats.rankdata(mat[i,:])
        
        ## score
        cutoff1 = int(np.sum(num_25bp) * 0.01)
        cutoff5 = int(np.sum(num_25bp) * 0.05)
        for i in range(1,mat.shape[0]):
            # 01.mse
            mseglobal = mse(mat[0,:],mat[i,:])
            # 02.pearson
            pear = np.corrcoef((mat[0,:],mat[i,:]))[0,1]
            # 03.spearman
            spear = np.corrcoef((mat_rank[0,:],mat_rank[i,:]))[0,1]
            # 04.mse prom
            mseprom = masked_mse(mat[0,:],mat[i,:],mask1_hg)
            # 05.mse gene
            msegene = masked_mse(mat[0,:],mat[i,:],mask2_hg)
            # 06.mse enh
            mseenh = masked_mse(mat[0,:],mat[i,:],mask3_hg)
            # 07.msevar
            msevar = ((mat[0,:] - mat[i,:]) ** 2).dot(var_hg)/var_hg.sum()
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
            list_out[i-1].write('%s\t' % the_id)
            list_out[i-1].write('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (mseglobal, pear, spear, mseprom, msegene, mseenh, msevar, mse1obs, mse1imp, match1, catch1obs, catch1imp))
            list_out[i-1].flush()

        bw0.close()
        bw1.close()
        bw3.close()
        del mat
        del mat_argsort
        del mat_rank

## close
for i in range(len(list_out)):
    list_out[i].close()


