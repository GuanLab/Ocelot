import os
import sys
import numpy as np
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
#,'DNase']
tier2=['H3K4me2','H2A.Z','H3K79me2','H4K20me1']
tier3=['H2AK5ac','H2BK120ac','H2BK5ac','H3K18ac','H3K23ac',\
    'H3K4ac','H3K79me1','H4K8ac','H2BK12ac','H3K14ac',\
    'H4K91ac','H2BK15ac','H3K9me1','H2BK20ac','H3K56ac',\
    'H4K5ac']
#,'H3K23me2','H2AK9ac','H3T11ph','H4K12ac']

the_template='template_unet_XXX_all'
for the_assay in tier1+tier2+tier3:
    the_name = 'unet_' + the_assay + '_all'
    os.system('cp -r ' + the_template + ' ' + the_name)
    os.system("sed -e 's/XXX/" + the_assay + "/g' " + the_template + "/bash.py > " + the_name + "/bash.py") 
    os.system("sed -e 's/XXX/" + the_assay + "/g' " + the_template + "/bash_pred.py > " + the_name + "/bash_pred.py") 

the_template='template_lgbm_XXX_all'
for the_assay in tier1+tier2+tier3:
    the_name = 'lgbm_' + the_assay + '_all'
    os.system('cp -r ' + the_template + ' ' + the_name)
    os.system("sed -e 's/XXX/" + the_assay + "/g' " + the_template + "/bash.py > " + the_name + "/bash.py")
    os.system("sed -e 's/XXX/" + the_assay + "/g' " + the_template + "/bash_pred.py > " + the_name + "/bash_pred.py")


