import os
import sys
import numpy as np
import time

tier0=['H3K4me1','H3K4me3','H3K36me3','H3K27me3','H3K9me3']
tier1=['H3K27ac','H3K9ac']
tier2=['H3K4me2','H2AFZ','H3K79me2','H4K20me1']
tier3=['H3F3A','H3K9me2']

# YYY is the row feature and XXX is the target
dict_map={}
dict_map['H3K4me1']=['H3K4me3','H3K36me3','H3K27me3']
dict_map['H3K4me3']=['H3K4me1','H3K36me3','H3K27me3']
dict_map['H3K36me3']=['H3K4me3','H3K27me3','H3K27ac']
dict_map['H3K27me3']=['H3K4me3','H3K36me3','H3K27ac']
dict_map['H3K9me3']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

dict_map['H3K27ac']=['H3K4me3','H3K36me3','H3K27me3']
dict_map['H3K9ac']=['H3K4me3','H3K27me3','H3K27ac']

dict_map['H3K4me2']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H2AFZ']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H3K79me2']=['H3K4me3','H3K36me3','H3K27ac']
dict_map['H4K20me1']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

dict_map['H3F3A']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']
dict_map['H3K9me2']=['H3K4me3','H3K36me3','H3K27me3','H3K27ac']

the_template='template_unet_YYY_XXX_all'
for the_X in tier0+tier1+tier2+tier3:
    for the_Y in dict_map[the_X]:
        the_name = 'unet_' + the_Y + '_' + the_X + '_all'
        os.system('cp -r ' + the_template + ' ' + the_name)
        os.system("sed -e 's/XXX/" + the_X + "/g; s/YYY/" + the_Y + "/g' " + the_template + "/bash.py > " + the_name + "/bash.py")
        os.system("sed -e 's/XXX/" + the_X + "/g; s/YYY/" + the_Y + "/g' " + the_template + "/bash_pred.py > " + the_name + "/bash_pred.py")

the_template='template_lgbm_YYY_XXX_all'
for the_X in tier0+tier1+tier2+tier3:
    for the_Y in dict_map[the_X]:
        the_name = 'lgbm_' + the_Y + '_' + the_X + '_all'
        os.system('cp -r ' + the_template + ' ' + the_name)
        os.system("sed -e 's/XXX/" + the_X + "/g; s/YYY/" + the_Y + "/g' " + the_template + "/bash.py > " + the_name + "/bash.py")
        os.system("sed -e 's/XXX/" + the_X + "/g; s/YYY/" + the_Y + "/g' " + the_template + "/bash_pred.py > " + the_name + "/bash_pred.py")


