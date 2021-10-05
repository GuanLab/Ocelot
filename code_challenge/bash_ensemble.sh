#!/bin/bash

python ensemble_l4na.py C_D &
python ensemble_l2na.py C_E &
python ensemble_l2na.py C_F &
python ensemble_l.py C_G &
python ensemble_l2na.py C_H &
python ensemble_lna.py C_I &
wait

python ensemble_l2na.py CH_D &
python ensemble_l2na.py CH_E &
python ensemble_l4na.py CH_F &
python ensemble_l.py CH_G &
python ensemble_l2na.py CH_I &
wait

python ensemble_l.py CDEH_G &
python ensemble_l2a.py CDEH_I &
python ensemble_l4a_v2.py DEFGHI_C &
python ensemble_l4a_v2.py DGH_C &
python ensemble_l.py DGH_F &
python ensemble_l4a.py DGH_I &
wait

python ensemble_l4a_v2.py F_C &
python ensemble_l2a.py F_D &
python ensemble_lna.py F_E &
python ensemble_l.py F_G &
python ensemble_l.py F_H &
python ensemble_l4a.py F_I &
python ensemble_a.py C12M01 C31M01 C38M01 &
wait

