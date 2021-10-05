#!/bin/bash

## require pre-processing data_challenge
## 0. prepare 
python bash_prepare.py

#### Ocelot ####
## 1. train and predict
bash train_pred.sh

## 2. ensemble prediction
python bash_stack.py
bash bash_ensemble.sh
#####################

#### ChromImpute ####
cd chromimpute
python bash_chromimpute.py
cd ..
#####################

## 3. evaluation
## finish ocelot/chromimpute and download gold standard files before evaluation
python score_hg.py



