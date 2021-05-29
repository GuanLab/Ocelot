## Ocelot: Improved Epigenome Imputation Reveals Asymmetric Predictive Relationships Across Histone Modifications

Ocelot is a machine learning approach to impute epigenomes across tissues and cell types. 
It ranked first in the [ENCODE Imputation Challenge](https://www.synapse.org/#!Synapse:syn17083203/wiki/604197) with high accuracy on held-out prospective data.
Beyond high predictive performance, it offers a new way to investigate the cross-histone regulations based on large-scale epigenomics datasets.
Please contact (hyangl@umich.edu or gyuanfan@umich.edu) if you have any questions or suggestions.

![Figure1](figure/fig1.png?raw=true "Title")

---

## Installation
Git clone a copy of code:
```
git clone https://github.com/GuanLab/Ocelot.git
```
## Required dependencies

* [python](https://www.python.org) (3.6.5)
* [numpy](http://www.numpy.org/) (1.13.3). It comes pre-packaged in Anaconda.
* [pyBigWig](https://github.com/deeptools/pyBigWig) A package for quick access to and create of bigwig files.
```
conda install pybigwig -c bioconda
```
* [lightgbm](https://lightgbm.readthedocs.io/en/latest/index.html)(2.3.0) A gradient boosting tree-based algorithm with fast training speed and high efficienty.
```
conda install -c conda-forge lightgbm
```
* [tensorflow](https://www.tensorflow.org/) (1.14.0) A popular deep learning package.
```
conda install tensorflow-gpu
```
* [keras](https://keras.io/) (2.2.5) A popular deep learning package using tensorflow backend.
```
conda install keras
```

## Dataset
* [The ENCODE Imputation Challenge dataset](https://www.synapse.org/#!Synapse:syn18143300)
* [Ocelot imputation for the ENCODE3 histone mark dataset](https://guanfiles.dcmb.med.umich.edu/Ocelot/imputation_encode3/)
* [Ocelot imputation for the Roadmap histone mark dataset](https://guanfiles.dcmb.med.umich.edu/Ocelot/imputation_roadmap/)

## Data processing and model building scripts for ENCODE3 imputation
* data_encode3
* code_encode3

## Data processing and model building scripts for Roadmap imputation
* data_roadmap 
* code_roadmap

## Example code for SHAP analysis
* code_shape

## Original code for ENCODE Imputation Challenge
* [winning algorithmn](https://github.com/GuanLab/ENCODE_imputation)



