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
pip install lightgbm
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

## Code of Ocelot and evaluation on the challenge data
* data_challenge
* code_challenge <br />


Reproducing all these imputations and evaluations requires considerable time even with super computing resources, we therefore also provide the processed data, trained models and predictions together with the reproducible scripts.
* 0a. [Ocelot - the challenge final submission](http://mitra.stanford.edu/kundaje/ic/round2/3393417/) or [npy format](https://guanfiles.dcmb.med.umich.edu/Ocelot/challenge_submission/)
* 0b. [Ensemble predictions without DNA](https://guanfiles.dcmb.med.umich.edu/Ocelot/ensemble_predictions_without_dna/)
* 1a. [Processed data](https://guanfiles.dcmb.med.umich.edu/Ocelot/processed_data/)
* 2a. [Trained lightGBM and neural network models and predictions](https://guanfiles.dcmb.med.umich.edu/Ocelot/models/)
* 2b. [Trained lightGBM and neural network models without DNA and predictions](https://guanfiles.dcmb.med.umich.edu/Ocelot/models_without_dna/)<br />


For benchmarking, predictions from Avocado and ChromImpute are also provided:
* 3a. [Avocado predictions](http://mitra.stanford.edu/kundaje/ic/avocado/)
* 4a. [ChromImpute models and predictions](https://guanfiles.dcmb.med.umich.edu/Ocelot/chromimpute/)


### Mapping between letter, id and histone mark in challenge
For simplicity, we map the epigeneic marks to captital letters as follows:
| letter |  id |      mark |
| ------ | --- | --------- |
| C      | M02 | DNase-seq |
| D      | M18 | H3K36me3  |
| E      | M17 | H3K27me3  |
| F      | M16 | H3K27ac   |
| G      | M20 | H3K4me1   |
| H      | M22 | H3K4me3   |
| I      | M29 | H3K9me3   |
| J      | M01 | ATAC-seq  |

For example, in the "CDEH_I" design, we used four marks (C, D, E, H) as cell type-specific features to predict mark I.

## Data processing and model building scripts for ENCODE3 imputation
* data_encode3
* code_encode3

## Data processing and model building scripts for Roadmap imputation
* data_roadmap 
* code_roadmap

## Code for SHAP analysis
* code_shap

## Code of Ocelot final submission to the ENCODE Imputation Challenge 
* [challenge solution](https://github.com/Hongyang449/ENCODE_imputation/tree/master/code_challenge)



