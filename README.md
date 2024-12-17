# Cox-Path: Biological Pathway-Informed Graph Neural Network for Cancer Survival Prediction
![Cox-Path](https://github.com/amazingma/Cox-Path/blob/main/figures/Cox-Path.svg)
## Introduction
Survival analysis is crucial for cancer prognosis, and accurate survival predictions can greatly benefit the clinical management of cancer patients. With the widespread application of deep learning and the decreasing cost of omics techniques, multi-omics data are increasingly being utilized to predict cancer prognoses more accurately. However, integrating different sources of omics data poses significant challenges, such as the high-dimensional small sample size problem and differences in omics modalities. To effectively integrate multi-omics data and make full use of the prior knowledge, in this paper we propose a new approach Cox-Path, a graph convolution network based on the pathway interaction network. Specifically, we use gene-pathway memberships and pathway-pathway interactions to construct a sparse neural network that is used to integrate different types of omics into a common space of pathways. Experiments demonstrate that our model improves the survival prediction across multiple cancers with interpretability.

## Getting Started
### 1. Clone the repo
```
git clone https://github.com/amazingma/Cox-Path.git
```
### 2. Create conda environment
```
conda env create --name coxpath --file=environment.yml
```

## Usage
### 1. Activate the created conda environment
```
source activate coxpath
```
### 2. Train the model
```
python train.py
```

## References
```
Teng Ma, Haochen Zhao, Qichang Zhao, and Jianxin Wang. 2024. Cox-Path: Biological Pathway-Informed Graph Neural Network for Cancer Survival Prediction. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics (BCB '24). Association for Computing Machinery, New York, NY, USA, Article 70, 1–6. https://doi.org/10.1145/3698587.3701397
```
