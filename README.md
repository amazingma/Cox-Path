# Cox-Path: A biologically-informed multi-omics model for cancer survival prediction
![Cox-Path](https://github.com/amazingma/Cox-Path/blob/main/figures/Cox-Path.pdf)
## Introduction
Survival analysis is a crucial means for cancer prognosis, and accurate predictions will greatly benefit the clinical management of cancer patients. In the paper, we propose a new approach Cox-Path, a graph convolution network based on the pathway interaction network. Specifically, we use gene-pathway memberships and pathway-pathway interactions to construct a sparse neural network that is used to integrate different types of omics into a common space of pathways. Experiments demonstrate that our model improves the prediction of survival analysis of multiple cancers with interpretability.

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
