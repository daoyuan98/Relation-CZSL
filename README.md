# Relation-CZSL
This repo contains the official implementation of paper "Relation-aware Compositional Zero-shot Learning for Attribute-Object Pair Recognition".

## Setup
- Install [Anaconda](https://www.anaconda.com/products/individual).
- Create environment using the configuration provided in the repo: 
```
conda env create -n RCZL -f pytorch.yaml
```
- Activate environment.
```
conda activate RCZL
```

## Evaluation
The snapshots of trained models for evaluation can be downloaded here: https://drive.google.com/drive/folders/1aMN2rlf6LWujW3HVLgS_WE3z5hmcbnvD?usp=sharing

Then, run 
```
eval.sh 
```
to evaluate our model.
