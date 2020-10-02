# Relation-CZSL
This repo contains the official implementation of paper "Relation-aware Compositional Zero-shot Learning for Attribute-Object Pair Recognition".

## Requirements
- Ubuntu 18.04
- Python 3.6+
- CUDA 10.0
- A CUDA-compatible GPU with 8GB+ memory

## Setup
- Install [Anaconda](https://www.anaconda.com/products/individual).
- Create an Anaconda environment using the configuration `pytorch.yaml` provided in the repo: 
```
conda env create -n RCZL -f pytorch.yaml
```
- Activate the environment.
```
conda activate RCZL
```

## Data and Model Preparation
- Follow the instructions in [TMN](https://github.com/facebookresearch/taskmodularnets#prerequisites) to prepare the data.
- Download the weights of models [here](https://drive.google.com/drive/folders/1aMN2rlf6LWujW3HVLgS_WE3z5hmcbnvD?usp=sharing).

## Evaluation
Modify variables at [line 4-8](https://github.com/daoyuan98/Relation-CZSL/blob/35a9a7b8ff8ab99658c56b152fb3391324a00a97/eval.sh#L4-L8) in `eval.sh` to proper batch size, path to model and data etc.
Then run 
```
eval.sh
```

## Training
Training scripts will be provided in future updates.
