# Relation-CZSL
This repo contains the official implementation of paper "Relation-aware Compositional Zero-shot Learning for Attribute-Object Pair Recognition".

## Requirements
- Ubuntu 18.04
- CUDA 10.0+
- A CUDA-compatible GPU with 8GB+ VRAM
- Anaconda/Miniconda
- 20GB HDD space

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
- For faster training and evaluation, download precomputed [features](https://drive.google.com/drive/folders/1w5yf8DPO-tAOSX3yAiXylH4qoaGgx8SC?usp=sharing), and place the file under `data/[mitstates|ut-zap50k]/`. Alternatively, remove the `--pre_feat` flag from `eval.sh` if you don't want to use the these features.
- Download the weights of models [here](https://drive.google.com/drive/folders/1aMN2rlf6LWujW3HVLgS_WE3z5hmcbnvD?usp=sharing).

## Evaluation
Modify variables at [line 4-8](https://github.com/daoyuan98/Relation-CZSL/blob/35a9a7b8ff8ab99658c56b152fb3391324a00a97/eval.sh#L4-L8) in `eval.sh` to a proper batch size (depending on the VRAM you have), the path to the model, and the path to the data etc.
Then run 
```
bash eval.sh
```

## Training
Training scripts will be provided as future updates.
