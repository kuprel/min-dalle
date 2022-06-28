#!/bin/bash

_pip=$(command -v pip pip3)

$_pip install -r requirements.txt

mkdir -p pretrained

# download vqgan
git lfs install
git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384 ./pretrained/vqgan

# download dalle-mini and dalle mega
$_pip install wandb
wandb login
wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14 
