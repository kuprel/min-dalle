#!/bin/bash

pip install -r requirements.txt

mkdir -p pretrained/vqgan

# download vqgan
git lfs install
git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384 ./pretrained/vqgan
# download the flax model
curl https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/resolve/main/flax_model.msgpack -L --output ./pretrained/vqgan/flax_model.msgpack

# download dalle-mini and dalle mega
wandb login --anonymously
wandb artifact get --root=./pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
wandb artifact get --root=./pretrained/dalle_bart_mega dalle-mini/dalle-mini/mega-1-fp16:v14 
